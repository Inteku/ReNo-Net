from xml.etree.ElementTree import TreeBuilder
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import SoundDatasets as set
import ConvAutoEncoder
import lstm_autoencoder_test
from functions import print_batch_loss, inverse_lmbe
import timeit
import torchaudio
import matplotlib.pyplot as plt
import tqdm
import os
from datetime import date

# Prompts:
# reset network? (0/1)
# 0: saved parameters will be loaded to the model
# 1: new model will be initialized and saved to new file after training
# train network? (0/1)
# 0: skips training
# 1: does training
# test network? (0/1)
# 0: skips testing
# 1: does testing

# INITIALIZING

print('\033c')
reset = bool(int(input('Reset network? (0/1) >')))
train = bool(int(input('Train network? (0/1) >')))
subset = bool(int(input('Use small Subset for testing/training? (0/1) >')))
test = bool(int(input('Test network? (0/1)  >')))
epochs = 3
if train:
 epochs = int(input('Number of epochs? >'))
device = 'cpu'
g, r, y, en = '\033[1m\033[92m', '\033[1m\033[31m', '\033[1m\033[33m', '\033[0m'

loadname = 'AE_BA_trained_LOG_19'
save_dir = os.path.dirname(os.getcwd()) + '/AE_savestates/'
today_date = date.today().strftime("%d%m%y")
exist_counter = 1
while os.path.exists(save_dir+today_date):
     today_date = date.today().strftime("%d%m%y")+'('+str(exist_counter)+')'
     exist_counter += 1
os.makedirs(save_dir+today_date, exist_ok=False)
savename = 'AEr2_' + today_date


# LOADING DATA # todo: richtiges test und trainset einfügen
train_set = set.ReverbLMBEsDataset(train=True)
test_set = set.ReverbLMBEsDataset(test=True)
valid_set = torch.utils.data.Subset(test_set, list(range(100))) #small validation set

if subset:
    train_set = torch.utils.data.Subset(train_set, list(range(100)))
    test_set = torch.utils.data.Subset(test_set, list(range(30)))

batch_size = 30
num_batches = train_set.__len__()//batch_size 

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

# AUTOENCODER SETUP

if reset:
    #AE = ConvAutoEncoder.Mel_AutoEncoder_280_regularized_2()
    AE = lstm_autoencoder_test.LSTMAutoencoder(input_size=128, cell_size=280, num_layers=1)
else:
    #AE = Mel_AutoEncoder_final()
    #AE.load_state_dict(torch.load(loadname)) #AE_BA_trained2 ist am besten trainiert allerdings mit falscher SR in trainingsdaten
    AE = torch.load(loadname)


# TRAINING
start = timeit.default_timer()

criterion = nn.MSELoss()
previous_loss = 1000.0

if train:   
    optimizer = optim.Adam(AE.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.LinearLR(optimizer)
    
    color = ''

    print('\n\033[47m\033[30m TRAINING \033[0m\n')
    print(f'Average loss will be {g}BETTER{en} or {r}WORSE{en} than the previous\n')
    

    loss_progress = []
    val_loss_progress = []
    for e in range(epochs):
        AE.train()
        running_loss = 0
        for idx, inputs in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = AE(inputs)

            train_loss = criterion(outputs, inputs)
            running_loss += train_loss.item()
            train_loss.backward()
            optimizer.step()
            
            #torch.nn.utils.clip_grad_norm.
            print_batch_loss(idx, e+1, num_batches, train_loss)

        #validation
        AE.eval()
        running_val_loss = 0
        c = 0
        for idx, inputs in enumerate(valid_loader):
            outputs = AE(inputs)
            val_loss = criterion(outputs,inputs)
            running_val_loss += val_loss
            c+=1
        val_loss_progress.append(float((running_val_loss/c)))
        
        loss_progress.append(float((running_loss/num_batches)))

        before_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        after_lr = optimizer.param_groups[0]['lr']

        if loss_progress[-1] < previous_loss:
            color = g
        elif loss_progress[-1] > previous_loss:
            color = r
        elif loss_progress[-1] == previous_loss:
            color = y

        print(end='\x1b[2K')

        new_best_tag = ''
        if val_loss_progress[-1] == min(val_loss_progress):
             new_best_tag = '*'
        #torch.save(AE, save_dir+today_date+'/'+savename+'_'+str(e+1)+new_best_tag)


        print(f"Epoch {e+1}: {color}{loss_progress[-1]}\033[0m\n"+ ' lr: '+str(before_lr)+' -> '+str(after_lr)+'\n validation loss: '+str(val_loss_progress[e]))
        previous_loss = loss_progress[-1]



stop = timeit.default_timer()

# TESTING

if test:
    print('\n\033[47m\033[30m TESTING \033[0m\n\n')
    AE.eval()
    num_batches = test_set.__len__()//batch_size 

    running_loss = 0
    for idx, inputs in enumerate(test_loader):
        outputs = AE(inputs)

        test_loss = criterion(outputs, inputs)
        test_loss = test_loss.item()
        running_loss += test_loss

        print_batch_loss(idx, 1, num_batches, test_loss)        

    avrg_loss = running_loss/num_batches

    if avrg_loss < previous_loss:
        color = '\033[1m\033[92m'
    elif avrg_loss > previous_loss:
        color = '\033[1m\033[31m'
    elif avrg_loss == previous_loss:
        color = '\033[1m\033[33m'

    print(end='\x1b[2K')
    print(f"Epoch 1: {color}{avrg_loss}\033[0m\n")
    previous_loss = avrg_loss

print('DONE')
if train:
    print(f'\nRuntime: {((stop-start)/60):.6f} min\nAvg. runtime per Epoch: {((stop-start)/epochs):.6f}s\n')



# VISUAL COMPARISON #übrig vom Praxisprojekt

AE.eval()
in_img = test_set.__getitem__(4)

print(f'input max: {in_img.max()}')
in_img_npy = in_img.detach().numpy().squeeze()


out_img = AE(in_img)#.detach().cpu()
print(f'output max: {out_img.max():.5f}')

out_img_npy = out_img.detach().numpy().squeeze()
print(f'loss: {criterion(out_img, in_img.unsqueeze(0)):.8f}')

if train:
    print(f'\nRuntime: {((stop-start)/60):.6f} min\nAvg. runtime per Epoch: {((stop-start)/epochs):.6f}s\n')


plt.figure()
plt.subplot(2,1,1)
plt.imshow(in_img_npy, origin='lower')
plt.xlabel('λ')
plt.ylabel('M')
#plt.title('Input LMBEs')
plt.colorbar()

plt.subplot(2,1,2)
plt.imshow(out_img_npy, origin='lower')
plt.title('Output LMBEs')
plt.colorbar()

plt.show()


if train:
    plt.figure()
    plt.plot(range(epochs), loss_progress)
    plt.plot(range(epochs),val_loss_progress)
    plt.legend(['Training loss', 'Validation loss'])
    plt.ylim(bottom=0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

# save decoded LMBE as soundfile to disk

#audio = inverse_lmbe(out_img.squeeze(0), 16e3)
#audio = torch.multiply(audio, 1/audio.max())

#torchaudio.save('decoded.wav', audio, int(16000))


