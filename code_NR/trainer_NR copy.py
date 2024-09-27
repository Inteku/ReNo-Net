import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import Datasets
import NR_Networks as net
from ConvAutoEncoder import Mel_AutoEncoder_280
import NR_Networks
from utilities import print_batch_loss
import timeit
#from NR_validation import validate
import random as rnd
import math
import ConvAutoEncoder
from datetime import date
import os

print('\033c')
batch_size = 1
learning_rate = 5e-5
# load decoder
DEC = torch.load('AE_BA_trained_280')
DEC.decode_only()
def repeater(x):
    return x
DEC = repeater

#initialize nn model
load = False#bool(int(input('Load trained Network? (0/1) >')))
train = True#bool(int(input('Train Network? (0/1) >')))
if train:
    epochs = int(input('How many epochs? >'))
test = False#bool(int(input('Test Network? (0/1) >'))) 

save_dir = os.getcwd() + '/Denoiser_saves/'
today_date = date.today().strftime("%d%m%y")
exist_counter = 1
while os.path.exists(save_dir+today_date):
     today_date = date.today().strftime("%d%m%y")+'('+str(exist_counter)+')'
     exist_counter += 1
os.makedirs(save_dir+today_date, exist_ok=False)
savename = 'DN3_trainedonDR_' + today_date


if load:
    NN = torch.load('best_NN42')
else:
    #NN = net.Learnable_Baseline()
    NN = net.Denoiser_3()

#import sets
dr = "/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/DR-S1L_candidate"
#dr = None
train_set = Datasets.Noise_Reduction_Dataset(train=True, shuffle_seed=555, dereverber=dr)
valid_set = Datasets.Noise_Reduction_Dataset(validate=True, shuffle_seed=555, dereverber=dr)
test_set = Datasets.Noise_Reduction_Dataset(test=True, shuffle_seed=555, dereverber=dr)
#valid_set = torch.utils.data.Subset(test_set, list(range(294)))
#test_set = torch.utils.data.Subset(test_set, list(range(294,test_set.__len__())))
num_batches = math.ceil(train_set.__len__()/batch_size)

#loss and optimizer
if train:
    NN.train()
    optimizer = optim.Adam(NN.parameters(), lr = learning_rate)
    #scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters = epochs, power = 2)

print(len(train_set))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle = False)
loss_function = nn.MSELoss()
#AE = torch.load('AE_BA_trained_280')
#AE.decode_only()

avg_losses = []
val_losses = []
lmbe_losses = []


print('\n\033[47m\033[30m Computing loss of untrained model \033[0m\n\nTrain set:\n')
running_loss = 0
c = 0
for idx, (sample, sim) in enumerate(train_loader):
    target = sample[:,0]
    inputs = sample[:,1:]
    Z_pred = NN(inputs, sim)
    loss = loss_function(DEC(Z_pred), DEC(target))
    running_loss += loss.item()
    print_batch_loss(idx, 0, num_batches, loss.item())
    c += 1
avg_losses.append(running_loss/c)
print('\n\nValidation set:\n')
running_loss = 0
c = 0
for idx, (sample, sim) in enumerate(valid_loader):
    target = sample[:,0]
    inputs = sample[:,1:]
    Z_pred = NN(inputs, sim)
    loss = loss_function(DEC(Z_pred), DEC(target))
    running_loss += loss.item()
    print_batch_loss(idx, 0, valid_set.__len__(), loss.item())
    c += 1
val_losses.append(running_loss/c)
 
previous_loss = 1000
g, r, y, en = '\033[1m\033[92m', '\033[1m\033[31m', '\033[1m\033[33m', '\033[0m'
color = g

alpha_progress = [1.2]
beta_progress = [0.2]

#training loop
if train:
    print('\n\n\033[47m\033[30m TRAINING \033[0m\n')
    print(f'Average loss will be {g}BETTER{en} or {r}WORSE{en} than the previous\n')

      
    start = timeit.default_timer()
    for e in range(epochs):
        NN.train()
        running_loss = 0
        for idx, (sample, sim) in enumerate(train_loader):

            target = sample[:,0]
            inputs = sample[:,1:]
            optimizer.zero_grad()
            Z_pred = NN(inputs, sim)
            loss = loss_function(DEC(Z_pred), DEC(target))
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            print_batch_loss(idx, e+1, num_batches, loss.item())

        ####
        #print('\n')
        #for name, param in NN.named_parameters():
        #    if param.requires_grad:
        #        print(f'{name} = {param.data:.5f}')
        #        if name == "alpha":
        #             alpha_progress.append(param.data.clone())
        #        if name == "beta":
        #             beta_progress.append(param.data.clone())
        #print(alpha_progress)

        avrg_loss = float((running_loss/num_batches))
        avg_losses.append(avrg_loss)
        before_lr = optimizer.param_groups[0]['lr']
        #scheduler.step()
        after_lr = optimizer.param_groups[0]['lr']

        if avrg_loss < previous_loss:
                color = g
        elif avrg_loss > previous_loss:
                color = r
        elif avrg_loss == previous_loss:
                color = y

        #validation
        NN.eval()
        running_val_loss = 0
        running_lmbe_loss = 0
        c = 0
        for idx, (sample, sim) in enumerate(valid_loader):

            target = sample[:,0]
            inputs = sample[:,1:]
            optimizer.zero_grad()
            Z_pred = NN(inputs, sim)
            val_loss = loss_function(DEC(Z_pred), DEC(target))

            running_val_loss += val_loss.item()
            c+=1
        
        val_losses.append(running_val_loss/c)
        lmbe_losses.append(running_lmbe_loss/c)


        print(end='\x1b[2K')

        new_best_tag = ''
        if val_losses[-1] == min(val_losses):
             new_best_tag = '*'
        torch.save(NN, save_dir+today_date+'/'+savename+'_'+str(e+1)+new_best_tag)

        print(f"Epoch {e+1}: {color}{avrg_loss:.8e}{en}\nValidation loss: \033[1m{val_losses[e]:.8e}{new_best_tag}{en}")
        print(f'η: {before_lr:.4e} → {after_lr:.4e}\n')
        previous_loss = avrg_loss

    stop = timeit.default_timer()

    print(f'\nRuntime: {((stop-start)/60):.6f} min\nAvg. runtime per Epoch: {((stop-start)/epochs):.6f}s\n')


NN.eval()
#test
'''
if test:
    print('\n\033[47m\033[30m TESTING \033[0m\n\n')
    NN.eval()
    c=0
    num_batches2 = math.ceil(test_set.__len__()/batch_size)

    running_loss = 0
    running_lmbe_loss = 0
    for idx, inputs in enumerate(test_loader):
        c+=1

        exp = inputs[5][0].item()
        seq = inputs[5][1].item()
        spk = inputs[5][2].item()

        Z_pred = NN(inputs[0], inputs[1], inputs[2], inputs[3],psi(exp, spk))
        test_loss = loss_function(DEC(Z_pred), DEC(inputs[4]))

        recon_pred = AE(Z_pred).squeeze(0)
        ref_data = np.load('/home/student/BA_Michael/work/LMBE_ref/exp_'+str(exp)+'/speaker_'+str(spk)+'/LMBE_'+str(seq)+'.npy')
        lmbe_ref = torch.Tensor(ref_data).unsqueeze(0).narrow(2, 0, 312)
        lmbe_loss = loss_function(recon_pred,lmbe_ref)
        running_lmbe_loss += lmbe_loss.item()
        
        running_loss += test_loss.item()

        print_batch_loss(idx, 1, num_batches2, test_loss)

    avrg_loss = running_loss/c
    avrg_lmbe_loss = running_lmbe_loss/c

    if avrg_loss < previous_loss:
        color = '\033[1m\033[92m'
    elif avrg_loss > previous_loss:
        color = '\033[1m\033[31m'
    elif avrg_loss == previous_loss:
        color = '\033[1m\033[33m'

    print(end='\x1b[2K')
    print('Test Loss: '+str(avrg_loss))
    print('Baseline test Loss: '+str(BL_loss()[0]))
    print('LMBE test loss: '+str(avrg_lmbe_loss))
    print('Baseline LMBE test Loss: '+str(BL_loss()[1]))
    previous_loss = avrg_loss
print('DONE')
'''


if train:
    plt.figure()
    plt.plot(range(epochs+1), avg_losses)
    plt.plot(range(epochs+1), val_losses)
    plt.legend(['Training loss', 'Validation loss'])
    plt.ylim(bottom=0)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(savename)
    plt.show()
#for trainable baseline only
   # plt.figure()
   # plt.plot(range(epochs+1), alpha_progress, color='r')
   # plt.plot(range(epochs+1), beta_progress, color='y')
   # plt.legend(['alpha', 'beta'])
   # plt.show()


#val_data = test_set.__getitem__(rnd.randint(0,test_set.__len__()-1))
#validate(val_data[5][0],val_data[5][1],val_data[5][2],NN(val_data[0],val_data[1],val_data[2],val_data[3],psi(inputs[5][0].item(),inputs[5][2].item())))


