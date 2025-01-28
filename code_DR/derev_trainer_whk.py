#DRM trainer with batch size 1
#change DRM parameters for different DRM structures
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import timeit
from math import ceil
from DereverbDataset import HighlevelNodeDataset
from SoundDatasets import CleanDataset
import Dereverb
#from ConvAutoEncoder_BA import Mel_AutoEncoder_280
import lstm_autoencoders
import ConvAutoEncoder
from utilities import print_batch_loss
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import os
print('\033c')
device = 'cpu'



g, r, y, en = '\033[1m\033[92m', '\033[1m\033[31m', '\033[1m\033[33m', '\033[0m'
criterion = nn.MSELoss()


#DR = Dereverb.ConvDereverberBest(feature_size=280)
DR = Dereverb.siameseDereverberMemb(feature_size=280, num_layers=1)


epochs = int(input("number of training epochs\n"))

#
#
#
#
#
 # Conv AE
#
#
#
#
#


date = '26-01-27_withDEC'
#DR = Dereverb.ConvDereverberBest(feature_size=280)
DR = Dereverb.siameseDereverberMemb(feature_size=280, num_layers=1)
#DR.load_state_dict(torch.load("/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/050724_siam2Layer_noDEC/DR_CAE-encoding/DR_cae_ep18*"))

# load data
dev_set = HighlevelNodeDataset(train=True)
devdata_size = len(dev_set)
traindata_size = int(devdata_size*0.8)
valdata_size = devdata_size - traindata_size
train_set, val_set = torch.utils.data.dataset.random_split(dev_set, [traindata_size, valdata_size])



#load Decoder

DEC = ConvAutoEncoder.Mel_AutoEncoder_280()
DEC.load_state_dict(torch.load('/home/student/Desktop/Code_BA_Intek/Trainierte Modelle/AE280_final'))
lmbe_set = CleanDataset(train=True)
print(len(lmbe_set))
DEC(lmbe_set[0])
DEC.eval()
DEC.decode_only()


#DEC = lambda x: x



optimizer = optim.Adam(DR.parameters(), lr=5e-5)

color = ''



print('\n\033[47m\033[30m BASELINES \033[0m\n')


torch.autograd.set_detect_anomaly(True)
a=0
#baseline
BL = Dereverb.Baseline()
ff_running_loss = 0


for idx in range(valdata_size):
    bl_data, bl_numInputs, exp_and_nodes = val_set[idx]
    bl_target = bl_data[0].unsqueeze(0)
    bl_inputs = bl_data[1:]

    exp = exp_and_nodes[0]
    node_list = exp_and_nodes[1]

    ff = BL.feature_fusion(exp, node_list, bl_inputs)

    ff_running_loss += criterion(DEC(ff), DEC(bl_target)).item()


ff_baseline = ff_running_loss / valdata_size

print(f'Baseline (FF):         {ff_baseline}\n')
#epoch 0
DR.eval()
#val
v_running_loss = 0
for idx in range(valdata_size):
    #ean: exp_and_nodes
    v_data, v_numInputs, v_ean = val_set[idx]
    v_target = v_data[0].unsqueeze(0)
    v_inputs = v_data[1:]
    v_running_loss += criterion(DEC(DR(v_inputs, v_ean).unsqueeze(0)), DEC(v_target)).item()

untrained_vloss = v_running_loss / valdata_size
print(f'Untrained net (val):   {untrained_vloss}\n')
#train
t_running_loss = 0
for idx in range(traindata_size):
    #ean: exp_and_nodes
    t_data, t_numInputs, t_ean = train_set[idx]
    t_target = t_data[0].unsqueeze(0)
    t_inputs = t_data[1:]
    t_running_loss += criterion(DEC(DR(t_inputs, t_ean).unsqueeze(0)), DEC(t_target)).item()

untrained_tloss = t_running_loss / traindata_size
print(f'Untrained net (train): {untrained_tloss}\n')

loss_progress2 = [0] * (epochs+1)
v_loss_progress2 = [0] * (epochs+1)
loss_progress2[0] = untrained_tloss
v_loss_progress2[0] = untrained_vloss

ff_progress2 = [ff_baseline] * (epochs+1)

os.makedirs(f'/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/{date}/DR_2025', exist_ok=True)
# training loop
start = timeit.default_timer()

previous_loss = 1000.0
best_vloss_ever = untrained_vloss
print('\n\033[47m\033[30m TRAINING: \033[0m\n')
print(f'Average loss will be {g}BETTER{en} or {r}WORSE{en} than the previous\n')
for e in range(epochs):
    #training
    DR.train()
    running_loss = 0
    shuffled_range = random.sample(list(range(traindata_size)), traindata_size)
    for idx in range(traindata_size):
        #if idx > 100:#
        #    break#
        data, numInputs, ean = train_set[shuffled_range[idx]]
        target = data[0].unsqueeze(0)
        inputs = data[1:]
        optimizer.zero_grad()
        #print(inputs.size(), target.size())
        outputs = DR(inputs, ean).unsqueeze(0)
        #print(outputs.size())
        train_loss = criterion(DEC(outputs), DEC(target))

        running_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
        #torch.nn.utils.clip_grad_norm()
        print_batch_loss(idx, e+1, traindata_size, train_loss.item())
    
    avrg_loss = loss_progress2[e+1] = running_loss/traindata_size

    if avrg_loss < previous_loss:
        color = g
    elif avrg_loss > previous_loss:
        color = r
    elif avrg_loss == previous_loss:
        color = y

    print(end='\x1b[2K')
    print(f"Epoch {e+1}: {color}{avrg_loss}\033[0m\n")
    previous_loss = avrg_loss


    #validation
    DR.eval()
    v_running_loss = 0
    for idx in range(valdata_size):
        v_data, v_numInputs, v_ean = val_set[idx]
        v_target = v_data[0].unsqueeze(0)
        v_inputs = v_data[1:]
        v_running_loss += criterion(DEC(DR(v_inputs, v_ean).unsqueeze(0)), DEC(v_target)).item()

    v_loss_progress2[e+1] = v_running_loss / valdata_size
    print(f'Validation loss: {v_loss_progress2[e+1]}\n')

    flag = ''
    if v_loss_progress2[e+1] < best_vloss_ever:
        best_vloss_ever = v_loss_progress2[e+1]
        flag = '*'
    torch.save(DR.state_dict(), f'/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/{date}/DR_siam1L_{str(e+1)}{flag}')

os.makedirs(f'/home/student/Documents/WHK_Projekt_1/code_DR/training_history/{date}', exist_ok=True)
np.save(f'/home/student/Documents/WHK_Projekt_1/code_DR/training_history/{date}/dr_train_progress_cae', loss_progress2)
np.save(f'/home/student/Documents/WHK_Projekt_1/code_DR/training_history/{date}/dr_val_progress_cae', v_loss_progress2)






stop = timeit.default_timer()
if epochs != 0:
    print(f'\nRuntime: {((stop-start)/60):.2f} min\nAvg. runtime per Epoch: {((stop-start)/epochs):.1f} s\n')


    t = list(range(epochs+1))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Erster Subplot
    axs[0].plot(t, loss_progress2, 'b')
    axs[0].plot(t, v_loss_progress2, color=(1,0.5,0))
    axs[0].plot(t, ff_progress2, 'g')
    axs[0].legend(['Training', 'Validation', 'Baseline (FF)'])
    axs[0].set_title('Conv Autoencoder')
    axs[0].set_xlabel('e')
    axs[0].set_ylabel('MSE')
    '''
    # Zweiter Subplot
    axs[1].plot(t, loss_progress, 'b')
    axs[1].plot(t, v_loss_progress, color=(1,0.5,0))
    axs[1].plot(t, ff_progress, 'g')
    axs[1].legend(['Training', 'Validation', 'Baseline (FF)'])
    axs[1].set_title('LSTM Autoencoder')
    axs[1].set_xlabel('e')
    axs[1].set_ylabel('MSE')

    # Dritter Subplot
    axs[2].plot(t, loss_progress3, 'b')
    axs[2].plot(t, v_loss_progress3, color=(1,0.5,0))
    axs[2].plot(t, ff_progress3, 'g')
    axs[2].legend(['Training', 'Validation', 'Baseline (FF)'])
    axs[2].set_title('ECAPA TDNN')
    axs[2].set_xlabel('e')
    axs[2].set_ylabel('Cosine Loss')
    '''
    # Layout anpassen
    plt.tight_layout()

    # Plot anzeigen
    plt.show()

