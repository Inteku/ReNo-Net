#DRM trainer with batch size 1
#change DRM parameters for different DRM structures
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import timeit
from math import ceil
from SoundDatasets import HighlevelNodeDataset, CleanDataset
import Dereverb
#from ConvAutoEncoder_BA import Mel_AutoEncoder_280
import lstm_autoencoders
from utilities import print_batch_loss
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
print('\033c')
device = 'cpu'

g, r, y, en = '\033[1m\033[92m', '\033[1m\033[31m', '\033[1m\033[33m', '\033[0m'
criterion = nn.MSELoss()

modelname = input("select model or type 'new' to create new model\n")

DR = Dereverb.ConvDereverber_v2()
if modelname == 'new':
    modelname = "new_model"   
else:
    DR.load_state_dict(torch.load(modelname))

epochs = int(input("number of training epochs\n"))

# load data
train_set = HighlevelNodeDataset(ae_name='AE-LSTMv4', train=True)
traindata_size = train_set.__len__()
val_set = torch.utils.data.Subset(HighlevelNodeDataset(ae_name='AE-LSTMv4', test=True), list(range(600,884)))
valdata_size = val_set.__len__()


#load Decoder
lmbe_set = torch.utils.data.Subset(CleanDataset(train=True), [0])
DEC = torch.load('LSTM_AE_v4_best', map_location='cpu')
DEC.set_decode(False)



optimizer = optim.Adam(DR.parameters(), lr=0.0005)

color = ''


if modelname == 'new_model':
    print('New model has been initialized with random parameters\n')
else:
    print('Pre-trained model has been loaded\n')

print('\n\033[47m\033[30m BASELINES \033[0m\n')


torch.autograd.set_detect_anomaly(True)
a=0
#baseline
BL = Dereverb.Baseline()
av_running_loss = 0
sn_running_loss = 0
ff_running_loss = 0
zr_running_loss = 0

for idx in range(valdata_size):
    bl_data, bl_numInputs, exp_and_nodes = val_set.__getitem__(idx)
    bl_target = bl_data[0].unsqueeze(0)
    bl_inputs = bl_data[1:]

    exp = exp_and_nodes[0]
    node_list = exp_and_nodes[1]
    a=bl_target
    av = BL.average(bl_inputs, bl_numInputs)
    sn = BL.single_node(bl_inputs)
    ff = BL.feature_fusion(exp, node_list, bl_inputs)

    av_running_loss += criterion(DEC(av), DEC(bl_target)).item()
    sn_running_loss += criterion(DEC(sn), DEC(bl_target)).item()
    ff_running_loss += criterion(DEC(ff), DEC(bl_target)).item()
    zr_running_loss += criterion(DEC(torch.zeros(1,280)), DEC(bl_target)).item()

av_baseline = av_running_loss / valdata_size
sn_baseline = sn_running_loss / valdata_size
ff_baseline = ff_running_loss / valdata_size
zr_baseline = zr_running_loss / valdata_size
print(f'Baseline (single):  {sn_baseline}\n')
print(f'Baseline (average): {av_baseline}\n')
print(f'Baseline (FF):      {ff_baseline}\n')
print(f'Zeros:              {zr_baseline}\n')

DR.eval()
v_running_loss = 0
for idx in range(valdata_size):
    #ean: exp_and_nodes
    v_data, v_numInputs, v_ean = val_set.__getitem__(idx)
    v_target = v_data[0].unsqueeze(0)
    v_inputs = v_data[1:]
    v_running_loss += criterion(DEC(DR(v_inputs, v_ean)), DEC(v_target)).item()
untrained_loss = v_running_loss / valdata_size
print(f'Untrained net:      {untrained_loss}\n')

loss_progress = [0] * (epochs)
v_loss_progress = [0] * (epochs+1)
v_loss_progress[0] = untrained_loss
av_progress = [av_baseline] * (epochs+1)
sn_progress = [sn_baseline] * (epochs+1)
ff_progress = [ff_baseline] * (epochs+1)
zr_progress = [zr_baseline] * (epochs+1)

# training loop
start = timeit.default_timer()

previous_loss = 1000.0
best_vloss_ever = untrained_loss
print('\n\033[47m\033[30m TRAINING: \033[0m\n')
print(f'Average loss will be {g}BETTER{en} or {r}WORSE{en} than the previous\n')
for e in range(epochs):
    #training
    DR.train()
    running_loss = 0
    shuffled_range = random.sample(list(range(traindata_size)), traindata_size)
    for idx in range(traindata_size):
        #if idx > 100:
        #    break
        data, numInputs, ean = train_set.__getitem__(shuffled_range[idx])
        target = data[0].unsqueeze(0)
        inputs = data[1:]
        optimizer.zero_grad()
        #print(inputs.size(), target.size())
        outputs = DR(inputs, ean)
        #print(outputs.size())
        train_loss = criterion(DEC(outputs), DEC(target))

        running_loss += train_loss.item()
        train_loss.backward()
        optimizer.step()
        #torch.nn.utils.clip_grad_norm()
        print_batch_loss(idx, e+1, traindata_size, train_loss.item())
    
    avrg_loss = loss_progress[e] = running_loss/traindata_size

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
        v_data, v_numInputs, v_ean = val_set.__getitem__(idx)
        v_target = v_data[0].unsqueeze(0)
        v_inputs = v_data[1:]
        v_running_loss += criterion(DEC(DR(v_inputs, v_ean)), DEC(v_target)).item()

    v_loss_progress[e+1] = v_running_loss / valdata_size
    print(f'Validation loss: {v_loss_progress[e+1]}\n')

    if v_loss_progress[e+1] < best_vloss_ever:
        best_vloss_ever = v_loss_progress[e+1]
        #torch.save(DR.state_dict(), '/home/student/Documents/BA/CDRMv2_saves_j/CDRMv2_2_'+str(e+1))


np.save('/home/student/Documents/BA/BA_prepost_t', loss_progress)
np.save('/home/student/Documents/BA/BA_prepost_v', v_loss_progress)

stop = timeit.default_timer()
if epochs != 0:
    print(f'\nRuntime: {((stop-start)/60):.2f} min\nAvg. runtime per Epoch: {((stop-start)/epochs):.1f} s\n')

    # save model
    savenew = input("Save trained model? (0/1)\n")
    #savenew = True
    if savenew != '0':
        new_modelname = input("Filename (type 0 to overwrite)\n")
        #new_modelname = 'DRv1_02'
        if new_modelname=='0':
            new_modelname = modelname
        torch.save(DR.state_dict(), new_modelname)

    #np.save('lossprogress_may4', loss_progress)
    #np.save('vallossprogress_may4', v_loss_progress)

    plt.figure()
    t = list(range(epochs+1))
    plt.plot(t[1:], loss_progress)
    plt.plot(t, v_loss_progress)
    plt.plot(t, av_progress)
    plt.plot(t, sn_progress)
    plt.plot(t, ff_progress)
    plt.plot(t, zr_progress)
    plt.ylim(bottom=0)
    plt.legend(['Training', 'Validation', 'Baseline (average)', 'Baseline (single)', 'Baseline (FF)', 'Zero'])
    plt.show()



