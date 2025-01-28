#train the SID for K cycles
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import timeit
from math import ceil
from IdentificationDataset import CleanID, FeatureFusionID, DereverberID, SingleNodeID
from IdentificationDataset2 import NoisyID
import SpeakerID
from utilities import print_batch_loss, accuracy
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
print('\033c')
g, r, y, en = '\033[1m\033[92m', '\033[1m\033[31m', '\033[1m\033[33m', '\033[0m'
yellow, orange, purple, lightpurple, coral = (1, 0.7, 0), (1, 0.4, 0), (0.8, 0, 1), (0.8, 0.6, 1), (1, 0, 0.4)


dr1_version = "siamese"
dr2_version = "siamese"
dr1_path = "/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/050724_siam1Layer_noDEC/DR_CAE-encoding/DR_cae_ep15*"
dr2_path = "/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/050724_siam2Layer_noDEC/DR_CAE-encoding/DR_cae_ep18*"
#load data
'''
train_set = CleanID(train=True)
val_set_clean = CleanID(validation=True)
val_set_ff = FeatureFusionID(validation=True)
val_set_dr1 = DereverberID(validation=True, model_path=dr1_path, num_layers=1, model_version=dr1_version)
val_set_dr2 = DereverberID(validation=True, model_path=dr2_path, num_layers=2, model_version=dr2_version)

test_set_clean = CleanID(test=True)
test_set_ff = FeatureFusionID(test=True)
test_set_dr1 = DereverberID(test=True, model_path=dr1_path, num_layers=1, model_version=dr1_version)
test_set_dr2 = DereverberID(test=True, model_path=dr2_path, num_layers=2, model_version=dr2_version)
'''

train_set = CleanID(train=True)#NoisyID(clean=True, train=True)
val_set_clean = NoisyID(clean=True, validation=True)
val_set_ff = NoisyID(noiseReductionMethod='baseline', dereverberationMethod='baseline', validation=True)
val_set_dr1 = NoisyID(noiseReductionMethod='baseline', dereverberationMethod='neural net', validation=True)
val_set_dr2 = NoisyID(noiseReductionMethod='neural net', dereverberationMethod='neural net', validation=True)
test_set_clean = NoisyID(clean=True, test=True)
test_set_ff = NoisyID(noiseReductionMethod='baseline', dereverberationMethod='baseline', test=True)
test_set_dr1 = NoisyID(noiseReductionMethod='baseline', dereverberationMethod='neural net', test=True)
test_set_dr2 = NoisyID(noiseReductionMethod='neural net', dereverberationMethod='neural net', test=True)


batch_size = 16
num_batches = ceil(train_set.__len__()/batch_size)
valdata_size = val_set_clean.__len__()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader_clean = torch.utils.data.DataLoader(val_set_clean, batch_size=val_set_clean.__len__(), shuffle=False)
val_loader_ff = torch.utils.data.DataLoader(val_set_ff, batch_size=val_set_ff.__len__(), shuffle=False)
val_loader_dr1 = torch.utils.data.DataLoader(val_set_dr1, batch_size=val_set_dr1.__len__(), shuffle=False)
val_loader_dr2 = torch.utils.data.DataLoader(val_set_dr2, batch_size=val_set_dr2.__len__(), shuffle=False)

test_loader_clean = torch.utils.data.DataLoader(test_set_clean, batch_size=test_set_clean.__len__(), shuffle=False)
test_loader_ff = torch.utils.data.DataLoader(test_set_ff, batch_size=test_set_ff.__len__(), shuffle=False)
test_loader_dr1 = torch.utils.data.DataLoader(test_set_dr1, batch_size=test_set_dr1.__len__(), shuffle=False)
test_loader_dr2 = torch.utils.data.DataLoader(test_set_dr2, batch_size=test_set_dr2.__len__(), shuffle=False)

criterion = nn.CrossEntropyLoss()
epochs_of_early_stopping = []
early_stopping_threshold = 3.45e-3


epochs = int(input('Number of epochs?\n'))

global_tloss = np.asarray([0.0] * epochs)
global_vloss_clean = np.asarray([0.0]*(epochs+1))
global_vloss_ff = np.asarray([0.0]*(epochs+1))
global_vloss_dr1 = np.asarray([0.0]*(epochs+1))
global_vloss_dr2 = np.asarray([0.0]*(epochs+1))
global_accuracy_clean = np.asarray([0.0]*(epochs+1))
global_accuracy_ff = np.asarray([0.0]*(epochs+1))
global_accuracy_dr1 = np.asarray([0.0]*(epochs+1))
global_accuracy_dr2 = np.asarray([0.0]*(epochs+1))

K = 1
clean_box = []
ff_box = []
dr1_box = []
dr2_box = []

for k in range(K):
    #load network
    ID = SpeakerID.Identifier_v1()
    optimizer = optim.Adam(ID.parameters(), lr=1e-4)

    val_loss_progress_clean = np.asarray([0.0]*(epochs+1))
    accuracy_progress_clean = np.asarray([0.0]*(epochs+1))
    for idx, (v_inputs, v_targets) in enumerate(val_loader_clean):
        v_outputs = ID(v_inputs)
        #print(v_outputs.size(), v_targets.size())
        #print(v_outputs.dtype, v_targets.dtype)
        untrained_vloss_clean = criterion(v_outputs.squeeze(), v_targets).item()
        val_loss_progress_clean[0] = untrained_vloss_clean
        accuracy_progress_clean[0] = accuracy(v_outputs, v_targets)


    val_loss_progress_ff = np.asarray([0.0]*(epochs+1))
    accuracy_progress_ff = np.asarray([0.0]*(epochs+1))
    for idx, (v_inputs, v_targets) in enumerate(val_loader_ff):
        v_outputs = ID(v_inputs)
        untrained_vloss_ff = criterion(v_outputs.squeeze(), v_targets).item()
        val_loss_progress_ff[0] = untrained_vloss_ff
        accuracy_progress_ff[0] = accuracy(v_outputs, v_targets)



    val_loss_progress_dr1 = np.asarray([0.0]*(epochs+1))
    accuracy_progress_dr1 = np.asarray([0.0]*(epochs+1))
    for idx, (v_inputs, v_targets) in enumerate(val_loader_dr1):
        v_outputs = ID(v_inputs)
        untrained_vloss_dr1 = criterion(v_outputs.squeeze(), v_targets).item()
        val_loss_progress_dr1[0] = untrained_vloss_dr1
        accuracy_progress_dr1[0] = accuracy(v_outputs, v_targets)



    val_loss_progress_dr2 = np.asarray([0.0]*(epochs+1))
    accuracy_progress_dr2 = np.asarray([0.0]*(epochs+1))
    for idx, (v_inputs, v_targets) in enumerate(val_loader_dr2):
        v_outputs = ID(v_inputs)
        untrained_vloss_dr2 = criterion(v_outputs.squeeze(), v_targets).item()
        val_loss_progress_dr2[0] = untrained_vloss_dr2
        accuracy_progress_dr2[0] = accuracy(v_outputs, v_targets)



    train_loss_progress = np.asarray([0.0]*epochs)

    # training loop
    training_achieved_100 = False
    early_stopping_achieved = False
    previous_loss = 1000.0
    
    print(f'\n\033[47m\033[30m {k+1} / {K} \033[0m\n')
    print(f'Average loss will be {g}BETTER{en} or {r}WORSE{en} than the previous\n')
    for e in range(epochs):
        train_running_loss = 0
        for idx, (inputs, target) in enumerate(train_loader):
            output = ID(inputs)
            #print(output.size(), target.size())
            #print(output.dtype, target.dtype)
            Loss = criterion(output.squeeze(), target)

            train_running_loss += Loss.item()
            Loss.backward()
            optimizer.step()
            print_batch_loss(idx, e+1, num_batches, Loss.item())

        avrg_loss = train_loss_progress[e] = train_running_loss/num_batches

        delta_loss = avrg_loss - previous_loss
        if delta_loss < 0:
            color = g
        elif delta_loss > 0:
            color = r
        elif delta_loss == 0:
            color = y

        

        print(end='\x1b[2K')
        print(f"Run {k+1}, Epoch {e+1}: {color}{avrg_loss}\033[0m\n")
        previous_loss = avrg_loss

        for idx, (v_inputs, v_targets) in enumerate(val_loader_clean):
            v_outputs = ID(v_inputs)
            val_loss_progress_clean[e+1] = criterion(v_outputs.squeeze(), v_targets).item()
            accuracy_progress_clean[e+1] = accuracy(v_outputs, v_targets)

        for idx, (v_inputs, v_targets) in enumerate(val_loader_ff):
            v_outputs = ID(v_inputs)
            val_loss_progress_ff[e+1] = criterion(v_outputs.squeeze(), v_targets).item()
            accuracy_progress_ff[e+1] = accuracy(v_outputs, v_targets)

        for idx, (v_inputs, v_targets) in enumerate(val_loader_dr1):
            v_outputs = ID(v_inputs)
            val_loss_progress_dr1[e+1] = criterion(v_outputs.squeeze(), v_targets).item()
            accuracy_progress_dr1[e+1] = accuracy(v_outputs, v_targets)

        for idx, (v_inputs, v_targets) in enumerate(val_loader_dr2):
            v_outputs = ID(v_inputs)
            val_loss_progress_dr2[e+1] = criterion(v_outputs.squeeze(), v_targets).item()
            accuracy_progress_dr2[e+1] = accuracy(v_outputs, v_targets)


        print(f'Validation loss: {val_loss_progress_clean[e+1]}\n')

        if e>50 and abs(delta_loss)<early_stopping_threshold and not early_stopping_achieved:
            early_stopping_achieved = True
            print(f'\nEarly Stopping at Epoch {e+1}')
            epochs_of_early_stopping.append(e+1)

            for idx, (t_inputs, t_targets) in enumerate(test_loader_clean):
                t_outputs = ID(t_inputs)
                clean_box.append(accuracy(t_outputs, t_targets))
            for idx, (t_inputs, t_targets) in enumerate(test_loader_ff):
                t_outputs = ID(t_inputs)
                ff_box.append(accuracy(t_outputs, t_targets))
            for idx, (t_inputs, t_targets) in enumerate(test_loader_dr1):
                t_outputs = ID(t_inputs)
                dr1_box.append(accuracy(t_outputs, t_targets))
            for idx, (t_inputs, t_targets) in enumerate(test_loader_dr2):
                t_outputs = ID(t_inputs)
                dr2_box.append(accuracy(t_outputs, t_targets))

            #print(1)

            #break
        '''
        if accuracy_progress_clean[e+1] == 1 and not training_achieved_100:
            print(2)
            print(f'k={k}, e+1={e+1}, len(ff_box)={len(ff_box)}, len(accpro_ff)={len(accuracy_progress_ff)}')
            training_achieved_100 = True
            ff_box[k] = accuracy_progress_ff[e+1]
            dr1_box[k] = accuracy_progress_dr1[e+1]
            dr2_box[k] = accuracy_progress_dr2[e+1]
        '''

    global_tloss += train_loss_progress
    global_vloss_clean += val_loss_progress_clean
    global_vloss_ff += val_loss_progress_ff
    global_vloss_dr1 += val_loss_progress_dr1
    global_vloss_dr2 += val_loss_progress_dr2


    global_accuracy_clean += accuracy_progress_clean
    global_accuracy_ff += accuracy_progress_ff
    global_accuracy_dr1 += accuracy_progress_dr1
    global_accuracy_dr2 += accuracy_progress_dr2

date = '050724'
if False:
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_L_train', global_tloss)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_L_sing', global_vloss_clean)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_L_ff', global_vloss_ff)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_L_clea', global_vloss_dr1)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_L_dr', global_vloss_dr2)

    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_A_clean', global_accuracy_clean)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_A_ff', global_accuracy_ff)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_A_clea', global_accuracy_dr1)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_A_dr', global_accuracy_dr2)

    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_Box_sing', clean_box)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_Box_ff', ff_box)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_Box_clea', dr1_box)
    np.save('/home/student/Documents/BA/id_multitrainer_saves/'+date+'/100_Box_dr', dr2_box)



plt.figure()
plt.boxplot([clean_box, ff_box, dr1_box, dr2_box])
plt.title('Accuracy Distribution')
plt.legend(['Clean', 'FF', 'DR 1Layer', 'DR 2Layer'])
plt.ylim(bottom=0)
plt.show()


plt.figure()
t = list(range(epochs+1))
plt.subplot(2,1,1)
plt.plot(t[:-1], global_tloss/K, color='k')
plt.plot(t, global_vloss_clean/K, color='g')
plt.plot(t, global_vloss_ff/K, color=orange)
plt.plot(t, global_vloss_dr1/K, color=purple)
plt.plot(t, global_vloss_dr2/K, color=lightpurple)
plt.vlines(x=epochs_of_early_stopping, ymin=0, ymax=1.7)
plt.title('Loss')
plt.legend(['Training', 'Clean Valid.', 'FF Valid.', 'DR 1L Valid.', 'DR 2L Valid.'])
plt.ylim(bottom=0)

plt.subplot(2,1,2)
plt.plot(t, global_accuracy_clean/K, color='g') 
plt.plot(t, global_accuracy_ff/K, color=orange)
plt.plot(t, global_accuracy_dr1/K, color=purple) 
plt.plot(t, global_accuracy_dr2/K, color=lightpurple) 
plt.plot(t, [1]*(epochs+1), color=(0,.3,0), linestyle='dashed')
plt.plot(t, [1/6]*(epochs+1), color=(.7,0,0), linestyle='dashed')
plt.title('Accuracy')
plt.legend(['Clean', 'FF', 'DR 1Lyr', 'DR 2Lyr'])
plt.ylim(bottom=0)
plt.show()