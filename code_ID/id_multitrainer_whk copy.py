#train the SID for K cycles
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import timeit
from math import ceil
from IdentificationDataset2 import NoisyID
import SpeakerID
from utilities import print_batch_loss, accuracy
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
print('\033c')
g, r, y, en = '\033[1m\033[92m', '\033[1m\033[31m', '\033[1m\033[33m', '\033[0m'
yellow, orange, brown, purple, lightpurple, coral = (1, 0.7, 0), (1, 0.4, 0), (0.5, 0.2, 0), (0.8, 0, 1), (0.8, 0.6, 1), (1, 0, 0.4)


#load data
train_set = NoisyID(clean=True, train=True)
val_set_list = []
val_set_list.append(NoisyID(clean=True, validation=True)) #clean val
val_set_list.append(NoisyID(validation=True, dereverberationMethod=None, noiseReductionMethod=None)) #noisy val
val_set_list.append(NoisyID(validation=True, dereverberationMethod="baseline", noiseReductionMethod="baseline", alphabeta=[1.2,0.2])) #default baseline val
val_set_list.append(NoisyID(validation=True, dereverberationMethod="baseline", noiseReductionMethod="baseline", alphabeta=[0.246,-0.754])) #optimal baseline val
val_set_list.append(NoisyID(validation=True, dereverberationMethod="baseline", noiseReductionMethod="neural net")) #neural NR val
val_set_list.append(NoisyID(validation=True, dereverberationMethod="neural net", noiseReductionMethod="neural net")) #neural DR + neural NR val

val_set_names = ["clean val.", "noisy val.", "baseline val.", "baseline+ val.", "Denoiser val.", "Dereverber+Denoiser val."]
val_set_colors = ['g', brown, orange, yellow, lightpurple, coral]
num_val_sets = len(val_set_list)

test_set_list = []
test_set_list.append(NoisyID(clean=True, test=True)) #clean test
test_set_list.append(NoisyID(test=True, dereverberationMethod=None, noiseReductionMethod=None)) #noisy test
test_set_list.append(NoisyID(test=True, dereverberationMethod="baseline", noiseReductionMethod="baseline", alphabeta=[1.2,0.2])) #default baseline test
test_set_list.append(NoisyID(test=True, dereverberationMethod="baseline", noiseReductionMethod="baseline", alphabeta=[0.246,-0.754])) #optimal baseline test
test_set_list.append(NoisyID(test=True, dereverberationMethod="baseline", noiseReductionMethod="neural net")) #neural NR test
test_set_list.append(NoisyID(test=True, dereverberationMethod="neural net", noiseReductionMethod="neural net")) #neural DR + neural NR test

test_set_names = ["clean", "noisy", "baseline", "baseline+", "Denoiser", "Dereverber+Denoiser"]
num_test_sets = len(test_set_list)

batch_size = 16
num_batches = ceil(train_set.__len__()/batch_size)
valdata_size = val_set_list[0].__len__()
testdata_size = test_set_list[0].__len__()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader_list = []
for val_set in val_set_list:
    val_loader_list.append(torch.utils.data.DataLoader(val_set, batch_size=valdata_size, shuffle=False))

test_loader_list = []
for test_set in test_set_list:
    test_loader_list.append(torch.utils.data.DataLoader(test_set, batch_size=testdata_size, shuffle=False))

criterion = nn.CrossEntropyLoss()
epochs_of_early_stopping = []
early_stopping_threshold = 3.45e-3


epochs = int(input('Number of epochs?\n'))

global_tloss = np.asarray([0.0] * epochs)

global_vloss_list = [np.asarray([0.0]*(epochs+1))] * num_val_sets

global_accuracy_list = [np.asarray([0.0]*(epochs+1))] * num_val_sets

K = 2

box_list = [[]] * num_test_sets

for k in range(K):
    #load network
    ID = SpeakerID.Identifier_v1()
    optimizer = optim.Adam(ID.parameters(), lr=0.00002)

    val_loss_progress_list = [np.asarray([0.0]*(epochs+1))] * num_val_sets
    accuracy_progress_list = [np.asarray([0.0]*(epochs+1))] * num_val_sets
    print(accuracy_progress_list[0])

    for v in range(num_val_sets):
        for idx, (v_inputs, v_targets) in enumerate(val_loader_list[v]):
            v_outputs = ID(v_inputs)
            print(v_inputs.size(), v_outputs.size(), v_targets.size())
            untrained_vloss = criterion(v_outputs.squeeze(), v_targets).item()
            val_loss_progress_list[v][0] += untrained_vloss
            accuracy_progress_list[v][0] += accuracy(v_outputs.squeeze(), v_targets)


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
            Loss = criterion(output, target)

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


        for v in range(num_val_sets):
            for idx, (v_inputs, v_targets) in enumerate(val_loader_list[v]):
                v_outputs = ID(v_inputs)
                val_loss_progress_list[v][e+1] = criterion(v_outputs, v_targets.squeeze()).item()
                accuracy_progress_list[v][e+1] = accuracy(v_outputs, v_targets)


        print(f'Validation loss: {val_loss_progress_list[0][e+1]}\n')

        if e>50 and abs(delta_loss)<early_stopping_threshold and not early_stopping_achieved:
            early_stopping_achieved = True
            print(f'\nEarly Stopping at Epoch {e+1}')
            epochs_of_early_stopping.append(e+1)

            for t in range(len(test_loader_list)):
                for idx, (t_inputs, t_targets) in enumerate(test_loader_list[t]):
                    t_outputs = ID(t_inputs)
                    box_list[t].append(accuracy(t_outputs, t_targets))

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
    for v in range(num_val_sets):
        global_vloss_list[v] += val_loss_progress_list[v]
        global_accuracy_list[v] += accuracy_progress_list[v]


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
plt.boxplot(box_list)
plt.title('Accuracy Distribution')
plt.legend(test_set_names)
plt.ylim(bottom=0)
plt.show()


plt.figure()
t = list(range(epochs+1))
plt.subplot(2,1,1)
plt.plot(t[:-1], global_tloss/K, color='k')
for v in range(num_val_sets):
    plt.plot(t, global_vloss_list[v]/K, color=val_set_colors[v])

plt.vlines(x=epochs_of_early_stopping, ymin=0, ymax=1.7)
plt.title('Loss')
legend_names = val_set_names.insert(0, "Training")
plt.legend(legend_names)
plt.ylim(bottom=0)

plt.subplot(2,1,2)
for t_ in range(num_test_sets):
    plt.plot(t, global_accuracy_list[t_]/K, color=val_set_colors[t_]) 
plt.plot(t, [1]*(epochs+1), color=(0,.3,0), linestyle='dashed')
plt.plot(t, [1/6]*(epochs+1), color=(.7,0,0), linestyle='dashed')
plt.title('Accuracy')
plt.legend(val_set_names)
plt.ylim(bottom=0)
plt.show()