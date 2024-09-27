import os
import numpy as np
import torch
import ID_Datasets
import matplotlib.pyplot as plt
#for root, dirs, files in os.walk('/home/student/BA_Michael/work/ID_data/NR_components/test/'):
 #   if not files and dirs[0][0:3] == 'seq':
#        for seq in range(4):
#            a=0
            #print(root+'/'+dirs[seq])

#i = np.load('/home/student/BA_Michael/work/ID_data/NR_components/test/3729/exp_189/seq_0/speaker_idx.npy')

#print(i)

#best_nn1 = torch.load('best_NN3')
#torch.save(best_nn1.state_dict(), 'best_NN3_dict')

NN1 = torch.load('best_NN1')
NN1.eval()

# load dataset
clean_set = ID_Datasets.clean_data(test=True)
nn1_set = ID_Datasets.NR_data(NN1)
dataset = nn1_set

S = nn1_set.__len__()
im = torch.zeros((S,280))
ids = torch.zeros((S))

for s in range(S):
    bn,id,_ = dataset.__getitem__(s)
    im[s, :] = bn.squeeze()
    print(f'Target speaker ID of sample {s}: ', id.detach().numpy())
    #ids[s] = id.squeeze()

#sorted_ids = torch.argsort(ids)
#im = im[sorted_ids]

plt.imshow(im.detach().numpy())
plt.show()
