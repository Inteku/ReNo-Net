import numpy as np
import torch
import matplotlib.pyplot as plt
import Datasets




dataset = Datasets.Noise_Reduction_Dataset()
D = dataset.__len__()
naughty_list = []
for i in range(D):
    sus = dataset.__getitem__(i)
    if torch.sum(torch.isnan(sus)) > 0:
        print(i)
        naughty_list.append(i)

#np.save('naughty_list', naughty_list)
'''

item1 = dataset.__getitem__(0)[1]
img = torch.zeros((D*4,280))
for sample in range(0,D*4-4,4):
    for inp in range(1,5):
        img[sample+inp,:] = dataset.__getitem__(sample//4)[inp]

plt.imshow(img, aspect='auto')
plt.colorbar()
plt.show()
'''

