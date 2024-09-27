import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from SoundDatasets import ReverbLMBEsDataset
import timeit

cwd = os.getcwd()
dataset = ReverbLMBEsDataset(train=True)
S = 100#dataset.__len__()
print(S)
img = torch.zeros((S,280))

#MSE = torch.nn.MSELoss()

date = '131223'
modelnames = sorted(os.listdir(os.path.dirname(cwd) + '/AE_savestates/' + date))
num_models = len(modelnames)
#test_loss = np.zeros(num_models, dtype=float)
#variance = np.zeros(num_models, dtype=float)
print(modelnames)



for model in range(1):#,num_models):
    #ENC = torch.load(os.path.dirname(cwd) + '/AE_savestates/'+date+'/' + modelnames[model])
    #ENC.encode_only()
    #DEC = torch.load(os.path.dirname(cwd) + '/AE_savestates/'+date+'/' + modelnames[model])
    #DEC.decode_only()
    ENC = torch.load('LSTM_AE_transfer_best', map_location='cpu')
    ENC.set_decode(False)

    running_loss = 0.0
    for s in range(S):
        start = timeit.default_timer()
        lmbe = dataset.__getitem__(s)
        bn = ENC(lmbe).squeeze()
        img[s,:] = bn
       # running_loss += MSE(DEC(bn), lmbe).item()


        print(f'Model {model}, Sample {s}', end='\r')
        if timeit.default_timer() - start > 5:
            break
    #test_loss[model] = running_loss/S

    #running_variance = 0.0
    #for column in range(280):
    #    running_variance += torch.var(img[:,column])
    #variance[model] = running_variance/280

    best_state = 5
    if model == 0:#best_state-1:
        plt.imshow(img.detach().numpy(), aspect='auto')
        plt.title(f'AE state-{best_state}')
        plt.show()

#test_loss = test_loss / np.max(test_loss)
#variance = variance / np.max(variance)
#plt.plot(range(1,num_models+1), test_loss)
#plt.plot(range(1,num_models+1), variance)
#plt.legend(['Loss', 'Variance'])
#plt.show()

