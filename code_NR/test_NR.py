import torch
import numpy as np
import matplotlib.pyplot as plt
import Datasets
from NR_Baseline import NR_baseline

dr = "/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/050824_siam1Layer_noDEC/DR_CAE-encoding/DR_cae_ep26*"
dr = "/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/DR-S1L_candidate"
#dr = None
testset = Datasets.Noise_Reduction_Dataset(test=True, shuffle_seed=555, dereverber=dr)
date = "110924(1)" #"260724(1)"
version = "DN3_trainedonDR_"
afterEpoch = 40
NR_NN = torch.load(f"/home/student/Documents/WHK_Projekt_1/code_NR/Denoiser_saves/{date}/{version}{date}_{afterEpoch}*")
for name, param in NR_NN.named_parameters():
    if param.requires_grad:
        print(f'{name} = {param.data}')
alpha = 0.246#0.9911#1.2
beta = -0.754#-0.0318#0.2
S = len(testset)
error_no_nr = [0]*S
error_baseline = [0]*S
error_nn = [0]*S
for s in range(S):
    # s == 1:
    #  continue
    sample, cos_sim = testset[s]
    y = sample[0]
    x = sample[1:]
    z_baseline = NR_baseline(x, alpha, beta, cos_sim)
    error_no_nr[s] = torch.nn.functional.mse_loss(y, x[0])
    error_baseline[s] = torch.nn.functional.mse_loss(y, z_baseline)
    error_nn[s] = torch.nn.functional.mse_loss(y, NR_NN(x.unsqueeze(0), cos_sim.unsqueeze(0)).squeeze()).item()

print(f"Test Loss\nno NR:     {sum(error_no_nr)/S:.4e}\nBaseline:  {sum(error_baseline)/S:.4e}\nDenoiser3: {sum(error_nn)/S:.4e}")
plt.figure()
#plt.ylim([0, 5])
plt.plot(error_no_nr)
plt.plot(error_baseline)
plt.plot(error_nn)
plt.legend(["No NR", "Baseline NR", "NN"])
plt.show()

sample, cos_sim = testset[5]
y = sample[0]
x = sample[1:]
ypred = NR_NN(x.unsqueeze(0), cos_sim.unsqueeze(0))
y = y.repeat(10,1)
x = x.detach().numpy()
ypred = ypred.repeat(10,1).detach().numpy()
plt.figure()
plt.subplot(1,3,1)
plt.imshow(x.T, aspect='auto')
plt.colorbar()
plt.title('Input')
plt.subplot(1,3,2)
plt.imshow(ypred.T, aspect='auto', vmin=0, vmax=1)
plt.colorbar()
plt.title('Prediction')
plt.subplot(1,3,3)
plt.imshow(y.T, aspect='auto')
plt.colorbar()
plt.title('Target')
plt.show()