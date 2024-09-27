import numpy as np
import matplotlib.pyplot as plt
import torch
import Datasets
from NR_Baseline import NR_baseline


trainset = Datasets.Noise_Reduction_Dataset(dereverber=None, shuffle_seed=555, train=True)
testset = Datasets.Noise_Reduction_Dataset(dereverber=None, shuffle_seed=555, test=True)
len_train = len(trainset)
len_test = len(testset)
'''
min_a = 0.6
max_a = 1.6
min_b = -0.4
max_b = 0.6
delta = 0.025

numsamples_a = int((max_a - min_a) / delta) + 1
numsamples_b = int((max_b - min_b) / delta) + 1

interval_a = np.linspace(min_a, max_a, numsamples_a)
interval_b = np.linspace(min_b, max_b, numsamples_b)

lossmap_train = np.zeros((numsamples_a, numsamples_b))
lossmap_test = np.zeros((numsamples_a, numsamples_b))

iterations = numsamples_a * numsamples_b
counter = 0
print("0.00%", end="\r")
for idx_a, a in enumerate(interval_a):
    for idx_b, b in enumerate(interval_b):
        
        loss = 0
        for sample_idx in range(len_train):
            sample, cos_sim = trainset[sample_idx]
            y = sample[0]
            x = sample[1:]
            z_baseline = NR_baseline(x, a, b, cos_sim)
            loss += torch.nn.functional.mse_loss(y, z_baseline)
        lossmap_train[idx_a, idx_b] = loss / len_train

        loss = 0
        for sample_idx in range(len_test):
            sample, cos_sim = testset[sample_idx]
            y = sample[0]
            x = sample[1:]
            z_baseline = NR_baseline(x, a, b, cos_sim)
            loss += torch.nn.functional.mse_loss(y, z_baseline)
        lossmap_test[idx_a, idx_b] = loss / len_test

        counter += 1
        print(f"{(counter/iterations*100):.2f}%", end="\r")

optimum_train = np.argmin(lossmap_train)
opt_a_train, opt_b_train = np.unravel_index(optimum_train, lossmap_train.shape)
opt_a_train = interval_a[opt_a_train]
opt_b_train = interval_b[opt_b_train]
print(f"\n\nLowest Train loss at:\nalpha = {opt_a_train}\nbeta = {opt_b_train}")
optimum_test = np.argmin(lossmap_test)
opt_a_test, opt_b_test = np.unravel_index(optimum_test, lossmap_test.shape)
opt_a_test = interval_a[opt_a_test]
opt_b_test = interval_b[opt_b_test]
print(f"\nLowest Test loss at:\nalpha = {opt_a_test}\nbeta = {opt_b_test}")

plt.figure()
plt.subplot(1,2,1)
plt.imshow(lossmap_train, extent=[min_a,max_a,min_b,max_b], origin="lower", cmap="plasma", vmax=0.0006)
plt.xlabel("α")
plt.ylabel("β")
plt.colorbar()
plt.title("Train set")
plt.subplot(1,2,2)
plt.imshow(lossmap_test, extent=[min_a,max_a,min_b,max_b], origin="lower", cmap="plasma", vmax=0.0006)
plt.xlabel("α")
plt.ylabel("β")
plt.colorbar()
plt.title("Test set")

plt.show()
'''
min_a = -0.5
max_a = 1.5
delta = 0.001
numsamples_a = int((max_a - min_a) / delta) + 1
interval_a = np.linspace(min_a, max_a, numsamples_a)

loss_curve = np.zeros(numsamples_a)
iterations = numsamples_a
counter = 0
for idx, a in enumerate(interval_a):
    b = a-1
    loss = 0
    for sample_idx in range(len_test):
        sample, cos_sim = testset[sample_idx]
        y = sample[0]
        x = sample[1:]
        z_baseline = NR_baseline(x, a, b, cos_sim)
        loss += torch.nn.functional.mse_loss(y, z_baseline)
    loss_curve[idx] = loss / len_test

    counter += 1
    print(f"{(counter/iterations*100):.2f}%", end="\r")


min_index = np.argmin(loss_curve)
a_opt = interval_a[min_index]
loss_opt = loss_curve[min_index]

plt.figure()
plt.plot(interval_a, loss_curve, color='r')
plt.xlabel('α')
plt.ylabel('L(α, α-1)')
plt.plot(a_opt, loss_opt, 'ko')
plt.text(a_opt, loss_opt*1.01, f'αₒₚₜ={a_opt:.4f} →βₒₚₜ={(a_opt-1):.4f}', fontsize=12, verticalalignment='bottom')
plt.show()