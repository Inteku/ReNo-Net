import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
from SoundDatasets import tSNE_dataset
import ConvAutoEncoder
from ecapa import get_ecapa_embeddings


cwd = os.getcwd()
lmbe_dataset = tSNE_dataset(LMBE=True, shuffle=True, seed=0)
wave_dataset = tSNE_dataset(wave=True, shuffle=True, seed=0)
wave_batch_dataset = tSNE_dataset(wave=True, shuffle=True, seed=0, cluster_batches=True)
S = len(lmbe_dataset)
print(S)
#S=30
data = torch.zeros((S,280))
means = torch.zeros((S,128))
ecapa_emb = torch.zeros((S,192))
labels_lmbe = torch.zeros(S)
labels_wave = torch.zeros(S)

AE_path = '/home/student/Documents/WHK_Projekt_1/code_AE/LSTM_AE_transfer_best'
#AE_path = '/home/student/Desktop/Code_BA_Intek/Trainierte Modelle/AE280_final'
#AE_path = '/home/student/Documents/WHK_Projekt_1/AE_savestates/131223/AEr2_131223_30*'
#ENC = ConvAutoEncoder.Mel_AutoEncoder_280_regularized_2()
#ENC.load_state_dict(torch.load(AE_path))

ENC = torch.load(AE_path, map_location=torch.device('cpu'))
ENC.set_decode(False)

print('\n')
for s in range(S):
    print(f'{100*s/S:.2f}%', end='\r')

    LMBE, spk_ID_lmbe = lmbe_dataset.__getitem__(s)
    WAVE, spk_ID_wave = wave_dataset.__getitem__(s)
    WAVE = WAVE.unsqueeze(0)#.detach().numpy()

    with torch.no_grad():
        bottleneck = ENC(torch.Tensor(LMBE))
    data[s,:] = bottleneck

    #time_mean = torch.mean(LMBE.squeeze(), dim=1)
    #means[s,:] = time_mean
    ecapa_emb[s,:] = torch.from_numpy(get_ecapa_embeddings(WAVE, classifier=None, fs=16000, evaluation_length=10))
    labels_lmbe[s] = spk_ID_lmbe
    labels_wave[s] = spk_ID_wave
print('100.00%\n')

S = len(lmbe_dataset)
'''
S2 = len(wave_batch_dataset)
print(S2)
ecapa_emb_batch = torch.zeros((S,192))
labels_batch = torch.zeros(S)
pointer = 0
for s in range(S2):
    print(f'{100*s/S2:.2f}%', end='\r')
    BATCH, spk_id_batch = wave_batch_dataset.__getitem__(s)
    b = BATCH.size(0)
    ecapa_emb_batch[pointer:pointer+b,:] = torch.from_numpy(get_ecapa_embeddings(BATCH, classifier=None, fs=16000, evaluation_length=10))
    labels_batch[pointer:pointer+b] = spk_id_batch*torch.ones(b)
    pointer += b
print('100.00%\n')
'''
plt.imshow(ecapa_emb.detach().numpy(), aspect='auto')
plt.colorbar()
#plt.show()   

print(data.size())
print(means.size())
print(ecapa_emb.size())
# t-SNE anwenden
tsne_ae = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_ae = tsne_ae.fit_transform(data)
#tsne_means = TSNE(n_components=2, random_state=42, perplexity=30)
#tsne_means = tsne_means.fit_transform(means)
tsne_emb = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_emb = tsne_emb.fit_transform(ecapa_emb)
#tsne_emb_batch = TSNE(n_components=2, random_state=42, perplexity=30)
#tsne_emb_batch = tsne_emb_batch.fit_transform(ecapa_emb_batch)

IDs = [1705,3729,4077,6341,6904,8230]

plt.subplot(1,2,1)
for id in range(6):
    plt.scatter(tsne_ae[labels_lmbe == IDs[id], 0], tsne_ae[labels_lmbe == IDs[id], 1], label=str(IDs[id]), marker='o', edgecolors='k')
plt.legend()
plt.title('t-SNE Verteilung der Bottlenecks')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')

plt.subplot(1,2,2)
for id in range(6):
    plt.scatter(tsne_emb[labels_wave == IDs[id], 0], tsne_emb[labels_wave == IDs[id], 1], label=str(IDs[id]), marker='o', edgecolors='k')
plt.legend()
plt.title('t-SNE Verteilung der Embeddings')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.show()









