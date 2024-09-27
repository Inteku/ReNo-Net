import torch
import numpy as np
import SoundDatasets
import Dereverb
import lstm_autoencoders
import ConvAutoEncoder
import matplotlib.pyplot as plt



#'''
#VIEW ONE DR SAMPLE WITH EVERY ENCODUNG TYPE

dataset = SoundDatasets.HighlevelNodeDataset(enc_name='AE-ConvM', test=True)

data, _, _ = dataset[0]
bright_level1 = torch.max(data)#*0.25
clean = data[0]#.unsqueeze(0)
nodes = data[1:]

img1 = clean.repeat(50,1)
img1 = torch.cat((img1, bright_level1*torch.ones(1,clean.size(0))),)
for i in range(10):
    img1 = torch.cat((img1, nodes[i].repeat(50,1)), dim=0)


dataset = SoundDatasets.HighlevelNodeDataset(enc_name='AE-LSTMv4', test=True)

data, _, _ = dataset[0]
bright_level2 = torch.max(data)
clean = data[0]#.unsqueeze(0)
nodes = data[1:]

img2 = clean.repeat(50,1)
img2 = torch.cat((img2, bright_level2*torch.ones(1,clean.size(0))),)
for i in range(10):
    img2 = torch.cat((img2, nodes[i].repeat(50,1)), dim=0)


dataset = SoundDatasets.HighlevelNodeDataset(enc_name='ECAPA', test=True)

data, _, _ = dataset[0]
bright_level3 = torch.max(data)
clean = data[0]#.unsqueeze(0)
nodes = data[1:]

img3 = clean.repeat(50,1)
img3 = torch.cat((img3, bright_level3*torch.ones(1,clean.size(0))),)
for i in range(10):
    img3 = torch.cat((img3, nodes[i].repeat(50,1)), dim=0)


fig, ax = plt.subplots(1,3)
im1 = ax[0].imshow(img1, vmax=bright_level1)
fig.colorbar(im1, ax=ax[0])
ax[0].set_title('Convolutional')
im2 = ax[1].imshow(img2, vmax=bright_level2)
fig.colorbar(im2, ax=ax[1])
ax[1].set_title('LSTM')
im3 = ax[2].imshow(img3, vmax=bright_level3)
fig.colorbar(im3, ax=ax[2])
ax[2].set_title('ECAPA')

plt.show()
#'''

'''
#COMPARE REVERBED/CLEAN/DEREVERBED SPECTROGRAMS

drset_cae = SoundDatasets.HighlevelNodeDataset(enc_name='AE-ConvM', test=True)
drset_lstm = SoundDatasets.HighlevelNodeDataset(enc_name='AE-LSTMv4', test=True)

DEC_cae = torch.load('AE_BA_trained_280', map_location='cpu')
DEC_cae.decode_only()

DEC_lstm = lstm_autoencoders.LSTMAutoencoder_v4(input_size=128, cell_size=280, num_layers=1)
DEC_lstm.load_state_dict(torch.load('LSTM_AE_v4_best_statedict', map_location='cpu'))
DEC_lstm.set_encode(False)

DR_cae = Dereverb.ConvDereverberBest(feature_size=280)
DR_lstm = Dereverb.ConvDereverberBest(feature_size=280)
#DR_cae.load_state_dict(torch.load('/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/030624/DR_CAE-encoding/DR_cae_ep7*'))
#DR_lstm.load_state_dict(torch.load('/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/030624/DR_LSTM-encoding/DR_ae_ep9*'))
DR_cae.load_state_dict(torch.load('/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/210624/DR_CAE-encoding/DR_cae_ep4*'))
DR_lstm.load_state_dict(torch.load('/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/210624/DR_LSTM-encoding/DR_ae_ep10*'))



sample_idx = 72
sample_cae, num_nodes, EaN = drset_cae[sample_idx]
sample_lstm, _, _ = drset_lstm[sample_idx]
print(num_nodes, EaN)

z_clean_cae = sample_cae[0]
z_reverb_cae = sample_cae[7]
z_dereverbed_cae = DR_cae(sample_cae[1:], EaN)

z_clean_lstm = sample_lstm[0]
z_reverb_lstm = sample_lstm[7]
z_dereverbed_lstm = DR_lstm(sample_lstm[1:], EaN)

print(f'{z_clean_lstm.shape}\n{z_reverb_lstm.shape}\n{z_dereverbed_lstm.shape}')

LMBE_cleanRec_cae = DEC_cae(z_clean_cae).squeeze().detach().numpy()
LMBE_reverbRec_cae = DEC_cae(z_reverb_cae).squeeze().detach().numpy()
LMBE_dereverbed_cae = DEC_cae(z_dereverbed_cae).squeeze().detach().numpy()

LMBE_cleanRec_lstm = DEC_lstm(z_clean_lstm.unsqueeze(0)).squeeze().detach().numpy()
LMBE_reverbRec_lstm = DEC_lstm(z_reverb_lstm.unsqueeze(0)).squeeze().detach().numpy()
LMBE_dereverbed_lstm = DEC_lstm(z_dereverbed_lstm.unsqueeze(0)).squeeze().detach().numpy()

print(f'{LMBE_cleanRec_cae.shape}\n{LMBE_reverbRec_cae.shape}\n{LMBE_dereverbed_cae.shape}')
print(f'{LMBE_cleanRec_lstm.shape}\n{LMBE_reverbRec_lstm.shape}\n{LMBE_dereverbed_lstm.shape}')

clean_source = np.load('/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/CLEAN_LMBE_DATA/exp_138/speaker_8230/seq_2.npy')
reverb_source = np.load('/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NODE_LMBE_DATA/exp_138/speaker_8230/node_0/seq_2.npy')

fig, ax = plt.subplots(3,3)

im00 = ax[0][0].imshow(clean_source, vmax=1, cmap='plasma', origin='lower')
ax[0][0].set_title('Source -> Clean')
im01 = ax[0][1].imshow(reverb_source, vmax=1, cmap='plasma', origin='lower')
ax[0][1].set_title('Reverbed')

im10 = ax[1][0].imshow(LMBE_cleanRec_cae, vmax=1, cmap='plasma', origin='lower')
ax[1][0].set_title('Conv AE -> Clean Reconstr.')
im11 = ax[1][1].imshow(LMBE_reverbRec_cae, vmax=1, cmap='plasma', origin='lower')
ax[1][1].set_title('Reverbed Reconstr.')
im12 = ax[1][2].imshow(LMBE_dereverbed_cae, vmax=1, cmap='plasma', origin='lower')
ax[1][2].set_title('Dereverbed')

im20 = ax[2][0].imshow(LMBE_cleanRec_lstm, vmax=1, cmap='plasma', origin='lower')
ax[2][0].set_title('LSTM AE -> Clean Reconstr.')
im21 = ax[2][1].imshow(LMBE_reverbRec_lstm, vmax=1, cmap='plasma', origin='lower')
ax[2][1].set_title('Reverbed Reconstr.')
im22 = ax[2][2].imshow(LMBE_dereverbed_lstm, vmax=1, cmap='plasma', origin='lower')
ax[2][2].set_title('Dereverbed')

plt.show()


'''