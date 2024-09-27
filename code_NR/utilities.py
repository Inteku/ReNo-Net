import torch
#import Datasets_NR as net
import torchaudio.transforms as T
import scipy as S
import numpy as np
from torch import nn
import json
from scipy.io import wavfile
import librosa
#import NR_Baseline as nr
#import Datasets_NR as set
#import NoiseReduction as nr
#mport ID_Datasets as idset
#from NoiseReduction import NR_baseline



def lmbe(signal, sr):
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )
    melspec = mel_spectrogram(signal)#.narrow(2, 0, 312)
    #melspec = torch.multiply(melspec, 1/peak)
    lmbe = np.log10(melspec)
    lmbe = torch.add(lmbe, lmbe.min().abs())
    lmbe = torch.multiply(lmbe, 1/lmbe.max())
    #print(torch.max(lmbe), torch.min(lmbe))
    return lmbe

def conv(audio,ir):
    audio = audio.astype(np.float32) #datentyp auf float ändern
    ir = S.signal.resample(ir, 16000) #impulsantort muss auf Abtsastrate der sprechnersignale herunter gesamplet werden
    signal = S.signal.convolve(audio, ir) # faltung um signal von sprecher an mic zu erhalten
    return signal

def print_batch_loss(idx, ep, total, loss):
    bar_len = 16
    percent_done = (idx+1)/total*100
    percent_done = round(percent_done, 1)

    done = round((idx+1)/(total/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    print(f'Epoch {ep}: [{done_str}{togo_str}] Loss {idx+1}/{total}: {loss:.12f}', end='\r')

def inverse_lmbe(melspec, sr):
    print(f'\nmelspec size: {melspec.size()}')
    spec = T.InverseMelScale(sample_rate=sr, n_mels=128, n_stft=1024//2+1, mel_scale="htk")(melspec)
    print(f'spec size: {spec.size()}')
    audio = T.GriffinLim(n_fft=1024)(spec)
    print(f'audio size: {audio.size()}')

    return audio

def feature_fusion(num_exp, num_clu, highlevels):

    #load .json-data
    load_prefix = '/home/student/BA_Michael/work/experiments/exp_'
    cluster_data = open(load_prefix + str(num_exp) + '/clusters.json')
    memberships_data = open(load_prefix + str(num_exp) + '/memberships.json')
    clusters = json.load(cluster_data)
    memberships = json.load(memberships_data)
    memberships = list(memberships.values())[0] #anders bin ich mit dem datentyp der meberships nicht klargekommen

    #calculating feature fusions for each cluster
    #Z_dach = [0] * len(clusters) #len(clusters) enspricht Anzahl der Clusters im experiment
    
    mics_incluster = clusters[str(num_clu)]
    sum = 0
    sum_memberships = 0
    for ctr_mic_incluster in range(len(mics_incluster)):
        num_mic = mics_incluster[ctr_mic_incluster]
        membership = memberships[num_mic]
        #print('Mic',num_mic,'has Membership value',membership)
        sum += membership * highlevels[ctr_mic_incluster]
        sum_memberships += membership
    
    Z_dach = sum/sum_memberships

    return Z_dach

'''
def BL_loss():
    AE = torch.load('AE_BA_trained_280')
    AE.decode_only()
    testset = set.Dataset4(test=True)
    testset = torch.utils.data.Subset(testset, list(range(294,testset.__len__())))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    alpha = 1.2
    beta = 0.2
    cos_sim = True
    loss_function = nn.MSELoss()
    c=0
    running_loss = 0
    running_lmbe_loss = 0

    for idx, inputs in enumerate(test_loader):
        exp = inputs[5][0].item()
        seq = inputs[5][1].item()
        spk = inputs[5][2].item()
       
        data_Zref = torch.Tensor(np.load('/home/student/BA_Michael/work/BN280_ref/exp_'+str(exp)+'/speaker_'+str(spk)+'/FF_seq_'+str(seq)+'.npy'))

        Z_pred = nr.NR_baseline(exp, seq, spk, alpha, beta, cos_sim = True)
        test_loss = loss_function(Z_pred, data_Zref)

        recon_pred = AE(Z_pred).squeeze(0)
        ref_data = np.load('/home/student/BA_Michael/work/LMBE_ref/exp_'+str(exp)+'/speaker_'+str(spk)+'/LMBE_'+str(seq)+'.npy')
        lmbe_ref = torch.Tensor(ref_data).unsqueeze(0).narrow(2, 0, 312)
        lmbe_loss = loss_function(recon_pred,lmbe_ref)

        running_lmbe_loss += lmbe_loss.item()
        running_loss += test_loss.item()
        c+=1

    return [running_loss/c, running_lmbe_loss/c]
'''
'''
def psi(exp,spk):
    psi = []
    pass_spkrs = []
    for i in range(4):
        if i != spk:
            pass_spkrs.append(i)
    sims = json.load(open('/home/student/BA_Michael/work/experiments/exp_'+str(exp)+'/similarities.json'))
    cossim_target_z2 = (1 + sims['similarities'][nr.dominant_mic(exp,spk)][nr.dominant_mic(exp,pass_spkrs[0])])/2
    cossim_target_z3 = (1 + sims['similarities'][nr.dominant_mic(exp,spk)][nr.dominant_mic(exp,pass_spkrs[1])])/2
    cossim_target_z4 = (1 + sims['similarities'][nr.dominant_mic(exp,spk)][nr.dominant_mic(exp,pass_spkrs[2])])/2
    
    sum_normal = (1-cossim_target_z2)+(1-cossim_target_z3)+(1-cossim_target_z4)
    psi.append((1-cossim_target_z2)/sum_normal)
    psi.append((1-cossim_target_z3)/sum_normal)
    psi.append((1-cossim_target_z4)/sum_normal)


    return psi
'''
'''
def test_SID(SID):
    #accuracy von noisy signalen
    #SID = torch.load('best_SIDclean')
    SID.eval()
    AE = torch.load('AE_BA_trained_280')
    AE.eval()
    Net1 = torch.load('best_NN42')
    #Net1.eval()
    Net2 = torch.load('NN2_187')
    #Net2.eval()
    Net3 = torch.load('NN3_187')
    #Net3.eval()
    loss_function = nn.CrossEntropyLoss()
    test_set = idset.clean_data(test = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    c=running_bl=ident_bl=running_ff=ident_ff=running_nn3=ident_nn3=running_nn2=ident_nn2=running_nn1=ident_nn1=running_clean=ident_clean=running_rev=ident_rev=0
    losses = []
    accs = [] # 0:ff  1:BL  2:Net3  3:Net4  4:Net5

    for idx, (inputs, targets, path) in enumerate(test_loader):
        id = path[0].split('test/')[1].split('/exp_')[0]
        exp = path[0].split('/exp_')[1].split('/FF_seq')[0]
        seq = path[0].split('/FF_seq')[1].split('.npy')[0]

        #find index (0-3) for active speaker
        jsondict = json.load(open('/home/student/BA_Michael/work/ae_sins_2.0_10sec_snr_measures/'+exp+'.json'))
        pass_spkrs = []
        for i in range(4):
            if jsondict['speaker_data'][i]['speaker'] == id:
                act_spkr = i
            else :
                pass_spkrs.append(i)

        #get all the bottlenecks for comparison
        Z_pred_bl = NR_baseline(int(exp),int(seq),act_spkr,1.2,0.2,cos_sim = True).squeeze(0).squeeze(0)
        Z_ff_act = np.load('/home/student/BA_Michael/work/FF280_noisy/exp_'+exp+'/speaker_'+str(act_spkr)+'/FF_seq_'+seq+'.npy')
        Z_ff_act = torch.Tensor(Z_ff_act).squeeze(0).squeeze(0)
        lmbe_rev = np.load('/home/student/BA_Michael/work/LMBE_ref/exp_'+exp+'/speaker_'+str(act_spkr)+'/LMBE_'+seq+'.npy')
        lmbe_rev = torch.Tensor(lmbe_rev).unsqueeze(0).narrow(2, 0, 312)
        out_ae = AE(lmbe_rev)
        Z_rev = AE.get_bottleneck().squeeze(0).squeeze(0)

        Z_ff_pass = []
        for j in range(3): 
            Z_ff = np.load('/home/student/BA_Michael/work/FF280_noisy/exp_'+str(exp)+'/speaker_'+str(pass_spkrs[j])+'/FF_seq_'+str(seq)+'.npy')
            Z_ff = torch.Tensor(Z_ff).squeeze(0).squeeze(0)
            Z_ff_pass.append(Z_ff)

        Z_pred_nn1 = Net1(Z_ff_act,Z_ff_pass[0],Z_ff_pass[1],Z_ff_pass[2],psi(exp,act_spkr))
        Z_pred_nn2 = Net2(Z_ff_act,Z_ff_pass[0],Z_ff_pass[1],Z_ff_pass[2],psi(exp,act_spkr))
        Z_pred_nn3 = Net3(Z_ff_act,Z_ff_pass[0],Z_ff_pass[1],Z_ff_pass[2],psi(exp,act_spkr))

        out_bl = SID(Z_pred_bl)
        loss_bl = loss_function(out_bl,targets)
        out_ff = SID(Z_ff_act)
        loss_ff = loss_function(out_ff,targets)
        out_nn1 = SID(Z_pred_nn1)
        loss_nn1 = loss_function(out_nn1,targets)
        out_nn2 =SID(Z_pred_nn2)
        loss_nn2 = loss_function(out_nn2,targets)
        out_nn3 = SID(Z_pred_nn3)
        loss_nn3 = loss_function(out_nn3,targets)
        out_rev = SID(Z_rev)
        loss_rev = loss_function(out_rev,targets)
        out_clean = SID(inputs)
        loss_clean = loss_function(out_clean,targets)

        running_bl += loss_bl.item()
        running_ff += loss_ff.item()
        running_nn1 += loss_nn1.item()
        running_nn2 += loss_nn2.item()
        running_nn3 += loss_nn3.item()
        running_rev += loss_rev.item()
        running_clean += loss_clean.item()

        ident_bl+=int(torch.argmax(out_bl)==targets) 
        ident_ff+=int(torch.argmax(out_ff)==targets)
        ident_nn1+=int(torch.argmax(out_nn1)==targets)
        ident_nn2+=int(torch.argmax(out_nn2)==targets)
        ident_nn3+=int(torch.argmax(out_nn3)==targets)
        ident_rev+=int(torch.argmax(out_rev)==targets)
        ident_clean+=int(torch.argmax(out_clean)==targets)
        c+=1


    losses.append(running_ff/c)
    accs.append((ident_ff/c)*100)
    losses.append(running_bl/c)
    accs.append((ident_bl/c)*100)
    losses.append(running_nn1/c)
    accs.append((ident_nn1/c)*100)
    losses.append(running_nn2/c)
    accs.append((ident_nn2/c)*100)
    losses.append(running_nn3/c)
    accs.append((ident_nn3/c)*100)
    losses.append(running_rev/c)
    accs.append((ident_rev/c)*100)
    losses.append(running_clean/c)
    accs.append((ident_clean/c)*100)

    return accs, losses

'''


def get_memberships(exp):
    load_prefix = '/home/student/Documents/BA/work/experiments/exp_'
    memberships_data = open(load_prefix + str(exp) + '/memberships.json')
    memberships = json.load(memberships_data) 
    memberships = list(memberships.values())[0]
    return memberships

