import torch
import torchaudio.transforms as T
import scipy as S
import numpy as np
from torch import nn
import json
from scipy.io import wavfile

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
    melspec = torch.log10(mel_spectrogram(signal)+1e-9)
    melspec = torch.add(melspec, -1*melspec.min())
    melspec = torch.multiply(melspec, 1/melspec.max())
   # peak = max(melspec.min().abs(), melspec.max())
    #melspec = torch.multiply(melspec, 1/peak)
    return melspec


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

def get_cluster_num(num_exp):
    load_prefix = '/home/student/Documents/Praxisprojekt_2022/work/experiments/exp_'
    cluster_data = open(load_prefix + str(num_exp) + '/clusters.json')
    clusters = json.load(cluster_data)

    return len(clusters)

def inverse_lmbe(melspec, sr):
    print(f'\nmelspec size: {melspec.size()}')
    spec = T.InverseMelScale(sample_rate=sr, n_mels=128, n_stft=1024//2+1, mel_scale="htk")(melspec)
    print(f'spec size: {spec.size()}')
    audio = T.GriffinLim(n_fft=1024)(spec)
    print(f'audio size: {audio.size()}')

    return audio

def generate_node(num_exp, num_seq, num_spk):
    #load data
    audio_data = open('/home/student/Documents/Praxisprojekt_2022/work/ae_sins_2.0_10sec_snr_measures/'+str(num_exp)+'.json')
    closest_clusters = open('/home/student/Documents/Praxisprojekt_2022/work/experiments/exp_'+str(num_exp)+'/closest_clusters.json')
    clusters_ = open('/home/student/Documents/Praxisprojekt_2022/work/experiments/exp_'+str(num_exp)+'/clusters.json')
    audiodata = json.load(audio_data)
    closestclusters = json.load(closest_clusters)
    clusters = json.load(clusters_)

    closestcluster = closestclusters['closest_clusters'][num_spk] #cluster, was am nächsten zu spezifiziertem sprecher ist
    mics_in_cluster = clusters[str(closestcluster)] #mikrofone in dem cluster

    #für alle mics in cluster nodesignale und LMBEs berechnen:
    LMBEs = torch.Tensor([])
    cluster_size = len(mics_in_cluster)
    for i_mic in range(cluster_size):

        #audiofiles und impulsantworten laden
        mic = mics_in_cluster[i_mic]
        fs, audio = wavfile.read('/home/student/Documents/Praxisprojekt_2022'+audiodata['data'][mic]['audio'][num_spk+4*num_seq])
        audio = audio/32768
        fs, ir = wavfile.read('/home/student/Documents/Praxisprojekt_2022'+audiodata['data'][mic]['ir'][num_spk])

        #nodesignal durch faltung erzeugen, auf 10s kürzen und zum Tensor umwandeln
        signal_mic = conv(audio,ir)[0:160000]
        signal_mic_t = torch.Tensor(signal_mic)
        #LMBEs erzeugen (funktion erwartet Tensor)
        if i_mic == 0:
            LMBEs = lmbe(signal_mic_t, 16000).narrow(1,0,312).unsqueeze(0)
        else:
            LMBEs = torch.cat((LMBEs, lmbe(signal_mic_t, 16000).narrow(1,0,312).unsqueeze(0)),0)


    return LMBEs, closestcluster, cluster_size


def generate_clean_reference(num_exp, num_seq):   
    #load data
    audio_data = open('/home/student/Documents/Praxisprojekt_2022/work/ae_sins_2.0_10sec_snr_measures/'+str(num_exp)+'.json')
    audiodata = json.load(audio_data)    
   
    for i in range(4):
        path = audiodata['data'][0]['audio'][i] #path.split('/') enthält info, ob ww-speaker   
        if path.split('/')[3] == 'ww_speakers_short':
            fs, audio = wavfile.read('/home/student/Documents/Praxisprojekt_2022'+audiodata['data'][0]['audio'][i+4*num_seq])
            LMBEs = lmbe(torch.Tensor(audio), 16000).narrow(1,0,312)
            speaker = i
    
    return LMBEs, speaker  

def melbe(signal, sr):
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
    peak = max(melspec.min().abs(), melspec.max())
    melspec = torch.multiply(melspec, 1/peak)
    return melspec

def dominant_mic(exp,speaker):
    closest_clusters = open('/home/student/BA/work/experiments/exp_'+str(exp)+'/closest_clusters.json')
    clusters = open('/home/student/BA/work/experiments/exp_'+str(exp)+'/clusters.json')
    memberships = open('/home/student/BA/work/experiments/exp_'+str(exp)+'/memberships.json')
    closest_clusters = json.load(closest_clusters)
    clusters = json.load(clusters)
    memberships = json.load(memberships)

    #find out cossine similarity of dominant mic in closest cluster
    cluster = closest_clusters['closest_clusters'][speaker]
    mics_incluster = clusters[str(cluster)]
    memberships_clu=[]
    for i in range(len(mics_incluster)):
        memberships_clu.append(memberships['memberships'][mics_incluster[i]])
    max_mem_idx = memberships_clu.index(max(memberships_clu))
    dominant_mic = mics_incluster[max_mem_idx]

    return dominant_mic

def find_in_list_of_list(mylist, id):
    for sub_list in mylist:
        if id in sub_list:
            return (mylist.index(sub_list), sub_list.index(id))
    return -1


def accuracy(output_batch, target_batch):
    N = output_batch.size(dim=0)
    correct_counter = 0
    for n in range(N):
        correct_counter += (torch.argmax(output_batch[n]) == target_batch[n].squeeze()).float()
    return correct_counter/N

def get_memberships(exp):
    load_prefix = '/home/student/Documents/BA/work/experiments/exp_'
    memberships_data = open(load_prefix + str(exp) + '/memberships.json')
    memberships = json.load(memberships_data) 
    memberships = list(memberships.values())[0]
    return memberships