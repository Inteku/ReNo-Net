import torch
import torchaudio
import librosa
import numpy as np
import os
import glob
import json
import utilities
import ConvAutoEncoder
import pickle
from sys import exit
import logging as log
from scipy.io import wavfile

def save_wav(signal, path):
    signal = np.array(signal)
    signal = (signal * 32767).astype(np.int16)
    wavfile.write(path, 16000, signal)
    return

def feature_fusion(exp, node_list, highlevels, feature_size=280):

    #load json-data
    load_prefix = '/home/student/Documents/BA/work/experiments/exp_'
    memberships_data = open(load_prefix + str(exp) + '/memberships.json')
    memberships = json.load(memberships_data) 
    memberships = list(memberships.values())[0]
    memberships = utilities.get_memberships(exp)

    sum = torch.zeros((1, feature_size))
    sum_memberships = 0
    for node_idx in range(len(node_list)):
        this_node = node_list[node_idx]
        membership = memberships[this_node]
        sum += membership * highlevels[node_idx].unsqueeze(0)
        sum_memberships += membership
    
    Z_dach = sum/sum_memberships

    return Z_dach

def feature_extraction(signal_tensor, AE, nodes=None, exp=None): #pro sample 4x feature extraction
    num_nodes = 1 if nodes is None else len(nodes)
    highlevel_tensor = torch.zeros((num_nodes, 280))
    for i in range(num_nodes):
        lmbe_spectogram = utilities.lmbe(signal_tensor[i], sr=16_000).unsqueeze(0)
        highlevel = AE(lmbe_spectogram.narrow(2,0,312))
        
        highlevel_tensor[i] = highlevel

    if nodes is None:
        fused = highlevel_tensor #target 1x280
    else:
        fused = feature_fusion(exp=exp, node_list=nodes, highlevels=highlevel_tensor)
    print(f"    \033[0;31mbefore: max={highlevel_tensor.max():.3f}, mean={highlevel_tensor.mean():.3f}\033[0m")
    print(f"    \033[0;31mafter: max={fused.max():.3f}, mean={fused.mean():.3f}\033[0m")
    return fused

def feature_extraction_2(node_signal, fs): #pro sample 4x feature extraction

    lmbe_spectrogram = utilities.lmbe(node_signal, fs)

    return lmbe_spectrogram

#AE_model = torch.load('AE_BA_trained_280')
AE_model = ConvAutoEncoder.Mel_AutoEncoder_280()
AE_model.load_state_dict(torch.load('/home/student/Desktop/Code_BA_Intek/Trainierte Modelle/AE280_final'))
AE_model.eval()
AE_model.encode_only()
print(f'model hash: {utilities.model_hash(AE_model)}')

def dominant_mic(exp,speaker):
    closest_clusters = open('/home/student/Desktop/Code_BA_Intek/work/experiments/exp_'+str(exp)+'/closest_clusters.json')
    clusters = open('/home/student/Desktop/Code_BA_Intek/work/experiments/exp_'+str(exp)+'/clusters.json')
    memberships = open('/home/student/Desktop/Code_BA_Intek/work/experiments/exp_'+str(exp)+'/memberships.json')
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

def get_cluster_similarities(exp, active_spk, n0_spk, n1_spk, n2_spk):
    sims = json.load(open('/home/student/Desktop/Code_BA_Intek/work/experiments/exp_'+str(exp)+'/similarities.json'))
    dominant_mic_active_cluster = dominant_mic(exp,active_spk)
    #similarity between Z(active) and Z(noise 0)
    sim_Za_Zn0 = (1 + sims['similarities'][dominant_mic_active_cluster][dominant_mic(exp,n0_spk)])/2
    sim_Za_Zn1 = (1 + sims['similarities'][dominant_mic_active_cluster][dominant_mic(exp,n1_spk)])/2
    sim_Za_Zn2 = (1 + sims['similarities'][dominant_mic_active_cluster][dominant_mic(exp,n2_spk)])/2

    return np.asarray([sim_Za_Zn0, sim_Za_Zn1, sim_Za_Zn2])


# Create train data for Noise Reduction Models or Identifier
#data_for = 'id'
data_for = 'nr'
loadpath_prefix = '/home/student/Desktop/Code_BA_Intek'
if data_for == 'id':
    savepath_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/ID_data/ID_FF_debugonly/'
elif data_for == 'nr':
    savepath_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NR/NR_FF/'###############
else:
    exit(1)
for active_speaker in range(4):
    for exp in range(200):
        print(f'\nExperiment: {exp+1}\n')
        experiment_dict = json.load(open('/home/student/Desktop/Code_BA_Intek/work/ae_sins/'+str(exp)+'.json'))
        

        speaker = experiment_dict['speaker_data'][active_speaker]['speaker']
        print(f'  >Active speaker: {active_speaker+1} | ID {speaker}')
        if data_for == 'id' and not int(speaker) in [1705, 3729, 4077, 6341, 6904, 8230]:
            print('    >skip')
            continue
        speaker_dir = savepath_prefix + speaker
        if not os.path.exists(speaker_dir):
           os.makedirs(speaker_dir)
        
        os.makedirs(speaker_dir + f'/exp_{exp}_as_{active_speaker}', exist_ok=True)
       

        closest_clusters_dict = json.load(open(loadpath_prefix+'/work/experiments/exp_'+str(exp)+'/closest_clusters.json'))
        closest_clusters = closest_clusters_dict['closest_clusters']

        clusters_dict = json.load(open(loadpath_prefix+'/work/experiments/exp_'+str(exp)+'/clusters.json'))
        clusters = []
        for c in range(4):
            clusters.append(clusters_dict[str(closest_clusters[c])])
        noise_speakers = [0,1,2,3]
        noise_speakers.pop(active_speaker) #list of speakers without active speaker
        input_order_index_list = noise_speakers
        input_order_index_list.insert(0, active_speaker)
        clusters_in_input_order = [clusters[i] for i in input_order_index_list]
        with open(speaker_dir + f'/exp_{exp}_as_{active_speaker}/NODES', 'wb') as fp:
            pickle.dump(clusters_in_input_order, fp)
        cluster_similarities = get_cluster_similarities(exp, active_speaker, noise_speakers[0], noise_speakers[1], noise_speakers[2])
        for seq in range(4):
            print(f'    >Sequence: {seq+1}')
            path_to_sample = speaker_dir+f'/exp_{exp}_as_{active_speaker}/seq_{seq}'
            os.makedirs(path_to_sample, exist_ok=True)
            target_audio = 0
            for input in range(4): #inputs for NR
                #break
                #print(f'-input {input}')
                is_active = (input==active_speaker)
                #print(is_active)
                if is_active:
                    #print('   active')
                    audio_path = experiment_dict['data'][0]['audio'][active_speaker+4*seq]
                    print(f"    audio: \033[0;34m{audio_path}\033[0m")
                    if audio_path == '/work/Clustering+WWD/non_ww_speakers_short/1705/142318/1705-142318-0038.wav':
                        print('+++++++++++++++++++++++++++')
                    audio_of_active_spk = np.zeros((1, 160_000), dtype=np.float32)
                    audio_of_active_spk[0, :159_999] = librosa.load(loadpath_prefix+audio_path, sr=16e3)[0]
                    _debug_target_audio_dir = audio_path#####
                    target_audio = torch.tensor(audio_of_active_spk, dtype=torch.float32)
                filename = ''
                id_of_inputcluster = closest_clusters[input]
                nodes_of_inputcluster = clusters[input]
                num_signals = len(nodes_of_inputcluster)
                node_signal_tensor = torch.zeros((num_signals,160_000), dtype=torch.float32)

                for spk in range(4): #all 4 speakers that contribute to overlap
                    print(f"      >overlap: {' '*(16-(4*input+spk))}#", end='\r')
                    cc = closest_clusters[spk]
                    nodes_in_cc = clusters[spk]
                    num_nodes_in_cc = len(nodes_in_cc)
                # node_signals_in_cc = torch.zeros((num_nodes_in_cc,16e4))
                    audio_path = experiment_dict['data'][0]['audio'][spk+4*seq]
                    #print(audio_path)
                    audio_of_this_spk = np.zeros((1, 160_000), dtype=np.float32)
                    audio_of_this_spk[0, :159_999] = librosa.load(loadpath_prefix+audio_path, sr=16e3)[0]
                    #if is_active:
                    #    print('   active')
                    #    target_audio = audio_of_this_spk[0:160_000]
                        

                    for node in range(num_signals): #for every node of inputcluster
                        ir_path = experiment_dict['data'][nodes_of_inputcluster[node]]['ir'][spk]
                        ir, _ = librosa.load(loadpath_prefix+ir_path)
                        node_signal = utilities.conv(audio_of_this_spk, ir)
                        
                        node_signal_tensor[node] = node_signal_tensor[node] + torch.tensor(node_signal[0:160_000], dtype=torch.float32)

                final_input = feature_extraction(signal_tensor=node_signal_tensor, AE=AE_model, exp=exp, nodes=nodes_of_inputcluster)
                if not is_active:
                    filename = f'Z_noise_{noise_speakers.index(input)}'
                    np.save(path_to_sample+'/'+filename, final_input.detach().numpy())
                    #save_wav(node_signal_tensor[0].detach(), f'{path_to_sample}/{filename}.wav')
                else:
                    filename = 'Z_active'
                    np.save(path_to_sample+'/'+filename, final_input.detach().numpy())
                    #save_wav(node_signal_tensor[0].detach(), f'{path_to_sample}/{filename}.wav')
            print('')
            final_target = feature_extraction(signal_tensor=target_audio, AE=AE_model)
            #final_target = utilities.bottleneck_from_dir(_debug_target_audio_dir)
            print(f"    \033[0;33mclean max: {final_target.max():.3f}\033[0m")
            np.save(path_to_sample+'/Z_target', final_target.detach().numpy())
            #save_wav(target_audio[0].detach(), f'{path_to_sample}/clean.wav')

            np.save(path_to_sample+'/cluster_similarities', cluster_similarities)
           # nodes_and_exp = nodes_in_cc
           # np.save(nodes_in_cc.append)




'''
        for spk in range(4):
            os.makedirs(savepath_prefix + str(spk), exist_ok = True)
            for seq in range(4):
                mic = dominant_mic(exp,spk)
                audio, fs = librosa.load(loadpath_prefix+json_dictionary['data'][mic]['audio'][spk+4*seq],sr = sr)
                ir, fs2 = librosa.load(loadpath_prefix+json_dictionary['data'][mic]['ir'][spk],sr = sr)
                signal_11s = conv(audio,ir)
                signal10 = signal_11s[0:160000]
                signal_10s = torch.Tensor(signal10)

                if log:
                    LMBE = np.asarray(lmbe(signal_10s, sr))
                    logsave = 'LMBE'
                else:
                    LMBE = np.asarray(melbe(signal_10s, sr))
                    logsave = 'MelBE'
                
                
                np.save(savepath_prefix + str(spk)+'/'+logsave+'_'+str(seq), LMBE)
        print('Data for exp '+str(exp)+' saved')
'''

