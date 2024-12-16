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

def feature_extraction(node_signal_tensor, fs, AE, exp=0, cluster=0, is_target=False): #pro sample 4x feature extraction
    num_nodes = node_signal_tensor.size(0)
    highlevel_tensor = torch.zeros((num_nodes, 280))
    for n in range(num_nodes):
        node_signal = node_signal_tensor[n,:]
        lmbe_spectogram = utilities.lmbe(node_signal, fs)
        AE.encode_only()
        highlevel = AE(lmbe_spectogram.unsqueeze(0).narrow(2,0,312))
        highlevel_tensor[n,:] = highlevel
    output = highlevel_tensor
    if not is_target:
        output = utilities.feature_fusion(exp, cluster, highlevel_tensor)
    return output

def feature_extraction_2(node_signal, fs): #pro sample 4x feature extraction

    lmbe_spectrogram = utilities.lmbe(node_signal, fs)

    return lmbe_spectrogram

#AE_model = torch.load('AE_BA_trained_280')
AE_model = ConvAutoEncoder.Mel_AutoEncoder_280()
AE_model.load_state_dict(torch.load('/home/student/Desktop/Code_BA_Intek/Trainierte Modelle/AE280_final'))


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


# Create train data for Noise Reduction Models

for active_speaker in range(4):
    for exp in range(200):
        print(f'\n\nExperiment {exp+1}:\n')
        experiment_dict = json.load(open('/home/student/Desktop/Code_BA_Intek/work/ae_sins/'+str(exp)+'.json'))
        loadpath_prefix = '/home/student/Desktop/Code_BA_Intek'
        savepath_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NR/NR_feature_fusion/'###############

        speaker = experiment_dict['speaker_data'][active_speaker]['speaker']
        print(f'Active speaker {active_speaker+1} | ID {speaker}')
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
            print(f'Sequence: {seq+1}')
            path_to_sample = speaker_dir+f'/exp_{exp}_as_{active_speaker}/seq_{seq}'
            os.makedirs(path_to_sample, exist_ok=True)
            target_audio = torch.Tensor(0)
            for input in range(4): #inputs for NR
                #print(f'-input {input}')
                is_active = (input==active_speaker)
                #print(is_active)
                if is_active:
                    #print('   active')
                    audio_path = experiment_dict['data'][0]['audio'][active_speaker+4*seq]
                    audio_of_active_spk, fs = librosa.load(loadpath_prefix+audio_path)
                    #print(f'   fs {fs}')
                    target_audio = torch.Tensor(audio_of_active_spk[0:160000]).squeeze().unsqueeze(0)
                filename = ''
                id_of_inputcluster = closest_clusters[input]
                nodes_of_inputcluster = clusters[input]
                num_signals = len(nodes_of_inputcluster)
                node_signal_tensor = torch.zeros((num_signals,160000))

                for spk in range(4): #all 4 speakers that contribute to overlap
                    print(f' -overlap speaker {spk}')
                    cc = closest_clusters[spk]
                    nodes_in_cc = clusters[spk]
                    num_nodes_in_cc = len(nodes_in_cc)
                # node_signals_in_cc = torch.zeros((num_nodes_in_cc,16e4))
                    audio_path = experiment_dict['data'][0]['audio'][spk+4*seq]
                    #print(audio_path)
                    audio_of_this_spk,_ = librosa.load(loadpath_prefix+audio_path)
                    #if is_active:
                    #    print('   active')
                    #    target_audio = audio_of_this_spk[0:160000]
                        

                    for node in range(num_signals): #for every node of inputcluster
                        ir_path = experiment_dict['data'][nodes_of_inputcluster[node]]['ir'][spk]
                        ir,_ = librosa.load(loadpath_prefix+ir_path)
                        node_signal = utilities.conv(audio_of_this_spk, ir)
                        
                        node_signal_tensor[node] = node_signal_tensor[node] + torch.Tensor(node_signal[0:160000])

                final_input = feature_extraction(node_signal_tensor, 16e3, AE_model, exp, id_of_inputcluster, is_target=False)
                if not is_active:
                    filename = 'Z_noise_'+str(noise_speakers.index(input))
                    np.save(path_to_sample+'/'+filename, final_input.detach().numpy())
                else:
                    filename = 'Z_active'
                    np.save(path_to_sample+'/'+filename, final_input.detach().numpy())
    
            final_target = feature_extraction(target_audio, 16e3, AE_model, is_target=True)
            np.save(path_to_sample+'/Z_target', final_target.detach().numpy())

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

