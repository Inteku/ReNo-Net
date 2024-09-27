import torch
import torchaudio
import librosa
import numpy as np
import os
import glob
import json
import utilities
import ConvAutoEncoder

def feature_extraction(node_signal_tensor, fs): #pro sample 4x feature extraction
    num_nodes = node_signal_tensor.size(0)
    LMBE_tensor = torch.zeros((num_nodes, 128, 312))
    for n in range(num_nodes):
        node_signal = node_signal_tensor[n,:]
        lmbe_spectogram = utilities.lmbe(node_signal, fs)
        LMBE = lmbe_spectogram.unsqueeze(0).narrow(2,0,312)
        LMBE_tensor[n,:, :] = LMBE
    return LMBE_tensor

#AE_model = torch.load('AE_BA_trained_280')
AE_model = ConvAutoEncoder.Mel_AutoEncoder_280()
AE_model.load_state_dict(torch.load('/home/student/Desktop/Code_BA_Intek/Trainierte Modelle/AE280_final'))


def dominant_mic(exp,speaker):
    closest_clusters = open('/home/student/Documents/WHK_Projekt_1/work/experiments/exp_'+str(exp)+'/closest_clusters.json')
    clusters = open('/home/student/Documents/WHK_Projekt_1/work/experiments/exp_'+str(exp)+'/clusters.json')
    memberships = open('/home/student/Documents/WHK_Projekt_1/work/experiments/exp_'+str(exp)+'/memberships.json')
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
    sims = json.load(open('/home/student/Documents/WHK_Projekt_1/work/experiments/exp_'+str(exp)+'/similarities.json'))
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
        experiment_dict = json.load(open('/home/student/Documents/WHK_Projekt_1/work/ae_sins/'+str(exp)+'.json'))
        loadpath_prefix = '/home/student/Documents/WHK_Projekt_1'
        savepath_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/waves_noisy/'###############

        speaker = experiment_dict['speaker_data'][active_speaker]['speaker'] #ID
        print(f'Active speaker {active_speaker+1} | ID {speaker}')
        speaker_dir = savepath_prefix + speaker
        if not os.path.exists(speaker_dir):
           os.makedirs(speaker_dir)
        
        os.makedirs(speaker_dir + '/exp_' + str(exp) + '_as_' + str(active_speaker), exist_ok=True)
       

        closest_clusters_dict = json.load(open(loadpath_prefix+'/work/experiments/exp_'+str(exp)+'/closest_clusters.json'))
        closest_clusters = closest_clusters_dict['closest_clusters']

        clusters_dict = json.load(open(loadpath_prefix+'/work/experiments/exp_'+str(exp)+'/clusters.json'))
        nodes_of_speakercluster = clusters_dict[str(closest_clusters[active_speaker])]
        
        #noise_speakers = [0,1,2,3]
        #noise_speakers.pop(active_speaker) #list of speakers without active speaker

        #cluster_similarities = get_cluster_similarities(exp, active_speaker, noise_speakers[0], noise_speakers[1], noise_speakers[2])

        for seq in range(4):
            print(f'Sequence: {seq+1}')
            sequence_dir = speaker_dir+'/exp_'+str(exp)+'_as_'+str(active_speaker)+'/seq_'+str(seq)
            os.makedirs(sequence_dir)
            #target_audio = torch.Tensor(0)
            #for input in range(4): #inputs for NR
            #is_active = (input==active_speaker)
            #filename = ''
            #id_of_inputcluster = closest_clusters[input]
            num_signals = len(nodes_of_speakercluster)
            node_signal_tensor = torch.zeros((num_signals,160000))

            for spk in range(4): #all 4 speakers that contribute to overlap
                #cc = closest_clusters[spk]
               # nodes_in_cc = clusters_dict[str(spk)]
                #num_nodes_in_cc = len(nodes_in_cc)
            # node_signals_in_cc = torch.zeros((num_nodes_in_cc,16e4))
                audio_path = experiment_dict['data'][0]['audio'][spk+4*seq]
                audio_of_this_spk,_ = librosa.load(loadpath_prefix+audio_path)
               # if is_active:
               #     target_audio = torch.Tensor(audio_of_this_spk[0:160000]).squeeze().unsqueeze(0)

                for node in range(num_signals): #for every node of inputcluster
                    ir_path = experiment_dict['data'][nodes_of_speakercluster[node]]['ir'][spk]
                    ir,_ = librosa.load(loadpath_prefix+ir_path)
                    node_signal = torch.Tensor(utilities.conv(audio_of_this_spk, ir))
                    node_signal_tensor[node] = node_signal_tensor[node] + torch.Tensor(node_signal[0:160000])

            for signal in range(num_signals):
                #####lmbe_spectogram = utilities.lmbe(node_signal_tensor[signal], 16e3).unsqueeze(0).narrow(2,0,312)
                filename = 'wave_ccNode_'+str(signal+1)
                np.save(sequence_dir+'/'+filename, node_signal_tensor[signal].detach().numpy())#lmbe_spectogram.detach().numpy())
                    

           # final_LMBEs = feature_extraction(node_signal_tensor, 16e3)


            #if not is_active:
            #    filename = 'Z_noise_'+str(noise_speakers.index(input))
            #    np.save(path_to_sample+'/'+filename, final_input.detach().numpy())
            #else:
            #    filename = 'Z_active'
            #    np.save(path_to_sample+'/'+filename, final_input.detach().numpy())
               ############# 
            #final_target = feature_extraction(target_audio, 16e3, AE_model, is_target=True)
            #np.save(path_to_sample+'/Z_target', final_target.detach().numpy())

            #np.save(path_to_sample+'/cluster_similarities', cluster_similarities)




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

