import torch
import torchaudio
import librosa
import json
from utilities import lmbe, conv, find_in_list_of_list
import numpy as np
from scipy.io import wavfile
import os
import timeit
import ConvAutoEncoder


start = timeit.default_timer()
print('\033c')
sr = 16000
flag = '_2025'

if __name__ == "__main__":

    #source
    folder_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NODE_LMBE_DATA/exp_'

    #load autoencoder
    AE = ConvAutoEncoder.Mel_AutoEncoder_280()
    AE.load_state_dict(torch.load('/home/student/Desktop/Code_BA_Intek/Trainierte Modelle/AE280_final'))
    AE.encode_only()

    isTest = bool(int(input('train (0), test (1)?')))


    train_speakers_preload = list(np.load('speaker_IDs_train.npy'))
    test_speakers_preload = list(np.load('speaker_IDs_test.npy'))
    train_speakers = []
    test_speakers = []
    for id in train_speakers_preload:
        train_speakers.append([id, []])
    for id in test_speakers_preload:
        test_speakers.append([id, []])

    print(test_speakers)

    root=folder_prefix+'0'
    dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    print(dirlist)
    for num_exp in range(200):
        print(f'\nExperiment {num_exp}')
        root=folder_prefix+str(num_exp)
        dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
        for dir in dirlist:
            dir_id = int(dir.split('_')[1])
            print(f'Speaker: {dir_id}')
            index = find_in_list_of_list(train_speakers, dir_id)
            set_of_id = 'Train'
            if index == -1:
                index = find_in_list_of_list(test_speakers, dir_id)
                set_of_id = 'Test'
            #print(index)
            print(f'Set: {set_of_id}, Index: {index[0]}')

            if set_of_id == 'Train' and index[1] != 1: #zweite bed. verhindet dass nicht der index einer gefundenen exp id als spk id interpretiert wird 
                train_speakers[index[0]][1].append(num_exp)
            elif set_of_id == 'Test' and index[1] != 1:
                test_speakers[index[0]][1].append(num_exp)

    print(train_speakers)

    isTargets = bool(int(input('\nrender inputs (0) or targets (1)')))

    if not isTargets:


        if isTest:
            setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/DR_data/HL_NODE{flag}/TEST/inputs/speaker_'
            set_speakers = test_speakers
        else:
            setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/DR_data/HL_NODE{flag}/TRAIN/inputs/speaker_'
            set_speakers = train_speakers

        
        for spk_id, exp_list in set_speakers:
            os.makedirs(setpath+str(spk_id), exist_ok=True)
            for exp in exp_list:
                os.makedirs(setpath+str(spk_id)+'/exp_'+str(exp), exist_ok=True)
                root = folder_prefix+str(exp)+'/speaker_'+str(spk_id)+'/'
                node_list = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
                print(node_list)
                for seq in range(4):
                    os.makedirs(setpath+str(spk_id)+'/exp_'+str(exp)+'/seq_'+str(seq))
                    for node in node_list:
                        node_feature = np.load(folder_prefix+str(exp)+'/speaker_'+str(spk_id)+'/'+node+'/seq_'+str(seq)+'.npy')
                        
                        
                        node_highlevel = AE(torch.Tensor(node_feature).unsqueeze(0).narrow(2, 0, 312)).detach().numpy()
                        print(f'\x1b[97m{node_highlevel.shape}\x1b[37m')
                        print(f'\x1b[92m{node_highlevel.max()}\x1b[37m')

                        savepath = setpath+str(spk_id)+'/exp_'+str(exp)+'/seq_'+str(seq)+'/'
                        np.save(savepath+node, node_highlevel)

    if isTargets:

        folder_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/CLEAN_LMBE_DATA/exp_'
        if isTest:
            setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/DR_data/HL_NODE{flag}/TEST/targets/speaker_'
            set_speakers = test_speakers
        else:
            setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/DR_data/HL_NODE{flag}/TRAIN/targets/speaker_'
            set_speakers = train_speakers

        
        for spk_id, exp_list in set_speakers:
            print(spk_id)
            os.makedirs(setpath+str(spk_id), exist_ok=True)
            for exp in exp_list:
                os.makedirs(setpath+str(spk_id)+'/exp_'+str(exp), exist_ok=True)

                for seq in range(4):
                    clean_feature = np.load(folder_prefix+str(exp)+'/speaker_'+str(spk_id)+'/seq_'+str(seq)+'.npy')
                    
                    
                    clean_highlevel = AE(torch.tensor(clean_feature).unsqueeze(0).narrow(2, 0, 312)).detach().numpy()
                    print(clean_highlevel.shape, clean_highlevel.max())

                    savepath = setpath+str(spk_id)+'/exp_'+str(exp)
                    np.save(savepath+'/seq_'+str(seq), clean_highlevel)