import torch
import torchaudio
import librosa
import json
from utilities import lmbe, conv, find_in_list_of_list
import numpy as np
from scipy.io import wavfile
import os
import timeit
import lstm_autoencoders
from ecapa import get_ecapa_embeddings

start = timeit.default_timer()
print('\033c')
isHighlevel = bool(int(input("save LMBEs (0) or Highlevel vectors (1)?\n")))
#if isHighlevel:
#    modelname = input("select AE model\n")
modelname = 'ae_950'

sr = 16000
loadpath_prefix = '/home/student/Documents/WHK_Projekt_1'
#i = 0

if not isHighlevel:
    isClean = bool(int(input('\nNode signals (0) or Clean signals (1)?\n')))


    if not isClean:
        print('\nrendering node LMBEs\n')
        #folder_prefix = '/home/student/Documents/BA/work/NODE_LMBE_DATA/exp_'
        folder_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NODE_TIME_DATA/exp_'

        for num_exp in range(200):
            print('\n\nExperiment', num_exp,':\n')
            json_dictionary = json.load(open('/home/student/Documents/WHK_Projekt_1/work/ae_sins/'+str(num_exp)+'.json'))
            speakers_in_exp = [json_dictionary['speaker_data'][0]['speaker'],
                            json_dictionary['speaker_data'][1]['speaker'],
                            json_dictionary['speaker_data'][2]['speaker'],
                            json_dictionary['speaker_data'][3]['speaker']]
            closest_clusters = json.load(open('/home/student/Documents/WHK_Projekt_1/work/experiments/exp_'+str(num_exp)+'/closest_clusters.json'))
            clusters = json.load(open('/home/student/Documents/WHK_Projekt_1/work/experiments/exp_'+str(num_exp)+'/clusters.json'))
            
            
            os.makedirs(folder_prefix+str(num_exp), exist_ok=True)

            for num_speaker in range(4):
                os.makedirs(folder_prefix+str(num_exp)+'/speaker_'+speakers_in_exp[num_speaker], exist_ok=True)
                cc_of_speaker = closest_clusters['closest_clusters'][num_speaker]
                mic_list_of_cc = list(clusters[str(cc_of_speaker)])
                for num_mic in mic_list_of_cc:
                    os.makedirs(folder_prefix+str(num_exp)+'/speaker_'+speakers_in_exp[num_speaker]+'/node_'+str(num_mic), exist_ok=True)
                    
                    for num_seq in range(4):
                        audio,fs = librosa.load(loadpath_prefix + json_dictionary['data'][num_mic]['audio'][num_speaker + 4*num_seq], sr=sr)

                        ir, fs2 = librosa.load(loadpath_prefix + json_dictionary['data'][num_mic]['ir'][num_speaker], sr=sr)
                        reverbed_sequence = conv(audio, ir)
                        reverbed_sequence = torch.Tensor(reverbed_sequence[0:160000]) #shorten to 10s

                        #reverbed_lmbe = np.asarray(torch.Tensor(lmbe(reverbed_sequence, sr)))

                        savepath = folder_prefix+str(num_exp)+'/speaker_'+speakers_in_exp[num_speaker]+'/node_'+str(num_mic)
                        #np.save(savepath+'/seq_'+str(num_seq), reverbed_lmbe)
                        np.save(savepath+'/seq_'+str(num_seq), np.asarray(reverbed_sequence.squeeze()))###


    if isClean:
        print('\nrendering clean LMBEs\n')
        #folder_prefix = '/home/student/Documents/BA/work/CLEAN_LMBE_DATA/exp_'
        folder_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/CLEAN_TIME_DATA/exp_'

        for num_exp in range(200):
            print('\n\nExperiment', num_exp,':\n')
            json_dictionary = json.load(open('/home/student/Documents/WHK_Projekt_1/work/ae_sins/'+str(num_exp)+'.json'))
            speakers_in_exp = [json_dictionary['speaker_data'][0]['speaker'],
                            json_dictionary['speaker_data'][1]['speaker'],
                            json_dictionary['speaker_data'][2]['speaker'],
                            json_dictionary['speaker_data'][3]['speaker']]
            closest_clusters = json.load(open('/home/student/Documents/WHK_Projekt_1/work/experiments/exp_'+str(num_exp)+'/closest_clusters.json'))
            clusters = json.load(open('/home/student/Documents/WHK_Projekt_1/work/experiments/exp_'+str(num_exp)+'/clusters.json'))

            
            
            os.makedirs(folder_prefix+str(num_exp))

            for num_speaker in range(4):
                os.makedirs(folder_prefix+str(num_exp)+'/speaker_'+speakers_in_exp[num_speaker])
                cc_of_speaker = closest_clusters['closest_clusters'][num_speaker]
                    
                for num_seq in range(4):
                    audio,fs = librosa.load(loadpath_prefix + json_dictionary['data'][0]['audio'][num_speaker + 4*num_seq], sr=sr)

                    audio = torch.Tensor(audio[0:160000]) #shorten to 10s

                    #clean_lmbe = np.asarray(torch.Tensor(lmbe(audio, sr)))

                    savepath = folder_prefix+str(num_exp)+'/speaker_'+speakers_in_exp[num_speaker]
                    #np.save(savepath+'/seq_'+str(num_seq), clean_lmbe)
                    np.save(savepath+'/seq_'+str(num_seq), np.asarray(audio.squeeze()))###



if isHighlevel:
    isEcapa = bool(int(input('\nuse AE encoding (0) or Ecapa embedding (1)')))

    #source folder
    if isEcapa:
        folder_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NODE_TIME_DATA/exp_'
    else:
        folder_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NODE_LMBE_DATA/exp_'

    #load autoencoder
    #AE = lstm_autoencoders.LSTMAutoencoder_v4_old()
    #AE.load_state_dict(torch.load(modelname))
    #AE.encode_only()
    AE = torch.load('LSTM_AE_v4_best', map_location='cpu')
    AE.set_decode(False)
    ae_name = 'LSTMv4'

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

    isTest = bool(int(input('\nrender training data (0) or test data (1)?\n')))
    isTargets = bool(int(input('\nrender inputs (0) or targets (1)')))

    if not isTargets:
        if isEcapa:
            if isTest:
                setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_ECAPA/TEST/inputs/speaker_'
                set_speakers = test_speakers
            else:
                setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_ECAPA/TRAIN/inputs/speaker_'
                set_speakers = train_speakers
        else:
            if isTest:
                setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_AE-{ae_name}/TEST/inputs/speaker_'
                set_speakers = test_speakers
            else:
                setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_AE-{ae_name}/TRAIN/inputs/speaker_'
                set_speakers = train_speakers

        
        for spk_id, exp_list in set_speakers:
            os.makedirs(setpath+str(spk_id))
            for exp in exp_list:
                os.makedirs(setpath+str(spk_id)+'/exp_'+str(exp))
                root = folder_prefix+str(exp)+'/speaker_'+str(spk_id)+'/'
                node_list = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
                print(node_list)
                for seq in range(4):
                    os.makedirs(setpath+str(spk_id)+'/exp_'+str(exp)+'/seq_'+str(seq))
                    for node in node_list:
                        node_feature = np.load(folder_prefix+str(exp)+'/speaker_'+str(spk_id)+'/'+node+'/seq_'+str(seq)+'.npy')
                        
                        if isEcapa:
                            print(f'++++++++++{node_feature.shape}++++++++++++++')
                            node_highlevel = get_ecapa_embeddings(torch.Tensor(node_feature).unsqueeze(0), classifier=None, fs=16000, evaluation_length=10)
                        else:
                            node_highlevel = AE(torch.Tensor(node_feature).unsqueeze(0).narrow(2, 0, 312)).detach().numpy()
                        #print(node_highlevel.shape, end='')

                        savepath = setpath+str(spk_id)+'/exp_'+str(exp)+'/seq_'+str(seq)+'/'
                        np.save(savepath+node, node_highlevel)

    if isTargets:

        if isEcapa:
            folder_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/CLEAN_TIME_DATA/exp_'
            if isTest:
                setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_ECAPA/TEST/targets/speaker_'
                set_speakers = test_speakers
            else:
                setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_ECAPA/TRAIN/targets/speaker_'
                set_speakers = train_speakers
        else:
            folder_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/CLEAN_LMBE_DATA/exp_'
            if isTest:
                setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_AE-{ae_name}/TEST/targets/speaker_'
                set_speakers = test_speakers
            else:
                setpath = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_AE-{ae_name}/TRAIN/targets/speaker_'
                set_speakers = train_speakers

        
        for spk_id, exp_list in set_speakers:
            print(spk_id)
            os.makedirs(setpath+str(spk_id))
            for exp in exp_list:
                os.makedirs(setpath+str(spk_id)+'/exp_'+str(exp))

                for seq in range(4):
                    clean_feature = np.load(folder_prefix+str(exp)+'/speaker_'+str(spk_id)+'/seq_'+str(seq)+'.npy')
                    
                    if isEcapa:
                        clean_highlevel = get_ecapa_embeddings(torch.Tensor(clean_feature).unsqueeze(0), classifier=None, fs=16000, evaluation_length=10)
                    else:
                        clean_highlevel = AE(torch.Tensor(clean_feature).unsqueeze(0).narrow(2, 0, 312)).detach().numpy()
                    print(clean_highlevel.shape, type(clean_highlevel), end="")

                    savepath = setpath+str(spk_id)+'/exp_'+str(exp)
                    np.save(savepath+'/seq_'+str(seq), clean_highlevel)