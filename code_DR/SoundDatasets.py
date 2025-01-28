import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
from utilities import lmbe
import random as rnd
import os


#sets for AE

class CleanDataset(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False):


        if not (train ^ test):
            print("ERROR, SELECT ONE SUBSET ('train' OR 'test')")
        elif train:
            self.folder_path = '/home/student/Documents/Praxisprojekt_2022/work/work/Clustering+WWD/non_ww_speakers_short/'
        elif test:
            self.folder_path = '/home/student/Documents/Praxisprojekt_2022/work/work/Clustering+WWD/ww_speakers_short/'
        
        self.data = []

        for path in Path(self.folder_path).rglob('*.wav'):
            self.data.append(path.joinpath(path.parent, path.name))
         #   print(path.joinpath(path.parent, path.name), '\n')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        
        signal,sr = torchaudio.load(file_path)
        
        spec = lmbe(signal, sr).narrow(2, 0, 312)
        return spec
        #dim 2 wird von 313 auf 312 gekürzt, damit beim pooling nicht gerundet wird


class ReverbMelBEsDataset(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False):

        if not (train ^ test):
            print("ERROR, SELECT ONE SUBSET ('train' OR 'test')")
        elif train:
            self.folder_path = '/home/student/BA_Michael/MelBEs_traindata/'
        elif test:
            self.folder_path = '/home/student/BA_Michael/MelBEs_testdata/'

        self.data = []

        for path in Path(self.folder_path).rglob('*.npy'):
            self.data.append(path.joinpath(path.parent, path.name))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        file_path = self.data[idx]
        loaded = np.load(file_path)
        return torch.Tensor(loaded).unsqueeze(0).narrow(2, 0, 312)
    
class ReverbLMBEsDataset(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False):

        if not (train ^ test):
            print("ERROR, SELECT ONE SUBSET ('train' OR 'test')")
        elif train:
            self.folder_path = '/home/student/BA_Michael/LMBEs_traindata/'
        elif test:
            self.folder_path = '/home/student/BA_Michael/LMBEs_testdata/'

        self.data = []

        for path in Path(self.folder_path).rglob('*.npy'):
            self.data.append(path.joinpath(path.parent, path.name))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        file_path = self.data[idx]
        loaded = np.load(file_path)
        return torch.Tensor(loaded).unsqueeze(0).narrow(2, 0, 312)
            

#praxisprojekt
class NodesignalDataset(torch.utils.data.Dataset):
    def __init__(self, Experiment=0):
        self.folder_path = '/home/student/Documents/Praxisprojekt_2022/work/feature_data/exp_' + str(Experiment) + '/'

        self.data = []

        for path in Path(self.folder_path).rglob('*.npy'):
            self.data.append(path.joinpath(path.parent, path.name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        loaded = np.load(file_path)
        return torch.Tensor(loaded).unsqueeze(0).narrow(2, 0, 312)
    
class rnd_dataset(torch.utils.data.Dataset):
    def __init__(self,train = False, test=False):
        rnd.seed(2)
        self.data = []

        if train:
            for i in range(1000):
                self.data.append(torch.rand(1,1,128,312))
        if test:
            for i in range(300):
                self.data.append(torch.rand(1,1,128,312))
    
    def __len__(self):
        return len(self.data)
    
    def __getidem__(self, idx):
        return self.data[idx]



class tSNE_dataset(torch.utils.data.Dataset):
    def __init__(self, LMBE=False, wave=False, shuffle=False, seed=0, cluster_batches=False):
        self.cluster_batches = cluster_batches
        if LMBE and cluster_batches:
            raise(ValueError("LMBE and cluster_batches not available simultaneously"))

        if LMBE:
            self.folder_path = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/LMBE_noisy/'
        if wave:
            self.folder_path = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/waves_noisy/'

        self.data = []
        self.labels = []

        if not cluster_batches:
            for path in Path(self.folder_path).rglob('*.npy'):
                speaker_id = int(path.parent.as_posix().split('sy/')[1].split('/ex')[0])
                
                if speaker_id in [1705,3729,4077,6341,6904,8230]:
                    self.data.append(path.joinpath(path.parent, path.name))
                    self.labels.append(speaker_id)
        
        if cluster_batches:
            for spk_dir in os.listdir(self.folder_path):
                speaker_id = int(spk_dir)
                if speaker_id in [1705,3729,4077,6341,6904,8230]:
                    #print(speaker_id, type(speaker_id))
                    spk_dir_path = os.path.join(self.folder_path, spk_dir)
                    for exp_dir in os.listdir(spk_dir_path):
                        exp_dir_path = os.path.join(spk_dir_path, exp_dir)
                        for seq_dir in os.listdir(exp_dir_path):
                            seq_dir_path = os.path.join(exp_dir_path, seq_dir)
                            this_cluster_this_sequence_list = []
                            for _, _, files in os.walk(seq_dir_path):
                                for file in files:
                                    this_cluster_this_sequence_list.append(os.path.join(seq_dir_path,file))

                            self.data.append(this_cluster_this_sequence_list)
                            self.labels.append(speaker_id)

        if shuffle:
            rnd.seed(seed)
            rnd.shuffle(self.data)
            rnd.seed(seed)
            rnd.shuffle(self.labels)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if not self.cluster_batches:
            file_path = self.data[idx]
            loaded = np.load(file_path)
            return torch.Tensor(loaded), self.labels[idx]
        
        if self.cluster_batches:
            file_list = self.data[idx]
            batch = torch.zeros((len(file_list), 160000))
            for i,file in enumerate(file_list):
                batch[i,:] = torch.Tensor(np.load(file))
            return batch, self.labels[idx]    
            
        
    

#sets for DR

class HighlevelNodeDataset(torch.utils.data.Dataset):
    def __init__(self, enc_name:str, train=False, test=False):
        self.data = []

        if not (train ^ test):
            print("ERROR, SELECT ONE SUBSET ('train' OR 'test')")
        elif train:
            self.input_path = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_{enc_name}/TRAIN/'
        elif test:
            self.input_path = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/HL_NODE_{enc_name}/TEST/'
        
        self.input_list = []

        self.feature_size = 192 if enc_name=='ECAPA' else 280

        for dirpath, dirnames, filenames in os.walk(self.input_path+'inputs/'):
            if not dirnames:
                self.input_list.append(dirpath)

        self.target_list = [0]*len(self.input_list)
        
        for idx in range(len(self.input_list)):
            self.target_list[idx] = self.input_path +'targets/'+ self.input_list[idx].split('inputs/')[1] + '.npy'

        self.exp_of_each_input = [0]*len(self.input_list)

        for idx in range(len(self.input_list)):
            self.exp_of_each_input[idx] = int(self.input_list[idx].split('exp_')[1].split('/seq')[0])

        #print(self.exp_of_each_input)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        #_, _, files = next(os.walk(self.input_list[818]))
        #node_count = len(files)
        target_vector = torch.Tensor(np.load(self.target_list[idx]))
        num_nodes = len(os.listdir(self.input_list[idx]+'/'))
        max_nodes = 10
        num_nodes = min(num_nodes, max_nodes)
        node_list = []

        Data = torch.zeros(max_nodes+1, self.feature_size)
        Data[0] = target_vector.squeeze()
        #Data = target_vector.squeeze().unsqueeze(0)
        for n, filename in enumerate(os.listdir(self.input_list[idx]+'/')):
            node_list.append(int(filename.split('de_')[1].split('.')[0]))
            node_vector = torch.Tensor(np.load(self.input_list[idx]+'/'+filename)).squeeze()#.unsqueeze(0)
            Data[n+1] = node_vector
            #Data = torch.cat((Data, node_vector), dim=0)
            if n >= max_nodes-1:
                break
        # lazy way to fill the remaining rows with zeros, not optimal (no pre allocation)
        #for _ in range(max_nodes-num_nodes):
        #    Data = torch.cat((Data, torch.zeros(1, 280)), dim=0)

        exp_and_nodes = [self.exp_of_each_input[idx], node_list]

        return Data, num_nodes, exp_and_nodes


