import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import sys
import os
import random
import json
import pickle
#sys.path.append("/home/student/Documents/WHK_Projekt_1/code_DR")
#from pydoc import importfile
#Dereverb = importfile('/home/student/Documents/WHK_Projekt_1/code_DR/Dereverb.py')
import Dereverb

print('\033c')


path_to_data = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/ID_data/'

class CleanID(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False, validation=False):
        self.classes = torch.Tensor([1705, 3729, 4077, 6341, 6904, 8230])

        if train:
            self.path = path_to_data+'ID_TRAIN/clean/'
        elif test:
            self.path = path_to_data+'ID_TEST/clean/'
        elif validation:
            self.path = path_to_data+'ID_VALIDATION/clean/'

        self.data_list = []

        for path in Path(self.path).rglob('*.npy'):
            self.data_list.append(path.joinpath(path.parent, path.name))

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        this_path = self.data_list[idx].absolute().as_posix()
        bottleneck = torch.Tensor(np.load(this_path)).squeeze()
        speaker_id = int(this_path.split('speaker_')[1].split('/exp')[0])
        target = torch.tensor([self.classes[i] == speaker_id for i in range(6)], dtype=torch.float32)
        #target = torch.argmax((self.classes == speaker_id).int())
        ##target = (self.classes == speaker_id).nonzero(as_tuple=True)[0]

        return bottleneck, target
    


class FeatureFusionID(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False, validation=False):
        self.classes = torch.Tensor([1705, 3729, 4077, 6341, 6904, 8230])

        if train:
            self.path = path_to_data+'ID_TRAIN/nodes/'
        elif test:
            self.path = path_to_data+'ID_TEST/nodes/'
        elif validation:
            self.path = path_to_data+'ID_VALIDATION/nodes/'

        self.input_list = []

        for dirpath, dirnames, filenames in os.walk(self.path):
            if not dirnames:
                self.input_list.append(dirpath)
        
        num_data = len(self.input_list)

       # self.target_list = torch.zeros(num_data, 6)
        self.target_list = torch.zeros(num_data)
        
        for idx in range(num_data):
            this_path = self.input_list[idx]
            speaker_id = int(this_path.split('speaker_')[1].split('/exp')[0])
            target = torch.argmax((self.classes == speaker_id).int())
            #target = (self.classes == speaker_id).nonzero(as_tuple=True)[0]
            self.target_list[idx] = target

        self.exp_of_each_input = [0]*num_data

        for idx in range(num_data):
            self.exp_of_each_input[idx] = int(self.input_list[idx].split('exp_')[1].split('/seq')[0])

        self.BL = Dereverb.Baseline()

    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, idx):
        node_list = []

        all_highlevels = torch.Tensor([])

        for n, filename in enumerate(os.listdir(self.input_list[idx]+'/')):
            node_list.append(int(filename.split('de_')[1].split('.')[0]))

            highlevel = torch.Tensor(np.load(self.input_list[idx]+'/'+filename)).squeeze().unsqueeze(0)
            
            all_highlevels = torch.cat((all_highlevels, highlevel), dim=0)
        

        exp = self.exp_of_each_input[idx]

        fused = self.BL.feature_fusion(exp, node_list, all_highlevels)

        return fused.squeeze(), self.target_list[idx].long()#, len(node_list) #remove third return




class DereverberID(torch.utils.data.Dataset):
    def __init__(self, model_path:str, model_version:str, num_layers=1, train=False, test=False, validation=False):
        self.classes = torch.Tensor([1705, 3729, 4077, 6341, 6904, 8230])

        self.model_path = model_path
        self.version = model_version.lower()
        if train:
            self.path = path_to_data+'ID_TRAIN/nodes/'
        elif test:
            self.path = path_to_data+'ID_TEST/nodes/'
        elif validation:
            self.path = path_to_data+'ID_VALIDATION/nodes/'

        self.input_list = []

        for dirpath, dirnames, filenames in os.walk(self.path):
            if not dirnames:
                self.input_list.append(dirpath)
        
        num_data = len(self.input_list)

       # self.target_list = torch.zeros(num_data, 6)
        self.target_list = torch.zeros(num_data)

        for idx in range(num_data):
            this_path = self.input_list[idx]
            speaker_id = int(this_path.split('speaker_')[1].split('/exp')[0])
            target = torch.argmax((self.classes == speaker_id).int())
            #target = (self.classes == speaker_id).nonzero(as_tuple=True)[0]
            self.target_list[idx] = target

        self.exp_of_each_input = [0]*num_data

        for idx in range(num_data):
            self.exp_of_each_input[idx] = int(self.input_list[idx].split('exp_')[1].split('/seq')[0])

        #self.DR = Dereverb.Dereverber_v7()
        #self.DR.load_state_dict(torch.load('/home/student/Documents/BA/DRv7_saves_may12/DRv7_1'))
        if self.version == 'convolutional':
            self.DR = Dereverb.ConvDereverberBest()
        if self.version == 'siamese':
            self.DR = Dereverb.siameseDereverberMemb(feature_size=280, num_layers=num_layers)

        self.DR.load_state_dict(torch.load(self.model_path))
        self.DR.eval()

    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, idx):
        num_nodes = len(os.listdir(self.input_list[idx]+'/'))
        h = len(os.listdir(self.input_list[idx]+'/'))
        max_nodes = 10
        num_nodes = min(num_nodes, max_nodes)
        node_list = []
        #print(self.input_list[idx])

        if self.version=='siamese':
            all_highlevels = torch.zeros(h, 280)
        else:
            all_highlevels = torch.zeros(max_nodes, 280)  

        for n, filename in enumerate(os.listdir(self.input_list[idx]+'/')):
            node_list.append(int(filename.split('de_')[1].split('.')[0]))
            node_vector = torch.Tensor(np.load(self.input_list[idx]+'/'+filename)).unsqueeze(0)
            #all_highlevels = torch.cat((all_highlevels, node_vector), dim=0)
            all_highlevels[n] = node_vector
            
            if (not self.version=='siamese' and n >= max_nodes-1) or (self.version=='siamese' and n>=num_nodes-1):
                break
        
        exp_and_nodes = [self.exp_of_each_input[idx], node_list]        

        if self.version == 'convolutional':
            dereverbed = self.DR(all_highlevels).squeeze()
        elif self.version == 'siamese':
            dereverbed = self.DR(all_highlevels, exp_and_nodes).squeeze()
        else:   
            dereverbed = self.DR(all_highlevels, exp_and_nodes).squeeze()

        return dereverbed, self.target_list[idx].long()#, num_nodes #remove thrid return
    










class SingleNodeID(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False, validation=False):
        self.classes = torch.Tensor([1705, 3729, 4077, 6341, 6904, 8230])

        if train:
            self.path = path_to_data+'ID_TRAIN/nodes/'
        elif test:
            self.path = path_to_data+'ID_TEST/nodes/'
        elif validation:
            self.path = path_to_data+'ID_VALIDATION/nodes/'

        self.input_list = []

        for dirpath, dirnames, filenames in os.walk(self.path):
            if not dirnames:
                self.input_list.append(dirpath)
        
        num_data = len(self.input_list)

        #self.target_list = torch.zeros(num_data, 6)
        self.target_list = torch.zeros(num_data)
        
        for idx in range(num_data):
            this_path = self.input_list[idx]
            speaker_id = int(this_path.split('speaker_')[1].split('/exp')[0])
            target = torch.argmax((self.classes == speaker_id).int())
            #target = (self.classes == speaker_id).nonzero(as_tuple=True)[0]
            self.target_list[idx] = target



    def __len__(self):
        return len(self.input_list)


    def __getitem__(self, idx):

        singlenode = torch.Tensor([])

        filename_first_node = os.listdir(self.input_list[idx]+'/')[0]
        singlenode = torch.Tensor(np.load(self.input_list[idx]+'/'+filename_first_node)).squeeze()

        return singlenode, self.target_list[idx].long()
    

class NoisyID(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False, validation=False, noiseReductionMethod=None, alphabeta=[1.2,0.2], dereverberationMethod=None):
        self.classes = torch.Tensor([1705, 3729, 4077, 6341, 6904, 8230])
        self.nr_method = noiseReductionMethod
        self.alphabeta = alphabeta
        self.dr_method = dereverberationMethod
        self.path = "/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/ID_data/ID_unfused/"
        self.input_list = []
        for dirpath, dirnames, filenames in os.walk(self.path):
            if not dirnames:
                self.input_list.append(dirpath)
        random.seed(0)
        random.shuffle(self.input_list)
        if train==True:
            self.input_list = self.input_list[0:333] #70%
        if validation==True:
            self.input_list = self.input_list[333:428] #20%
        if test==True:
            self.input_list = self.input_list[428:476] #10%

        if self.nr_method.lower() == "neural net":
            if self.dr_method.lower() == "neural net":
                self.Denoiser = torch.load("/home/student/Documents/WHK_Projekt_1/code_NR/Denoiser_saves/DN3_trainedOnDR")
                self.Dereverber = torch.load("/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/DR-S1L_candidate")
            else:
                self.Denoiser = torch.load("/home/student/Documents/WHK_Projekt_1/code_NR/Denoiser_saves/DN3_trainedOnFF")

    def __len__(self):
        return len(self.input_list)
    
    def nr_baseline(self, inputs:torch.tensor, psi:torch.tensor, alpha, beta):
        Z_c1_active = inputs[0]
        Z_c2 = inputs[1]
        Z_c3 = inputs[2]
        Z_c4 = inputs[3]
        epsilon = 1/(torch.sum(psi,1))
        weighted_noise = (psi[:,0].unsqueeze(1)*Z_c2 + psi[:,1].unsqueeze(1)*Z_c3 + psi[:,2].unsqueeze(1)*Z_c4)
            
        Z_NR = alpha*Z_c1_active - epsilon*beta*weighted_noise
        
        return Z_NR
    
    def dr_baseline(self, inputs:torch.tensor, exp, nodes):
        #feature fusion
        memberships = self.get_memberships(exp)

        sum = torch.zeros(1,280)
        sum_memberships = 0
        for node_idx in range(len(nodes)):
            this_node = nodes[node_idx]
            membership = memberships[this_node]
            sum += membership * inputs[node_idx].unsqueeze(0)
            sum_memberships += membership
        
        Z_dach = sum/sum_memberships

        return Z_dach
    
    def get_memberships(exp):
        load_prefix = '/home/student/Documents/BA/work/experiments/exp_'
        memberships_data = open(load_prefix + str(exp) + '/memberships.json')
        memberships = json.load(memberships_data) 
        memberships = list(memberships.values())[0]
        return memberships
    
    def __getitem__(self, idx):
        item_path = self.input_list[idx]
        speaker_id = int(item_path.split('unfused_/')[1].split('/exp')[0])
        target = torch.argmax((self.classes == speaker_id).int()).long()

        #experiment number and node list of this sample
        exp = int(item_path.split("exp_"))[1].split("_as")[0]
        with open(item_path.split("seq")[0]+"NODES", 'rb') as file:
                nodes = pickle.load(file)

        Z_c1 = torch.Tensor(np.load(item_path+"Z_active.npy"))
        Z_c2 = torch.Tensor(np.load(item_path+"Z_noise_0.npy"))
        Z_c3 = torch.Tensor(np.load(item_path+"Z_noise_1.npy"))
        Z_c4 = torch.Tensor(np.load(item_path+"Z_noise_2.npy"))
        psi = torch.Tensor(np.load(item_path+"cluster_similarities.npy"))

        dereverbed_clusters = torch.zeros((4,280))
        #DEREVERBERATION STAGE
        if self.dr_method == None:
            dereverbed_clusters[0] = Z_c1[0]
            dereverbed_clusters[1] = Z_c2[0]
            dereverbed_clusters[2] = Z_c3[0]
            dereverbed_clusters[3] = Z_c4[0]
        elif self.dr_method.lower() == "baseline":           
            dereverbed_clusters[0] = self.dr_baseline(Z_c1, exp, nodes)
            dereverbed_clusters[1] = self.dr_baseline(Z_c2, exp, nodes)
            dereverbed_clusters[2] = self.dr_baseline(Z_c3, exp, nodes)
            dereverbed_clusters[3] = self.dr_baseline(Z_c4, exp, nodes)
        elif self.dr_method.lower() == "neural net":           
            dereverbed_clusters[0] = self.Dereverber(Z_c1, [exp, nodes])
            dereverbed_clusters[1] = self.Dereverber(Z_c2, [exp, nodes])
            dereverbed_clusters[2] = self.Dereverber(Z_c3, [exp, nodes])
            dereverbed_clusters[3] = self.Dereverber(Z_c4, [exp, nodes])
        #NOISE REDUCTION STAGE
        if self.nr_method == None:
            inputvec = dereverbed_clusters[0]
        elif self.nr_method.lower() == "baseline":
            inputvec = self.nr_baseline(dereverbed_clusters, psi, self.alphabeta[0], self.alphabeta[1])
        elif self.nr_method.lower() == "neural net":
            inputvec = self.Denoiser(dereverbed_clusters, psi)

        return inputvec

h = CleanID(train=True)
i,t = h[9]
print(i.size(), t)
#a = FeatureFusionID(test=True)
#for n in range(a.__len__()):
#    i, t, c = a.__getitem__(n)
#    print(c)