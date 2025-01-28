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


class NoisyID(torch.utils.data.Dataset): #all in one dataset
    def __init__(self, clean=False, train=False, test=False, validation=False, noiseReductionMethod=None, alphabeta=[1.2,0.2], dereverberationMethod=None):
        self.classes = torch.Tensor([1705, 3729, 4077, 6341, 6904, 8230])
        empirical_mean_max = 35.67011642456055
        self.normalization_factor = 1/empirical_mean_max
        self.nr_method = noiseReductionMethod
        self.alphabeta = alphabeta
        self.dr_method = dereverberationMethod
        self.is_clean_data = clean
        self.dir = "ID_unfused_aereset"#########
        self.path = "/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/ID_data/"+self.dir+"/"
        self.input_list = []
        for dirpath, dirnames, filenames in os.walk(self.path):
            if not dirnames:
                self.input_list.append(dirpath)
        random.seed(0)
        random.shuffle(self.input_list)
        nan_idx_list = [453, 357, 164, 121]
        for nan_idx in nan_idx_list:
            self.input_list.pop(nan_idx)
        if train==True:
            self.input_list = self.input_list[0:333] #70%
        if validation==True:
            self.input_list = self.input_list[333:428] #20%
        if test==True:
            self.input_list = self.input_list[428:472] #10%

        self.Dereverber = Dereverb.siameseDereverberMemb(feature_size=280, num_layers=1)
        self.Dereverber.load_state_dict(torch.load("/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/DR-S1L_candidate"))
        if self.nr_method == "neural net":
            if self.dr_method == "neural net":
                self.Denoiser = torch.load("/home/student/Documents/WHK_Projekt_1/code_NR/Denoiser_saves/DN3_temp_onDR")
            else:
                self.Denoiser = torch.load("/home/student/Documents/WHK_Projekt_1/code_NR/Denoiser_saves/DN3_temp_onFF")

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
    
    def get_memberships(self, exp):
        load_prefix = '/home/student/Documents/BA/work/experiments/exp_'
        memberships_data = open(load_prefix + str(exp) + '/memberships.json')
        memberships = json.load(memberships_data) 
        memberships = list(memberships.values())[0]
        return memberships
    
    def __getitem__(self, idx):
        item_path = self.input_list[idx]
        #print(item_path)
        speaker_id = int(item_path.split(self.dir+'/')[1].split('/exp')[0])
        #id_target = torch.argmax((self.classes == speaker_id).int()).long()
        id_target = torch.Tensor((self.classes == speaker_id).int().float())

        id_target = list(id_target).index(1)

        #experiment number and node list of this sample
        exp = int(item_path.split("exp_")[1].split("_as")[0])
        with open(item_path.split("seq")[0]+"NODES", 'rb') as file:
                nodes = pickle.load(file)

        #cluster matrices
        Z_c1 = torch.Tensor(np.load(item_path+"/Z_active.npy"))
        Z_c2 = torch.Tensor(np.load(item_path+"/Z_noise_1.npy"))
        Z_c3 = torch.Tensor(np.load(item_path+"/Z_noise_2.npy"))
        Z_c4 = torch.Tensor(np.load(item_path+"/Z_noise_3.npy"))
        #target
        z_clean = torch.Tensor(np.load(item_path+"/Z_target.npy"))
        psi = torch.Tensor(np.load(item_path+"/cluster_similarities.npy")).unsqueeze(0)

        if self.is_clean_data:
            return z_clean , nn.functional.one_hot(torch.tensor(id_target), num_classes=6).float(), item_path

        dereverbed_clusters = torch.zeros((4,280))
        #DEREVERBERATION STAGE
        if self.dr_method == None:
            dereverbed_clusters[0] = Z_c1[0]
            dereverbed_clusters[1] = Z_c2[0]
            dereverbed_clusters[2] = Z_c3[0]
            dereverbed_clusters[3] = Z_c4[0]
        elif self.dr_method == "baseline":           
            dereverbed_clusters[0] = self.dr_baseline(Z_c1, exp, nodes[0])
            dereverbed_clusters[1] = self.dr_baseline(Z_c2, exp, nodes[1])
            dereverbed_clusters[2] = self.dr_baseline(Z_c3, exp, nodes[2])
            dereverbed_clusters[3] = self.dr_baseline(Z_c4, exp, nodes[3])
        elif self.dr_method == "neural net":                
            dereverbed_clusters[0] = self.Dereverber(Z_c1, [exp, nodes[0]])
            dereverbed_clusters[1] = self.Dereverber(Z_c2, [exp, nodes[1]])
            dereverbed_clusters[2] = self.Dereverber(Z_c3, [exp, nodes[2]])
            dereverbed_clusters[3] = self.Dereverber(Z_c4, [exp, nodes[3]])
        #NOISE REDUCTION STAGE
        dereverbed_clusters *= self.normalization_factor
        if self.nr_method == None:
            inputvec = dereverbed_clusters[0]
        elif self.nr_method == "baseline":
            inputvec = self.nr_baseline(dereverbed_clusters, psi, self.alphabeta[0], self.alphabeta[1]).squeeze()
        elif self.nr_method == "neural net":
            inputvec = self.Denoiser(dereverbed_clusters, psi).squeeze()

        return inputvec, nn.functional.one_hot(torch.tensor(id_target), num_classes=6).float(), item_path

S_clean = NoisyID(clean=True, train=True)
S_no_no = NoisyID(noiseReductionMethod=None, dereverberationMethod=None)
S_bl_bl = NoisyID(noiseReductionMethod="baseline", dereverberationMethod="baseline")
S_bl_nn = NoisyID(noiseReductionMethod="neural net", dereverberationMethod="baseline")
S_nn_nn = NoisyID(noiseReductionMethod="neural net", dereverberationMethod="neural net")

S = S_bl_nn
inp, trg, *_ = S[100]
print(inp.size(), trg)
'''
inp_nancount = 0
trg_nancount = 0
nan_list = []
for idx in range(len(S)):
    inp, trg = S[idx]
    if torch.isnan(inp).any():
        inp_nancount += 1
        nan_list.append(idx)
    if torch.isnan(trg).any():
        trg_nancount += 1

print(f"# of nan inputs =  {inp_nancount} of {len(S)}\n# of nan targets = {trg_nancount} of {len(S)}")
print(nan_list)
'''


