import torch
import numpy as np
import os
import random
import Dereverb
import pickle
print('\033c')

class Noise_Reduction_Dataset(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False, validate=False, dereverber=None, test_size=300, validation_size=300, shuffle_seed=280):
        
        self.isDereverber = False
        if dereverber == None: #clusters are fused by feature fusion / use FF set
            path_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NR/NR_feature_fusion/'
        else: #clusters are fused using dereverber / use unfused set
            path_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NR/NR_unfused/'
            self.DR_model = Dereverb.siameseDereverberMemb(feature_size=280, num_layers=1)
            self.DR_model.load_state_dict(torch.load(dereverber))
            self.isDereverber = True
        self.all_paths_to_samples = []
        for dirPath, dirNames, fileNames in os.walk(path_prefix):
            if dirNames == []:
                self.all_paths_to_samples.append(dirPath)

        naughty_list = np.load('naughty_list.npy') #enthält indices der samples die nan sind
        #print(naughty_list[::-1])
        #entferne nan samples
        #self.all_paths_to_samples = [element for index, element in enumerate(self.all_paths_to_samples) if index not in naughty_list]
        for n in naughty_list[::-1]:
            #print(n)
            self.all_paths_to_samples.pop(n)
        total_len = len(self.all_paths_to_samples)

        random.seed(shuffle_seed)
        #random.shuffle(self.all_paths_to_samples)

        if not (train ^ test ^ validate):
            print("\n\033[1m\033[33mWARNING: No subset selected. Entire set will be returned.\033[0m\n")
        elif train:
            n = 0 #von
            m = total_len - validation_size - test_size #bis
            self.all_paths_to_samples = self.all_paths_to_samples[n:m]
        elif validate:
            n = total_len - validation_size - test_size
            m = total_len - test_size
            self.all_paths_to_samples = self.all_paths_to_samples[n:m]
        elif test:
            n = total_len - test_size
            m = total_len
            self.all_paths_to_samples = self.all_paths_to_samples[n:m]

        empirical_mean_max = 35.67011642456055
        self.normalization_factor = 1/empirical_mean_max


    def __len__(self):
        return len(self.all_paths_to_samples)
    
    def __getitem__(self, idx):
        path_to_sample = self.all_paths_to_samples[idx]
       
        sample = torch.zeros(5,280)
        #target
        sample[0,:] = torch.Tensor(np.load(path_to_sample + '/Z_target.npy'))
        if not self.isDereverber:            
            #inputs
            #sample[1,:] = 20*torch.log10(torch.tensor(np.load(path_to_sample + '/Z_active.npy'))+1) 
            #sample[2,:] = 20*torch.log10(torch.tensor(np.load(path_to_sample + '/Z_noise_0.npy'))+1)
            #sample[3,:] = 20*torch.log10(torch.tensor(np.load(path_to_sample + '/Z_noise_1.npy'))+1) 
            #sample[4,:] = 20*torch.log10(torch.tensor(np.load(path_to_sample + '/Z_noise_2.npy'))+1) 
            sample[1,:] = torch.Tensor(np.load(path_to_sample + '/Z_active.npy'))
            sample[2,:] = torch.Tensor(np.load(path_to_sample + '/Z_noise_1.npy'))
            sample[3,:] = torch.Tensor(np.load(path_to_sample + '/Z_noise_2.npy'))
            sample[4,:] = torch.Tensor(np.load(path_to_sample + '/Z_noise_3.npy')) #0,1,2->1,2,3
            sample = sample.detach()
            sample *= self.normalization_factor
        else:
            with open(path_to_sample.split('seq')[0]+'NODES', 'rb') as fp:
                nodes = pickle.load(fp)
            exp = int(path_to_sample.split('exp_')[1].split('_as')[0])

            unfused_active = torch.Tensor(np.load(path_to_sample + '/Z_active.npy'))
            sample[1,:] = self.DR_model(unfused_active, [exp, nodes[0]])

            unfused_noise0 = torch.Tensor(np.load(path_to_sample + '/Z_noise_1.npy'))
            sample[2,:] = self.DR_model(unfused_noise0, [exp, nodes[1]])
            unfused_noise1 = torch.Tensor(np.load(path_to_sample + '/Z_noise_2.npy'))
            sample[3,:] = self.DR_model(unfused_noise1, [exp, nodes[2]])
            unfused_noise2 = torch.Tensor(np.load(path_to_sample + '/Z_noise_3.npy')) #changed obo error 0,1,2->1,2,3
            sample[4,:] = self.DR_model(unfused_noise2, [exp, nodes[3]])

            sample *= self.normalization_factor

        similarities = torch.Tensor(np.load(path_to_sample + '/cluster_similarities.npy'))
        #row 0: target
        #row 1: active input
        #row 2-4: noise inputs

    
        return sample, similarities

#dr = "/home/student/Documents/WHK_Projekt_1/code_DR/Dereverber_saves/DR-S1L_candidate"
#dataset_dr = Noise_Reduction_Dataset(dereverber=dr)
dataset_ff = Noise_Reduction_Dataset(dereverber=None)
#sample_dr, _ = dataset_dr[5]
#sample_ff, _ = dataset_ff[5]

#print(len(dataset_ff))


