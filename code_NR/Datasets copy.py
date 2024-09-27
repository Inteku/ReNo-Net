import torch
import numpy as np
import os
import random
print('\033c')

class Noise_Reduction_Dataset(torch.utils.data.Dataset):
    def __init__(self, train=False, test=False, validate=False, test_size=300, validation_size=300, shuffle_seed=280):

        path_prefix = '/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/NR/NR_featurefusion/' ################
        self.all_paths_to_samples = []
        for dirPath, dirNames, fileNames in os.walk(path_prefix):
            if dirNames == []:
                self.all_paths_to_samples.append(dirPath)

        naughty_list = np.load('naughty_list.npy') #enthält indices der samples die nan sind
        #entferne nan samples
        self.all_paths_to_samples = [element for index, element in enumerate(self.all_paths_to_samples) if index not in naughty_list]
        total_len = len(self.all_paths_to_samples)

        random.seed(shuffle_seed)
        random.shuffle(self.all_paths_to_samples)

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
        #inputs
        #sample[1,:] = 20*torch.log10(torch.tensor(np.load(path_to_sample + '/Z_active.npy'))+1) 
        #sample[2,:] = 20*torch.log10(torch.tensor(np.load(path_to_sample + '/Z_noise_0.npy'))+1)
        #sample[3,:] = 20*torch.log10(torch.tensor(np.load(path_to_sample + '/Z_noise_1.npy'))+1) 
        #sample[4,:] = 20*torch.log10(torch.tensor(np.load(path_to_sample + '/Z_noise_2.npy'))+1) 
        sample[1,:] = torch.Tensor(np.load(path_to_sample + '/Z_active.npy'))
        sample[2,:] = torch.Tensor(np.load(path_to_sample + '/Z_noise_0.npy'))
        sample[3,:] = torch.Tensor(np.load(path_to_sample + '/Z_noise_1.npy'))
        sample[4,:] = torch.Tensor(np.load(path_to_sample + '/Z_noise_2.npy'))

        similarities = torch.Tensor(np.load(path_to_sample + '/cluster_similarities.npy'))
        #row 0: target
        #row 1: active input
        #row 2-4: noise inputs

        sample *= self.normalization_factor

        return sample, similarities


    

