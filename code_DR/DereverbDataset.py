import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class HighlevelNodeDataset(torch.utils.data.Dataset):
    def __init__(self, flag:str='_2025', train=False, test=False):
        self.data = []

        if not (train ^ test):
            print("ERROR, SELECT ONE SUBSET ('train' OR 'test')")
        elif train:
            self.input_path = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/DR_data/HL_NODE{flag}/TRAIN/'
        elif test:
            self.input_path = f'/home/student/Documents/WHK_Projekt_1/work_remastered/DATA/DR_data/HL_NODE{flag}/TEST/'
        
        self.input_list = []
        self.feature_size = 280


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
    

if __name__ == "__main__":
    S = HighlevelNodeDataset(flag='_2025', test=True)
    data, n, ean = S[10]
    print(data.shape, n, ean)
    print([data[i].max().item() for i in range(11)])
    
    im = np.zeros((len(S), 280))
    for idx in range(len(S)):
        im[idx] = S[idx][0][0]

    plt.imshow(im, cmap='gray', vmax=1, aspect='auto')
    plt.colorbar()
    plt.show()
