from re import I
import torch
from torch import nn
#from SoundDatasets import HighlevelNodeDataset
import json
from utilities import get_memberships
print('\033c')

class Baseline():
    def average(self, all_x, numInputs):
        x = torch.zeros(1,280)
        for inputs_idx in range(numInputs):
            x += all_x[inputs_idx].unsqueeze(0)
        x = x / numInputs
        return x
    
    def single_node(self, all_x):
        return all_x[0].unsqueeze(0)
    
    def feature_fusion(self, num_exp, node_list, highlevels, feature_size=280):

        #load json-data
        #load_prefix = '/home/student/Documents/BA/work/experiments/exp_'
        #memberships_data = open(load_prefix + str(num_exp) + '/memberships.json')
        #memberships = json.load(memberships_data) 
        #memberships = list(memberships.values())[0]
        memberships = get_memberships(num_exp)

        sum = torch.zeros(1,feature_size)
        sum_memberships = 0
        for node_idx in range(len(node_list)):
            this_node = node_list[node_idx]
            membership = memberships[this_node]
            sum += membership * highlevels[node_idx].unsqueeze(0)
            sum_memberships += membership
        
        Z_dach = sum/sum_memberships

        return Z_dach
    

class siameseDereverberMemb(nn.Module):
    def __init__(self, feature_size, num_layers, is_Ecapa=False, **kwargs):
        super().__init__()
        self.is_Ecapa = is_Ecapa

        self.feature_size = feature_size

        self.Layers = []
        for l in range(num_layers):
            self.Layers.append(nn.Linear(in_features=feature_size, out_features=feature_size))
        self.Layers = nn.ModuleList(self.Layers)


    def forward(self, all_x, exp_and_nodes):
        activation = nn.Sigmoid()
        if self.is_Ecapa:
            activation = nn.Tanhshrink()

        exp = exp_and_nodes[0]
        nodes = exp_and_nodes[1]
        n = len(nodes)
        memberships = torch.Tensor(get_memberships(exp))
        this_memberships = memberships[nodes]

        weighted_sum = torch.zeros(self.feature_size)
        
        for node in range(n):
            x = all_x[node]
            for Layer in self.Layers:
                x = Layer(x)
                x = activation(x)
            weighted_sum += this_memberships[node] * x
        
        membership_sum = sum(this_memberships)

        return weighted_sum / membership_sum
    

