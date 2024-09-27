from re import I
import torch
from torch import nn
#from SoundDatasets import HighlevelNodeDataset
import json
from utilities import get_memberships
print('\033c')


    

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
    

