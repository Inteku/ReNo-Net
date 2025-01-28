from re import I
import torch
from torch import nn
import keras.layers as kr
from SoundDatasets import HighlevelNodeDataset
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


class Dereverber_v1(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.Layers = [nn.Linear(in_features=420, out_features=840),
                       nn.Linear(in_features=840, out_features=840),
                       nn.Dropout(0.4),
                       nn.Linear(in_features=840, out_features=420)]
        self.Layers = nn.ModuleList(self.Layers)

        
    def forward(self, all_x, numInputs):

        numSamples = all_x.size(dim=0)
        # create empty tensor to collect the output of each sample
        OUTPUT = torch.zeros(numSamples, 1, 420)
        #each sample
        for sample_idx in range(numSamples): 
            inputs_in_this_sample = numInputs[sample_idx]
            x_sample = all_x[sample_idx].unsqueeze(0)
            outputs_of_this_sample = torch.zeros(1, 1, 420)
            #each input vector
            for input_idx in range(inputs_in_this_sample):          
                x = x_sample[:,input_idx,:]
                #each layer
                for L in self.Layers:
                    x = L(x)
                    x = torch.relu(x)

                outputs_of_this_sample += x.clone()
            
            # append the mean output of this sample to output
            OUTPUT[sample_idx, :, :] = outputs_of_this_sample / inputs_in_this_sample #.squeeze(0)?


        return OUTPUT
        





class Dereverber_v2(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.Layers = [nn.Linear(in_features=420, out_features=480),
                       nn.Dropout(0.1),
                       nn.Linear(in_features=480, out_features=420)]
        self.Layers = nn.ModuleList(self.Layers)

        
    def forward(self, all_x, numInputs):

        numSamples = all_x.size(dim=0)
        # create empty tensor to collect the output of each sample
        OUTPUT = torch.zeros(numSamples, 1, 420)
        #each sample
        for sample_idx in range(numSamples): 
            inputs_in_this_sample = numInputs[sample_idx]
            x_sample = all_x[sample_idx].unsqueeze(0)
            outputs_of_this_sample = torch.zeros(1, 1, 420)
            #each input vector
            for input_idx in range(inputs_in_this_sample):          
                x = x_sample[:,input_idx,:]
                #each layer
                for L in self.Layers:
                    x = L(x)
                    x = torch.relu(x)

                outputs_of_this_sample += x.clone()
            
            # append the mean output of this sample to output
            OUTPUT[sample_idx, :, :] = outputs_of_this_sample / inputs_in_this_sample #.squeeze(0)?


        return OUTPUT







class Dereverber_v3(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.preLayers = [nn.Linear(in_features=420, out_features=420),
                          nn.Dropout(0.1),
                          nn.Linear(in_features=420, out_features=420)]
        
        self.postLayers = [nn.Linear(in_features=420, out_features=840),
                           nn.Dropout(0.2),
                           nn.Linear(in_features=840, out_features=420)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        
    def forward(self, all_x, numInputs):

        numSamples = all_x.size(dim=0)
        # create empty tensor to collect the output of each sample
        OUTPUT = torch.zeros(numSamples, 1, 420)
        #each sample
        for sample_idx in range(numSamples): 
            inputs_in_this_sample = numInputs[sample_idx]
            x_sample = all_x[sample_idx].unsqueeze(0)
            outputs_of_this_sample = torch.zeros(1, 1, 420)
            #each input vector
            for input_idx in range(inputs_in_this_sample):          
                x = x_sample[:,input_idx,:]
                #each layer
                for L in self.preLayers:
                    x = L(x)
                    x = torch.relu(x)

                outputs_of_this_sample += x.clone()
            
            # send the mean through the post layers
            y = outputs_of_this_sample / inputs_in_this_sample
            for L in self.postLayers:
                y = L(y)
                y = torch.relu(y)

            OUTPUT[sample_idx, :, :] = y

        return OUTPUT
    





class Dereverber_v4(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.preLayers = [nn.Linear(in_features=280, out_features=280)]#,
                          #nn.Dropout(0.1),
                          #nn.Linear(in_features=280, out_features=280)]
        
        self.postLayers = [nn.Linear(in_features=280, out_features=560),
                           nn.Linear(in_features=560, out_features=560),
                           nn.Dropout(0.2),
                           nn.Linear(in_features=560, out_features=280)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        
    def forward(self, all_x, numInputs):

        numSamples = all_x.size(dim=0)
        # create empty tensor to collect the output of each sample
        OUTPUT = torch.zeros(numSamples, 1, 280)
        #each sample
        for sample_idx in range(numSamples): 
            inputs_in_this_sample = numInputs[sample_idx]
            x_sample = all_x[sample_idx].unsqueeze(0)
            outputs_of_this_sample = torch.zeros(1, 1, 280)
            #each input vector
            for input_idx in range(inputs_in_this_sample):          
                x = x_sample[:,input_idx,:]
                #each layer
                for L in self.preLayers:
                    x = L(x)
                    x = torch.relu(x)

                outputs_of_this_sample += x.clone()
            
            # send the mean through the post layers
            y = outputs_of_this_sample / inputs_in_this_sample
            for L in self.postLayers:
                y = L(y)
                y = torch.relu(y)

            OUTPUT[sample_idx, :, :] = y.clone()

        return OUTPUT
    






class Dereverber_v5(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.preLayers = [nn.Linear(in_features=280, out_features=280),
                          nn.Dropout(0.3),
                          nn.Linear(in_features=280, out_features=280)]
        
        self.postLayers = [nn.Linear(in_features=280, out_features=280),
                           nn.Linear(in_features=280, out_features=280),
                           nn.Dropout(0.3),
                           nn.Linear(in_features=280, out_features=280),
                           nn.Linear(in_features=280, out_features=280)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        
    def forward(self, all_x, numInputs):

        # create empty tensor to collect the output of each sample
        y = torch.zeros(1, 280)
 
        x_sample = all_x
        #each input vector
        for input_idx in range(numInputs):          
            x = x_sample[input_idx]
            #each layer
            for L in self.preLayers:
                x = L(x)
                x = torch.relu(x)

            y += x.clone()
        
        # send the mean through the post layers
        y = y / numInputs
        for L in self.postLayers:
            y = L(y)
            y = torch.relu(y)

        OUTPUT = y.clone()

        return OUTPUT
    









class Dereverber_v6(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.preLayers = [nn.Linear(in_features=280, out_features=280),
                          nn.Dropout(0.3)]
        
        self.postLayers = [nn.Linear(in_features=280, out_features=280),
                           nn.Dropout(0.3)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        #self.postLayers = nn.ModuleList(self.postLayers)

        
    def forward(self, all_x, numInputs):

        # create empty tensor to collect the output of each sample
        y = torch.zeros(1, 280)
 
        x_sample = all_x
        #each input vector
        for input_idx in range(numInputs):          
            x = x_sample[input_idx]
            #each layer
            for L in self.preLayers:
                x = L(x)
                x = torch.relu(x)

            y += x.clone()
        
        # send the mean through the post layers
        y = y / numInputs
        #for L in self.postLayers:
        #    y = L(y)
        #    y = torch.relu(y)


        return y
    





class Dereverber_v7(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.preLayers = [nn.Linear(in_features=280, out_features=280),
                          nn.Dropout(0.4)]
        
        self.postLayers = [nn.Linear(in_features=280, out_features=280),
                           nn.Dropout(0.3)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        
    def forward(self, all_x, numInputs):

        # create empty tensor to collect the output of each sample
        y = torch.zeros(1, 280)
 
        x_sample = all_x
        #each input vector
        for input_idx in range(numInputs):          
            x = x_sample[input_idx]
            #each layer
            for L in self.preLayers:
                x = L(x)
                x = torch.relu(x)#tanh

            y += x.clone()
        
        # send the mean through the post layers
        y = y / numInputs
        for L in self.postLayers:
            y = L(y)
            y = torch.relu(y)#sigmoid


        return y
    



class ConvDereverber_v2(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
                            #1x10x280
        self.preLayers = [nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, dilation=1, stride=1),
                            #4x10x280
                          nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, padding=2, dilation=1, stride=1)]
                            #1x10x280
        
        self.postLayers = [nn.Linear(in_features=2800, out_features=280)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        
    def forward(self, all_x):

        # create empty tensor to collect the output of each sample
        #y = torch.zeros(1, 280)
 
        x = all_x.unsqueeze(0)

        for L in self.preLayers:
            #print(x.size())
            x = L(x)
            x = torch.tanh(x)
        
        x = torch.flatten(x)
        #print(x.size())
        for L in self.postLayers:
            x = L(x)
            x = torch.sigmoid(x)


        return x
    
#CDRM: Convolutional DeReverber with Memberships
class ConvDereverberMemb_v1(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
                            #1x10x280
        self.preLayers = [nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, dilation=1, stride=1),
                            #4x10x280
                          nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, padding=2, dilation=1, stride=1)]
                            #1x10x280
        
        self.postLayers = [nn.Linear(in_features=2800, out_features=280)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        
    def forward(self, all_x, exp_and_nodes):

        exp = exp_and_nodes[0]
        nodes = exp_and_nodes[1]
        n = len(nodes)
        memberships = torch.Tensor(get_memberships(exp))
        this_memberships = memberships[nodes]
        sorted_indices = torch.argsort(this_memberships, dim=0, descending=True).int()
        all_x[0:n] = all_x[0:n][sorted_indices]
 
        x = all_x.unsqueeze(0)

        for L in self.preLayers:
            #print(x.size())
            x = L(x)
            x = torch.tanh(x)
        
        x = torch.flatten(x)
        #print(x.size())
        for L in self.postLayers:
            x = L(x)
            x = torch.sigmoid(x)


        return x
    



#CDRMS: Convolutional DeReverber with Memberships, with Spacing
class ConvDereverberMemb_v2(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
                            #1x10x280
        self.preLayers = [nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, dilation=1, stride=1),
                            #4x10x280
                          nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, padding=2, dilation=1, stride=1)]
                            #1x10x280
        
        self.postLayers = [nn.Linear(in_features=2800, out_features=280)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        #permutation creates roughly equal spacing between the vectors, for every number of vectors (1-10) 
        self.permutation_matrix = [[0,1,2,3,4,5,6,7,8,9],
                                   [0,2,3,4,5,6,7,8,9,1],
                                   [0,3,4,5,1,6,7,8,9,2],
                                   [0,4,5,1,6,7,2,8,9,3],
                                   [0,5,1,6,2,7,3,8,9,4],
                                   [0,1,6,2,7,3,8,4,9,5],
                                   [0,1,7,2,3,8,4,5,9,6],
                                   [0,1,2,8,3,4,9,5,6,7],
                                   [0,1,2,3,9,4,5,6,7,8],
                                   [0,1,2,3,4,5,6,7,8,9]]

        
    def forward(self, all_x, exp_and_nodes):

        exp = exp_and_nodes[0]
        nodes = exp_and_nodes[1]
        n = len(nodes)
        memberships = torch.Tensor(get_memberships(exp))
        this_memberships = memberships[nodes]
        sorted_indices = torch.argsort(this_memberships, dim=0, descending=True).int()
        all_x[0:n] = all_x[0:n][sorted_indices]

        all_x = all_x[self.permutation_matrix[n-1]]
 
        x = all_x.unsqueeze(0)

        for L in self.preLayers:
            #print(x.size())
            x = L(x)
            x = torch.tanh(x)
        
        x = torch.flatten(x)
        #print(x.size())
        for L in self.postLayers:
            x = L(x)
            x = torch.sigmoid(x)


        return x
    
#CDRMD: Convolutional DeReverber with Memberships, with Duplication
class ConvDereverberMemb_v3(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
                            #1x10x280
        self.preLayers = [nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, dilation=1, stride=1),
                            #4x10x280
                          nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, padding=2, dilation=1, stride=1)]
                            #1x10x280
        
        self.postLayers = [nn.Linear(in_features=2800, out_features=280)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        #permutation creates roughly equal spacing between the vectors, for every number of vectors (1-10) 
        self.permutation_matrix = [[0,0,0,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,1,1,1,1,1],
                                   [0,0,0,0,1,1,1,2,2,2],
                                   [0,0,0,1,1,1,2,2,3,3],
                                   [0,0,1,1,2,2,3,3,4,4],
                                   [0,0,1,1,2,2,3,3,4,5],
                                   [0,0,1,1,2,2,3,4,5,6],
                                   [0,0,1,1,2,3,4,5,6,7],
                                   [0,0,1,2,3,4,5,6,7,8],
                                   [0,1,2,3,4,5,6,7,8,9]]

        
    def forward(self, all_x, exp_and_nodes):

        exp = exp_and_nodes[0]
        nodes = exp_and_nodes[1]
        n = len(nodes)
        memberships = torch.Tensor(get_memberships(exp))
        this_memberships = memberships[nodes]
        sorted_indices = torch.argsort(this_memberships, dim=0, descending=True).int()
        all_x[0:n] = all_x[0:n][sorted_indices]

        all_x = all_x[self.permutation_matrix[n-1]]
 
        x = all_x.unsqueeze(0)

        for L in self.preLayers:
            #print(x.size())
            x = L(x)
            x = torch.tanh(x)
        
        x = torch.flatten(x)
        #print(x.size())
        for L in self.postLayers:
            x = L(x)
            x = torch.sigmoid(x)


        return x
    





#CDRM: Convolutional DeReverber with Memberships, variable feature size
class ConvDereverberBest(nn.Module):
    
    def __init__(self, feature_size, **kwargs):
        super().__init__()
                            #1x10x280=feature_size
        self.preLayers = [nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, dilation=1, stride=1),
                            #4x10x280
                          nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, padding=2, dilation=1, stride=1)]
                            #1x10x280
        
        self.postLayers = [nn.Linear(in_features=10*feature_size, out_features=feature_size)]
        
        self.preLayers = nn.ModuleList(self.preLayers)
        self.postLayers = nn.ModuleList(self.postLayers)

        
    def forward(self, all_x, exp_and_nodes):

        exp = exp_and_nodes[0]
        nodes = exp_and_nodes[1]
        n = len(nodes)
        memberships = torch.Tensor(get_memberships(exp))
        this_memberships = memberships[nodes]
        sorted_indices = torch.argsort(this_memberships, dim=0, descending=True).int()
        all_x[0:n] = all_x[0:n][sorted_indices]
 
        x = all_x.unsqueeze(0)

        for L in self.preLayers:
            #print(x.size())
            x = L(x)
            x = torch.tanh(x)
        
        x = torch.flatten(x)
        #print(x.size())
        for L in self.postLayers:
            x = L(x)
            x = torch.sigmoid(x)


        return x
    

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
        activation = nn.ReLU()#nn.Sigmoid() #fatal
        if self.is_Ecapa:
            activation = nn.Tanhshrink()

        exp = exp_and_nodes[0]
        nodes = exp_and_nodes[1]
        n = len(nodes)
        memberships = torch.tensor(get_memberships(exp))
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
    

if __name__ == "__main__":
    dr = siameseDereverberMemb(280, num_layers=1)
    x = torch.ones(10,280)
    ean = [91, [22, 24, 26]]
    x = dr(x, ean)
