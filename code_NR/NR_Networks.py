import torch
import torch.nn as nn
import numpy as np


class Denoiser_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.active_layer = nn.Linear(280,280)
        self.noise_layer = nn.Linear(280,280)
        self.noise_dropout = nn.Dropout(0.2)
        self.sum_layer = nn.Linear(280,280)

    def forward(self, inputs):
        act = self.active_layer(inputs[:,0,:])
        act = nn.Sigmoid()(act)

        noise1 = self.noise_layer(inputs[:,1,:])
        noise1 = nn.Sigmoid()(noise1)
        noise1 = self.noise_dropout(noise1)

        noise2 = self.noise_layer(inputs[:,2,:])
        noise2 = nn.Sigmoid()(noise2)
        noise2 = self.noise_dropout(noise2)

        noise3 = self.noise_layer(inputs[:,3,:])
        noise3 = nn.Sigmoid()(noise3)
        noise3 = self.noise_dropout(noise3)

        noise_sum = noise1 + noise2 + noise3
        noise_sum = self.sum_layer(noise_sum)
        noise_sum = nn.Sigmoid()(noise_sum)

        return act - noise_sum

class Denoiser_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_layer = nn.Linear(280,280)
        self.noise_dropout = nn.Dropout(0.2)

    def forward(self, inputs, psi):
        active = inputs[:,0,:]
        
        noise1 = self.noise_layer(inputs[:,1,:])
        noise1 = nn.Sigmoid()(noise1)
        noise1 = self.noise_dropout(noise1)

        noise2 = self.noise_layer(inputs[:,2,:])
        noise2 = nn.Sigmoid()(noise2)
        noise2 = self.noise_dropout(noise2)

        noise3 = self.noise_layer(inputs[:,3,:])
        noise3 = nn.Sigmoid()(noise3)
        noise3 = self.noise_dropout(noise3)

        noise_estimate = psi[:,0].unsqueeze(1)*noise1 + psi[:,1].unsqueeze(1)*noise2 + psi[:,2].unsqueeze(1)*noise3
        #noise_estimate = noise_estimate / torch.sum(psi, 1)

        #return active - noise_estimate #>=25.07.
        return nn.Sigmoid()(active - noise_estimate) #<=23.07.
        
    
class Learnable_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.2))
        self.beta = nn.Parameter(torch.tensor(0.2))

    def forward(self, inputs, psi):
        Z_c1_active = inputs[:,0]
        Z_c2 = inputs[:,1]
        Z_c3 = inputs[:,2]
        Z_c4 = inputs[:,3]

        epsilon = 1/(torch.sum(psi,1))
        weighted_noise = (psi[:,0].unsqueeze(1)*Z_c2 + psi[:,1].unsqueeze(1)*Z_c3 + psi[:,2].unsqueeze(1)*Z_c4)
            

        Z_NR = self.alpha*Z_c1_active - epsilon*self.beta*weighted_noise
        #Z_NR = 1.2*Z_c1_active - epsilon*0.2*weighted_noise + 1e-9*(self.alpha+self.beta)

        return Z_NR
    

class Denoiser_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layers = nn.ModuleList([nn.Linear(280, 140),
                                     nn.Linear(140, 70),
                                     nn.Linear(70, 140),
                                     nn.Linear(140, 280)])
        self.Activation = [nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.Tanh()]
        self.alpha = nn.Parameter(torch.tensor(1.2))
        self.beta = nn.Parameter(torch.tensor(0.2))

    def forward(self, inputs:torch.tensor, psi:torch.tensor):
        if not inputs.dim() == 3 and not psi.dim() == 2:
            raise ValueError("Inputs expected to be 3-dimesnional, Similarities excpected to be 2-dimensional")
        active = inputs[:,0,:]
        noise1 = inputs[:,1,:]
        noise2 = inputs[:,2,:]
        noise3 = inputs[:,3,:]
        for L, A in zip(self.Layers, self.Activation):
            noise1 = A(L(noise1))
            noise2 = A(L(noise2))
            noise3 = A(L(noise3))
        epsilon = 1/(torch.sum(psi,1))
        psi = psi.unsqueeze(2)
        weigted_noise = psi[:,0] * noise1 + psi[:,1] * noise2 + psi[:,2] * noise3
        #print(f"{self.alpha.size()}, {active.size()}, {epsilon.size()}, {self.beta.size()}, {weigted_noise.size()}")
        result = self.alpha * active + epsilon.unsqueeze(1) * self.beta * weigted_noise
        return result
