import torch
import torch.nn as nn


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
        active = inputs[0,:]
        noise1 = inputs[1,:]
        noise2 = inputs[2,:]
        noise3 = inputs[3,:]
        for L, A in zip(self.Layers, self.Activation):
            noise1 = A(L(noise1))
            noise2 = A(L(noise2))
            noise3 = A(L(noise3))
        epsilon = 1/(torch.sum(psi,1))
        psi = psi.unsqueeze(2)
        weigted_noise = psi[:,0] * noise1 + psi[:,1] * noise2 + psi[:,2] * noise3

        result = self.alpha * active + epsilon.unsqueeze(1) * self.beta * weigted_noise
        return result
