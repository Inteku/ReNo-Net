import torch
from torch import nn
import keras.layers as kr
print('\033c')


class Identifier_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layers = [nn.Linear(280, 128),
                       nn.Linear(128, 64),
                       nn.Linear(64, 32),
                       nn.Linear(32, 6)]
        
        self.Layers = nn.ModuleList(self.Layers)

        self.Activations = [torch.relu,
                            torch.relu,
                            torch.relu,
                            nn.Softmax()]
    #def softmx(self, input):
    #    return torch.nn.functional.softmax(input)

    def forward(self, x):
        for L, A in zip(self.Layers, self.Activations):
            x = L(x)
            x = A(x)

        return x    