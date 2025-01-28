from re import I
import torch
import numpy as np
from torch import nn
print('\033c')


class Mel_AutoEncoder_280(nn.Module):

    def __init__(self, **kwargs):

        super().__init__()
        self.bottleneck_vector = []
        self.encode = True
        self.decode = True
        self.indices_1 = []
        self.indices_2 = []
        self.indices_3 = []
        # INPUT
        # 1 x 128 x 312
        self.encode_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # 6 x 124 x 308
        self.encode_2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 6 x 62 x 154
        self.encode_3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 16 x 58 x 150
        self.encode_4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 16 x 29 x 75
        self.encode_5 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=6)
        # 8 x 24 x 70
        self.encode_6 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 8 x 12 x 35
        # flatten
        # 1 x 1 x 3360
        self.encode_7 = nn.Linear(in_features=3360, out_features=840)
        # 1 x 1 x 840
        self.encode_8 = nn.Linear(in_features=840, out_features=420)
        # 1 x 1 x 420
        self.encode_bn = nn.Linear(in_features=420, out_features=280)
        # 1 x 1 x 280
        self.dropout = nn.Dropout(0.2)
        self.decode_bn = nn.Linear(in_features=280, out_features=420)
        # 1 x 1 x 420
        self.decode_8 = nn.Linear(in_features=420, out_features=840)
        # 1 x 1 x 840
        self.decode_7 = nn.Linear(in_features=840, out_features=3360)
        # 1 x 1 x 3360
        # unflatten
        # 8 x 12 x 35
        self.decode_6 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 8 x 24 x 70
        self.decode_5 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=6)
        # 16 x 29 x 75
        self.decode_4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 16 x 58 x 150
        self.decode_3 = nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5)
        # 6 x 62 x 154
        self.decode_2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # 6 x 124 x 308
        self.decode_1 = nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=5)
        # OUTPUT
        # 1 x 128 x 312

    def forward(self, x):
        if self.encode:
            x = x.reshape(-1, 1, 128, 312)
            # ENCODE
            x = torch.sigmoid(self.encode_1(x))
            x, self.indices_1 = self.encode_2(x) #no need for relu when pooling
            x = torch.sigmoid(self.encode_3(x))
            x, self.indices_2 = self.encode_4(x) #no need for relu when pooling
            x = torch.relu(self.encode_5(x))
            x, self.indices_3 = self.encode_6(x) #no need for relu when pooling

            x = torch.reshape(x, (-1,1,1,3360))

            x = torch.relu(self.encode_7(x))
            x = torch.relu(self.encode_8(x))
            x = torch.relu(self.encode_bn(x))
            # BOTTLENECK
            self.bottleneck_vector = x.detach().clone()

        if self.training and self.encode and self.decode:
            x = self.dropout(x)

        if self.decode:
            # DECODE
            x = torch.relu(self.decode_bn(x))
            x = torch.relu(self.decode_8(x))
            x = torch.relu(self.decode_7(x))

            x = torch.reshape(x, (-1,8,12,35))

            x = self.decode_6(x, self.indices_3)
            x = torch.relu(self.decode_5(x))
            x = self.decode_4(x, self.indices_2) #no need for relu when unpooling
            x = torch.sigmoid(self.decode_3(x))
            x = self.decode_2(x, self.indices_1) #no need for relu when unpooling
            x = torch.sigmoid(self.decode_1(x))
        
        return x

    def get_bottleneck(self):
        return self.bottleneck_vector
    
    def encode_only(self):
        self.encode = True
        self.decode = False

    def decode_only(self):
        self.encode = False
        self.decode = True
    
    def encode_decode(self):
        self.encode = True
        self.decode = True




