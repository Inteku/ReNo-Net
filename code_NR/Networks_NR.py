### ORIGINAL AUS BA MICHAEL ###
import torch
import torch.nn as nn
import numpy as np
#import Datasets_NR as set

#Networks 1-7 are used with features of size 420

class Speaker_ID(nn.Module):
    def __init__(self):
        super().__init__()
        self.Layer1 = nn.Linear(280,128)
        self.Layer2 = nn.Linear(128,64)
        self.Layer3 = nn.Linear(64,32)
        self.Layer4 = nn.Linear(32,6)
    
    def forward(self, x):
        x = torch.relu(self.Layer1(x))
        x = torch.relu(self.Layer2(x))
        x = torch.relu(self.Layer3(x))
        x = nn.functional.softmax(self.Layer4(x))

        return(x)


class Network1(nn.Module):
    def __init__(self):
        super().__init__()

        self.noise1 = nn.Linear(420,420)
        self.noise2 = nn.Linear(420,420)
        self.dropout = nn.Dropout(0.05)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):
        out_2 = torch.sigmoid(self.noise1(Z_c2))
        out_2 = self.dropout(out_2)
        out_3 = torch.sigmoid(self.noise1(Z_c3))
        out_3 = self.dropout(out_3)
        out_4 = torch.sigmoid(self.noise1(Z_c4))
        out_4 = self.dropout(out_4)

        out_sum = (out_2+out_3+out_4)/3
        out_sum = torch.sigmoid(self.noise2(out_sum))

        return Z_c1 - out_sum

class Network2(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.Linear(420, 420)
        self.noise1 = nn.Linear(420, 420)
        self.noise2 = nn.Linear(420, 420)

        self.dropout = nn.Dropout(0.05)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):
        out_act = self.act(Z_c1)
        out_act = torch.sigmoid(out_act)
        out_act = self.dropout(out_act)
         
        out2 = torch.sigmoid(self.noise1(Z_c2))
        out2 = torch.sigmoid(self.noise2(out2))
        out2 = self.dropout(out2)
        
        out3 = torch.sigmoid(self.noise1(Z_c3))
        out3 = torch.sigmoid(self.noise2(out3))
        out3 = self.dropout(out3)

        out4 = torch.sigmoid(self.noise1(Z_c4))
        out4 = torch.sigmoid(self.noise2(out4))
        out4 = self.dropout(out4)
        
        return out_act - (out2+out3+out4)/3

class Network3(nn.Module):
    def __init__(self):
        super().__init__()

        self.act = nn.Linear(420,420)

        self.noise1 = nn.Linear(420,840)
        self.noise2 = nn.Linear(840,840)
        self.noise3 = nn.Linear(840,420)

        self.noise_sum1 = nn.Linear(420,840)
        self.noise_sum2 = nn.Linear(840,420)

        self.dropout = nn.Dropout(0.4)
    
    def forward(self, Z_target, Z_c2, Z_c3, Z_c4,psi):
        out_act = torch.sigmoid(self.act(Z_target))
        out_act = self.dropout(out_act)

        out_2 = torch.sigmoid(self.noise1(Z_c2))
        out_2 = torch.sigmoid(self.noise2(out_2))
        out_2 = torch.sigmoid(self.noise3(out_2))
        out_2 = self.dropout(out_2)

        out_3 = torch.sigmoid(self.noise1(Z_c3))
        out_3 = torch.sigmoid(self.noise2(out_3))
        out_3 = torch.sigmoid(self.noise3(out_3))
        out_3 = self.dropout(out_3)

        out_4 = torch.sigmoid(self.noise1(Z_c4))
        out_4 = torch.sigmoid(self.noise2(out_4))
        out_4 = torch.sigmoid(self.noise3(out_4))
        out_4 = self.dropout(out_4)

        out_sum = out_2+out_3+out_4

        out_sum = torch.sigmoid(self.noise_sum1(out_sum))
        out_sum = torch.sigmoid(self.noise_sum2(out_sum))

        return out_act - out_sum

class Network4(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise = nn.Linear(420,420)
        self.noise_sum = nn.Linear(420,420)
        self.act = nn.Linear(420,420)
        self.dropout = nn.Dropout(0.5)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

        out_act = torch.sigmoid(self.act(Z_c1))
        #out_act = self.dropout(out_act)

        out_2 = torch.sigmoid(self.noise(Z_c2))
        #out_2 = self.dropout(out_2)
        out_3 = torch.sigmoid(self.noise(Z_c3))
        #out_3 = self.dropout(out_3)
        out_4 = torch.sigmoid(self.noise(Z_c4))
        #out_4 = self.dropout(out_4)

        Z_sum = out_2+out_3+out_4

        out_sum = torch.relu(self.noise_sum(Z_sum))
        out_sum = self.dropout(out_sum)
        
        return out_act - out_sum


class Network6(nn.Module): #net 5 uses alpha, beta and the cossim similarities
    def __init__(self):
        super().__init__()
        self.noise1 = nn.Linear(420,1680)
        self.noise2 = nn.Linear(1680,1680)
        self.noise3 = nn.Linear(1680,420)

        self.act1 = nn.Linear(420,1680)
        self.act2 = nn.Linear(1680,420)

        self.sum1 = nn.Linear(420,1680)
        self.sum2 = nn.Linear(1680,420)

        self.dropout = nn.Dropout(0.1)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

        Z_c1 = 1.2 * Z_c1
        Z_c1 = torch.sigmoid(self.act1(Z_c1))
        Z_c1 = self.dropout(Z_c1)
        Z_c1 = torch.relu(self.act2(Z_c1))
        
        Z_c2 = psi[0] * Z_c2
        Z_c2 = torch.sigmoid(self.noise1(Z_c2))
        Z_c2 = torch.relu(self.noise2(Z_c2))
        Z_c2 = self.dropout(Z_c2)
        Z_c2 = torch.relu(self.noise3(Z_c2))

        Z_c3 = psi[1] * Z_c3
        Z_c3 = torch.sigmoid(self.noise1(Z_c3))
        Z_c3 = torch.relu(self.noise2(Z_c3))
        Z_c3 = self.dropout(Z_c3)
        Z_c3 = torch.relu(self.noise3(Z_c3))

        Z_c4 = psi[2] * Z_c4
        Z_c4 = torch.sigmoid(self.noise1(Z_c4))
        Z_c4 = torch.relu(self.noise2(Z_c4))
        Z_c4 = self.dropout(Z_c4)
        Z_c4 = torch.relu(self.noise3(Z_c4))

        Z_sum = Z_c2 + Z_c3 + Z_c4
        Z_sum = 0.2 * Z_sum
        Z_sum = torch.sigmoid(self.sum1(Z_sum))
        Z_sum = torch.sigmoid(self.sum2(Z_sum))
        #Z_sum = self.dropout(Z_sum)

        return Z_c1 - Z_sum
    

class Network5(nn.Module): #net 5 uses alpha, beta and the cossim similarities
    def __init__(self):
        super().__init__()
        self.noise = nn.Linear(420,420)
        self.act = nn.Linear(420,420)
        self.sum = nn.Linear(420,420)
        self.dropout = nn.Dropout(0.2)
    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

        Z_c1 = 1.2 * Z_c1
        Z_c1 = torch.sigmoid(self.act(Z_c1))
        Z_c1 = self.dropout(Z_c1)

        Z_c2 = psi[0] * Z_c2
        Z_c2 = torch.sigmoid(self.noise(Z_c2))
        Z_c2 = self.dropout(Z_c2)

        Z_c3 = psi[1] * Z_c3
        Z_c3 = torch.sigmoid(self.noise(Z_c3))
        Z_c3 = self.dropout(Z_c3)

        Z_c4 = psi[2] * Z_c4
        Z_c4 = torch.sigmoid(self.noise(Z_c4))
        Z_c4 = self.dropout(Z_c4)

        Z_sum = Z_c2 + Z_c3 + Z_c4
        Z_sum = 0.2 * Z_sum
        Z_sum = torch.sigmoid(self.sum(Z_sum))
        #Z_sum = self.dropout(Z_sum)

        return Z_c1 - Z_sum
    
class Network7(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise1 = nn.Linear(1260,2520)
        self.noise2 = nn.Linear(2520,2520)
        self.noise3 = nn.Linear(2520,1680)
        self.noise4 = nn.Linear(1680,840)
        self.noise5 = nn.Linear(840,420)

        self.act1 = nn.Linear(420,840)
        self.act2 = nn.Linear(840,420)

        self.out = nn.Linear(420,420)

        self.dropout = nn.Dropout(0.2)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

        Z_c1 = torch.sigmoid(self.act1(Z_c1))
        Z_c1 = torch.sigmoid(self.act2(Z_c1))

        Z_noise = torch.cat((Z_c2,Z_c3, Z_c4), dim=3)

        Z_noise = torch.sigmoid(self.noise1(Z_noise))
        Z_noise = torch.sigmoid(self.noise2(Z_noise))
        Z_noise = self.dropout(Z_noise)
        Z_noise = torch.sigmoid(self.noise3(Z_noise))
        Z_noise = torch.sigmoid(self.noise4(Z_noise))
        Z_noise = torch.sigmoid(self.noise5(Z_noise))

        Z_out = Z_c1 - Z_noise
        Z_out = torch.relu(self.out(Z_out))

        return Z_out
    
class Network4_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise = nn.Linear(280,280)
        self.noise_sum = nn.Linear(280,280)
        self.act = nn.Linear(280,280)
        self.dropout = nn.Dropout(0.4)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

        out_act = torch.tanh(self.act(Z_c1))
        out_act = self.dropout(out_act)

        out_2 = torch.tanh(self.noise(Z_c2))
        out_2 = self.dropout(out_2)
        out_3 = torch.tanh(self.noise(Z_c3))
        out_3 = self.dropout(out_3)
        out_4 = torch.tanh(self.noise(Z_c4))
        out_4 = self.dropout(out_4)

        Z_sum = out_2+out_3+out_4

        out_sum = torch.tanh(self.noise_sum(Z_sum))
        #out_sum = self.dropout(out_sum)
        
        return out_act - out_sum
    

class Network7_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise1 = nn.Linear(840,1680)
        self.noise2 = nn.Linear(1680,840)
        self.noise3 = nn.Linear(840,560)
        self.noise4 = nn.Linear(560,280)

        self.act1 = nn.Linear(280,560)
        self.act2 = nn.Linear(560,280)

        self.out = nn.Linear(280,280)

        self.dropout = nn.Dropout(0.4)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

        Z_c1 = torch.tanh(self.act1(Z_c1))
        Z_c1 = torch.tanh(self.act2(Z_c1))
        Z_c1 = self.dropout(Z_c1)

        Z_noise = torch.cat((Z_c2,Z_c3, Z_c4), dim=3)

        Z_noise = torch.tanh(self.noise1(Z_noise))
        Z_noise = torch.tanh(self.noise2(Z_noise))
        Z_noise = torch.tanh(self.noise3(Z_noise))
        Z_noise = torch.tanh(self.noise4(Z_noise))
        Z_noise = self.dropout(Z_noise)

        Z_out = Z_c1 - Z_noise
        Z_out = torch.tanh(self.out(Z_out))

        return Z_out

class Network5_2(nn.Module): #net 5 uses alpha, beta and the cossim similarities
    def __init__(self):
        super().__init__()
        self.noise = nn.Linear(280,280)
        self.dropout = nn.Dropout(0.3)
    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

        Z_c1 = 1.2 * Z_c1
        
        Z_c2 = psi[0] * Z_c2
        Z_c2 = torch.tanh(self.noise(Z_c2))
        Z_c2 = self.dropout(Z_c2)

        Z_c3 = psi[1] * Z_c3
        Z_c3 = torch.tanh(self.noise(Z_c3))
        Z_c3 = self.dropout(Z_c3)

        Z_c4 = psi[2] * Z_c4
        Z_c4 = torch.tanh(self.noise(Z_c4))
        Z_c4 = self.dropout(Z_c4)

        Z_sum = Z_c2 + Z_c3 + Z_c4
        Z_sum = 0.2 * Z_sum
        #Z_sum = self.dropout(Z_sum)

        return Z_c1 - Z_sum

class Network3_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.act1 = nn.Linear(280,560)
        self.act2 = nn.Linear(560,560)
        self.act3 = nn.Linear(560,280)

        self.noise1 = nn.Linear(280,560)
        self.noise2 = nn.Linear(560,560)
        self.noise3 = nn.Linear(560,280)


        self.noise_sum1 = nn.Linear(280,560)
        self.noise_sum2 = nn.Linear(560,280)

        self.output1 = nn.Linear(280,560)
        self.output2 = nn.Linear(560,280)

        self.dropout = nn.Dropout(0.4)
    
    def forward(self, Z_target, Z_c2, Z_c3, Z_c4,psi):
        out_act = torch.tanh(self.act1(Z_target))
        out_act = torch.tanh(self.act2(out_act))
        out_act = torch.tanh(self.act3(out_act))
        out_act = self.dropout(out_act)

        out_2 = torch.tanh(self.noise1(Z_c2))
        out_2 = torch.tanh(self.noise2(out_2))
        out_2 = torch.tanh(self.noise3(out_2))
        out_2 = self.dropout(out_2)

        out_3 = torch.tanh(self.noise1(Z_c3))
        out_3 = torch.tanh(self.noise2(out_3))
        out_3 = torch.tanh(self.noise3(out_3))
        out_3 = self.dropout(out_3)

        out_4 = torch.tanh(self.noise1(Z_c4))
        out_4 = torch.tanh(self.noise2(out_4))
        out_4 = torch.tanh(self.noise3(out_4))
        out_4 = self.dropout(out_4)

        out_sum = out_2+out_3+out_4

        out_sum = torch.tanh(self.noise_sum1(out_sum))
        out_sum = torch.tanh(self.noise_sum2(out_sum))

        out = out_act-out_sum
        out = torch.tanh(self.output1(out))
        out = torch.tanh(self.output2(out))


        return out
    

class Network2_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.Linear(280,280)
        self.noise1 = nn.Linear(280,280)
        self.noise2 = nn.Linear(280,280)
        self.noise3 = nn.Linear(280,280)
        self.noise_sum = nn.Linear(280,280)
        self.out = nn.Linear(280,280)
        self.dropout = nn.Dropout(0.3)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

       act = torch.sigmoid(self.act(Z_c1))
       act = self.dropout(act)

       noise_1 = torch.sigmoid(self.noise1(Z_c2))
       noise_1 = self.dropout(noise_1)
       noise_2 = torch.sigmoid(self.noise2(Z_c3))
       noise_2 = self.dropout(noise_2)
       noise_3 = torch.sigmoid(self.noise3(Z_c4))
       noise_3 = self.dropout(noise_3)

       noise_sum = noise_1+noise_2+noise_3
       noise_sum =torch.sigmoid(self.noise_sum(noise_sum))
       

       out = act-noise_sum
       out = torch.relu(self.out(out))

       return out
    

class Network6_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise = nn.Linear(280,280)
        self.noise_sum = nn.Linear(280,280)
        self.act = nn.Linear(280,280)
        self.out = nn.Linear(280,280)
        self.dropout = nn.Dropout(0.4)

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):

        Z_c1 = 1.2 * Z_c1
        out_act = torch.tanh(self.act(Z_c1))
        out_act = self.dropout(out_act)

        Z_c2 = psi[0] * Z_c2
        out_2 = torch.tanh(self.noise(Z_c2))
        out_2 = self.dropout(out_2)

        Z_c3 = psi[1] * Z_c3
        out_3 = torch.tanh(self.noise(Z_c3))
        out_3 = self.dropout(out_3)

        Z_c4 = psi[2] * Z_c4
        out_4 = torch.tanh(self.noise(Z_c4))
        out_4 = self.dropout(out_4)

        Z_sum = 0.2*(out_2+out_3+out_4)
        out_sum = torch.tanh(self.noise_sum(Z_sum))
        
        out = out_act - out_sum
        out = torch.tanh(self.out(out))
        
        return out
    
class Network1_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Z_c1, Z_c2, Z_c3, Z_c4,psi):
        Z=torch.Zeros(4,280)
        Z[1]=Z_c1
        Z[2]=Z_c2
        Z[3]=Z_c3
        Z[4]=Z_c4
        
