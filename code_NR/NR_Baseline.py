import torch
import numpy as np

def NR_baseline(x, alpha, beta, cos_sim):
    
    Z_c1_active = x[0]
    Z_c2 = x[1]
    Z_c3 = x[2]
    Z_c4 = x[3]

    epsilon = 1/(torch.sum(cos_sim))
    weighted_noise = (cos_sim[0]*Z_c2 + cos_sim[1]*Z_c3 + cos_sim[2]*Z_c4)
        

    Z_NR = alpha*Z_c1_active - epsilon*beta*weighted_noise

    return Z_NR
