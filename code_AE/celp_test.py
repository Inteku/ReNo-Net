import numpy as np
import torch
import SoundDatasets
import os
import celp_codec.celp as celp

dataset = SoundDatasets.tSNE_dataset(wave=True, shuffle=True, seed=0)

frame, id = dataset[0]
frame_len = frame.shape[0]

major_len = 160
minor_len = 40
num_minorframes = frame_len//minor_len
num_majorframes = frame_len//major_len

ENC = celp.CELP(frame_length=frame_len, n_subframes=major_len/minor_len)

for N in range(num_majorframes):
    major_frame = frame[N*major_len:(N+1)*major_len]
    LPC , fc_indexes , fc_amplifs , ac_indexes , ac_amplifs = ENC.encode(major_frame)





print(fc_amplifs.shape)