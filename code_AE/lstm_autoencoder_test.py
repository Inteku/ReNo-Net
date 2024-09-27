import torch
import torch.nn as nn
import math

def sp(x): #size print
    if False:
        print(x.size())

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, cell_size, num_layers, segment_size=24):
        super(LSTMAutoencoder, self).__init__()
        self.segment_size = segment_size
        self.cell_size = cell_size
        self.is_decode = True

        self.outer_encoder = nn.LSTM(input_size=input_size,
                                    hidden_size=cell_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        self.inner_encoder = nn.LSTM(input_size=cell_size,
                                    hidden_size=cell_size,
                                    num_layers=num_layers,
                                    batch_first=True)

        self.inner_decoder = nn.LSTM(input_size=cell_size,
                                    hidden_size=cell_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        self.outer_decoder = nn.LSTM(input_size=cell_size,
                                    hidden_size=input_size,
                                    num_layers=num_layers,
                                    batch_first=True)

    def set_decode(self, new:bool):
        self.is_decode = new

    def forward(self, inp):
        if inp.dim() != 3:
            #raise ValueError(f'Expected input to be 3-dimensional (batches, bins, time), got {inp.dim()}')
            inp = inp.squeeze()
        batch_size = inp.size(0)
        f_bins = inp.size(1)
        T_total = inp.size(2)
        sp(inp)
        inp = torch.swapaxes(inp, 1, 2)
        sp(inp)
        num_segments = math.ceil(T_total/self.segment_size)
        cell_state_tensor = torch.zeros((batch_size, num_segments, self.cell_size))
        dt = self.segment_size
        for s in range(num_segments):
            start = s*dt
            end = min((s+1)*dt, T_total)
            segment = inp[:,start:end,:]
            if s==0:
                sp(segment)
            # Outer Encoder
            _, (_, cell_state) = self.outer_encoder(segment)
            if s==0:
                sp(cell_state)
            cell_state = torch.swapaxes(cell_state, 0, 1)
            if s==0:
                sp(cell_state)
            cell_state_tensor[:,s,:] = cell_state.squeeze()

        sp(cell_state_tensor)
        # Inner Encoder
        _, (_, cell_bottleneck) = self.inner_encoder(cell_state_tensor)
        sp(cell_bottleneck)
        # Bottleneck
        if not self.is_decode:
            return cell_bottleneck.squeeze()
        # Inner Decoder
        cell_bottleneck = torch.swapaxes(cell_bottleneck, 0, 1)
        sp(cell_bottleneck)
        inner_decoder_output, _ = self.inner_decoder(cell_bottleneck.repeat(1, num_segments, 1))
        sp(inner_decoder_output)
        reconstructed_input = torch.zeros((batch_size, f_bins, T_total))
        for s in range(num_segments):
            start = s*dt
            end = min((s+1)*dt, T_total)
            encoded_segment = inner_decoder_output[:,s,:].unsqueeze(1)
            if s==0:
                sp(encoded_segment)
            reconstructed_segment, _ = self.outer_decoder(encoded_segment.repeat(1, end-start, 1))
            if s==0:
                sp(reconstructed_segment)
            reconstructed_input[:,:,start:end] = torch.swapaxes(reconstructed_segment, 1, 2)
        sp(reconstructed_input)
        return reconstructed_input

L = LSTMAutoencoder(input_size=128, cell_size=280, num_layers=1)
L.set_decode(False)
ip = torch.randn((10,128,312))
op = L(ip)
#print(op.size())


