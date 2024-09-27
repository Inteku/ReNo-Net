import torch
import torch.nn as nn
import math

def sp(x): #size print
    #print(x.size())
    return

class LSTMAutoencoder_v1(nn.Module):
    def __init__(self, input_size, cell_size, num_layers, segment_size=24):
        super(LSTMAutoencoder_v1, self).__init__()
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
        #print(inp.size())
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
        cell_state_tensor = torch.zeros((batch_size, num_segments, self.cell_size)).to(inp.device)
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



class LSTMAutoencoder_v2(nn.Module):
    def __init__(self, input_size, cell_size, num_layers, segment_size=24):
        super(LSTMAutoencoder_v2, self).__init__()
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
        #print(inp.size())
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
        hidden_state_tensor = torch.zeros((batch_size, num_segments, self.cell_size)).to(inp.device)
        dt = self.segment_size
        for s in range(num_segments):
            start = s*dt
            end = min((s+1)*dt, T_total)
            segment = inp[:,start:end,:]
            if s==0:
                sp(segment)
            # Outer Encoder
            _, (hidden_state, _) = self.outer_encoder(segment)
            if s==0:
                sp(hidden_state)
            hidden_state = torch.swapaxes(hidden_state, 0, 1)
            if s==0:
                sp(hidden_state)
            hidden_state_tensor[:,s,:] = hidden_state.squeeze()

        sp(hidden_state_tensor)
        # Inner Encoder
        _, (hidden_bottleneck, _) = self.inner_encoder(hidden_state_tensor)
        sp(hidden_bottleneck)
        # Bottleneck
        if not self.is_decode:
            return hidden_bottleneck.squeeze()
        # Inner Decoder
        hidden_bottleneck = torch.swapaxes(hidden_bottleneck, 0, 1)
        sp(hidden_bottleneck)
        inner_decoder_output, _ = self.inner_decoder(hidden_bottleneck.repeat(1, num_segments, 1))
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
    


class LSTMAutoencoder_v3(nn.Module):
    def __init__(self, input_size, cell_size, num_layers, segment_size=24):
        super(LSTMAutoencoder_v3, self).__init__()
        self.segment_size = segment_size
        self.cell_size = cell_size
        self.is_decode = True

        self.outer_encoder = nn.LSTM(input_size=input_size,
                                    hidden_size=cell_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        self.outer_decoder = nn.LSTM(input_size=cell_size,
                                    hidden_size=input_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        self.encode_bottleneck = nn.Linear(in_features=13*cell_size, out_features=cell_size)
        self.decode_bottleneck = nn.Linear(in_features=cell_size, out_features=13*cell_size)

    def set_decode(self, new:bool):
        self.is_decode = new

    def forward(self, inp):
        #print(inp.size())
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
        cell_state_tensor = torch.zeros((batch_size, num_segments, self.cell_size)).to(inp.device)
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
        flat_features = torch.flatten(cell_state_tensor, start_dim=1)
        sp(flat_features)
        bottleneck = self.encode_bottleneck(flat_features)
        bottleneck = nn.Tanh()(bottleneck)
        sp(bottleneck)
        if not self.is_decode:
            return bottleneck
        flat_features = self.decode_bottleneck(bottleneck)
        flat_features = nn.Tanh()(flat_features)
        sp(flat_features)
        decoder_input = torch.unflatten(flat_features, dim=1, sizes=(num_segments, self.cell_size))
        sp(decoder_input)

        reconstructed_input = torch.zeros((batch_size, f_bins, T_total))
        for s in range(num_segments):
            start = s*dt
            end = min((s+1)*dt, T_total)
            encoded_segment = decoder_input[:,s,:].unsqueeze(1)
            if s==0:
                sp(encoded_segment)
            reconstructed_segment, _ = self.outer_decoder(encoded_segment.repeat(1, end-start, 1))
            if s==0:
                sp(reconstructed_segment)
            reconstructed_input[:,:,start:end] = torch.swapaxes(reconstructed_segment, 1, 2)
        sp(reconstructed_input)
        return reconstructed_input




class LSTMAutoencoder_v4_old(nn.Module):
    def __init__(self, input_size, cell_size, num_layers, segment_size=24):
        super(LSTMAutoencoder_v4, self).__init__()
        self.segment_size = segment_size
        self.cell_size = cell_size
        self.is_decode = True

        self.outer_encoder = nn.LSTM(input_size=input_size,
                                    hidden_size=cell_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        self.outer_decoder = nn.LSTM(input_size=cell_size,
                                    hidden_size=input_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        self.encode_bottleneck = nn.Linear(in_features=13*cell_size, out_features=cell_size)
        self.decode_bottleneck = nn.Linear(in_features=cell_size, out_features=13*cell_size)

    def set_decode(self, new:bool):
        self.is_decode = new

    def forward(self, inp):
        #print(inp.size())
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
        cell_state_tensor = torch.zeros((batch_size, num_segments, self.cell_size)).to(inp.device)
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
        flat_features = torch.flatten(cell_state_tensor, start_dim=1)
        sp(flat_features)
        bottleneck = self.encode_bottleneck(flat_features)
        bottleneck = nn.Tanh()(bottleneck)
        sp(bottleneck)
        if not self.is_decode:
            return bottleneck
        flat_features = self.decode_bottleneck(bottleneck)
        flat_features = nn.Tanh()(flat_features)
        sp(flat_features)
        decoder_input = torch.unflatten(flat_features, dim=1, sizes=(num_segments, self.cell_size))
        sp(decoder_input)

        current_device = next(self.outer_decoder.parameters()).device
        decoder_input_repeated = torch.zeros((batch_size, T_total, self.cell_size)).to(current_device)
        sp(decoder_input_repeated)
        for s in range(num_segments):
            start = s*dt
            end = min((s+1)*dt, T_total)
            decoder_input_repeated[:,start:end,:] = decoder_input[:,s,:].unsqueeze(1).repeat(1, end-start, 1)

        reconstructed_input,_ = self.outer_decoder(decoder_input_repeated)
        sp(reconstructed_input)
        reconstructed_input = torch.swapaxes(reconstructed_input, 1, 2)
        sp(reconstructed_input)

        return reconstructed_input
    

class LSTMAutoencoder_v4(nn.Module):
    def __init__(self, input_size, cell_size, num_layers, segment_size=24):
        super(LSTMAutoencoder_v4, self).__init__()
        self.segment_size = segment_size
        self.cell_size = cell_size
        self.is_decode = True

        self.spectral_mean = torch.Tensor([0.5866, 0.5835, 0.5796, 0.5871, 0.5975, 0.6065, 0.6209, 0.6352, 0.6463, 0.6571, 0.6708, 0.6830, 0.6974, 0.6967, 0.6881, 0.6796,
                                           0.6699, 0.6688, 0.6667, 0.6632, 0.6701, 0.6812, 0.6760, 0.6843, 0.6828, 0.6773, 0.6789, 0.6683, 0.6683, 0.6582, 0.6514, 0.6454,
                                           0.6339, 0.6255, 0.6178, 0.6080, 0.6046, 0.6014, 0.5979, 0.5937, 0.5898, 0.5860, 0.5840, 0.5818, 0.5795, 0.5777, 0.5764, 0.5749,
                                           0.5743, 0.5728, 0.5700, 0.5722, 0.5680, 0.5679, 0.5640, 0.5655, 0.5631, 0.5642, 0.5676, 0.5701, 0.5722, 0.5742, 0.5748, 0.5735,
                                           0.5712, 0.5684, 0.5666, 0.5630, 0.5607, 0.5588, 0.5541, 0.5538, 0.5513, 0.5509, 0.5515, 0.5526, 0.5529, 0.5533, 0.5512, 0.5485,
                                           0.5442, 0.5395, 0.5357, 0.5319, 0.5307, 0.5293, 0.5287, 0.5275, 0.5270, 0.5263, 0.5251, 0.5236, 0.5230, 0.5207, 0.5192, 0.5175,
                                           0.5150, 0.5122, 0.5089, 0.5055, 0.5021, 0.4991, 0.4957, 0.4917, 0.4865, 0.4814, 0.4759, 0.4705, 0.4656, 0.4605, 0.4554, 0.4486,
                                           0.4414, 0.4347, 0.4307, 0.4289, 0.4278, 0.4269, 0.4235, 0.4195, 0.4178, 0.4193, 0.4203, 0.4194, 0.4170, 0.4105, 0.3831, 0.2974])

        self.outer_encoder = nn.LSTM(input_size=input_size,
                                    hidden_size=cell_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        self.outer_decoder = nn.LSTM(input_size=cell_size,
                                    hidden_size=input_size,
                                    num_layers=num_layers,
                                    batch_first=True)
        
        self.encode_bottleneck = nn.Linear(in_features=13*cell_size, out_features=cell_size)
        self.decode_bottleneck = nn.Linear(in_features=cell_size, out_features=13*cell_size)

    def set_decode(self, new:bool):
        self.is_decode = new

    def add_mean(self, x, factor):
        batch_size = x.size(0)
        T = x.size(1)
        mean_spectrum = self.spectral_mean.repeat(batch_size, T, 1).to(x.device)
        
        return torch.add(x, factor*mean_spectrum)

    def forward(self, inp):
        #print(inp.size())
        if inp.dim() != 3:
            #raise ValueError(f'Expected input to be 3-dimensional (batches, bins, time), got {inp.dim()}')
            inp = inp.squeeze()
            inp = inp.unsqueeze(0)
        batch_size = inp.size(0)
        f_bins = inp.size(1)
        T_total = inp.size(2)
        
        sp(inp)
        inp = torch.swapaxes(inp, 1, 2)
        sp(inp)
        #de-mean
        inp = self.add_mean(inp, -1)
        num_segments = math.ceil(T_total/self.segment_size)
        cell_state_tensor = torch.zeros((batch_size, num_segments, self.cell_size)).to(inp.device)
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
        flat_features = torch.flatten(cell_state_tensor, start_dim=1)
        sp(flat_features)
        bottleneck = self.encode_bottleneck(flat_features)
        bottleneck = nn.Tanh()(bottleneck)
        sp(bottleneck)
        if not self.is_decode:
            return bottleneck
        flat_features = self.decode_bottleneck(bottleneck)
        flat_features = nn.Tanh()(flat_features)
        sp(flat_features)
        decoder_input = torch.unflatten(flat_features, dim=1, sizes=(num_segments, self.cell_size))
        sp(decoder_input)

        current_device = next(self.outer_decoder.parameters()).device
        decoder_input_repeated = torch.zeros((batch_size, T_total, self.cell_size)).to(current_device)
        sp(decoder_input_repeated)
        for s in range(num_segments):
            start = s*dt
            end = min((s+1)*dt, T_total)
            decoder_input_repeated[:,start:end,:] = decoder_input[:,s,:].unsqueeze(1).repeat(1, end-start, 1)

        reconstructed_input,_ = self.outer_decoder(decoder_input_repeated)
        sp(reconstructed_input)
        reconstructed_input = self.add_mean(reconstructed_input, 1)
        reconstructed_input = torch.swapaxes(reconstructed_input, 1, 2)
        sp(reconstructed_input)
        
        return reconstructed_input


L = LSTMAutoencoder_v4(input_size=128, cell_size=280, num_layers=1)
L.set_decode(True)
ip = torch.randn((10,128,312))
op = L(ip)
#print(op.size())


