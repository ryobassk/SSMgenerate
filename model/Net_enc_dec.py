#torchライブラリ
import torch
from torch import nn
import torch.optim as optimizers
#modelのライブラリ
from model.Attentionmodel import Encoder
from model.Attentionmodel import Decoder
#Encoder-decoderモデル
class EncoderDecoder(nn.Module):
    def __init__(self,
                 input_dim1, input_dim2, input_dim3, 
                 hidden_dim,
                 output_dim1, output_dim2, output_dim3,
                 maxlen=20,
                 num_layers=2,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.encoder = Encoder(input_dim1, input_dim2, input_dim3, 
                               hidden_dim, num_layers=2, device=device)
        #note予測のデコーダ
        self.decoder1 = Decoder(hidden_dim, output_dim1, 
                                num_layers=2, device=device)
        #len予測のデコーダ
        self.decoder2 = Decoder(hidden_dim, output_dim2, 
                                num_layers=2, device=device)
        #chord予測のデコーダ
        self.decoder3 = Decoder(hidden_dim, output_dim3, 
                                num_layers=2, device=device)
        self.maxlen = maxlen
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.output_dim3 = output_dim3

    def forward(self, phase,
                x_note, x_len, x_chord, chord=None, t_num=None,
                t_note=None, t_len=None, t_chord=None, 
                use_teacher_forcing=False):
        batch_size = x_note.size(1)
        hs, states = self.encoder(x_note, x_len, x_chord)
        y = torch.ones((1, batch_size),
                       dtype=torch.long,
                       device=self.device)
        if phase == 'chord':
            len_target_sequences = 8
            output = torch.zeros((len_target_sequences,
                                  batch_size,
                                  self.output_dim3),
                                 device=self.device)
            for t in range(len_target_sequences):
                out, states, _ = self.decoder3(y, hs, states, source=x_note)
                output[t] = out
                if use_teacher_forcing and chord is not None:
                    y = chord[t].unsqueeze(0)
                else:
                    y = out.max(-1)[1]
            return output
        
        if phase == 'note':
            if t_note is not None:
                len_target_sequences = t_note.size(0)
            else:
                len_target_sequences = self.maxlen
                
            y = torch.ones((1, batch_size),
                       dtype=torch.long,
                       device=self.device)
            output = torch.zeros((len_target_sequences,
                                  batch_size,
                                  self.output_dim1),
                                 device=self.device)
            _, states3, hs_dec3 = self.decoder3(chord, hs, states, source=x_note)
            for t in range(len_target_sequences):                
                out, states, _ = self.decoder1(y, hs_dec3, states, source=chord)
                output[t] = out
                if use_teacher_forcing and t_note is not None:
                    y = t_note[t].unsqueeze(0)
                else:
                    y = out.max(-1)[1]
            return output
        
        if phase == 'len':
            _, states3, hs_dec3 = self.decoder3(chord, hs, states, source=x_note)
            _, states1, hs_dec1 = self.decoder1(t_note, hs_dec3, states, source=chord)
            if t_len is not None:
                len_target_sequences = t_len.size(0)
            else:
                len_target_sequences = self.maxlen
            y = torch.ones((1, batch_size),
                       dtype=torch.long,
                       device=self.device)
            output = torch.zeros((len_target_sequences,
                                  batch_size,
                                  self.output_dim2),
                                 device=self.device)
            for t in range(len_target_sequences):
                out, states, _ = self.decoder2(y, hs_dec1, states, source=t_note)
                output[t] = out
                if use_teacher_forcing and t_len is not None:
                    y = t_len[t].unsqueeze(0)
                else:
                    y = out.max(-1)[1]
            return output
            
            
    
            