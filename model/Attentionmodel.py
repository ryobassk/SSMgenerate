#torchライブラリ
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

#attention層ライブラリ
from model.layers import Attention

class Encoder(nn.Module):
    def __init__(self,
                 input_dim1, input_dim2, input_dim3, 
                 hidden_dim,
                 num_layers=1,
                 dropout=0,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding1 = nn.Embedding(input_dim1, hidden_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(input_dim2, hidden_dim, padding_idx=0)
        self.embedding3 = nn.Embedding(input_dim3, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim*3, hidden_dim*3, num_layers)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x1, x2, x3):
        len_source_sequences = (x1.t() > 0).sum(dim=-1)

        
        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)
        x3 = self.embedding3(x3)

        x123 = torch.cat((x1, x2, x3),2)
        x123 = pack_padded_sequence(x123, len_source_sequences.cpu())
        
        h, states = self.lstm(x123)
        h, _ = pad_packed_sequence(h)

        return h, states

class Decoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 output_dim,
                 num_layers=1,
                 dropout=0,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(output_dim, hidden_dim*3, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim*3, hidden_dim*3, num_layers)
        self.attn = Attention(hidden_dim*3, hidden_dim*3, device=self.device)
        self.out = nn.Linear(hidden_dim*3, output_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x, hs, states, source=None, enc=None):
        x = self.embedding(x)
        ht, states = self.lstm(x, states)
        ht = self.attn(ht, hs, source=source)
        y = self.out(ht)        
        
        return y, states, ht
