#torchライブラリ
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

#attention層ライブラリ
from model.layers import Attention

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_layers=1,
                 dropout=0,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        len_source_sequences = (x.t() > 0).sum(dim=-1)
        x = self.embedding(x)
        x = pack_padded_sequence(x, len_source_sequences)
        h, states = self.lstm(x)
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
        self.embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.attn = Attention(hidden_dim, hidden_dim, device=self.device)
        self.out = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x, hs, states, source=None):
        x = self.embedding(x)
        ht, states = self.lstm(x, states)
        ht = self.attn(ht, hs, source=source)
        y = self.out(ht)
        return y, states
