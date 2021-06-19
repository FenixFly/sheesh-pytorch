import torch
import torch.nn as nn
import gc
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len].clone().detach().to(x.device)
        return x


class CustomTransformer(nn.Module):
    def __init__(self, input_feature_size, hidden_size,n_outputs,n_head,n_encoders):
        super(CustomTransformer, self).__init__()
        print('CustomTransformer')

        self.input_feature_size= input_feature_size
        self.hidden_size = hidden_size

        self.first = nn.Sequential(
            nn.Conv1d(in_channels=input_feature_size, out_channels=hidden_size // 2, kernel_size=5, stride=1, padding=2,
                      bias=False),
            nn.BatchNorm1d(num_features=hidden_size // 2, affine=True),  # try different momentum and batch size
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size // 2, out_channels=hidden_size, kernel_size=5, stride=1, padding=2,
                      bias=False),

        )

        self.pe = PositionalEncoder(d_model=hidden_size,max_seq_len=4096)

        self.encoder = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model=hidden_size,nhead=n_head,dim_feedforward=1024) for i in range(n_encoders)]
        )

        self.clfconv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, stride=1,
                      padding=2,
                      bias=False),
            nn.BatchNorm1d(num_features=hidden_size, affine=True),
            nn.Tanh(),
            nn.Conv1d(in_channels=hidden_size, out_channels=n_outputs, kernel_size=5, stride=1,
                      padding=2,
                      bias=True),

        )


    def forward(self, x):
        input = x['signal']
        N, C, W = input.shape

        if 'paddingsignal' in x:
            pad_left, pad_right = x['paddingsignal']
            pad_left = pad_left.cpu().numpy()
            pad_right = pad_right.cpu().numpy()

            mask = np.zeros(shape=())


        conv = self.first(input)
        conv = conv.permute(2,0,1)
        #conv = self.pe(conv)


        mem = self.encoder(conv)

        res = self.clfconv(mem.permute(1,2,0))
        res = torch.sigmoid(res)


        return {
            'prediction': res,
        }

