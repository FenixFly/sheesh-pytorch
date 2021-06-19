import torch
import torch.nn as nn
import gc
import torch.nn.functional as F

class CustomLSTM(nn.Module):
    def __init__(self, input_feature_size, hidden_size,n_outputs):
        super(CustomLSTM, self).__init__()
        print('CustomLSTM')

        self.input_feature_size= input_feature_size
        self.hidden_size = hidden_size

        self.first = nn.Sequential(
            nn.Conv1d(in_channels=input_feature_size,out_channels=hidden_size//2,kernel_size=5,stride=1,padding=2,
                               bias=False),
            nn.BatchNorm1d(num_features=hidden_size//2,affine=True), #try different momentum and batch size
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_size//2, out_channels=hidden_size, kernel_size=5, stride=1, padding=2,
                      bias=False),
            nn.BatchNorm1d(num_features=hidden_size, affine=False),

        )
        self.lstm = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,bidirectional=True,dropout=0.1,num_layers=1)

        self.clfconv = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size*2, out_channels=hidden_size, kernel_size=5, stride=1,
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

        conv = self.first(input)

        hidden0 = (torch.zeros(2*self.lstm.num_layers, N, self.hidden_size).to(input.device),
                   torch.zeros(2*self.lstm.num_layers, N, self.hidden_size).to(input.device))  # clean out hidden state
        conv = conv.permute(2,0,1)
        out, hidden = self.lstm(conv, hidden0)

        res = self.clfconv(out.permute(1,2,0))
        res = torch.sigmoid(res)


        return {
            'prediction': res,
        }

