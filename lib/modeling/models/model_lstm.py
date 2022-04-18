import os
import sys
import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Long Short Term Memory network
    """

    def __init__(self, in_dim, hidden_dim, out_dim,
                 aggregate='avg', bidirectional=True):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers=2,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim*2, out_dim)
        self.aggregate = aggregate

    def forward(self, seq):
        '''
        seq: tensor(LxNxD) [L:length, N:batch, D:dim] 
        '''
        out, (ht, ct) = self.lstm(seq)
        if self.aggregate == 'last':
            out = out[-1] # NxD
        if self.aggregate == 'avg':
            out = torch.mean(out, dim=0) # LxNxD -> NxD

        # out = self.fc(out)
        return out


def build_lstm(args):
    model = LSTM(in_dim=300, hidden_dim=256, out_dim=256)
    return model