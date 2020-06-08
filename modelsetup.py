# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:17:30 2020

@author: USER
"""
import torch
from torch import nn
class LSTM_Net(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout=0.5):
        super(LSTM_Net, self).__init__()
        #self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        #self.output_size = output_size
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_layer_size, output_size),
                                         nn.Sigmoid())
    def forward(self, inputs):
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

