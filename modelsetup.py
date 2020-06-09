# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:17:30 2020

@author: USER
"""
import torch
from torch import nn
from torch.autograd import Variable
class LSTM_Net(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, n_layers): #dropout=0.5
        super(LSTM_Net, self).__init__()
        
        self.hidden_layer_size = hidden_layer_size
        self.n_layers = n_layers
        #self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_layer_size, n_layers, batch_first=True)
        
        self.classifier = nn.Sequential( #nn.Dropout(dropout),
                                         nn.Linear(hidden_layer_size, output_size),
                                         nn.Sigmoid())
        #self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
        #                    torch.zeros(1,1,self.hidden_layer_size))
        
        #self.hidden_state = Variable(torch.randn(1, 1, self.hidden_layer_size), requires_grad=False).double().cuda()
        #self.cell_state = Variable(torch.randn(1, 1, self.hidden_layer_size), requires_grad=False).double().cuda()
        #self.hidden = (self.hidden_state, self.cell_state)
        
         
    def forward(self, inputs, hidden):
        #outputs, self.hidden = self.lstm(inputs.view(len(inputs) ,1, -1), self.hidden)
        batch_size = inputs.size(0)
        outputs, hidden = self.lstm(inputs.view(len(inputs) ,1, -1), hidden)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        predictions = self.classifier(outputs.view(len(inputs), -1))
        #x = outputs[:, -1, :] 
        #x = self.classifier(x)
        return predictions
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_layer_size).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_layer_size).zero_().cuda())
        return hidden
