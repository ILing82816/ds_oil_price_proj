# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:48:53 2020

@author: USER
"""
import torch

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        test_h = model.init_hidden(batch_size)
        for i, inputs in enumerate(test_loader):
            test_h = tuple([each.data for each in test_h])
            inputs = inputs.to(device, dtype=torch.float) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            outputs = model(inputs, test_h)
            ret_output.append(outputs.item())
            #ret_output += outputs.tolist()   
    return ret_output
