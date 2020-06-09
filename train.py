# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:52:17 2020

@author: USER
"""
import torch
from torch import nn
import torch.optim as optim
import numpy as np


def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    t_batch = len(train) 
    v_batch = len(valid)
    
    loss = nn.MSELoss() # 定義損失函數，這裡我們使用 binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr) # 將模型的參數給 optimizer，並給予適當的 learning rate
    lower_loss = np.Inf
    for epoch in range(n_epoch):
        total_loss = 0
        h = model.init_hidden(batch_size)
        # 這段做 training
        model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數
        for i, (inputs, labels) in enumerate(train):
            h = tuple([e.data for e in h])
            inputs = inputs.to(device, dtype=torch.float) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
            optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
            #model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
            #            torch.zeros(1, 1, model.hidden_layer_size))
            
            #model.hidden_state = torch.randn(1, 1, model.hidden_layer_size).cuda()
            #model.cell_state = torch.randn(1, 1, model.hidden_layer_size).cuda()
            #model.hidden = (model.hidden_state, model.cell_state)
            
            outputs = model(inputs, h) # 將 input 餵給模型
            
            #outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
            batch_loss = loss(outputs, labels) # 計算此時模型的 training loss
            batch_loss.backward() # 算 loss 的 gradient
            optimizer.step() # 更新訓練模型的參數
            
            total_loss += batch_loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} '.format(
            	epoch+1, i+1, t_batch, batch_loss.item()), end='\r')
        print('\nTrain | Loss:{:.5f} '.format(total_loss/t_batch))

        # 這段做 validation
        model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
        with torch.no_grad():
            total_loss = 0
            val_h = model.init_hidden(batch_size)
            for i, (inputs, labels) in enumerate(valid):
                val_h = tuple([each.data for each in val_h])
                inputs = inputs.to(device, dtype=torch.float) # device 為 "cuda"，將 inputs 轉成 torch.cuda.LongTensor
                labels = labels.to(device, dtype=torch.float) # device 為 "cuda"，將 labels 轉成 torch.cuda.FloatTensor，因為等等要餵進 criterion，所以型態要是 float
                outputs = model(inputs, val_h) # 將 input 餵給模型
                #outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
                batch_loss = loss(outputs, labels) # 計算此時模型的 validation loss
                
                total_loss += batch_loss.item()

            print("Valid | Loss:{:.5f} ".format(total_loss/v_batch))
            if total_loss/v_batch < lower_loss:
                # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/oil_price.model".format(model_dir))
                print('saving model with loss {:.3f}'.format(total_loss/v_batch))
        print('-----------------------------------------------')

