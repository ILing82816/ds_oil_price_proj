# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:12:40 2020

@author: USER
"""
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import preprocess as p
import pricedata as d
import math
import torch
from torch.utils.data import DataLoader
import modelsetup as msp
import train as tr


# Set up some path and parameters.
## path
path_prefix = 'D:/USA 2020 summer/Machine Learning/ds_oil_price_proj'
model_dir = path_prefix # model directory for checkpoint model
## torch.cuda.is_available() ，if is true then device will be "cuda"，if false then device will be "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## the size of batch、epoch、learning rate
batch_size = 1
epoch =30
lr = 0.001




#load data
print("loading data ...")
df = pd.read_csv(os.path.join(path_prefix, 'data/data_clean.csv'),usecols = ['index','wti_price'])






#data preprocess
print("Data preprocess ...")
##Scale
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(df['wti_price'].values.reshape(-1,1))
##train_test split
train = dataset[: math.floor(len(dataset) * 0.8), :]
val = dataset[math.floor(len(dataset) * 0.8): , :]
##transform data to be supervised learning
look_back = 1
trainX, trainY = p.create_dataset(train, look_back)
valX, valY = p.create_dataset(val, look_back)
##To dataset for Dataloader
train_dataset = d.PriceDataset(X=trainX, y=trainY)
val_dataset = d.PriceDataset(X=valX, y=valY)
## To batch of tensors
train_loader = DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
val_loader = DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)






# Set up model and train model
print("Start training...")
model = msp.LSTM_Net(input_size=1, hidden_layer_size=100, output_size=1, n_layers=1)
model = model.to(device) 
tr.training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)




#Prediction
print('\nload model ...')
import predict as pre
val = dataset[math.floor(len(dataset) * 0.8): , :]
val_dataset = d.PriceDataset(X=valX, y=None)
val_loader = DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)
model = torch.load(os.path.join(model_dir, 'pre_LSTM.model'))
val_outputs = pre.testing(batch_size, val_loader, model, device)
##invert scaling
import numpy as np
val_pred = scaler.inverse_transform(np.array(val_outputs).reshape(-1, 1))
valY = scaler.inverse_transform(np.array(valY).reshape(-1, 1))
print(mean_absolute_error(valY, val_pred))



## Draw
x = np.arange(math.floor(len(dataset) * 0.8), math.floor(len(dataset) * 0.8)+1303, 1)
plt.plot(df["wti_price"],color="red",Label='real wti_price')
plt.plot(x, val_pred, color="blue",Label='wti_price_prediction (MAE: 1.08)')
plt.legend()
plt.show()

## Zoom in
plt.title("Ture vs Prediction")
plt.plot(df["wti_price"][math.floor(len(dataset) * 0.8):math.floor(len(dataset) * 0.8)+1303],color="red",Label='wti_price')
plt.plot(x, val_outputs1, color="blue",Label='wti_price_prediction(MAE: 1.08)')
plt.legend()
plt.show()


##pridict 5/1: get 19.998 real is 19.78
test = dataset[-1:]
test_dataset = d.PriceDataset(X=test, y=None)
test_loader = DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)
test_outputs = pre.testing(batch_size, test_loader, model, device)
test_pred = scaler.inverse_transform(np.array(test_outputs).reshape(-1, 1))

print('\nfinish prediction')