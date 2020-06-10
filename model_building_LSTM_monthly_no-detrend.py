# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:00:52 2020

@author: USER
"""
# Set up some path and parameters.
## path
import os
import torch
path_prefix = 'D:/USA 2020 summer/Machine Learning/ds_oil_price_proj'
model_dir = path_prefix # model directory for checkpoint model
# 通過 torch.cuda.is_available() 的回傳值進行判斷是否有使用 GPU 的環境，如果有的話 device 就設為 "cuda"，沒有的話就設為 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定義batch 大小、要訓練幾個 epoch、learning rate 的值
batch_size = 1
epoch =30
lr = 0.001




#load data
print("loading data ...")
import pandas as pd
wti = pd.read_csv(os.path.join(path_prefix, 'data/Crude Oil WTI Futures Historical Data.csv'),usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
wti.rename(columns={'Price':'wti_price',"Vol.":"wti_volumn"}, 
                 inplace=True)
wti1 = pd.read_csv(os.path.join(path_prefix, 'data/Crude Oil WTI Futures Historical Data (1).csv'),usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
wti1.rename(columns={'Price':'wti_price',"Vol.":"wti_volumn"}, 
                 inplace=True)
wti_oil_price= pd.concat([wti1,wti])
wti_oil_price["wti_volumn"] = wti_oil_price["wti_volumn"].apply(lambda x : None if x =='-'  
                                         else (x))
wti_oil_price_monthly = wti_oil_price.resample('M').mean()




# data EDA
import matplotlib.pyplot as plt
plt.plot(wti_oil_price["wti_price"],color="red",Label='wti_price')
plt.legend()
plt.show()

plt.title("wti price monthly")
plt.plot(wti_oil_price_monthly["wti_price"],color="red",Label='wti_price_monthly')
plt.legend()
plt.savefig('wti_price_monthly.png')
plt.show()






#data preprocess
print("Data preprocess ...")
##Scale
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(wti_oil_price_monthly['wti_price'].values.reshape(-1,1))

##train_test split
import math
train = dataset[: math.floor(len(dataset) * 0.8), :]
val = dataset[math.floor(len(dataset) * 0.8): , :]

##transform data to be supervised learning
import preprocess as p
look_back = 1
trainX, trainY = p.create_dataset(train, look_back)
valX, valY = p.create_dataset(val, look_back)
##To dataset for Dataloader
import pricedata as d
train_dataset = d.PriceDataset(X=trainX, y=trainY)
val_dataset = d.PriceDataset(X=valX, y=valY)

## 把 data 轉成 batch of tensors

from torch.utils.data import DataLoader
train_loader = DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)
val_loader = DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)





# Set up model and train model
print("Start training...")
import modelsetup as msp
import train as tr
model = msp.LSTM_Net(input_size=1, hidden_layer_size=100, output_size=1, n_layers=1)
model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）
tr.training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)




#Prediction
print('\nload model ...')
import predict as pre
val = dataset[math.floor(len(dataset) * 0.8): , :]
val_dataset = d.PriceDataset(X=valX, y=None)
val_loader = DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)
model = torch.load(os.path.join(model_dir, 'oil_price.model'))
val_outputs = pre.testing(batch_size, val_loader, model, device)
##invert scaling
import numpy as np
val_outputs1 = scaler.inverse_transform(np.array(val_outputs).reshape(-1, 1))
##invert differencing
raw_data = wti_oil_price_monthly['wti_price'].reset_index()

## Draw
x = np.arange(253, 315, 1)
plt.plot(raw_data["wti_price"],color="red",Label='wti_price_monthly')
plt.plot(x, val_outputs1, color="blue",Label='wti_price_prediction')
plt.legend()
plt.show()

## Zoom in
plt.title("Ture vs Prediction")
plt.plot(raw_data["wti_price"][253:315],color="red",Label='wti_price_monthly')
plt.plot(x, val_outputs1, color="blue",Label='wti_price_prediction')
plt.legend()
plt.show()


##pridict June. 20
test = dataset[-3:]
test_dataset = d.PriceDataset(X=test, y=None)
test_loader = DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)
test_outputs = pre.testing(batch_size, test_loader, model, device)
test_outputs1 = scaler.inverse_transform(np.array(test_outputs).reshape(-1, 1))
#actual_prediction1 = p.inverse_difference(settlement_date['wti_price'], outputs[:,0], 3) 

x1 = np.arange(314, 317, 1)
plt.title("Ture vs Prediction")
plt.plot(settlement_date['wti_price'][253:316],color="red",Label='wti_price_monthly')
plt.plot(x, val_outputs1, color="blue",Label='wti_price_monthly_prediction')
plt.plot(x1, test_outputs1, color="orange",Label='wti_price_monthly_prediction')
plt.legend()
plt.savefig('wti_price_monthly_prediction.png')
plt.show() 




print('\nfinish prediction')

