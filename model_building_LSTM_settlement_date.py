# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 08:29:39 2020

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
epoch =50
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
wti_oil_price = wti_oil_price.reset_index()

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
settlement_date = pd.DataFrame(columns = ["Date", "wti_price", "wti_volumn"])
start_date = datetime(1994, 2, 20) 
for i in range (315):
    start_date = start_date + relativedelta(months=+1)
    if start_date.weekday() == 5:
        use_date = start_date + timedelta(days=+2)
    elif start_date.weekday() == 6:
        use_date = start_date + timedelta(days=+1)
    else:
        use_date = start_date
    if wti_oil_price[wti_oil_price['Date'] == use_date].empty:
        use_date = use_date + timedelta(days=+1)
    settlement_date = settlement_date.append(wti_oil_price[wti_oil_price['Date'] == use_date], ignore_index=True)




# data EDA
import matplotlib.pyplot as plt
plt.title("wti price in settlement date")
plt.plot(settlement_date["wti_price"])
plt.legend()
plt.savefig('wti_price_settlement_date.png')
plt.show()

'''
## ADF/KPSS
print("Check ADF/KPSS ...")
import preprocess as p
p.adf_test(settlement_date['wti_price'])  
#Based upon the significance level of 0.05 and the p-value of ADF test, the null hypothesis can not be rejected. Hence, the series is non-stationary.
p.kpss_test(settlement_date['wti_price']) 
#Based upon the significance level of 0.05 and the p-value of the KPSS test, the null hypothesis can be rejected. Hence, the series is non-stationary.
### detrend
#wti_oil_price_monthly['wti_price_diff'] = wti_oil_price_monthly['wti_price'] - wti_oil_price_monthly['wti_price'].shift(1)
print("De-Trend ...")
wti_oil_price_diff = p.difference(settlement_date['wti_price'], interval=1)
plt.title("de-trend of wti price")
plt.plot(wti_oil_price_diff,color="blue",Label='wti_price_settlement_date_diff')
plt.legend()
plt.savefig('de-trend_wti_price_settlement_date.png')
plt.show()

print("Check ADF/KPSS ...")
p.adf_test(wti_oil_price_diff)
#Based upon the p-value of ADF test, there is evidence for rejecting the null hypothesis in favor of the alternative. Hence, the series is strict stationary now.
p.kpss_test(wti_oil_price_diff)
#Based upon the p-value of KPSS test, the null hypothesis can not be rejected. Hence, the series is stationary.

##ACF/PACF
print("Check ACF/PACF ...")
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, ax = plt.subplots(2, figsize=(10,6))
ax[0] = plot_acf(settlement_date['wti_price'], ax=ax[0])
ax[1] = plot_pacf(settlement_date['wti_price'], lags=20, ax=ax[1])
plt.legend()
plt.savefig('ACF_PACF_settlement_date.png')
plt.show()
##Based ACF and PACF, there is AR(2). 
'''




#data preprocess
print("Data preprocess ...")
##Scale
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(wti_oil_price_diff.values.reshape(-1,1))

##train_test split
import math
train = dataset[: math.floor(len(dataset) * 0.8), :]
val = dataset[math.floor(len(dataset) * 0.8): , :]

##transform data to be supervised learning
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
outputs = pre.testing(batch_size, val_loader, model, device)
##invert scaling
import numpy as np
outputs = scaler.inverse_transform(np.array(outputs).reshape(-1, 1))
##invert differencing
actual_prediction = p.inverse_difference(settlement_date['wti_price'], outputs[:,0], 61) 

#yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
## Draw
plt.plot(settlement_date['wti_price'],color="red",Label='wti_price_settlement_date')
plt.plot(actual_prediction, color="blue",Label='wti_price_settlement_date_prediction')
plt.legend()
plt.show()

## Zoom in
plt.title("Ture vs Prediction")
plt.plot(settlement_date['wti_price'][253:314],color="red",Label='wti_price_settlement_date')
plt.plot(actual_prediction, color="blue",Label='wti_price_settlement_date_prediction')
plt.legend()
plt.savefig('wti_price_settlement_date_prediction.png')
plt.show()    
    
##pridict June. 20
test = dataset[-3:]
test_dataset = d.PriceDataset(X=test, y=None)
test_loader = DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False)
outputs = pre.testing(batch_size, test_loader, model, device)
outputs = scaler.inverse_transform(np.array(outputs).reshape(-1, 1))
actual_prediction1 = p.inverse_difference(settlement_date['wti_price'], outputs[:,0], 3) 

#plt.title("Ture vs Prediction")
#plt.plot(settlement_date['wti_price'][253:314],color="red",Label='wti_price_settlement_date')
#plt.plot(actual_prediction, color="blue",Label='wti_price_settlement_date_prediction')
#plt.legend()
#plt.savefig('wti_price_settlement_date_prediction.png')
#plt.show() 
print('\nfinish prediction')    