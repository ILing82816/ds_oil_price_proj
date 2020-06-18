# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:58:28 2020

@author: USER
"""
import pandas as pd
import os
#path
path_prefix = 'D:/USA 2020 summer/Machine Learning/ds_oil_price_proj'
#Depend on the data eda to take some features.
df = pd.read_csv(os.path.join(path_prefix, 'data/data_clean.csv'),usecols = ["monetary_base", "cpi","fed_fund", 'saudi_production', 'wti_price'])




# data preparation (to numpy, normalization) We use the pervious five day as features, and use the sixth day wti_price as target.
import numpy as np

data=df.to_numpy()


X = np.empty([6518, 5*5], dtype=float)
Y = np.empty([6518, 1], dtype=float)
for day in range(6518):
    X[day, :]=data[day:day+5, :].reshape(1,-1)
    Y[day, 0]=data[day+5, 3]


mean_X = np.mean(X, axis=0)
std_X= np.std(X, axis=0)
for i in range(len(X)):
    for j in range(len(X[0])):
        if std_X[j] !=0:
            X[i][j]=(X[i][j]-mean_X[j])/std_X[j]



# train_test split
import math
x_train = X[: math.floor(len(X) * 0.8), :]
y_train = Y[: math.floor(len(Y) * 0.8), :]
x_val = X[math.floor(len(X) * 0.8): , :]
y_val = Y[math.floor(len(Y) * 0.8): , :]



#Multiple linear regression (use mean absolute error to evaluation)
## sklearn Linear Regression
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

model_linear = LinearRegression()
model_linear.fit(x_train, y_train)
#print(np.mean(cross_val_score(model_linear, x_train, y_train, scoring="neg_mean_absolute_error")))

y_train_pred = model_linear.predict(x_train)
print(mean_absolute_error(y_train, y_train_pred))
y_val_pred = model_linear.predict(x_val)
print(mean_absolute_error(y_val, y_val_pred))  
y_pred = np.append(y_train_pred, y_val_pred, axis=0)
import matplotlib.pyplot as plt
plt.plot(y_pred, label='prediction (MAE:0.82)')
plt.plot(Y, label='real')
plt.legend()
plt.show()

##predict 5/1: get 18.70 real is 19.78
x_test = df.loc[6519:, :].to_numpy().reshape(1,-1)
for j in range(len(x_test[0])):
        if std_X[j] !=0:
            x_test[0][j]=(x_test[0][j]-mean_X[j])/std_X[j]
y_test_pred = model_linear.predict(x_test)



## Gradient descent
dim = 5*5+1
w = np.zeros([dim, 1])
x_train = np.concatenate((np.ones([math.floor(6518*0.8), 1]), x_train), axis = 1).astype(float)
learning_rate = 5
iter_time = 1000
sig = np.zeros([dim, 1])
eps = 0.0000000001
T=[]
Loss=[]
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train, 2))/math.floor(6518*0.8))
    if(t%100==0):
        T.append(t)
        Loss.append(loss)
    gradient = 2*np.dot(x_train.transpose(), np.dot(x_train,w)-y_train)
    sig += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(sig+eps)

x_val = np.concatenate((np.ones([math.ceil(6518*0.2), 1]), x_val), axis = 1).astype(float)
loss = np.sqrt(np.sum(np.power(np.dot(x_val, w) - y_val, 2))/math.ceil(6518*0.2))
print(loss)




