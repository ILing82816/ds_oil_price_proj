# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:58:28 2020

@author: USER
"""
import pandas as pd

df = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/data_clean.csv')
df = df.drop(["index"], axis=1)



# data preparation (to numpy, normalization)
import numpy as np

data=df.to_numpy()
#X = np.empty([1303, 19*5], dtype=float)
#Y = np.empty([1303, 1], dtype=float)
#for week in range(1303):
#    X[week, :]=data[5*(week):5*(week+1), :].reshape(1,-1)
#    Y[week, 0]=data[5*(week+1), 13]

X = np.empty([6518, 19*5], dtype=float)
Y = np.empty([6518, 1], dtype=float)
for day in range(6518):
    X[day, :]=data[day:day+5, :].reshape(1,-1)
    Y[day, 0]=data[day+5, 13]





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



#Multiple linear regression
## Gradient descent
dim = 19*5+1
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
        print(str(t) + ":" + str(loss))
        T.append(t)
        Loss.append(loss)
    gradient = 2*np.dot(x_train.transpose(), np.dot(x_train,w)-y_train)
    sig += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(sig+eps)

import matplotlib.pyplot as plt
plt.plot(T,Loss,color="red", label='learning rate=5')
plt.legend()
plt.show()

x_val = np.concatenate((np.ones([math.ceil(6518*0.2), 1]), x_val), axis = 1).astype(float)
loss = np.sqrt(np.sum(np.power(np.dot(x_val, w) - y_val, 2))/math.ceil(6518*0.2))
print(loss)

## statsmodels
import statsmodels.api as sm

X_sm = sm.add_constant(X) 
model = sm.OLS(Y, X_sm)
print(model.fit().summary())

## sklearn Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model_linear = LinearRegression()
model_linear.fit(x_train, y_train)
print(np.mean(cross_val_score(model_linear, x_train, y_train, scoring="neg_mean_squared_error")))
print(np.mean(cross_val_score(model_linear, x_train, y_train, scoring="neg_mean_absolute_error")))
print(np.mean(cross_val_score(model_linear, x_train, y_train, scoring="r2")))

y_pred = model_linear.predict(x_val)
print(mean_squared_error(y_val, y_pred))
print(mean_absolute_error(y_val, y_pred))
print(r2_score(y_val, y_pred))


# ARMA model

# RNN (LSTM)

