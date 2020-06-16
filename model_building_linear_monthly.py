# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 13:47:24 2020

@author: USER
"""
import pandas as pd

df = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/data_clean.csv')

date = df["index"].values
year = [my_str.split("-")[0] for my_str in date]
month = [my_str.split("-")[1] for my_str in date]
day = [my_str.split("-")[2] for my_str in date]
df["year"] = year
df["month"] = month
df["day"] = day
df['year']=df['year'].astype('int')
df['month']=df['month'].astype('int')
df = df.drop(["index"], axis=1)
day_data = df.groupby(['year','month'])["day"].count()
amountofday =min(df.groupby(['year','month'])["day"].count())
print(amountofday)


# data preparation (to numpy, normalization)
import numpy as np
import math

data=df.to_numpy()

X = np.empty([8+24*12+4, 19*20], dtype=float)
Y = np.empty([8+24*12+4, 1], dtype=float)    
day = 0
month = 0
for len_day in day_data:
    not_day = len_day-20
    X[month, :] = data[day:day+len_day-not_day, 0:19].reshape(1,-1)
    Y[month, 0] = data[day+math.floor(len_day*2/3), 13]
    day += len_day
    month += 1

X =  np.delete(X,299, 0)     
Y = np.delete(Y, 0, 0)      


mean_X = np.mean(X, axis=0)
std_X= np.std(X, axis=0)
for i in range(len(X)):
    for j in range(len(X[0])):
        if std_X[j] !=0:
            X[i][j]=(X[i][j]-mean_X[j])/std_X[j]
            

# train_test split
x_train = X[: math.floor(len(X) * 0.8), :]
y_train = Y[: math.floor(len(Y) * 0.8), :]
x_val = X[math.floor(len(X) * 0.8): , :]
y_val = Y[math.floor(len(Y) * 0.8): , :]



#Multiple linear regression
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

## Gradient descent
dim = 19*20+1
w = np.zeros([dim, 1])
x_train = np.concatenate((np.ones([math.floor(299*0.8), 1]), x_train), axis = 1).astype(float)
learning_rate = 10
iter_time = 2000
sig = np.zeros([dim, 1])
eps = 0.0000000001
T=[]
Loss=[]
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train, 2))/math.floor(299*0.8))
    if(t%100==0):
        print(str(t) + ":" + str(loss))
        T.append(t)
        Loss.append(loss)
    gradient = 2*np.dot(x_train.transpose(), np.dot(x_train,w)-y_train)
    sig += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(sig+eps)

import matplotlib.pyplot as plt
plt.plot(T,Loss,color="red", label='learning rate=10')
plt.legend()
plt.show()

x_val = np.concatenate((np.ones([math.ceil(299*0.2), 1]), x_val), axis = 1).astype(float)
loss = np.sqrt(np.sum(np.power(np.dot(x_val, w) - y_val, 2))/math.ceil(299*0.2))
print(loss)