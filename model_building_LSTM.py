# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:47:25 2020

@author: USER
"""

import pandas as pd

df = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/data_clean.csv')


# data preparation



# train_test split
import math
x_train = X[: math.floor(len(X) * 0.8), :]
y_train = Y[: math.floor(len(Y) * 0.8), :]
x_val = X[math.floor(len(X) * 0.8): , :]
y_val = Y[math.floor(len(Y) * 0.8): , :]


