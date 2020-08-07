# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:42:15 2020

@author: USER
"""
import pandas as pd
import os
# data path
path_prefix = 'D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data'

#Economic Indicators
monetary_base = pd.read_csv(os.path.join(path_prefix, 'BOGMBASEW.csv'),parse_dates =["DATE"], index_col ="DATE")
monetary_base.rename(columns={'BOGMBASEW':'monetary_base'}, 
                 inplace=True)
monetary_base.index = monetary_base.index+ pd.DateOffset(-2)
monetary_base = monetary_base.resample('D').ffill()

cpi = pd.read_csv(os.path.join(path_prefix, "CPI.csv"), usecols = ["Label", "Value"],parse_dates =["Label"], index_col ="Label")
cpi.rename(columns={'Value':'cpi'}, 
                 inplace=True)
cpi = cpi.resample('D').ffill()

fed_fund = pd.read_csv(os.path.join(path_prefix, 'fed-funds-rate-historical-chart.csv'),skiprows=15,parse_dates =["date"], index_col ="date")
fed_fund.rename(columns={' value':'fed_fund'}, 
                 inplace=True)


#Crude Oil Price
wti = pd.read_csv(os.path.join(path_prefix, 'Crude Oil WTI Futures Historical Data.csv'),usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
wti.rename(columns={'Price':'wti_price',"Vol.":"wti_volumn"}, 
                 inplace=True)
wti1 = pd.read_csv(os.path.join(path_prefix, 'Crude Oil WTI Futures Historical Data (1).csv'),usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
wti1.rename(columns={'Price':'wti_price',"Vol.":"wti_volumn"}, 
                 inplace=True)
wti_oil_price= pd.concat([wti1,wti])

#Crude Oil Production
saudi_production = pd.read_csv(os.path.join(path_prefix, 'saudi-arabia-crude-oil-production-chart.csv'),skiprows=15,parse_dates =["date"], index_col ="date")
saudi_production.rename(columns={' value':'saudi_production'}, 
                 inplace=True)
saudi_production = saudi_production.resample('D').ffill()


#combine and clean data
df= pd.concat([monetary_base,cpi,fed_fund,wti_oil_price,saudi_production], axis=1)
df=df.loc['1995-05-01':"2020-06-30"]

df = df.reset_index()
df = df[df['index'].dt.weekday < 5]
#df["brent_volumn"] = df["brent_volumn"].apply(lambda x : None if x =='-'  
#                                         else (x))
df["wti_volumn"] = df["wti_volumn"].apply(lambda x : None if x =='-'  
                                         else (x))
#df['brent_volumn']= df['brent_volumn'].astype('float64')
df['wti_volumn']= df['wti_volumn'].astype('float64')
#df['brent_volumn']= df['brent_volumn']*1000
df['wti_volumn']= df['wti_volumn']*1000
df=df.fillna(method='pad')

df.to_csv(os.path.join(path_prefix, 'data_clean.csv'), index=False)