# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:41:37 2020

@author: USER
"""
import pandas as pd
import os
import math
#Stock
Dow_Jones = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/^DJI.csv',usecols = ["Date", "Close","Volume"],parse_dates =["Date"], index_col ="Date")
Dow_Jones.rename(columns={'Close':'dj_price','Volume':'dj_vloumn'}, 
                 inplace=True)

S_P = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/^GSPC.csv',usecols = ["Date", "Close","Volume"], parse_dates =["Date"], index_col ="Date")
S_P.rename(columns={'Close':'sp_price','Volume':'sp_vloumn'}, 
                 inplace=True)



#Bond
bond_1year = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/1-year-treasury-rate-yield-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
bond_1year.rename(columns={' value':'bond_1year'}, 
                 inplace=True)

bond_5year = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/5-year-treasury-bond-rate-yield-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
bond_5year.rename(columns={' value':'bond_5year'}, 
                 inplace=True)

bond_10year = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/10-year-treasury-bond-rate-yield-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
bond_10year.rename(columns={' value':'bond_10year'}, 
                 inplace=True)

bond_30year = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/30-year-treasury-bond-rate-yield-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
bond_30year.rename(columns={' value':'bond_30year'}, 
                 inplace=True)


#Economic Indicators
monetary_base = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/BOGMBASEW.csv',parse_dates =["DATE"], index_col ="DATE")
monetary_base.rename(columns={'BOGMBASEW':'monetary_base'}, 
                 inplace=True)
monetary_base.index = monetary_base.index+ pd.DateOffset(-2)
monetary_base = monetary_base.resample('D').ffill()

cpi = pd.read_csv("D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/CPI.csv", usecols = ["Label", "Value"],parse_dates =["Label"], index_col ="Label")
cpi.rename(columns={'Value':'cpi'}, 
                 inplace=True)
cpi = cpi.resample('D').ffill()

fed_fund = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/fed-funds-rate-historical-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
fed_fund.rename(columns={' value':'fed_fund'}, 
                 inplace=True)



#Crude Oil Price
brent_oil = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/Brent Oil Futures Historical Data.csv',usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
brent_oil.rename(columns={'Price':'brent_price',"Vol.":"brent_volumn"}, 
                 inplace=True)
brent_oil1 = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/Brent Oil Futures Historical Data (1).csv',usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
brent_oil1.rename(columns={'Price':'brent_price',"Vol.":"brent_volumn"}, 
                 inplace=True)
brent_oil_price= pd.concat([brent_oil1,brent_oil])


wti = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/Crude Oil WTI Futures Historical Data.csv',usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
wti.rename(columns={'Price':'wti_price',"Vol.":"wti_volumn"}, 
                 inplace=True)
wti1 = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/Crude Oil WTI Futures Historical Data (1).csv',usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
wti1.rename(columns={'Price':'wti_price',"Vol.":"wti_volumn"}, 
                 inplace=True)
wti_oil_price= pd.concat([wti1,wti])


Oil_Contract = pd.read_excel(open('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/RCLC1d.xls', 'rb'),
              sheet_name='Data 1', skiprows=2, usecols = ["Date", "Cushing, OK Crude Oil Future Contract 1 (Dollars per Barrel)"],parse_dates =["Date"], index_col ="Date") 
Oil_Contract.rename(columns={'Cushing, OK Crude Oil Future Contract 1 (Dollars per Barrel)':'Oil_Contract'}, 
                 inplace=True) 



#Crude Oil Production
saudi_production = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/saudi-arabia-crude-oil-production-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
saudi_production.rename(columns={' value':'saudi_production'}, 
                 inplace=True)
saudi_production = saudi_production.resample('D').ffill()

us_production = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/us-crude-oil-production-historical-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
us_production.rename(columns={' value':'us_production'}, 
                 inplace=True)
us_production = us_production.resample('D').ffill()




#Gasoline and fuel price
fuel_price = pd.read_excel(open('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/psw18vwall.xls', 'rb'),
              sheet_name='Data 1', skiprows=2, usecols = ["Date", "Weekly U.S. No 2 Diesel Retail Prices  (Dollars per Gallon)"],parse_dates =["Date"], index_col ="Date")
fuel_price.rename(columns={'Weekly U.S. No 2 Diesel Retail Prices  (Dollars per Gallon)':'fuel_price'}, 
                 inplace=True)
fuel_price = fuel_price.resample('D').ffill()

Gasoline_price = pd.read_excel(open('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/pswrgvwall.xls', 'rb'),
              sheet_name='Data 1', skiprows=2, usecols = ["Date", "Weekly U.S. Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)"],parse_dates =["Date"], index_col ="Date")
Gasoline_price.rename(columns={'Weekly U.S. Regular Conventional Retail Gasoline Prices  (Dollars per Gallon)':'Gasoline_price'}, 
                 inplace=True) 
Gasoline_price = Gasoline_price.resample('D').ffill() 


#combine and clean data
df= pd.concat([Dow_Jones,S_P,bond_1year,bond_5year,bond_10year,bond_30year,monetary_base,cpi,fed_fund,brent_oil_price,wti_oil_price,Oil_Contract,saudi_production,us_production,fuel_price,Gasoline_price], axis=1)
df=df.loc['1995-05-01':'2020-05-01']
df.index.to_series().dt.dayofweek
df = df.reset_index()
df = df[df['index'].dt.weekday < 5]
df["brent_volumn"] = df["brent_volumn"].apply(lambda x : None if x =='-'  
                                         else (x))
df["wti_volumn"] = df["wti_volumn"].apply(lambda x : None if x =='-'  
                                         else (x))
df['brent_volumn']= df['brent_volumn'].astype('float64')
df['wti_volumn']= df['wti_volumn'].astype('float64')
df['brent_volumn']= df['brent_volumn']*1000
df['wti_volumn']= df['wti_volumn']*1000
df=df.fillna(method='pad')
print(df)

#split to train data set and test data set
train = df.loc[: math.floor(len(df) * 0.8), :]
test = df.loc[math.floor(len(df) * 0.8)+1: , :]
train.to_csv(os.path.join('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data', 'train.csv'), index=False)
test.to_csv(os.path.join('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data', 'test.csv'), index=False)



