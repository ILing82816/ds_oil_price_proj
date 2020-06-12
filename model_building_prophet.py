# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 08:52:06 2020

@author: USER
"""
# Set up some path and parameters.
## path
import os
path_prefix = 'D:/USA 2020 summer/Machine Learning/ds_oil_price_proj'




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
wti_oil_price = wti_oil_price.drop(['wti_volumn'], axis=1)  
wti_oil_price=wti_oil_price.rename(columns={"Date": "ds", "wti_price": "y"})
 


'''
#Baseline Model
##Train
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
model_baseline = Prophet()
model_baseline.fit(wti_oil_price)

future_baseline = model_baseline.make_future_dataframe(periods=365)
forecast_baseline = model_baseline.predict(future_baseline)
#predict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
fig1_baseline = model_baseline.plot(forecast_baseline)
a_baseline = add_changepoints_to_plot(fig1_baseline.gca(), model_baseline, forecast_baseline)
fig2_baseline = model_baseline.plot_components(forecast_baseline)

##Evaluation
import predict as p
from fbprophet.diagnostics import cross_validation
baseline_cv = cross_validation(model_baseline, initial='730 days', period='180 days', horizon = '365 days')
mape_baseline = p.mean_absolute_percentage_error(baseline_cv.y, baseline_cv.yhat)
'''


#imporvement Model
##Train
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
election = pd.DataFrame({
  'holiday': 'election',
  'ds': pd.to_datetime(['1996-11-01', '2000-11-01', '2004-11-01',
                        '2008-11-01', '2012-11-01', '2016-11-01',
                        '2020-11-01']),
  'lower_window': -120,
  'upper_window': 30,
})
events = pd.DataFrame({
  'holiday': 'event',
  'ds': pd.to_datetime(['2020-03-09']),
  'lower_window': 0,
  'upper_window': 60,
})
holidays = pd.concat((election, events))


model_imporvement = Prophet(holidays=holidays, weekly_seasonality=False).add_seasonality(name='monthly', period=30.5, fourier_order=5)
#weekly_seasonality=False,  .add_seasonality(name='monthly', period=30.5, fourier_order=5),  changepoint_range=0.9, changepoint_prior_scale=0.5, .add_country_holidays(country_name='US')

model_imporvement.fit(wti_oil_price)

future_imporvement = model_imporvement.make_future_dataframe(periods=365)
forecast_imporvement = model_imporvement.predict(future_imporvement)
#predict = forecast_imporvement[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
fig1_imporvement = model_imporvement.plot(forecast_imporvement)
a_imporvement = add_changepoints_to_plot(fig1_imporvement.gca(), model_imporvement, forecast_imporvement)
fig2_imporvement = model_imporvement.plot_components(forecast_imporvement)

##Evaluation
import predict as p
from fbprophet.diagnostics import cross_validation
imporvement_cv = cross_validation(model_imporvement, initial='730 days', period='180 days', horizon = '365 days')
mape_imporvement = p.mean_absolute_percentage_error(imporvement_cv.y, imporvement_cv.yhat)





