# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 08:52:06 2020

@author: USER
"""
# Set up some path and parameters.
## path
import os
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import predict as p
from fbprophet.diagnostics import cross_validation
from pandas import Series
#path
path_prefix = 'D:/USA 2020 summer/Machine Learning/ds_oil_price_proj'



'''
#Baseline Model
##prepare data
print("loading data ...")
wti = pd.read_csv(os.path.join(path_prefix, 'data/data_clean.csv'),usecols = ["index", "wti_price"],parse_dates =["index"])
wti=wti.rename(columns={'index':'ds', "wti_price": "y"})

##Train
print("Start training...")
model_baseline = Prophet()
model_baseline.fit(wti)

future_baseline = model_baseline.make_future_dataframe(periods=365)
forecast_baseline = model_baseline.predict(future_baseline)
#predict = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
fig1_baseline = model_baseline.plot(forecast_baseline)
a_baseline = add_changepoints_to_plot(fig1_baseline.gca(), model_baseline, forecast_baseline)
fig2_baseline = model_baseline.plot_components(forecast_baseline)

##Evaluation
baseline_cv = cross_validation(model_baseline, initial='730 days', period='180 days', horizon = '365 days')
mape_baseline = p.mean_absolute_error(baseline_cv.y, baseline_cv.yhat)





#imporvement Model--Season
##prepare data
print("loading data ...")
wti = pd.read_csv(os.path.join(path_prefix, 'data/data_clean.csv'),usecols = ["index", "wti_price"],parse_dates =["index"])
wti=wti.rename(columns={'index':'ds', "wti_price": "y"})

## set up season
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
crisis = pd.DataFrame({
  'holiday': 'crisis',
  'ds': pd.to_datetime(['2008-05-20']),#'2000-03-10',"2003-03-20", ,'2010-09-15'
  'lower_window': -120,
  'upper_window': 360,
})
holidays = pd.concat((election, events, crisis))

##Train
print("Start training...")
model_imporvement = Prophet(holidays=holidays, weekly_seasonality=False, yearly_seasonality=20).add_seasonality(name='monthly', period=30.5, fourier_order=5)
#weekly_seasonality=False,  .add_seasonality(name='monthly', period=30.5, fourier_order=5),  changepoint_range=0.9, changepoint_prior_scale=0.5, .add_country_holidays(country_name='US')

model_imporvement.fit(wti)

future_imporvement = model_imporvement.make_future_dataframe(periods=365)

forecast_imporvement = model_imporvement.predict(future_imporvement)
#predict = forecast_imporvement[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
fig1_imporvement = model_imporvement.plot(forecast_imporvement)
a_imporvement = add_changepoints_to_plot(fig1_imporvement.gca(), model_imporvement, forecast_imporvement)
fig2_imporvement = model_imporvement.plot_components(forecast_imporvement)

##Evaluation
imporvement_cv = cross_validation(model_imporvement, initial='730 days', period='180 days', horizon = '365 days')
mape_imporvement = p.mean_absolute_error(imporvement_cv.y, imporvement_cv.yhat)
'''




#imporvement Model--Additional regressors
##prepare data
print("loading data ...")

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)


df = pd.read_csv(os.path.join(path_prefix, 'data/data_clean.csv'),usecols = ["index","monetary_base", "cpi","fed_fund", 'saudi_production', 'wti_price'])


monetary_base_diff = difference(df['monetary_base'], interval=1)
monetary_base_diff = monetary_base_diff.to_frame()
monetary_base_diff =monetary_base_diff.append({0:0}, ignore_index=True)
monetary_base_diff['adjust'] = 0
for i in range(len(monetary_base_diff)):
    monetary_base_diff['adjust'].loc[i+1] = monetary_base_diff[0].loc[i]

df["monetary_base_diff"] = monetary_base_diff['adjust']

cpi_diff = difference(df['cpi'], interval=1)
cpi_diff = cpi_diff.to_frame()
cpi_diff =cpi_diff.append({0:0}, ignore_index=True)
cpi_diff['adjust'] = 0
for i in range(len(cpi_diff)):
    cpi_diff['adjust'].loc[i+1] = cpi_diff[0].loc[i]

df["cpi_diff"] = cpi_diff['adjust']

data_with_regressors= df.drop(['monetary_base', 'cpi'], axis=1)
data_with_regressors=data_with_regressors.rename(columns={'index':'ds', "wti_price": "y"})


## set up season
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
crisis = pd.DataFrame({
  'holiday': 'crisis',
  'ds': pd.to_datetime(['2008-05-20']),#'2000-03-10',"2003-03-20", ,'2010-09-15'
  'lower_window': -120,
  'upper_window': 360,
})
holidays = pd.concat((election, events, crisis))

##Train
print("Start training...")
model_imporvement2 = Prophet(holidays=holidays, weekly_seasonality=False, yearly_seasonality=20)
model_imporvement2.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_imporvement2.add_regressor('monetary_base_diff')
model_imporvement2.add_regressor('cpi_diff')
model_imporvement2.add_regressor('fed_fund')
model_imporvement2.add_regressor('saudi_production')

model_imporvement2.fit(data_with_regressors)

future_imporvement2 = model_imporvement2.make_future_dataframe(periods=365)
future_imporvement2["monetary_base_diff"] = data_with_regressors["monetary_base_diff"]
future_imporvement2["monetary_base_diff"] = future_imporvement2["monetary_base_diff"].fillna(method='pad')
future_imporvement2["cpi_diff"] = data_with_regressors["cpi_diff"]
future_imporvement2["cpi_diff"]= future_imporvement2["cpi_diff"].fillna(method='pad')
future_imporvement2["fed_fund"] = data_with_regressors["fed_fund"]
future_imporvement2["fed_fund"] = future_imporvement2["fed_fund"].fillna(method='pad')
future_imporvement2["saudi_production"] = data_with_regressors["saudi_production"]
future_imporvement2["saudi_production"] = future_imporvement2["saudi_production"].fillna(method='pad')

forecast_imporvement2 = model_imporvement2.predict(future_imporvement2)
fig1_imporvement2 = model_imporvement2.plot(forecast_imporvement2)
a_imporvement2 = add_changepoints_to_plot(fig1_imporvement2.gca(), model_imporvement2, forecast_imporvement2)
fig2_imporvement2 = model_imporvement2.plot_components(forecast_imporvement2)

##Evaluation
imporvement2_cv = cross_validation(model_imporvement2, initial='730 days', period='180 days', horizon = '365 days')
#mape_imporvement2 = p.mean_absolute_percentage_error(imporvement2_cv.y, imporvement2_cv.yhat)
mae_imporvement2 = p.mean_absolute_error(imporvement2_cv.y, imporvement2_cv.yhat)




# flask_API--store the best model (model_improvement2)
import pickle
with open(os.path.join(path_prefix, 'FlaskApI/pre_prophet_model.pckl'), 'wb') as fout:
    pickle.dump(model_imporvement2, fout)

data_with_regressors.to_csv(os.path.join(path_prefix, 'FlaskApI/data_with_regressor.csv'), index=False)
forecast_imporvement2.to_csv(os.path.join(path_prefix, 'Optimization/Forcast_20210118.csv'), index=False)




