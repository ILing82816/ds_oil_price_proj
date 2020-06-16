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
df = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/data_clean.csv')







'''
#Baseline Model
##prepare data
wti = pd.read_csv(os.path.join(path_prefix, 'data/data_clean.csv'),usecols = ["index", "wti_price"],parse_dates =["index"])
wti=wti.rename(columns={'index':'ds', "wti_price": "y"})

##Train
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
model_baseline = Prophet()
model_baseline.fit(wti)

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







#imporvement Model--Season
##prepare data
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

#def is_nfl_season(ds):
#    date = pd.to_datetime(ds)
#    return (date.year > 2009 and date.year < 2014)

#wti['on_season'] = wti['ds'].apply(is_nfl_season)
#wti['off_season'] = ~wti['ds'].apply(is_nfl_season)

##Train
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
model_imporvement = Prophet(holidays=holidays, weekly_seasonality=False, yearly_seasonality=20).add_seasonality(name='monthly', period=30.5, fourier_order=5)
#weekly_seasonality=False,  .add_seasonality(name='monthly', period=30.5, fourier_order=5),  changepoint_range=0.9, changepoint_prior_scale=0.5, .add_country_holidays(country_name='US')

model_imporvement.fit(wti)

future_imporvement = model_imporvement.make_future_dataframe(periods=365)
#future_imporvement['on_season'] = future_imporvement['ds'].apply(is_nfl_season)
#future_imporvement['off_season'] = ~future_imporvement['ds'].apply(is_nfl_season)

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
'''




#imporvement Model--Additional regressors
##prepare data
import pandas as pd
data_with_regressors = pd.read_csv(os.path.join(path_prefix, 'data/data_clean.csv'),usecols = ["index", "wti_price", 'monetary_base','cpi','brent_price',"fuel_price"],parse_dates =["index"])
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
  'upper_window': 270,
})
holidays = pd.concat((election, events, crisis))

#def is_nfl_season(ds):
#    date = pd.to_datetime(ds)
#    return (date.year > 2009 and date.year < 2014)

#wti['on_season'] = wti['ds'].apply(is_nfl_season)
#wti['off_season'] = ~wti['ds'].apply(is_nfl_season)

##Train
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
model_imporvement2 = Prophet(holidays=holidays, weekly_seasonality=False, yearly_seasonality=20, seasonality_mode='multiplicative')
model_imporvement2.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_imporvement2.add_regressor('monetary_base', mode='multiplicative')
model_imporvement2.add_regressor('cpi', mode='multiplicative')
model_imporvement2.add_regressor('brent_price', mode='multiplicative')
model_imporvement2.add_regressor('fuel_price', mode='multiplicative')
#model_imporvement2.add_regressor('saudi_production', mode='multiplicative')
#weekly_seasonality=False,  .add_seasonality(name='monthly', period=30.5, fourier_order=5),  changepoint_range=0.9, changepoint_prior_scale=0.5, .add_country_holidays(country_name='US')

model_imporvement2.fit(data_with_regressors)

future_imporvement2 = model_imporvement2.make_future_dataframe(periods=365)
future_imporvement2["monetary_base"] = data_with_regressors["monetary_base"]
future_imporvement2["monetary_base"] = future_imporvement2["monetary_base"].fillna(method='pad')
future_imporvement2["cpi"] = data_with_regressors["cpi"]
future_imporvement2["cpi"]= future_imporvement2["cpi"].fillna(method='pad')
future_imporvement2["brent_price"] = data_with_regressors["brent_price"]
future_imporvement2["brent_price"] = future_imporvement2["brent_price"].fillna(method='pad')
future_imporvement2["fuel_price"] = data_with_regressors["fuel_price"]
future_imporvement2["fuel_price"] = future_imporvement2["fuel_price"].fillna(method='pad')
#future_imporvement2["saudi_production"] = data_with_regressors["saudi_production"]
#future_imporvement2["saudi_production"] = future_imporvement2["saudi_production"].fillna(method='pad')
#future_imporvement['on_season'] = future_imporvement['ds'].apply(is_nfl_season)
#future_imporvement['off_season'] = ~future_imporvement['ds'].apply(is_nfl_season)

forecast_imporvement2 = model_imporvement2.predict(future_imporvement2)
#predict = forecast_imporvement[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
fig1_imporvement2 = model_imporvement2.plot(forecast_imporvement2)
a_imporvement2 = add_changepoints_to_plot(fig1_imporvement2.gca(), model_imporvement2, forecast_imporvement2)
fig2_imporvement2 = model_imporvement2.plot_components(forecast_imporvement2)

##Evaluation
import predict as p
from fbprophet.diagnostics import cross_validation
imporvement2_cv = cross_validation(model_imporvement2, initial='730 days', period='180 days', horizon = '365 days')
mape_imporvement2 = p.mean_absolute_percentage_error(imporvement2_cv.y, imporvement2_cv.yhat)




