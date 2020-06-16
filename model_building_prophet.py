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
import preprocess as p
#data_with_regressors = pd.read_csv(os.path.join(path_prefix, 'data/data_clean.csv'),usecols = ["index", "wti_price", 'monetary_base','cpi','fed_fund'],parse_dates =["index"])
#data_with_regressors=data_with_regressors.rename(columns={'index':'ds', "wti_price": "y"})
monetary_base = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/BOGMBASEW.csv',parse_dates =["DATE"], index_col ="DATE")
monetary_base.rename(columns={'BOGMBASEW':'monetary_base'}, 
                 inplace=True)
monetary_base.index = monetary_base.index+ pd.DateOffset(-2)
monetary_base = monetary_base.reset_index()
monetary_base_diff = p.difference(monetary_base['monetary_base'], interval=1)
monetary_base_diff = monetary_base_diff.to_frame()
monetary_base_diff =monetary_base_diff.append({0:0}, ignore_index=True)
monetary_base_diff['adjust'] = 0
for i in range(len(monetary_base_diff)):
    monetary_base_diff['adjust'].loc[i+1] = monetary_base_diff[0].loc[i]

monetary_base["monetary_base_diff"] = monetary_base_diff['adjust']
monetary_base = monetary_base.set_index('DATE')    
monetary_base = monetary_base.resample('D').ffill()

cpi = pd.read_csv("D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/CPI.csv", usecols = ["Label", "Value"],parse_dates =["Label"], index_col ="Label")
cpi.rename(columns={'Value':'cpi'}, 
                 inplace=True)
cpi = cpi.reset_index()
cpi_diff = p.difference(cpi['cpi'], interval=1)
cpi_diff = cpi_diff.to_frame()
cpi_diff =cpi_diff.append({0:0}, ignore_index=True)
cpi_diff['adjust'] = 0
for i in range(len(cpi_diff)):
    cpi_diff['adjust'].loc[i+1] = cpi_diff[0].loc[i]

cpi["cpi_diff"] = cpi_diff['adjust']
cpi = cpi.set_index('Label')
cpi = cpi.resample('D').ffill()

fed_fund = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/fed-funds-rate-historical-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
fed_fund.rename(columns={' value':'fed_fund'}, 
                 inplace=True)

#bond_5year = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/5-year-treasury-bond-rate-yield-chart.csv',skiprows=15,parse_dates =["date"], index_col ="date")
#bond_5year.rename(columns={' value':'bond_5year'}, 
#                 inplace=True)

wti = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/Crude Oil WTI Futures Historical Data.csv',usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
wti.rename(columns={'Price':'wti_price',"Vol.":"wti_volumn"}, 
                 inplace=True)
wti1 = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/data/Crude Oil WTI Futures Historical Data (1).csv',usecols = ["Date", "Price", "Vol."],parse_dates =["Date"], index_col ="Date")
wti1.rename(columns={'Price':'wti_price',"Vol.":"wti_volumn"}, 
                 inplace=True)
wti_oil_price= pd.concat([wti1,wti])

data_with_regressors= pd.concat([wti_oil_price["wti_price"],monetary_base["monetary_base_diff"],cpi["cpi_diff"],fed_fund], axis=1)
data_with_regressors=data_with_regressors.loc['1995-05-01':"2020-04-30"]
data_with_regressors = data_with_regressors.reset_index()
data_with_regressors = data_with_regressors[data_with_regressors['index'].dt.weekday < 5]
data_with_regressors=data_with_regressors.fillna(method='pad')
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

#def is_nfl_season(ds):
#    date = pd.to_datetime(ds)
#    return (date.year > 2009 and date.year < 2014)

#wti['on_season'] = wti['ds'].apply(is_nfl_season)
#wti['off_season'] = ~wti['ds'].apply(is_nfl_season)

##Train
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
model_imporvement2 = Prophet(holidays=holidays, weekly_seasonality=False, yearly_seasonality=20)
model_imporvement2.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_imporvement2.add_regressor('monetary_base_diff')
model_imporvement2.add_regressor('cpi_diff')
model_imporvement2.add_regressor('fed_fund')
#model_imporvement2.add_regressor('bond_5year')
#model_imporvement2.add_regressor('fuel_price', mode='multiplicative')
#model_imporvement2.add_regressor('saudi_production', mode='multiplicative')
#weekly_seasonality=False,  .add_seasonality(name='monthly', period=30.5, fourier_order=5),  changepoint_range=0.9, changepoint_prior_scale=0.5, .add_country_holidays(country_name='US')

model_imporvement2.fit(data_with_regressors)

future_imporvement2 = model_imporvement2.make_future_dataframe(periods=365)
future_imporvement2["monetary_base_diff"] = data_with_regressors["monetary_base_diff"]
future_imporvement2["monetary_base_diff"] = future_imporvement2["monetary_base_diff"].fillna(method='pad')
future_imporvement2["cpi_diff"] = data_with_regressors["cpi_diff"]
future_imporvement2["cpi_diff"]= future_imporvement2["cpi_diff"].fillna(method='pad')
future_imporvement2["fed_fund"] = data_with_regressors["fed_fund"]
future_imporvement2["fed_fund"] = future_imporvement2["fed_fund"].fillna(method='pad')
#future_imporvement2["bond_5year"] = data_with_regressors["bond_5year"]
#future_imporvement2["bond_5year"] = future_imporvement2["bond_5year"].fillna(method='pad')
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



