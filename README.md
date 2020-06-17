# Data Science Crude Oil WTI Price Estimator: Project Overview
* Created a tool that estimates data science crude oil wti price (MAPE ~27\%) to help investors realize oil price in the future when they invest oil-related stocks.
* Collected 25 years economic indicators, Saudi Arabia production of oil and wti price from:
  * https://www.macrotrends.net/
  * https://beta.bls.gov/
  * https://fred.stlouisfed.org/
  * https://www.investing.com/
* Engineered features from the economic indicators to de-trend the value put on python, excel.
* Optimized Linear, Long Short-term Memory, and Prophet tuning parameters, seasonality and add_regressor, to reach the best model.
* Built a client facing API using flask

## Code and Resources Used
**Python Version:** 3.7
**Packages:** pandas, numpy, sklearn, statsmodels, torce, fbprophet, matplotlib, seaborn, flask, json, pickle
**For Web Framework Requirements:** `pip install -r requirements.txt`
**Model Building Github:** https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb
**Model Building Article:** https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3
**Flask Productionization:** https://github.com/abaranovskis-redsamurai/automation-repo/blob/master/forecast/ironsteel_forecast_prophet.ipynb

## Data Collecting
With each date, we got the following:
* WTI Oil Price
* Monetary Base
* CPI
* Federal Funds Rate
* Saudi Arabia production of oil

## Data Cleaning
After collecting the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
* Parsed datetime data out of date
* Made a new column for the variation of CPI and Monetary Base
* Resampled the data from monthly to daily
* Removed rows with weekends of date
* Filled the previous data in missing data.

## EDA
I looked at the distributions and autocorrelation of the data, the correlation with the various variables. Below are a few highlights from the figures.


## Model Building

## Model performance

## Productionization
