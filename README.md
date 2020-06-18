# Data Science Crude Oil WTI Price Estimator: Project Overview
* Created a tool that estimates data science crude oil wti price (MAE ~/$15) to help investors realize oil price in the future when they invest oil-related stocks.
* Collected 25 years economic indicators, price and production of oil, stock, bond from:
  * https://www.eia.gov/
  * https://finance.yahoo.com/
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
* WTI Crude Oil Price
* Brent Crude Oil Price
* Regular Gasoline price
* Diesel Fuel price
* Saudi Arabia Oil Production
* US Crude Oil Production
* S&P 500
* Dow Jones Industrial Average
* 1/5/10/30 year Treasury rate
* Monetary Base
* CPI
* Federal Funds Rate

## Data Cleaning
After collecting the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
* Parsed datetime data out of date
* Made a new column for the variation of CPI and Monetary Base
* Resampled the data from monthly to daily
* Removed rows with weekends of date
* Filled the previous data in missing data.

## EDA
I looked at the distributions and autocorrelation of the data, the correlation with the various variables. Below are a few highlights from the figures.
Distribution of WTI Price:
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/distribution_wti_price.png "distribution")  
Autocorrelation of WTI Price: There are AR(3)
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/ACF_PACF.png "ACF")  
Correlation with other features:
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/Features_corr.png "correlation")  

## Model Building
First, I normalized the data. I also split the data into train and tests sets with a test size of 20%.  
I tried three different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.  
I tried three different models:  
* **Linear Regression** - Baseline for the model
* **Long Short-term Memory (LSTM)** - Because the history of oil price would affect current oil price, I thought a memorable model like long short-term memory would be effective.
* **Prophet** - Again, with the time series data, I thought that this would be a good fit. Also, prophet can predict not only one period but more.   

## Model performance
Depend on the trend of oil price in the future, investors decide the strategies of investment. Although the Linear Regression model far outperformed the other approaches on the test and validation sets, the Prophet model is more practical.
* **Prophet:** MAE = 14.56   
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/prediction_prophet.png "prophet")   
* **Linear Regression:** MAE = 0.82  
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/prediction_linear.png "linear")  
* **Long Short-term Memory (LSTM):** MAE = 1.08  
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/prediction_LSTM.png "LSTM")

## Productionization
In this step, I built a flask API endpoint that was hosted on a local webserver by following along with the tutorial in the reference section above. The API endpoint takes in a request with the day of prediction and returns a list of estimated WTI Price.
