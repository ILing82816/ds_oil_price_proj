# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:57:38 2020

@author: USER
"""
import pickle
from fbprophet import Prophet
from flask import Flask, jsonify, request
import pandas as pd
#from flask_cors import CORS, cross_origin

app = Flask(__name__)
#CORS(app)

@app.route("/forecast", methods=['POST'])
def predict():
    data_with_regressors = pd.read_csv('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/FlaskAPI/data_with_regressor.csv')
    with open('D:/USA 2020 summer/Machine Learning/ds_oil_price_proj/forecast_model.pckl', 'rb') as fin:
        m2 = pickle.load(fin)
    
    horizon = int(request.json['horizon'])
    
    future2 = m2.make_future_dataframe(periods=horizon)
    future2["monetary_base_diff"] = data_with_regressors["monetary_base_diff"]
    future2["monetary_base_diff"] = future2["monetary_base_diff"].fillna(method='pad')
    future2["cpi_diff"] = data_with_regressors["cpi_diff"]
    future2["cpi_diff"]= future2["cpi_diff"].fillna(method='pad')
    future2["fed_fund"] = data_with_regressors["fed_fund"]
    future2["fed_fund"] = future2["fed_fund"].fillna(method='pad')
    future2["saudi_production"] = data_with_regressors["saudi_production"]
    future2["saudi_production"] = future2["saudi_production"].fillna(method='pad')
    forecast2 = m2.predict(future2)
    
    data = forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-horizon:]
    
    ret = data.to_json(orient='records', date_format='iso')
    
    return ret

# running REST interface, port=3000 for direct test
if __name__ == "__main__":
    app.run(debug=False)