import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
#import talib as tl
import mplfinance as fplt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, Bidirectional, RNN, GRU
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


#returns true if the ADF test p value is less than 0.05 (meaning the series is stationary)
def check_stationary(series):
    result = adfuller(series)
    return result[1] <= 0.05 

def create_arima_model(data):
    df = data['df']
    df['close_diff'] = df['adjclose']
    d = 0

    #apply differecing if needed
    is_stationary = check_stationary(df['close_diff'])
    while not is_stationary:
        d += 1
        df['close_diff'] = df['close_diff'].diff()
        is_stationary = check_stationary(df['close_diff'].dropna())
        
    print('d value: ', d)

    #plot_acf(df['close_diff'].dropna(), lags=40)
    #plt.show()

    #plot_pacf(df['close_diff'].dropna(), lags=40)
    #plt.show()

    p = 27
    q = 27


    model = ARIMA(df['adjclose'], order=(p, d, q))
    return model

        

    