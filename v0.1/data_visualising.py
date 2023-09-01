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
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

def plotCandlestick(df, n_days=1):

    print(df)
    df.head()

    #TODO this is used for the n day aspect of the candlestick
    #learnt from bottom 2 links on this page: https://github.com/matplotlib/mplfinance/wiki/Plotting-Too-Much-Data
    df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)


    df['date'] = pd.to_datetime(df['date'])
    resample_value = str(n_days,) + 'D'
    aggregation = {'open'  :'first',
                   'high'  :'max',
                   'low'   :'min',
                   'close' :'last',
                   'volume':'sum'}

    resampled_df = df.resample(resample_value, on='date').agg(aggregation)

    # learnt from guide linked on canvas
    fplt.plot(
            resampled_df,
            type='candle',
            title='Amazon Stock Price Candlesticks',
            ylabel='Price ($)'
        )
    
    return


def plotBoxplot(df):
    # There should be one boxplot per column, so a boxplot for each of open, high, low, close
    # Used this gude to learn about boxplot: https://www.geeksforgeeks.org/box-plot-in-python-using-matplotlib/ 
    data = [df['open'], df['high'], df['low'], df['close'], df['adjclose']]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(["open", "high", "low", "close", "adjclose"])
    bp = ax.boxplot(data, patch_artist = True)

    colors = ['#0000DD', '#00DD00',
          '#DDDD00', '#DD00DD', '#DD4400']
 
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.show()
    
    return