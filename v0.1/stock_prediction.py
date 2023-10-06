# File: stock_prediction.py
# Authors: Cheong Koo and Bao Vo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 25/07/2023 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

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
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from data_processing import *
from data_visualising import *
from model_creation import *
from arima_model_creation import *


#Load Data Variables
PREDICTION_DAYS = 50
COMPANY = 'AMZN'
DATA_START = '2015-01-01'
DATA_END = '2022-12-30'
SPLIT_BY_DATE = True
TEST_SIZE = 0.2
FEATURE_COLUMNS = ['adjclose', 'volume', 'open', 'high', 'low']
STORE_DATA = True
LOAD_DATA = True

#Create Model Variables
N_LAYERS = 2
CELL = GRU
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 128
EPOCHS = 20

#Multi-step variable
LOOKUP_DAYS = 1
COLUMN_PREDICTION = 'adjclose'


#make start and end date
data = load_data(data_start=DATA_START, data_end=DATA_END, ticker=COMPANY, n_steps=PREDICTION_DAYS, split_by_date=SPLIT_BY_DATE, test_size=TEST_SIZE, 
                 feature_columns=FEATURE_COLUMNS, store_data=STORE_DATA, load_data=LOAD_DATA, lookup_step=LOOKUP_DAYS)

#plotCandlestick(data['test_df'], 30)
#plotBoxplot(data['test_df'])

scaler = data['column_scaler']
scaler = scaler[COLUMN_PREDICTION]
x_train = data['X_train']
y_train = data['y_train']
x_test = data['X_test']
y_test = data['y_test']



#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = create_model(PREDICTION_DAYS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
arima_model = create_arima_model(data)


# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

print('model fitting')
arima_fit = arima_model.fit()
print('model fitted')

forecast = arima_fit.forecast(steps=LOOKUP_DAYS)

# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data

#test_data = data['test_df']
#PRICE_VALUE = 'adjclose'

# The above bug is the reason for the following line of code
#test_data = test_data[1:]

actual_prices = np.squeeze(scaler.inverse_transform(np.expand_dims(y_test, axis=0)))

#total_dataset = data['df']
#print(data['df'])
#print(data['test_df'])
# total_dataset = total_dataset[PRICE_VALUE]
#model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

#model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

#model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
#x_test = []
#for x in range(PREDICTION_DAYS, len(model_inputs)):
#    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

#x_test = np.array(x_test)
#x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


#real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
#real_data = np.array(real_data)
#real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
#real_data = y_test

#prediction = model.predict(real_data)
#prediction = scaler.inverse_transform(prediction)
#print(f"Prediction: {prediction}")

# retrieve the last sequence from data
last_sequence = data["last_sequence"][-PREDICTION_DAYS:]
# expand dimension
last_sequence = np.expand_dims(last_sequence, axis=0)
# get the prediction (scaled from 0 to 1)
prediction = model.predict(last_sequence)
# get the price (by inverting the scaling)
predicted_price = scaler.inverse_transform(prediction)[0][0]


ARIMA_WEIGHT = 1
LSTM_WEIGHT = 1

ensemble_prediction = ((forecast * ARIMA_WEIGHT) + (predicted_price * LSTM_WEIGHT)) / 2
print(f"Future price after {LOOKUP_DAYS} days is {predicted_price:.2f}$")
print(f"Future ensemble price after {LOOKUP_DAYS} days is {ensemble_prediction:}$")




# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??