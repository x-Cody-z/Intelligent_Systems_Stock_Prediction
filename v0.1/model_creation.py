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



def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    """
    Creates the prediction model based on the following parameters
    Params:
        sequence_length (int): the sequence length/ window size
        n_features (int): the number of feature columns used for training the model (close, open, high, low, etc.)
        units (int): the size of each layer
        cell (str): the name of each layer (LSTM, RNN, GRU, etc.)
        n_layers (int): the number of layers in the model
        dropout (float): the dropout ratio as a fraction (0.3 = 30% dropout)
        loss (str): the loss method used to compile the model
        optimzier (str): the optimizer used ot compile the model
        didirectional (bool): whether or not each layer should be bidirectional
    """


    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model