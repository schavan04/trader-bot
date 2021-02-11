import abc
import math
import time
import threading
from datetime import datetime, timedelta

from alpaca_trade_api.rest import APIError

import pandas as pd
import numpy as np
import matplotlib
import scipy
import sklearn

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def sma(data, window, label=None):
    if label is None:
        label = 'SMA'
    col = data.columns[0]
    data[label] = data[col].rolling(window=window).mean()
    return data

def ema(data, window, label=None):
    if label is None:
        label = 'EMA'
    col = data.columns[0]
    data[label] = data[col].ewm(span=window, adjust=False).mean()
    return data

def std(data, window, label=None):
    if label is None:
        label = 'STD'
    col = data.columns[0]
    data[label] = data[col].rolling(window=window).std()
    return data

def bbands(data, weight):
    col = data.columns[0]
    data['upperbband'] = data['EMA'] + (weight * data['STD'])
    data['lowerbband'] = data['EMA'] - (weight * data['STD'])
    return data
