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

import stats

class Basic(object):
    def __init__(self, held, hit_top, hit_bottom):
        self.held = held
        self.hit_top = hit_top
        self.hit_bottom = hit_bottom

    def run_stats(self, symbol, data):
        stats.sma(data, 10, 'SMA10')
        stats.sma(data, 30, 'SMA30')

        return data

    def get_action(self, datapoint, last):

        symbol = datapoint.columns[0]
        last_trade_price = last

        if datapoint['SMA10'].item() > datapoint['SMA30'].item() and not self.held[symbol]:
            return 'buy'
        elif datapoint['SMA10'].item() < datapoint['SMA30'].item() and self.held[symbol]:
            return 'sell'
        else:
            return None

    def get_signal_points(self, data):
        symbol = data.columns[0]

        buy_signal = []
        sell_signal = []
        held = False
        hit_top = False

        for i in range(len(data)):
            if data['SMA10'][i] > data['SMA30'][i] and not held:
                buy_signal.append(data[symbol][i])
                sell_signal.append(np.nan)
                held = True
            elif data['SMA10'][i] < data['SMA30'][i] and held:
                buy_signal.append(np.nan)
                sell_signal.append(data[symbol][i])
                held = False
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)

        return (buy_signal, sell_signal)
