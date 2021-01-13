import abc
import math
import time
import threading

import pandas as pd
import numpy as np
import matplotlib
import scipy
import sklearn

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import data_filters

class Strategy(abc.ABC):
    @abc.abstractmethod
    def __init__(self, interface):
        self.interface = interface

    @abc.abstractmethod
    def system_loop(self):
        pass


class ShellSystemTest(Strategy):
    def __init__(self, interface):
        super().__init__(interface)

    def get_buyable_stocks(self, quantity, price):
        symbols = data_filters.get_yf_gainers(quantity, price)
        approved = data_filters.check_rating(symbols)
        return approved

    def system_loop(self):
        symbols = self.get_buyable_stocks(5, 200)
        for x in symbols:
            print(x)


class Basic(Strategy):
    def __init__(self, interface):
        super().__init__(interface)

    def time_until_market_close():
        clock = self.interface.get_clock()
        time_to_close = (clock.next_close - clock.timestamp).total_seconds()
        print(time_to_close)
        return time_to_close

    def sleep_until_market_open():
        clock = self.interface.get_clock()
        if not clock.is_open:
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            print(f'Sleeping for {time_to_open} seconds...')
            time.sleep(round(time_to_open))

    def get_buyable_stocks(self, quantity, price):
        time.sleep(1)
        symbols = data_filters.get_yf_gainers(quantity, price)
        approved = data_filters.check_rating(symbols)
        return approved

    def get_barset(self, symbol):
        data = self.interface.get_apca_data(symbol)
        df = data[symbol].df
        bar = {
            'high': float(df['high'].max()),
            'low': float(df['low'].min())
        }
        return bar

    def place_multi_order(self, symbol, quantity, direction):
        # if self.time_until_market_close() > 120:
            bar = self.get_barset(symbol)
            margin = float(bar['high']) - float(bar['low'])

            if direction == 'buy':
                stop_loss = bar['high'] - margin
                take_profit = bar['high'] + margin
            elif direction == 'sell':
                stop_loss = bar['low'] + margin
                take_profit = bar['low'] - margin

            sl = dict(stop_price = str(stop_loss))
            tp = dict(limit_price = str(take_profit))

            # Add a check for trying to sell stock we don't have

            self.interface.multi_order(symbol, quantity, direction, sl, tp)
            return True
        # else:
        #     return False

    def system_loop(self):
        # approved_symbols = self.get_buyable_stocks(5, 150)
        # print("Approved stocks:")
        # for x in approved_symbols:
        #     print(x)
        # self.place_multi_order(approved_symbols[0], 5, 'buy')
        # print(f'Bought 5 of {approved_symbols[0]}')
        pass


class MovingAvgDay(Strategy):
    def __init__(self, interface):
        super().__init__(interface)

    def time_until_market_close(self):
        clock = self.interface.get_clock()
        return (clock.next_close - clock.timestamp).total_seconds()

    def sleep_until_market_open(self):
        clock = self.interface.get_clock()
        if not clock.is_open:
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            print(time_to_open)
            time.sleep(round(time_to_open))

    def get_buyable_stocks(self, quantity, price):
        symbols = data_filters.get_yf_gainers(quantity, price)
        return symbols

    def run_ratings_filter(self, symbols):
        approved = data_filters.check_rating(symbols)
        return approved

    def get_barset(self, symbol, timeframe=None, limit=None):
        if timeframe is None:
            timeframe = 'day'
        if limit is None:
            limit = 1000
        data = self.interface.get_apca_data(symbol, timeframe, limit)
        df = data[symbol].df
        return df

    def get_high_low(self, symbol):
        data = self.interface.get_apca_data(symbol, '1Min', 60)
        df = data[symbol].df

        high = float(df['high'].max())
        low = float(df['low'].min())

        margin = high - low
        print(data[symbol].df)

        loss = low + margin
        profit = low - margin

        return (loss, profit)

    def plot_switches(self, data, symbol):
        signal_price_buy = []
        signal_price_sell = []
        flag = -1 # -1 is nothing, 0 means has already sent sell signal, 1 means has already sent buy signal

        for i in range(len(data)):
            if data['SMA30'][i] > data['SMA100'][i]:
                if flag != 1:
                    signal_price_buy.append(data[symbol][i])
                    signal_price_sell.append(np.nan)
                    flag = 1
                else:
                    signal_price_buy.append(np.nan)
                    signal_price_sell.append(np.nan)
            elif data['SMA30'][i] < data['SMA100'][i]:
                if flag != 0:
                    signal_price_buy.append(np.nan)
                    signal_price_sell.append(data[symbol][i])
                    flag = 0
                else:
                    signal_price_buy.append(np.nan)
                    signal_price_sell.append(np.nan)
            else:
                signal_price_buy.append(np.nan)
                signal_price_sell.append(np.nan)

        return (signal_price_buy, signal_price_sell)

    def check_trend(self, symbol):
        barset = self.get_barset(symbol, timeframe='day', limit=1000)

        # 30-day simple moving average
        sma30 = pd.DataFrame()
        sma30['close'] = barset['close'].rolling(window=30).mean()

        # 100-day simple moving average
        sma100 = pd.DataFrame()
        sma100['close'] = barset['close'].rolling(window=100).mean()

        data = pd.DataFrame()
        data[f'{symbol}'] = barset['close']
        data['SMA30'] = sma30['close']
        data['SMA100'] = sma100['close']

        switch_points = self.plot_switches(data, symbol)
        data['buy_signal_price'] = switch_points[0]
        data['sell_signal_price'] = switch_points[1]

        print(data.tail(1))
        data = data.tail(500)

        # plt.figure(figsize=(12.5, 4.5))
        # plt.plot(data[f'{symbol}'], label=f"{symbol} close", alpha=0.35)
        # plt.plot(data['SMA30'], label="SMA30", alpha=0.35)
        # plt.plot(data['SMA100'], label="SMA100", alpha=0.35)
        # plt.scatter(data.index, data['buy_signal_price'], label='Buy', marker='^', color='green')
        # plt.scatter(data.index, data['sell_signal_price'], label='Sell', marker='v', color='red')
        # plt.title(f"{symbol} close price history and signals")
        # plt.xlabel("Date")
        # plt.ylabel("Close Price")
        # plt.legend(loc='upper left')
        # plt.show()

        if data.tail(1)['SMA30'][0] > data.tail(1)['SMA100'][0]:
            return 'buy'
        elif data.tail(1)['SMA30'][0] < data.tail(1)['SMA100'][0]:
            return 'sell'
        else:
            return None

    def system_test(self):
        direction = self.check_trend('PLUG')
        p_loss, p_profit = self.get_high_low('PLUG')

        print(direction)
        print(p_loss)
        print(p_profit)
        print(self.interface.get_last_quote('PLUG'))

    def system_loop(self):
        self.sleep_until_market_open()

        gainers = self.get_buyable_stocks(5, 200) # TODO: Refactor the scraper in data_filters
        todays_focus = self.run_ratings_filter(gainers) # /                    /
        print(todays_focus) #                         <--/ This should have 5 /

        while(self.interface.get_clock().is_open and self.time_until_market_close() > 120):
            time.sleep(1)

            positions = self.interface.list_all_positions()
            owned = []
            for i in range(0, len(positions)):
                owned.append(positions[i].symbol)
            print(f'Positions: {owned}')

            if positions:
                for position in positions:
                    # direction = self.check_trend(position.symbol)
                    # p_loss, p_profit = self.get_high_low(position.symbol)

                    # yf_rating = len(self.run_ratings_filter([position.symbol])) # TODO: Refactor the rating filter in data_filters
                    # yf_sell = (yf_rating == 0) # Boolean value

                    if position.unrealized_plpc < -0.35 or position.unrealized_plpc > 0.5:
                        self.interface.market_order(symbol=position.symbol, quantity=position.qty, direction='sell')

            buying_power = self.get_account().buying_power
            for symbol in todays_focus:
                direction = self.check_trend(symbol)

                if direction == 'buy' and symbol not in owned:
                    qty = math.floor((buying_power / len(symbols)) / self.interface.get_last_quote(symbol))
                    if qty > 0:
                        try:
                            self.interface.market_order(symbol=symbol, quantity=qty, direction='buy')
                        except Exception as e:
                            print(f"Error: {symbol} is not tradable")

        for position in self.interface.list_all_positions():
            self.interface.market_order(symbol=position.symbol, quantity=position.qty, direction='sell')
        time.sleep(self.time_until_market_close() + 1)


class BollingerShortTerm(Strategy):
    def __init__(self, interface):
        super().__init__(interface)

    def time_until_market_close(self):
        clock = self.interface.get_clock()
        return (clock.next_close - clock.timestamp).total_seconds()

    def sleep_until_market_open(self):
        clock = self.interface.get_clock()
        if not clock.is_open:
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            print(f'* Sleeping for {time_to_open} seconds')
            time.sleep(round(time_to_open))

    def get_buyable_stocks(self, price):
        symbols = data_filters.get_yf_gainers(price)
        return symbols

    def run_ratings_filter(self, symbols, quantity):
        approved = data_filters.check_rating(symbols, quantity)
        return approved

    def plot_switches(self, data, symbol):
        buy_signal = []
        sell_signal = []
        flag = 1 # 0 means not held, 1 means holding

        for i in range(len(data)):

            if data[f'{symbol}'][i] <= data['lower'][i]:
                if flag != 1:
                    buy_signal.append(data[symbol][i])
                    sell_signal.append(np.nan)
                    flag = 1
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            elif data[f'{symbol}'][i] >= data['upper'][i]:
                if flag != 0:
                    buy_signal.append(np.nan)
                    sell_signal.append(data[symbol][i])
                    flag = 0
                else:
                    buy_signal.append(np.nan)
                    sell_signal.append(np.nan)
            else:
                buy_signal.append(np.nan)
                sell_signal.append(np.nan)

        return (buy_signal, sell_signal)

    def calculate_band(self, symbol):
        raw = self.interface.get_apca_data(symbol, timeframe='minute', limit=250)
        barset = raw[symbol].df

        data = pd.DataFrame()
        data[f'{symbol}'] = barset['close']
        data['SMA'] = data[f'{symbol}'].rolling(window=10).mean()
        data['EMA'] = data[f'{symbol}'].ewm(span=10, adjust=False).mean()
        data['STD'] = data[f'{symbol}'].rolling(window=10).std()
        data['upper'] = data['EMA'] + (1.5 * data['STD'])
        data['lower'] = data['EMA'] - (1.5 * data['STD'])

        return data

    def plot_band(self, data):
        switch_points = self.plot_switches(data, data.columns[0])
        data['buy_signal'] = switch_points[0]
        data['sell_signal'] = switch_points[1]

        plt.figure(figsize=(16, 9))
        plt.plot(data[f'{data.columns[0]}'], label=f"{data.columns[0]} close", alpha=1)
        # plt.plot(data['SMA'], label="SMA", alpha=1)
        plt.plot(data['EMA'], label="EMA", alpha=0.35, color='red')
        # plt.plot(data['STD'], label="STD", alpha=1)
        plt.plot(data['lower'], alpha=0.5, color='red')
        plt.plot(data['upper'], alpha=0.5, color='red')
        plt.scatter(data.index, data['buy_signal'], label='Buy', marker='^', color='green')
        plt.scatter(data.index, data['sell_signal'], label='Sell', marker='v', color='red')
        plt.title(f"{data.columns[0]} close price history and signals")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend(loc='upper left')
        plt.show()

    def system_test(self):
        self.plot_band(self.calculate_band('ABNB'))

    def system_loop(self):
        self.sleep_until_market_open()
        print("* Market open, initializing system")

        for position in self.interface.list_all_positions():
            self.interface.market_order(symbol=position.symbol, quantity=position.qty, direction='sell')

        gainers = self.get_buyable_stocks(200) # TODO: Refactor the scraper in data_filters
        todays_focus = self.run_ratings_filter(gainers, 5) # /                    /
        print(todays_focus) #                         <--/ This should have 5 /

        buying_power = self.interface.get_account().buying_power
        print(buying_power)
        individual_power = math.floor((float(buying_power) / float(len(todays_focus))))

        print("* Beginning loop...")

        while self.interface.get_clock().is_open and self.time_until_market_close() > 120:
            time.sleep(1)

            positions = self.interface.list_all_positions()
            owned = []
            for i in range(0, len(positions)):
                owned.append(positions[i].symbol)
            # print(f'Positions: {owned}')

            if positions:
                for position in positions:

                    data = self.calculate_band(position.symbol).tail(1)
                    if data[f'{position.symbol}'].item() >= data['upper'].item():
                        order = self.interface.market_order(symbol=position.symbol, quantity=position.qty, direction='sell')
                        print(f'--> Placed sell order for {order.qty} of {order.symbol}')
                    else:
                        # print(position.symbol + " not sold")
                        pass

            buying_power = self.interface.get_account().buying_power
            for symbol in todays_focus:
                data = self.calculate_band(symbol).tail(1)
                if data[f'{symbol}'].item() <= data['lower'].item() and symbol not in owned:
                    qty = individual_power / self.interface.get_last_quote(symbol)
                    if qty > 0:
                        try:
                            order = self.interface.market_order(symbol=symbol, quantity=qty, direction='buy')
                            print(f'--> Placed buy order for {order.qty} of {order.symbol}')
                        except Exception as e:
                            print(f"Error: {symbol} is not tradable")
                else:
                    # print(symbol + " not bought")
                    pass

        for position in self.interface.list_all_positions():
            self.interface.market_order(symbol=position.symbol, quantity=position.qty, direction='sell')
        time.sleep(self.time_until_market_close() + 1)
