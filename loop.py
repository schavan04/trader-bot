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

import data_filters
import strategies
import stats

class Loop(object):
    def __init__(self, interface):
        self.NUMBER = 5
        self.interface = interface
        self.held = {}
        self.hit_top = {}
        self.hit_bottom = {}
        self.strategy = strategies.Basic(self.held, self.hit_top, self.hit_bottom)

    def place_order(self, symbol, dir):
        if not dir:
            print(" None")
        else:
            if self.positions and dir == 'sell':
                for position in self.positions:
                    if position.symbol == symbol:
                        qty = position.qty
            elif dir == 'buy':
                qty = int(self.individual_power / self.interface.get_last_trade(symbol).price)
            try:
                order = self.interface.market_order(symbol=symbol, quantity=qty, direction=dir)
                self.held[symbol] = not self.held[symbol]
                print(f" {dir.capitalize()} {order.qty} share(s)")
            except APIError as e:
                self.held[symbol] = not self.held[symbol]
                if self.held[symbol]:
                    self.hit_top[symbol] = not self.hit_top[symbol]
                print(" Error:", e)

    def get_data(self, symbol, start=None, end=None):
        if start is None or end is None:
            raw = self.interface.get_apca_data(symbol)
        else:
            raw = self.interface.get_apca_data(symbol, start=start, end=end)

        if not raw:
            return None

        barset = raw[symbol].df

        data = pd.DataFrame()
        data[f'{symbol}'] = barset['close']
        data = self.strategy.run_stats(symbol, data)

        return data

    def get_action(self, datapoint):
        symbol = datapoint.columns[0]
        last = self.interface.get_last_trade(symbol).price
        return self.strategy.get_action(datapoint, last)

    def get_signal_points(self, data):
        return self.strategy.get_signal_points(data)

    def graph(self, symbol):
        yesterday = datetime.today() - timedelta(days=1)
        today = datetime.today()
        now = datetime.now()
        market_open = pd.Timestamp(today.replace(hour=9, minute=30, second=0).strftime('%Y-%m-%d-%H:%M:%S'), tz='America/New_York').isoformat()
        market_close = pd.Timestamp(today.replace(hour=16, minute=0, second=0).strftime('%Y-%m-%d-%H:%M:%S'), tz='America/New_York').isoformat()
        start = pd.Timestamp((now - timedelta(hours=1)).strftime('%Y-%m-%d-%H:%M:%S'), tz='America/New_York').isoformat()
        end = pd.Timestamp((now).strftime('%Y-%m-%d-%H:%M:%S'), tz='America/New_York').isoformat()

        data = self.get_data(symbol, market_open, market_close)

        print(data)
        print(self.interface.get_last_trade(symbol).price)

        switch_points = self.get_signal_points(data)
        data['buy_signal'] = switch_points[0]
        data['sell_signal'] = switch_points[1]

        plt.figure(figsize=(16, 9))
        plt.plot(data[f'{data.columns[0]}'], label=f"{data.columns[0]} close", alpha=1)
        plt.plot(data['SMA10'], label="SMA10", alpha=1)
        plt.plot(data['SMA30'], label="SMA30", alpha=1)

        plt.scatter(data.index, data['buy_signal'], label='Buy', marker='^', color='green')
        plt.scatter(data.index, data['sell_signal'], label='Sell', marker='v', color='red')

        plt.title(f"{data.columns[0]} close price history and signals")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend(loc='upper left')

        plt.show()
        return None

    def system_graph(self, focus):
        for s in focus:
            self.graph(s)

    def system_test(self):
        pass

    def system_loop(self):
        print("* Waiting for market to open...")
        time_to_open = self.interface.time_until_market_open()
        print(f'    @ {time_to_open} seconds until open')
        time.sleep(time_to_open)

        print("* Market open, initializing system...")
        time.sleep(3)

        print("$ Liquidating all positions...")
        self.positions = self.interface.list_all_positions()
        if self.positions:
            for position in self.positions:
                order = self.interface.market_order(symbol=position.symbol, quantity=position.qty, direction='sell')
                print(f"    -  {order.symbol}: Liquidate {order.qty} shares")
        else:
            print("    - No positions!")
        time.sleep(3)

        self.total_buying_power = self.interface.get_account().cash
        self.individual_power = math.floor((float(self.total_buying_power) / float(self.NUMBER)) - 5.0)

        print("* Waiting for 60 mins...")
        time.sleep(3600)

        print(f"$ Getting today's top {self.NUMBER} gainers under ${self.individual_power}...")
        gainers = self.interface.get_buyable_stocks(self.individual_power)
        self.todays_focus = self.interface.run_ratings_filter(gainers, self.NUMBER)
        time.sleep(3)

        print("$ Today's parameters:")
        print(f"    - Stocks to focus on: {self.todays_focus}")
        print(f"    - Buying power per stock: ${self.individual_power}")
        time.sleep(3)

        # print("$ Buying all focused stocks...")
        # for symbol in self.todays_focus:
        #     qty = int(self.individual_power / self.interface.get_last_trade(symbol).price)
        #     try:
        #         order = self.interface.market_order(symbol=symbol, quantity=qty, direction='buy')
        #         print(f"    - {order.symbol}: Buy {order.qty} shares")
        #     except APIError as e:
        #         print(" Error:", e)
        # time.sleep(3)

        print("* Beginning algorithm loop...")

        for symbol in self.todays_focus:
            self.held[symbol] = False
            self.hit_top[symbol] = False

        while self.interface.get_clock().is_open and self.interface.time_until_market_close() > 30:
            self.last_loop = (datetime.utcnow() - timedelta(hours=8)).strftime('%b %d, %Y at %-I:%M:%S %p')
            print(f"@ {self.last_loop}:")

            owned = []
            for key, value in self.held.items():
                if value:
                    owned.append(key)
            print(f"    - Held: {owned}")

            self.positions = self.interface.list_all_positions() # HERE FOR A REASON

            for symbol in self.todays_focus:
                print(f"    - {symbol}:", end='')

                data = self.get_data(symbol)
                if data is None:
                    print(f" Data not available")
                elif not data.empty:
                    last_datapoint = data.tail(1)
                    dir = self.get_action(last_datapoint)
                    # print(current_datapoint)
                    self.place_order(symbol, dir)
                else:
                    print(f" Data empty")

            time.sleep(10)

        print("$ Market closing soon, liquidating all positions...")
        self.positions = self.interface.list_all_positions()
        if self.positions:
            for position in self.positions:
                order = self.interface.market_order(symbol=position.symbol, quantity=position.qty, direction='sell')
                print(f"    -  {order.symbol}: Liquidate {order.qty} shares")
        else:
            print("    - No positions!")

        # print("* Waiting until close...")
        time_to_close = self.interface.time_until_market_close()
        # print(f'    @ {time_to_close} seconds until close')
        time.sleep(time_to_close)
        #END

class MovingAvgDay(Loop):
    def __init__(self, interface):
        super().__init__(interface)

    def sleep_until_market_open(self):
        clock = self.interface.get_clock()
        if not clock.is_open:
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            print(time_to_open)
            time.sleep(round(time_to_open))

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
