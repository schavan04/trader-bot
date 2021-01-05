import abc
import time
import threading

import pandas as pd
import numpy as np

import data_scrapers

class PMStrategy(abc.ABC):
    @abc.abstractmethod
    def __init__(self, interface):
        self.interface = interface

    @abc.abstractmethod
    def system_loop(self):
        pass

class ShellSystemTest(PMStrategy):
    def __init__(self, interface):
        super().__init__(interface)

    def get_tradeable_stocks(self):
        symbols = data_scrapers.get_yf_gainers(quantity=5, price=200)
        return symbols

    def system_loop(self): # gets top 5 gainers from YF and buys the first one
        symbols = self.get_tradeable_stocks()
        self.interface.market_order(str(symbols[0]), 1, 'buy')

class BasicStrategy(PMStrategy):
    def __init__(self, interface):
        super().__init__(interface)

    def time_until_market_close():
        clock = self.interface.get_clock()
        return (clock.next_close - clock.timestamp).total_seconds()

    def sleep_until_market_open():
        clock = self.interface.get_clock()
        if not clock.is_open:
            time_to_open = (clock.next_open - clock.timestamp).total_seconds()
            time.sleep(round(time_to_open))

    def get_tradeable_stocks(self):
        symbols = data_scrapers.get_yf_gainers(quantity=5, price=200)
        return symbols

    def get_barset(self, symbol):
        data = self.interface.get_data(symbol)
        df = data[symbol].df
        bar = {
            'high': float(df['high'].max()),
            'low': float(df['low'].min())
        }
        return bar

    def place_order(self, symbol, quantity, direction):
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

            self.interface.limit_order(symbol, quantity, direction, sl, tp)
            return True
        # else:
        #     return False

    def system_loop(self):
        symbols = self.get_tradeable_stocks()
        self.place_order(symbols[1], 1, 'buy')
