import abc
import time
import threading

import pandas as pd
import numpy as np

import data_filters

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

    def get_buyable_stocks(self, quantity, price):
        symbols = data_filters.get_yf_gainers(quantity, price)
        approved = []
        for symbol in symbols:
            rating = data_filters.check_rating(symbol)
            if rating < 2.4:
                approved.append(symbol)
        return approved

    def system_loop(self):
        symbols = self.get_buyable_stocks(3, 200)
        for x in symbols:
            print(x)

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

    def get_buyable_stocks(self, quantity, price):
        time.sleep(1)
        symbols = data_filters.get_yf_gainers(quantity, price)
        approved = data_filters.check_rating(symbols)
        return approved

    def get_barset(self, symbol):
        data = self.interface.get_data(symbol)
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

            self.interface.limit_order(symbol, quantity, direction, sl, tp)
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
