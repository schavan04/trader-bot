import math

import auth
import data_filters

paper = auth.paper
alpha = auth.av
yf = auth.yf

class MainInterface(object):
    def __init__(self, api):
        self.api = api

    def get_account(self):
        return paper.get_account()

    def get_clock(self):
        clock = self.api.get_clock()
        return clock

    def time_until_market_close(self):
        clock = self.get_clock()
        if clock.is_open:
            time_to_close = math.ceil((clock.next_close - clock.timestamp).total_seconds())
            return time_to_close
        else:
            return 0

    def time_until_market_open(self):
        clock = self.get_clock()
        if not clock.is_open:
            time_to_open = math.ceil((clock.next_open - clock.timestamp).total_seconds())
            return time_to_open
        else:
            return 0

    def get_buyable_stocks(self, price):
        symbols = data_filters.get_yf_gainers(price)
        return symbols

    def run_ratings_filter(self, symbols, quantity):
        approved = data_filters.check_rating(symbols, quantity, self.api)
        return approved

    def get_apca_data(self, symbol, timeframe=None, limit=None, start=None, end=None, after=None, until=None):
        if timeframe is None:
            timeframe = 'minute'
        if limit is None:
            limit = 500
        try:
            symbol_data = self.api.get_barset(symbol, timeframe=timeframe, limit=limit, start=start, end=end, after=after, until=until)
            return symbol_data
        except error as e:
            print(e)
            return None

    def get_asset(self, symbol):
        asset = self.api.get_asset(symbol)
        return asset

    def get_last_quote(self, symbol):
        last = self.api.get_last_quote(symbol)
        return last

    def get_last_trade(self, symbol):
        last = self.api.get_last_trade(symbol)
        return last

    # def multi_order(self, symbol, quantity, direction, stop_price, limit_price):
    #     order = self.api.submit_order(symbol=symbol,
    #                                 qty=quantity,
    #                                 side=direction,
    #                                 time_in_force='day',
    #                                 type='market',
    #                                 order_class='bracket',
    #                                 stop_loss=stop_price,
    #                                 take_profit=limit_price)
    #     return order

    def market_order(self, symbol, quantity, direction):
        order = self.api.submit_order(symbol=symbol,
                                    qty=quantity,
                                    side=direction,
                                    time_in_force='day',
                                    type='market')
        return order

    def limit_order(self, symbol, quantity, direction, price):
        order = self.api.submit_order(symbol=symbol,
                                    qty=quantity,
                                    side=direction,
                                    time_in_force='day',
                                    type='limit',
                                    limit_price=price)
        return order

    def list_all_positions(self):
        positions = self.api.list_positions()
        return positions
