import auth

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

    def get_apca_data(self, symbol, timeframe=None, limit=None, start=None, end=None, after=None, until=None):
        if timeframe is None:
            timeframe = 'minute'
        if limit is None:
            limit = 500
        symbol_data = self.api.get_barset(symbol, timeframe=timeframe, limit=limit, start=start, end=end, after=after, until=until)
        return symbol_data

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
