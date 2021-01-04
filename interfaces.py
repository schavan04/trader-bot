import abc

class TradingInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, api):
        self.api = api

    @abc.abstractmethod
    def get_clock(self):
        pass

    @abc.abstractmethod
    def get_data(self, symbol, timeframe):
        pass

    @abc.abstractmethod
    def limit_order(self, symbol, direction, quantity, stop_price, limit_price):
        pass

    @abc.abstractmethod
    def market_order(self, symbol, direction, quantity):
        pass

    @abc.abstractmethod
    def list_all_orders(self):
        pass

    @abc.abstractmethod
    def get_order_by_id(self, id):
        pass

    @abc.abstractmethod
    def list_all_positions(self):
        pass

    @abc.abstractmethod
    def get_position(self, symbol):
        pass

class Alpaca(TradingInterface):
    def __init__(self, api):
        super().__init__(api)

    def get_clock(self):
        clock = self.api.get_clock()
        return clock

    def get_data(self, symbol, timeframe=None):
        if timeframe is None:
            timeframe = 'minute'
        symbol_data = self.api.get_barset(symbol, timeframe, limit=10)
        return symbol_data

    def limit_order(self, symbol, quantity, direction, stop_price, limit_price):
        order = self.api.submit_order(symbol=symbol,
                                    qty=quantity,
                                    side=direction,
                                    time_in_force='day',
                                    type='market',
                                    order_class='bracket',
                                    stop_loss=stop_price,
                                    take_profit=limit_price)
        return order

    def market_order(self, symbol, quantity, direction):
        order = self.api.submit_order(symbol=symbol,
                                    qty=quantity,
                                    side=direction,
                                    time_in_force='day',
                                    type='market')
        return order

    def list_all_orders(self):
        orders = self.api.list_orders()
        return orders

    def get_order_by_id(self, id):
        order = self.api.get_order(id)
        return order

    def list_all_positions(self):
        positions = self.api.list_positions()
        return positions

    def get_position(self, symbol):
        position = self.api.get_position(symbol)
        return position
