import abc

class TradingInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, api):
        self.api = api;

    @abc.abstractmethod
    def place_buy_order(self, symbol, quantity):
        pass

    @abc.abstractmethod
    def place_sell_order(self, symbol, quantity):
        pass

class Alpaca(TradingInterface):
    def __init__(self, api):
        self.api = api

    def place_buy_order(self, symbol, quantity):
        self.api.submit_order(symbol=symbol,
                            qty=quantity,
                            side='buy',
                            time_in_force='gtc',
                            type='market')

    def place_sell_order(self, symbol, quantity):
        self.api.submit_order(symbol=symbol,
                            qty=quantity,
                            side='sell',
                            time_in_force='gtc',
                            type='market')
