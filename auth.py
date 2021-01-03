import os

import pandas as pd

from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi

# Alpaca setup
key = os.getenv('ALPACA_KEY')
sec = os.getenv('SECRET_KEY')
url = "https://paper-api.alpaca.markets"
paper = tradeapi.REST(key, sec, url, api_version='v2')
# account = paper.get_account()
# print(account.status, "")

av_key = os.getenv('ALPHAVANTAGE_API_KEY')
av = TimeSeries(output_format='pandas')
