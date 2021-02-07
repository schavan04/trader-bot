import os
import json

from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi
import yfinance as yf

import keys

for key, value in keys.k.items():
    os.environ[key] = value

# Alpaca setup
key = os.environ.get('APCA_API_KEY_ID')
sec = os.environ.get('APCA_API_SECRET_KEY')
url = "https://paper-api.alpaca.markets"
paper = tradeapi.REST(key, sec, url, api_version='v2')

# Alpha Vantage setup
av_key = os.environ.get('ALPHAVANTAGE_API_KEY')
av = TimeSeries(output_format='pandas')
