import os
import time
from datetime import datetime
import pandas as pd

from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi

import scraper

# Clear the console
os.system('clear')

# Alpha Vantage setup
av_key = os.getenv('ALPHAVANTAGE_API_KEY')
av = TimeSeries(output_format='pandas')

# Alpaca setup
key = os.getenv('ALPACA_KEY')
sec = os.getenv('SECRET_KEY')
url = "https://paper-api.alpaca.markets"
alpaca = tradeapi.REST(key, sec, url, api_version='v2')
account = alpaca.get_account()
print(account.status, "")

# print(av.get_daily_adjusted('AAPL'))
# print(alpaca.get_barset('AAPL', 'day').df)

# Use Selenium to scrape Yahoo Finance for symbols of highest gainers by amount
symbols = scraper.yfgainers(5, 200)

os.system('clear')
data = av.get_daily_adjusted(symbols[0])
print(data)
