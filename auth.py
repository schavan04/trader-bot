import os
import json

import pandas as pd

from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi

with open('keys.json', 'r') as f:
    data = json.load(f)

for key, value in data.items():
    os.environ[key] = value

# Alpaca setup
key = os.environ.get('APCA_API_KEY_ID')
sec = os.environ.get('SECRET_KEY')
url = "https://paper-api.alpaca.markets"
paper = tradeapi.REST(key, sec, url, api_version='v2')
# account = paper.get_account()
# print(account.status, "")

av_key = os.environ.get('ALPHAVANTAGE_API_KEY')
av = TimeSeries(output_format='pandas')
