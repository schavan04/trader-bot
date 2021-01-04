import os
import json

from alpha_vantage.timeseries import TimeSeries
import alpaca_trade_api as tradeapi

# Storing the api keys in a json file and setting them as environment variables
# because I am lazy and don't want to set them every time through command line
with open('keys/keys.json', 'r') as f:
    data = json.load(f)

for key, value in data.items():
    os.environ[key] = value

# Alpaca setup
key = os.environ.get('APCA_API_KEY_ID')
sec = os.environ.get('APCA_API_SECRET_KEY')
url = "https://paper-api.alpaca.markets"
paper = tradeapi.REST(key, sec, url, api_version='v2')

# Alpha Vantage setup
av_key = os.environ.get('ALPHAVANTAGE_API_KEY')
av = TimeSeries(output_format='pandas')
