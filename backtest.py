import os
import sys

import auth
import interfaces
import strategies

import backtrader as bt
import alpaca_backtrader_api
from datetime import datetime

import keys

for key, value in keys.items():
    os.environ[key] = value

key = os.environ.get('APCA_API_KEY_ID')
sec = os.environ.get('APCA_API_SECRET_KEY')

ALPACA_API_KEY = key
ALPACA_SECRET_KEY = sec
ALPACA_PAPER = True

cerebro = bt.Cerebro()

print(f"Starting portfolio value: {cerebro.broker.getvalue()}")

cerebro.run()

print(f"Final portfolio value: {cerebro.broker.getvalue()}")
