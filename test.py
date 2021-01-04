import auth
import pandas as pd

api = auth.paper

candlesticks = api.get_barset('WDC', 'minute', limit=10)
df = candlesticks['WDC'].df

data = df['low'].min()
print(candlesticks['WDC'].df)
