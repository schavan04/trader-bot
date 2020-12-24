from alpha_vantage.timeseries import TimeSeries
app = TimeSeries(output_format='pandas')

aapl = app.get_daily_adjusted('AAPL')
print(aapl)
