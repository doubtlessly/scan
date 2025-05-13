# test_features.py

import pandas as pd
import ccxt
from modules.features import build_feature_vector

# Initialize Kraken (geo-friendly)
exchange = ccxt.kraken({'enableRateLimit': True})

print("Fetching last 100 1h bars and order book snapshot for BTC/USD via Kraken…")
# OHLCV
ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', since=None, limit=100)
# Order book
order_book = exchange.fetch_order_book('BTC/USD', limit=100)

# Build DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Compute features (now including order_book_imbalance)
fv = build_feature_vector(df, order_book)

# Print the last 10 rows of our new signals
print(fv[['order_book_imbalance','rsi','macd_line','signal_line','macd_histogram']].tail(10))
