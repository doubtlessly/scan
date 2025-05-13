# Altcoin Scanner — Current Progress

## Project Layout
altcoin_scanner/
├── config.py # Exchange + symbol settings
├── main.py # Test harness for features
├── modules/
│ ├── exchange.py # CCXT wrapper (fetch order-book, trades, OHLCV)
│ ├── features.py # Order-flow + VWAP + TA (EMA, ATR) + feature-vector builder
│ └── config.py # (duplicate of root config if you prefer)
└── requirements.txt # Dependencies


## What’s Done
1. **Exchange Module**  
   – Configurable CCXT client (MEXC swap markets by default)  
   – `get_order_book(symbol, limit)` returns bids/asks + latency  

2. **Feature Engineering Module**  
   – `order_book_imbalance` (bid vs ask volume)  
   – `compute_vwap` over last N trades  
   – `fetch_ohlcv` + `ema` + `atr` from 1m candles  
   – `build_feature_vector` → `{imbalance, vwap, ema, atr, mid_price, spread}`  

3. **Test Harnesses in `main.py`**  
   – Stepwise tests for order-book, then features, then full feature vector  

## Next Steps
1. **Signal Generation**  
   – Take this feature vector, feed into a LightGBM/XGBoost model  
   – Output a “breakout probability” + risk filter rules  
2. **Backtesting Module**  
3. **Execution & Risk Manager**  
4. **Telegram Alerts & ChatGPT Feedback Loop**

---

### 2. (Optional) Push to GitHub

1. In the Replit console:
   ```bash
   git add .
   git commit -m "Add README and current modules"
   git remote add origin https://github.com/<your-username>/altcoin_scanner.git
   git push -u origin master
