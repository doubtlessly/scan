import pandas as pd
import numpy as np

# --- Existing Feature Functions ---

def order_book_imbalance(order_book: dict) -> float:
    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])
    bid_vol = sum([float(b[1]) for b in bids])
    ask_vol = sum([float(a[1]) for a in asks])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)


def compute_vwap(df: pd.DataFrame) -> float:
    price_volume = df['close'] * df['volume']
    return price_volume.sum() / (df['volume'].sum() + 1e-9)


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

# --- New Momentum & Trend Indicators ---

def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd_line': macd_line,
        'signal_line': signal_line,
        'macd_histogram': histogram
    })

# --- Statistical & Volatility Features ---

def compute_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change()


def compute_rolling_statistics(prices: pd.Series, window: int) -> pd.DataFrame:
    roll_mean = prices.rolling(window).mean()
    roll_std = prices.rolling(window).std()
    return pd.DataFrame({
        f'ma_{window}': roll_mean,
        f'std_{window}': roll_std
    })


def compute_bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    percent_b = (prices - lower) / (upper - lower + 1e-9)
    width = (upper - lower) / (rolling_mean + 1e-9)
    return pd.DataFrame({
        'bb_percent_b': percent_b,
        'bb_width': width
    })

# --- Build Full Feature Vector ---

def build_feature_vector(df: pd.DataFrame, order_book: dict = None) -> pd.DataFrame:
    """
    Aggregates a comprehensive set of features:
    - Price-based: returns, moving averages, volatility
    - Momentum: RSI, MACD
    - Volatility: ATR, Bollinger Bands
    - Liquidity: VWAP, order-book imbalance
    """
    features = {}

    # Price-derived features
    features['return_1'] = compute_returns(df['close'])
    # Rolling stats for mean and volatility
    stats_5 = compute_rolling_statistics(df['close'], 5)
    stats_10 = compute_rolling_statistics(df['close'], 10)
    stats_20 = compute_rolling_statistics(df['close'], 20)
    features.update(stats_5.to_dict(orient='list'))
    features.update(stats_10.to_dict(orient='list'))
    features.update(stats_20.to_dict(orient='list'))

    # Volatility indicators
    features['atr'] = compute_atr(df)
    bb = compute_bollinger_bands(df['close'])
    features['bb_percent_b'] = bb['bb_percent_b']
    features['bb_width'] = bb['bb_width']

    # Momentum indicators
    features['rsi'] = compute_rsi(df['close'])
    macd_df = compute_macd(df['close'])
    features['macd_line'] = macd_df['macd_line']
    features['signal_line'] = macd_df['signal_line']
    features['macd_histogram'] = macd_df['macd_histogram']

    # Liquidity & flow
    features['vwap'] = compute_vwap(df)
    if order_book is not None:
        features['order_book_imbalance'] = order_book_imbalance(order_book)
    else:
        features['order_book_imbalance'] = np.nan

    # Build DataFrame
    return pd.DataFrame(features)
