# evaluate_strategy.py

import ccxt
import joblib
import json
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier

from backtester import run_backtest
from modules.features import build_feature_vector, compute_atr

# Configuration constants
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE_PCT = 0.01  # 1% risk per trade
NEUTRAL_ZONE = 0.1        # hysteresis zone around threshold
SYMBOLS = ['BTC/USD', 'ETH/USD', 'ADA/USD']
TIMEFRAME = '1h'
LIMIT = 2000


def train_model_for_symbol(symbol):
    """
    Train and persist a model and threshold for a given symbol.
    Returns: (model, threshold, feature_list)
    """
    ex = ccxt.kraken({'enableRateLimit': True})
    # 1. Fetch OHLCV and order book
    ohlcv = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
    ob = ex.fetch_order_book(symbol, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 2. Build features
    tech = build_feature_vector(df, ob)

    # Funding rate stub (0 if unsupported)
    try:
        fr = ex.fetch_funding_rate(symbol)
        tech['funding_rate'] = fr.get('fundingRate', 0)
    except Exception:
        tech['funding_rate'] = 0
    # Time-of-day
    tech['hour'] = tech.index.hour

    # Target: next-bar direction
    tech['next_close'] = df['close'].shift(-1)
    tech['target'] = (tech['next_close'] > df['close']).astype(int)
    tech.drop(columns=['next_close'], inplace=True)

    data = tech.dropna()
    X = data.drop(columns=['target'])
    y = data['target']

    # 3. Train/val split
    n = len(X)
    i1, i2 = int(n*0.6), int(n*0.8)
    X_train, y_train = X.iloc[:i1], y.iloc[:i1]
    X_val, y_val = X.iloc[i1:i2], y.iloc[i1:i2]

    # 4. Tune HistGB
    tscv = TimeSeriesSplit(n_splits=5)
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'l2_regularization': [0.0, 1.0, 10.0],
        'max_bins': [255]
    }
    base_hgb = HistGradientBoostingClassifier(loss='log_loss')
    search = RandomizedSearchCV(base_hgb, param_dist, n_iter=20, cv=tscv,
                                scoring='accuracy', n_jobs=-1, random_state=0)
    search.fit(X_train, y_train)
    hgb = search.best_estimator_

    # 5. Build ensemble
    rf = RandomForestClassifier(n_estimators=200, max_depth=7,
                                class_weight='balanced', random_state=0, n_jobs=-1)
    cb = CatBoostClassifier(iterations=200, learning_rate=0.05,
                             depth=6, verbose=0, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5)
    clf = VotingClassifier(
        estimators=[('hgb',hgb),('rf',rf),('cb',cb),('knn',knn)],
        voting='soft', weights=[3,1,1,1], n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 6. Optimize threshold with neutral zone
    probs = clf.predict_proba(X_val)[:,1]
    best_t, best_acc = 0.5, 0.0
    for t in np.linspace(0.3, 0.7, 41):
        pred = np.zeros_like(probs, dtype=int)
        pred[probs > t + NEUTRAL_ZONE] = 1
        acc = accuracy_score(y_val, (pred==1).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"{symbol} Validation accuracy: {best_acc:.4f} at threshold {best_t:.2f}")

    feature_list = X.columns.tolist()
    joblib.dump({'model':clf,'threshold':best_t,'feature_list':feature_list},
                f'model_{symbol.replace("/","_")}.pkl')
    return clf, best_t, feature_list


def evaluate_symbol(symbol, model, threshold, feature_list):
    """
    Generate signals and backtest for a given symbol.
    """
    print(f"--- Evaluating {symbol} ---")
    ex = ccxt.kraken({'enableRateLimit': True})
    ohlcv = ex.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT)
    ob = ex.fetch_order_book(symbol, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # construct static order book snapshot
    ob_snapshot = ob
    order_book_snapshots = {t: ob_snapshot for t in df.index}

    # funding rates stub
    funding_rates = pd.Series(0, index=df.index)

    tech = build_feature_vector(df, ob_snapshot)
    try:
        fr = ex.fetch_funding_rate(symbol)
        tech['funding_rate'] = fr.get('fundingRate',0)
    except Exception:
        tech['funding_rate'] = 0
    tech['hour'] = tech.index.hour

    features = tech[feature_list].dropna()
    probs = model.predict_proba(features)[:,1]
    raw = pd.Series(0, index=features.index)
    raw[probs > threshold + NEUTRAL_ZONE] = 1
    raw[probs < threshold - NEUTRAL_ZONE] = -1
    idx = df.index.intersection(raw.index)
    signals = raw.reindex(idx)
    price_aligned = df.loc[idx]

    results, metrics = run_backtest(
        price_aligned,
        signals,
        order_book_snapshots,
        funding_rates,
        fee=0.0005,
        initial_capital=INITIAL_CAPITAL,
        stop_loss_atr=1.0,
        take_profit_atr=2.0,
        risk_pct=RISK_PER_TRADE_PCT
    )
    print(f"Metrics for {symbol}: {metrics}")
    return metrics


def main():
    print("=== Training and evaluating multi-asset models ===")
    summary = {}
    for sym in SYMBOLS:
        clf, thresh, feats = train_model_for_symbol(sym)
        metrics = evaluate_symbol(sym, clf, thresh, feats)
        summary[sym] = metrics
    print("=== Summary ===")
    print(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
