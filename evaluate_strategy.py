# evaluate_strategy.py

import joblib
import pandas as pd

from backtester import run_backtest
from train_model import fetch_alternative_data, fetch_price_and_orderbook, engineer_features


def load_model(path='model.pkl'):
    """
    Load the persisted ensemble model and optimal threshold.
    """
    data = joblib.load(path)
    return data['model'], data['threshold']


def generate_signals(model, threshold, symbol='BTC/USD', timeframe='1h', limit=1000):
    """
    Fetch data, engineer features, and generate trading signals based on the model.

    Returns:
    - price_df: DataFrame of historical prices (for backtesting)
    - signals: Series of integer signals (1=long, 0=flat)
    """
    # 1. Fetch alternative and price/order data
    on_chain, sentiment = fetch_alternative_data(symbol, timeframe, limit)
    price_df, order_book = fetch_price_and_orderbook(symbol, timeframe, limit)

    # 2. Engineer features
    features = engineer_features(price_df, order_book, on_chain, sentiment)

    # 3. Align feature set to training schema
    #    Use HistGB inside the voting ensemble to get original feature names
    hgb = model.named_estimators_['hgb']
    feature_cols = list(hgb.feature_names_in_)
    X = features[feature_cols]

    # 4. Predict probabilities and form signals
    probs = model.predict_proba(X)[:, 1]
    signals = pd.Series((probs > threshold).astype(int), index=price_df.index[:len(probs)])
    return price_df, signals


def main():
    # Load model and threshold
    model, threshold = load_model('model.pkl')

    # Generate signals
    price_df, signals = generate_signals(model, threshold)
    # Map to positions: 1=long, 0=flat
    position_signals = signals

    # Align price data to signals
    common_idx = price_df.index.intersection(position_signals.index)
    price_aligned = price_df.loc[common_idx]
    signals_aligned = position_signals.loc[common_idx]

    # Run backtest on aligned data
    results, metrics = run_backtest(
        price_aligned,
        signals_aligned,
        fee=0.0005,
        slippage=0.0002,
        initial_capital=10000.0
    )

    # Display performance metrics
    print("Backtest Performance Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    # Save the equity curve
    results.to_csv('backtest_results.csv')
    print("Equity curve saved to backtest_results.csv")

if __name__ == '__main__':
    main()
