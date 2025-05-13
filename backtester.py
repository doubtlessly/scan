# backtester.py

import pandas as pd
import numpy as np

def run_backtest(price_df: pd.DataFrame, signals: pd.Series, fee: float = 0.0005, slippage: float = 0.0002, initial_capital: float = 10000.0):
    """
    Simple backtester for futures signals. Assumes long-only or short-only positions per signal.

    Parameters:
    - price_df: DataFrame with 'close' prices indexed by timestamp.
    - signals: Series of trading signals (+1 for long, -1 for short, 0 for flat), aligned to price_df index.
    - fee: round-trip fee fraction (e.g. 0.0005 = 0.05%).
    - slippage: assumed slippage per trade.
    - initial_capital: starting capital.

    Returns a results DataFrame with equity curve and trade details.
    """
    prices = price_df['close']
    # Shift signals so entry happens at next open
    positions = signals.shift(1).fillna(0)

    # Calculate returns
    returns = prices.pct_change().fillna(0)
    # Strategy returns
    strat_ret = positions * returns
    # Adjust for fees and slippage where positions change
    trades = positions.diff().abs() > 0
    # Subtract fees + slippage each time a trade occurs
    cost = trades.astype(float) * (fee + slippage)
    strat_ret_adj = strat_ret - cost

    # Equity curve
    equity = (1 + strat_ret_adj).cumprod() * initial_capital

    results = pd.DataFrame({
        'price': prices,
        'position': positions,
        'return': returns,
        'strat_return': strat_ret,
        'strat_return_adj': strat_ret_adj,
        'equity': equity
    })

    # Compute performance metrics
    total_return = equity.iloc[-1] / initial_capital - 1
    annualized_return = (1 + total_return) ** (365 * 24 / len(equity)) - 1
    annualized_vol = strat_ret_adj.std() * np.sqrt(365 * 24)
    sharpe = annualized_return / (annualized_vol + 1e-9)

    metrics = {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': (equity / equity.cummax() - 1).min()
    }

    return results, metrics
