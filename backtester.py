# backtester.py

import pandas as pd
import numpy as np

def run_backtest(
    price_df: pd.DataFrame,
    signals: pd.Series,
    order_book_snapshots: dict,
    funding_rates: pd.Series = None,
    fee: float = 0.0005,
    initial_capital: float = 10000.0,
    stop_loss_atr: float = 1.0,
    take_profit_atr: float = 2.0,
    risk_pct: float = 0.01
) -> (pd.DataFrame, dict):
    """
    Backtester with dynamic slippage based on order-book depth and optional funding rate costs.
    """
    prices = price_df['close']
    high = price_df['high']; low = price_df['low']; prev_close = price_df['close'].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()

    equity = initial_capital
    cash = initial_capital
    position = 0.0
    entry_price = np.nan
    entry_atr = np.nan

    records = []
    for t, price in prices.items():
        sig = signals.get(t, 0)
        depth = order_book_snapshots.get(t, {'bids': [], 'asks': []})
        # sum quantities from bids safely
        bid_qty = sum(level[1] for level in depth.get('bids', []) if len(level) >= 2)
                # sum quantities from asks safely
        ask_qty = sum(level[1] for level in depth.get('asks', []) if len(level) >= 2)
        avg_depth = np.mean([bid_qty, ask_qty]) if (bid_qty + ask_qty) > 0 else 0

        if position == 0 and sig != 0:
            risk_amount = cash * risk_pct
            pos_size = risk_amount / (atr.loc[t] + 1e-9)
            position = sig * pos_size
            entry_price = price
            entry_atr = atr.loc[t]
            slippage = min(0.005, abs(position) / (avg_depth + 1e-9))
            cost = fee * abs(position * price) + slippage * abs(position * price)
            cash -= cost
        elif position != 0:
            adverse = (entry_price - price) if position > 0 else (price - entry_price)
            if adverse >= stop_loss_atr * entry_atr:
                cash += position * price
                pos_size = abs(position * price)
                slippage = min(0.005, pos_size / (avg_depth + 1e-9))
                cost = fee * pos_size + slippage * pos_size
                cash -= cost
                position = 0
            else:
                profit = (price - entry_price) if position > 0 else (entry_price - price)
                if profit >= take_profit_atr * entry_atr:
                    cash += position * price
                    pos_size = abs(position * price)
                    slippage = min(0.005, pos_size / (avg_depth + 1e-9))
                    cost = fee * pos_size + slippage * pos_size
                    cash -= cost
                    position = 0
                elif sig != np.sign(position) and sig != 0:
                    cash += position * price
                    pos_size = abs(position * price)
                    slippage = min(0.005, pos_size / (avg_depth + 1e-9))
                    cost = fee * pos_size + slippage * pos_size
                    cash -= cost
                    risk_amount = cash * risk_pct
                    pos_size = risk_amount / (atr.loc[t] + 1e-9)
                    position = sig * pos_size
                    entry_price = price
                    entry_atr = atr.loc[t]

        if funding_rates is not None and position < 0:
            fr = funding_rates.get(t, 0)
            cash -= abs(position * price) * fr / 24

        equity = cash + position * price
        records.append({'timestamp': t, 'price': price, 'position': position, 'equity': equity})

    results = pd.DataFrame(records).set_index('timestamp')
    strat_ret = results['equity'].pct_change().fillna(0)

        # Total return
    total_ret = results['equity'].iloc[-1] / initial_capital - 1
    # Log-based annualization to handle negative moves gracefully
    strat_ret = results['equity'].pct_change().fillna(0)
    log_ret = np.log1p(strat_ret)
    hours_per_year = 365 * 24
    annual_log_return = log_ret.mean() * hours_per_year
    ann_ret = np.expm1(annual_log_return)
    # Annual volatility
    ann_vol = strat_ret.std() * np.sqrt(hours_per_year)
    # Sharpe using log returns
    sharpe = annual_log_return / (log_ret.std() * np.sqrt(hours_per_year) + 1e-9)
    # Max drawdown
    max_dd = (results['equity'] / results['equity'].cummax() - 1).min()

    metrics = {
        'Total Return': total_ret,
        'Annualized Return': ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }
    return results, metrics
