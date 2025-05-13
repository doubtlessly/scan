# modules/risk.py
from .config import (
    ENABLE_POSITION_SIZING,
    RISK_PER_TRADE_PERCENT,
    MAX_LEVERAGE,
    ATR_MULTIPLIER,
    SIM_ACCOUNT_BALANCE,
)

def compute_position_size(fv: dict, balance: float = None) -> dict:
    """
    Given a feature_vector with 'atr' and 'mid_price',
    returns {'quantity': float, 'leverage': float}.
    """
    balance = balance if balance is not None else SIM_ACCOUNT_BALANCE

    if not ENABLE_POSITION_SIZING:
        return {'quantity': None, 'leverage': None}

    atr   = fv.get('atr', 0.0)
    price = fv.get('mid_price', 0.0)
    if atr <= 0 or price <= 0 or balance <= 0:
        return {'quantity': 0.0, 'leverage': 0.0}

    # 1) compute risk amount in USDT
    risk_amount = balance * RISK_PER_TRADE_PERCENT

    # 2) contracts = risk_amount / (ATR * ATR_MULTIPLIER)
    quantity = risk_amount / (atr * ATR_MULTIPLIER)

    # 3) compute notional and cap by max leverage
    notional = quantity * price
    max_notional = balance * MAX_LEVERAGE
    if notional > max_notional:
        notional = max_notional
        quantity = notional / price

    # 4) implied leverage
    leverage = notional / balance

    return {
        'quantity': round(quantity, 6),  # round for readability
        'leverage': round(leverage, 2),
    }
