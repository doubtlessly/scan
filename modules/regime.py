# modules/regime.py
import time
import ccxt.async_support as ccxt
import pandas as pd
from .config import (
    ENABLE_REGIME_FILTER,
    REGIME_PARAMS,
    REGIME_THRESHOLDS,
    EXCHANGE_ID,
    EXCHANGE_PARAMS,
)

class RegimeFilter:
    def __init__(self):
        p = REGIME_PARAMS
        self.sym    = p['symbol']
        self.period = p['momentum_period']
        self.thresh = p['momentum_threshold']
        ex_cls = getattr(ccxt, EXCHANGE_ID)
        self.ex = ex_cls(EXCHANGE_PARAMS)

    async def fetch_funding(self):
        fr = await self.ex.fetch_funding_rate(self.sym)
        return fr.get('fundingRate', 0.0)

    async def fetch_momentum(self):
        # get OHLCV minute bars
        raw = await self.ex.fetch_ohlcv(self.sym, '1m', limit=self.period+1)
        df = pd.DataFrame(raw, columns=['ts','o','h','l','c','v'])
        # percent change from oldest to newest
        return (df['c'].iloc[-1] - df['c'].iloc[0]) / df['c'].iloc[0]

    async def evaluate(self):
        if not ENABLE_REGIME_FILTER:
            return 'neutral', REGIME_THRESHOLDS['neutral']
        try:
            fr  = await self.fetch_funding()
            mom = await self.fetch_momentum()
        finally:
            await self.ex.close()
        # decide regime
        if fr > 0 and mom > self.thresh:
            r = 'bull'
        elif fr < 0 and mom < self.thresh:
            r = 'bear'
        else:
            r = 'neutral'
        return r, REGIME_THRESHOLDS[r]
