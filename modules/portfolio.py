# modules/portfolio.py
import numpy as np
import pandas as pd
from .config import (
    ENABLE_PORTFOLIO_OPT,
    PORTFOLIO_METHOD,
    CORRELATION_LOOKBACK,
    EXCHANGE_ID,
    EXCHANGE_PARAMS
)
import ccxt.async_support as ccxt

class PortfolioOptimizer:
    def __init__(self):
        self.enabled = ENABLE_PORTFOLIO_OPT

    async def _fetch_returns(self, symbols):
        """
        Fetch last CORRELATION_LOOKBACK+1 minutes of close
        for each symbol, compute log returns series.
        """
        ex_cls = getattr(ccxt, EXCHANGE_ID)
        ex = ex_cls(EXCHANGE_PARAMS)
        data = {}
        for sym in symbols:
            ohlcv = await ex.fetch_ohlcv(sym, '1m', limit=CORRELATION_LOOKBACK+1)
            df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
            data[sym] = np.log(df['c']).diff().dropna().values
        await ex.close()
        return data  # dict[sym] = array of returns

    def _risk_parity_weights(self, cov):
        # approximate via inverse volatility
        iv = 1 / np.sqrt(np.diag(cov))
        w = iv / iv.sum()
        return w

    def _mean_variance_weights(self, cov, mu):
        # w ∝ inv(cov) * mu ; then normalize to sum=1
        inv = np.linalg.pinv(cov)
        w = inv.dot(mu)
        if (w < 0).any():
            # no shorting: clip & renormalize
            w = np.clip(w, 0, None)
        return w / w.sum()

    async def allocate(self, signals):
        """
        signals: list of dicts each with 'symbol' and 'probability'
        Returns signals with an added 'weight' field summing to 1.
        """
        if not self.enabled or not signals:
            # equal‐weight fallback
            ew = 1/len(signals) if signals else {}
            return [{**s, 'weight': ew} for s in signals]

        syms = [s['symbol'] for s in signals]
        # 1) fetch returns
        rets = await self._fetch_returns(syms)
        # 2) build return matrix
        R = np.vstack([rets[s] for s in syms]).T       # shape (T, N)
        cov = np.cov(R, rowvar=False)                  # (N,N)
        # 3) expected returns = normalized probabilities
        mu = np.array([s['probability'] for s in signals])
        mu = mu / mu.sum()

        # 4) choose method
        if PORTFOLIO_METHOD == 'mean_variance':
            w = self._mean_variance_weights(cov, mu)
        else:
            w = self._risk_parity_weights(cov)

        # 5) attach weights
        out = []
        for sig, weight in zip(signals, w):
            out.append({**sig, 'weight': float(weight)})
        return out
