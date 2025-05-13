# modules/features.py
import asyncio
import pandas as pd
from .exchange import ExchangeClient, get_order_book
from .config import (
    SYMBOL,
    ORDER_BOOK_LIMIT,
    ENABLE_FUNDING_FEATURE,
    ENABLE_MULTITF,
    MTF_TIMEFRAMES,
    MTF_LIMIT,
    ENABLE_RSI,
    RSI_PERIOD,
    ENABLE_MACD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    ENABLE_MULTI_PHASE,
    PHASE1_TIMEFRAMES,
    PHASE1_LIMIT,
    PHASE1_THRESHOLD,
)

class FeatureEngineer:
    @staticmethod
    def order_book_imbalance(ob: dict) -> float:
        bid_vol = sum(entry[1] for entry in ob['bids'])
        ask_vol = sum(entry[1] for entry in ob['asks'])
        total   = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total else 0.0

    @staticmethod
    async def compute_vwap(client: ExchangeClient, symbol: str, limit: int = 200) -> float:
        trades = await client.fetch_trades(symbol, limit)
        df = pd.DataFrame(trades)
        if df.empty or df['amount'].sum() == 0:
            return 0.0
        return (df['price'] * df['amount']).sum() / df['amount'].sum()

    @staticmethod
    async def fetch_ohlcv_df(client: ExchangeClient, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        raw = await client.fetch_ohlcv(symbol, timeframe, limit)
        return pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])

    @staticmethod
    def ema(df: pd.DataFrame, period: int = 14) -> pd.Series:
        return df['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low   = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close  = (df['low']  - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        ma_up   = up.ewm(span=period, adjust=False).mean()
        ma_down = down.ewm(span=period, adjust=False).mean()
        rs = ma_up / ma_down.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd_histogram(series: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
        ema_fast   = series.ewm(span=fast, adjust=False).mean()
        ema_slow   = series.ewm(span=slow, adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line= macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    @staticmethod
    async def funding_rate(client: ExchangeClient, symbol: str) -> float:
        if not ENABLE_FUNDING_FEATURE:
            return 0.0
        data = await client.fetch_funding_rate(symbol)
        return data.get('fundingRate', 0.0)

    @staticmethod
    async def fetch_multi_tf_features(client: ExchangeClient, symbol: str) -> dict:
        out = {}
        tasks   = [client.fetch_ohlcv(symbol, tf, MTF_LIMIT) for tf in MTF_TIMEFRAMES]
        results = await asyncio.gather(*tasks)
        for tf, raw in zip(MTF_TIMEFRAMES, results):
            df = pd.DataFrame(raw, columns=['timestamp','open','high','low','close','volume'])
            if df.empty:
                out[f'{tf}_momentum'] = 0.0
                out[f'{tf}_atr']      = 0.0
                if ENABLE_RSI:   out[f'{tf}_rsi']     = 50.0
                if ENABLE_MACD:  out[f'{tf}_macdhist'] = 0.0
            else:
                out[f'{tf}_momentum'] = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]
                out[f'{tf}_atr']      = FeatureEngineer.atr(df).iloc[-1]
                if ENABLE_RSI:
                    out[f'{tf}_rsi']     = FeatureEngineer.rsi(df['close'], period=RSI_PERIOD).iloc[-1]
                if ENABLE_MACD:
                    out[f'{tf}_macdhist']= FeatureEngineer.macd_histogram(
                        df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL
                    ).iloc[-1]
        return out

    @staticmethod
    async def phase1_momentum_filter(symbol: str) -> bool:
        if not ENABLE_MULTI_PHASE:
            return True
        client = ExchangeClient()
        try:
            tasks = [client.fetch_ohlcv(symbol, tf, PHASE1_LIMIT) for tf in PHASE1_TIMEFRAMES]
            raws  = await asyncio.gather(*tasks)
            for tf, raw in zip(PHASE1_TIMEFRAMES, raws):
                df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
                if df.empty:
                    return False
                mom = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]
                if mom <= PHASE1_THRESHOLD:
                    return False
            return True
        finally:
            await client.close()

    @staticmethod
    async def build_feature_vector(
        symbol: str = None,
        ob_limit: int = None,
        vwap_limit: int = 200,
        ohlcv_timeframe: str = '1m',
        ohlcv_limit: int = 100,
        ema_period: int = 14,
        atr_period: int = 14,
    ) -> dict:
        symbol   = symbol or SYMBOL
        ob_limit = ob_limit or ORDER_BOOK_LIMIT

        client = ExchangeClient()
        try:
            ob         = await client.fetch_order_book(symbol, ob_limit)
            imb        = FeatureEngineer.order_book_imbalance(ob)
            vwap       = await FeatureEngineer.compute_vwap(client, symbol, vwap_limit)

            df1m       = await FeatureEngineer.fetch_ohlcv_df(client, symbol, ohlcv_timeframe, ohlcv_limit)
            latest_ema = FeatureEngineer.ema(df1m, period=ema_period).iloc[-1]
            latest_atr = FeatureEngineer.atr(df1m, period=atr_period).iloc[-1]

            best_bid   = ob['bids'][0][0] if ob['bids'] else 0
            best_ask   = ob['asks'][0][0] if ob['asks'] else 0
            mid_price  = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
            spread     = best_ask - best_bid if best_bid and best_ask else 0

            fr         = await FeatureEngineer.funding_rate(client, symbol)
            mtf        = await FeatureEngineer.fetch_multi_tf_features(client, symbol) if ENABLE_MULTITF else {}

            return {
                'imbalance':    imb,
                'vwap':         vwap,
                'ema':          latest_ema,
                'atr':          latest_atr,
                'mid_price':    mid_price,
                'spread':       spread,
                'funding_rate': fr,
                **mtf
            }
        finally:
            await client.close()
