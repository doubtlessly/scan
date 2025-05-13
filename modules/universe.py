# modules/universe.py
import ccxt.async_support as ccxt
from .config import EXCHANGE_ID, EXCHANGE_PARAMS, UNIVERSE_SIZE, MIN_VOLUME

async def get_universe():
    exchange_cls = getattr(ccxt, EXCHANGE_ID)
    ex = exchange_cls(EXCHANGE_PARAMS)
    try:
        tickers = await ex.fetch_tickers()
    finally:
        await ex.close()

    vols = []
    for sym, data in tickers.items():
        # only USDT-margined swaps look like SYMBOL/BASE:USDT
        if not sym.endswith(':USDT'):
            continue
        # get 24h quote volume
        qv = data.get('quoteVolume') or data.get('quoteVolume24h') or 0
        if qv >= MIN_VOLUME:
            vols.append((sym, qv))

    # sort by volume descending, take top N
    top = sorted(vols, key=lambda x: x[1], reverse=True)[:UNIVERSE_SIZE]
    return [sym for sym, _ in top]
