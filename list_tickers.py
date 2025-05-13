# list_tickers.py
import asyncio
import ccxt.async_support as ccxt
from modules.config import EXCHANGE_ID, EXCHANGE_PARAMS

async def main():
    ex = getattr(ccxt, EXCHANGE_ID)(EXCHANGE_PARAMS)
    tickers = await ex.fetch_tickers()
    await ex.close()

    # How many tickers total?
    print("Total tickers:", len(tickers))

    # Show first 20 symbols and their quoteVolume
    for i, (sym, data) in enumerate(tickers.items()):
        if i == 20: break
        qv = data.get('quoteVolume') or data.get('quoteVolume24h') or 0
        print(f"{sym:12s} quoteVolume={qv}")
if __name__ == "__main__":
    asyncio.run(main())
