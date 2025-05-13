# modules/exchange.py
import ccxt.async_support as ccxt
from .config import EXCHANGE_ID, EXCHANGE_PARAMS
from .utils import retry_async

class ExchangeClient:
    """
    Persistent async CCXT client with built-in retries.
    Close only once when you’re done with all your fetches.
    """
    def __init__(self):
        cls = getattr(ccxt, EXCHANGE_ID)
        self.exchange = cls(EXCHANGE_PARAMS)

    @retry_async(attempts=3, initial_delay=0.5, max_delay=5.0)
    async def fetch_order_book(self, symbol: str, limit: int):
        return await self.exchange.fetch_order_book(symbol, limit)

    @retry_async(attempts=3, initial_delay=0.5, max_delay=5.0)
    async def fetch_trades(self, symbol: str, limit: int):
        return await self.exchange.fetch_trades(symbol, limit)

    @retry_async(attempts=3, initial_delay=0.5, max_delay=5.0)
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int):
        return await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    @retry_async(attempts=3, initial_delay=0.5, max_delay=5.0)
    async def fetch_funding_rate(self, symbol: str):
        return await self.exchange.fetch_funding_rate(symbol)

    async def close(self):
        await self.exchange.close()

# convenience one-off wrapper with retries
@retry_async(attempts=3, initial_delay=0.5, max_delay=5.0)
async def get_order_book(symbol: str, limit: int):
    client = ExchangeClient()
    try:
        return await client.fetch_order_book(symbol, limit)
    finally:
        await client.close()
