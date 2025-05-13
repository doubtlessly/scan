# modules/trader.py
import sqlite3
import time
from .config import DB_PATH, ENABLE_LIVE_TRADING, LIVE_EXCHANGE_ID, LIVE_EXCHANGE_PARAMS
import ccxt.async_support as ccxt
from .utils import retry_async

class LiveTrader:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_table()
        if ENABLE_LIVE_TRADING:
            cls = getattr(ccxt, LIVE_EXCHANGE_ID)
            self.exchange = cls(LIVE_EXCHANGE_PARAMS)
        else:
            self.exchange = None

    def _create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
          id INTEGER PRIMARY KEY,
          ts INTEGER,
          symbol TEXT,
          side TEXT,
          price REAL,
          quantity REAL,
          pnl REAL
        )""")
        self.conn.commit()

    @retry_async()
    async def place_order(self, symbol: str, side: str, quantity: float) -> float:
        now = int(time.time())
        if not ENABLE_LIVE_TRADING:
            # simulate a fill at mid-price
            from .exchange import get_order_book
            ob = await get_order_book(symbol, 1)
            price = (ob['bids'][0][0] + ob['asks'][0][0]) / 2
        else:
            order = await self.exchange.create_market_order(symbol, side, quantity)
            fills = order.get('fills', [])
            price = sum(f['price'] * f['size'] for f in fills) / sum(f['size'] for f in fills)
        # log it
        self.conn.execute(
            "INSERT INTO live_trades (ts,symbol,side,price,quantity,pnl) VALUES (?,?,?,?,?,NULL)",
            (now, symbol, side, price, quantity)
        )
        self.conn.commit()
        return price

    async def close(self):
        if self.exchange:
            await self.exchange.close()

    def update_live_pnls(self):
        # mark pnl for any open live trades
        cur = self.conn.execute(
            "SELECT id,symbol,side,price,quantity FROM live_trades WHERE pnl IS NULL"
        )
        rows = cur.fetchall()
        for tid, sym, side, entry, qty in rows:
            # get mid-price synchronously via get_order_book
            import asyncio
            from .exchange import get_order_book
            ob = asyncio.get_event_loop().run_until_complete(get_order_book(sym, 1))
            mid = (ob['bids'][0][0] + ob['asks'][0][0]) / 2
            pnl = (mid - entry) * qty if side == 'buy' else (entry - mid) * qty
            self.conn.execute(
                "UPDATE live_trades SET pnl = ? WHERE id = ?",
                (pnl, tid)
            )
        self.conn.commit()
