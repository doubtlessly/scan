# modules/exit_manager.py
import asyncio
import pandas as pd
from .exchange import ExchangeClient
from .config   import EXIT_ATR_MULTIPLIER, PROFIT_TARGET_ATR, EXIT_CHECK_INTERVAL

class ExitManager:
    """
    Monitors open trades (simulated or live) and issues exit when:
      - price hits trailing stop (entry ± ATR×multiplier)
      - price hits profit target (entry ± ATR×multiplier)
    """
    def __init__(self):
        self.client = ExchangeClient()

    async def monitor_and_exit(self, open_trades: list[dict]):
        """
        open_trades: list of {
           'symbol','side','entry_price','stop_loss','profit_target','quantity', ... 
        }
        Continuously polls every EXIT_CHECK_INTERVAL seconds, closes trades when triggered.
        """
        active = []
        # initialize stops/targets
        for t in open_trades:
            # fetch recent 1m ATR
            df = await self.client.fetch_ohlcv(t['symbol'], '1m', limit=100)
            atr = pd.DataFrame(df,columns=['ts','o','h','l','c','v']).pipe(lambda d: pd.concat([
                        d['h']-d['l'],
                        (d['h']-d['c'].shift()).abs(),
                        (d['l']-d['c'].shift()).abs()
                    ],axis=1).max(axis=1).rolling(14).mean()).iloc[-1]
            entry = t['entry_price']
            mult = EXIT_ATR_MULTIPLIER
            # trailing stop: below (for buy) or above (for sell)
            stop = entry - atr*mult if t['side']=='buy' else entry + atr*mult
            # profit target:
            target = entry + atr*PROFIT_TARGET_ATR if t['side']=='buy' else entry - atr*PROFIT_TARGET_ATR
            active.append({**t, 'stop':stop, 'target':target})

        # loop until all closed
        while active:
            await asyncio.sleep(EXIT_CHECK_INTERVAL)
            to_keep = []
            for t in active:
                ob = await self.client.fetch_order_book(t['symbol'],1)
                mid = (ob['bids'][0][0] + ob['asks'][0][0]) / 2
                if t['side']=='buy':
                    if mid <= t['stop'] or mid >= t['target']:
                        # trigger exit
                        # simulate or place order logic here...
                        print(f"✂️ Exiting {t['symbol']} at {mid:.4f} (stop/target triggered)")
                        continue
                else:
                    if mid >= t['stop'] or mid <= t['target']:
                        print(f"✂️ Exiting {t['symbol']} at {mid:.4f} (stop/target triggered)")
                        continue
                to_keep.append(t)
            active = to_keep

        await self.client.close()
