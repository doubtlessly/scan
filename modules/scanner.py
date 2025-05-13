# modules/scanner.py
import os
import asyncio
import time
import ccxt.async_support as ccxt
from ccxt.base.errors import ExchangeError, RequestTimeout

from .config          import (
    ENABLE_MULTI_PHASE,
    PHASE1_TIMEFRAMES,
    PHASE1_THRESHOLD,
    ENABLE_LIVE_TRADING,
    LIVE_EXCHANGE_ID,
    LIVE_EXCHANGE_PARAMS,
    SIM_ACCOUNT_BALANCE,
    CIRCUIT_BREAKER_LOOKBACK_TRADES,
    ENABLE_PRE_SCAN_FEEDBACK,
    FEEDBACK_LOOKBACK_TRADES,
)
from .universe        import get_universe
from .features        import FeatureEngineer
from .regime          import RegimeFilter
from .signals         import SignalGenerator
from .portfolio       import PortfolioOptimizer
from .simulator       import Simulator
from .notifier        import Notifier
from .online_model    import OnlineModelHandler
from .feedback        import FeedbackEngine
from .trader          import LiveTrader
from .risk_controller import RiskController
from .exit_manager    import ExitManager

import json

class Scanner:
    def __init__(self):
        # Core services
        self.regime_filt = RegimeFilter()
        self.sim         = Simulator()
        self.notifier    = Notifier()
        self.online      = OnlineModelHandler()
        self.feedback    = FeedbackEngine()
        self.trader      = LiveTrader()
        # Risk & exits
        self.risk_ctrl   = RiskController()
        self.exit_mgr    = ExitManager()

    async def phase1_filter(self, symbols):
        if not ENABLE_MULTI_PHASE:
            return symbols
        print(f"▶ Phase 1 filter on {PHASE1_TIMEFRAMES}, require momentum > {PHASE1_THRESHOLD}")
        sem = asyncio.Semaphore(10)
        async def check(sym):
            async with sem:
                for attempt in (1,2):
                    try:
                        ok = await FeatureEngineer.phase1_momentum_filter(sym)
                        break
                    except ExchangeError as e:
                        if 'too fast' in str(e).lower() and attempt==1:
                            await asyncio.sleep(1)
                            continue
                        return False
                print(f"   {sym:12s} → {'KEEP' if ok else 'DROP'}", end='\r')
                return sym if ok else None
        results  = await asyncio.gather(*[check(s) for s in symbols])
        filtered = [s for s in results if s]
        print(f"\n✔ Phase 1 down to {len(filtered)} symbols\n")
        return filtered

    async def generate_signals(self, symbols, thresholds):
        print("▶ Generating signals:")
        gen   = SignalGenerator(thresholds=thresholds)
        sem   = asyncio.Semaphore(20)
        start = time.perf_counter()

        async def work(sym, idx, total):
            async with sem:
                try:
                    sig = await gen.generate_signal(sym)
                except (ExchangeError, RequestTimeout):
                    print(f"\n⚠️  {sym} failed, skipping")
                    return None
                elapsed = time.perf_counter() - start
                fv      = sig['feature_vector']
                print(
                    f"[{idx}/{total}] {sym:15s} "
                    f"prob={sig['probability']:.4f} "
                    f"imb={fv['imbalance']:+.3f} "
                    f"spread={fv['spread']:.3f} "
                    f"({elapsed:.1f}s)",
                    end='\r'
                )
                return sig

        total   = len(symbols)
        tasks   = [work(s, i+1, total) for i,s in enumerate(symbols)]
        results = await asyncio.gather(*tasks)
        signals = [s for s in results if s]
        print(f"\n▶ Signal generation done in {time.perf_counter() - start:.1f}s\n")
        return signals

    async def run(self):
        # ── 0) Pre-scan ChatGPT feedback ─────────────────────────────────────────
        if ENABLE_PRE_SCAN_FEEDBACK:
            summary = self.sim.summary_since(FEEDBACK_LOOKBACK_TRADES)
            cur = self.sim.conn.execute("""
              SELECT params,pnl FROM simulated_trades
              WHERE exit_ts IS NOT NULL
              ORDER BY id DESC LIMIT ?
            """, (FEEDBACK_LOOKBACK_TRADES,))
            recent = [
                {"fv": json.loads(r[0]), "pnl": r[1]} for r in cur.fetchall()
            ]
            advice = await self.feedback.pre_scan_advice(summary, recent)
            print(f"🧠 Pre-scan advice: {advice.get('summary','—')}")
            await self.notifier.send_feedback(advice)
            # Apply tweaks if provided
            tweaks = advice.get("tweaks", {})
            if tweaks.get("entry"):
                SignalGenerator.thresholds['entry'] = tweaks['entry']
            if tweaks.get("exit"):
                SignalGenerator.thresholds['exit'] = tweaks['exit']

        # ── 1) Health & account checks ────────────────────────────────────────────
        print("▶ Health & account checks…")
        key = os.getenv("OPENAI_API_KEY")
        print(f"  🔑 OpenAI key loaded: {bool(key)}")
        if not key:
            print("  ❌ ChatGPT disabled")
        print(f"  ℹ️  Sim balance: {SIM_ACCOUNT_BALANCE:.2f} USDT")

        if ENABLE_LIVE_TRADING:
            print("  ℹ️  Fetching KuCoin Futures balance…", end=' ')
            try:
                Ex = getattr(ccxt, LIVE_EXCHANGE_ID)
                ex = Ex(LIVE_EXCHANGE_PARAMS)
                acct = await ex.fetch_balance()
                bal = acct['total'].get('USDT',0.0)
                positions = acct.get('positions',[])
                await ex.close()
                print(f"{bal:.2f} USDT, {len(positions)} positions")
            except Exception as e:
                print("FAILED:", e)
        else:
            print("  ℹ️  Live trading disabled")
        print("")

        # ── 2) Regime filter ───────────────────────────────────────────────────────
        regime, thresholds = await self.regime_filt.evaluate()
        print(f"📊 Regime: {regime.upper()} — thresholds: {thresholds}\n")

        # ── 3) Universe fetch ─────────────────────────────────────────────────────
        symbols = await get_universe()
        print(f"✔ Universe size: {len(symbols)} symbols\n")

        # ── 4) Phase-1 screen ────────────────────────────────────────────────────
        symbols = await self.phase1_filter(symbols)

        # ── 5) Circuit-breaker check ─────────────────────────────────────────────
        recent = self.sim.summary_since(CIRCUIT_BREAKER_LOOKBACK_TRADES)
        allowed = self.risk_ctrl.update_and_check(
            sim_pnl=recent['sum_pnl'], live_pnl=0.0
        )
        if not allowed:
            print("⛔ Entry paused due to drawdown. Exiting scan.")
            return

        # ── 6) Signal generation ────────────────────────────────────────────────
        signals = await self.generate_signals(symbols, thresholds)
        for s in signals:
            s['score'] = int(s['probability'] * 10)
        winners = [s for s in signals if s['score'] >= int(thresholds['entry'] * 10)]
        print(f"✔ {len(winners)}/{len(signals)} passed score filter\n")

        # ── 7) Portfolio & sizing ────────────────────────────────────────────────
        po        = PortfolioOptimizer()
        allocated = await po.allocate(winners)
        for s in allocated:
            price = s['feature_vector']['mid_price']
            s['quantity'] = (s['weight'] * SIM_ACCOUNT_BALANCE) / price if price else 0
        print("✔ Allocated & sized positions\n")

        # ── 8) Simulated trades ─────────────────────────────────────────────────
        self.sim.log_signals(allocated)
        await self.sim.update_outcomes(horizon_s=300)
        sim_summary = self.sim.summary()
        print(f"✔ Simulation summary: {sim_summary}\n")

        # ── 9) Live trading & exits ──────────────────────────────────────────────
        live_results = []
        if ENABLE_LIVE_TRADING:
            print("▶ Placing live orders…")
            for s in allocated:
                price = await self.trader.place_order(s['symbol'], s['side'], s['quantity'])
                s['live_price'] = price
                live_results.append(s)
            await self.trader.close()
            self.trader.update_live_pnls()
            print(f"✔ Placed {len(live_results)} live orders\n")
            print("▶ Monitoring exits (ATR stops & targets)…")
            await self.exit_mgr.monitor_and_exit(live_results)
            print("✔ All live trades exited\n")

        # ── 10) Online learning & feedback ────────────────────────────────────────
        self.online.update_from_simulator(self.sim)
        new_t = await self.feedback.suggest_thresholds(sim_summary)
        print(f"✔ ChatGPT thresholds: entry={new_t['entry']:.3f}, exit={new_t['exit']:.3f}\n")

        # ── 11) Post-trade circuit-breaker ───────────────────────────────────────
        live_pnl = sum(t.get('pnl',0.0) for t in live_results)
        self.risk_ctrl.update_and_check(
            sim_pnl=sim_summary['avg_pnl']*sim_summary['total'],
            live_pnl=live_pnl
        )

        # ── 12) Notification ─────────────────────────────────────────────────────
        await self.notifier.send(simulated=allocated, live=live_results)
        print("✅ Scan complete")
