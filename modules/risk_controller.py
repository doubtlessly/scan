# modules/risk_controller.py
import sqlite3
import time
from .config import DB_PATH, SIM_ACCOUNT_BALANCE, CIRCUIT_BREAKER_THRESHOLD

class RiskController:
    """
    Tracks cumulative equity (sim + live), maintains peak equity,
    and vetoes new entries if drawdown exceeds threshold.
    """

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._ensure_table()
        # load peak equity
        cur = self.conn.execute("SELECT value FROM metrics WHERE key='peak_equity'")
        row = cur.fetchone()
        if row:
            self.peak_equity = row[0]
        else:
            self.peak_equity = SIM_ACCOUNT_BALANCE
            self._set_metric('peak_equity', self.peak_equity)

    def _ensure_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
          key   TEXT PRIMARY KEY,
          value REAL
        )""")
        self.conn.commit()

    def _set_metric(self, key: str, value: float):
        self.conn.execute("""
          INSERT INTO metrics(key,value) VALUES(?,?)
          ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        self.conn.commit()

    def update_and_check(self, sim_pnl: float, live_pnl: float = 0.0) -> bool:
        """
        Call at end of each run.
        - sim_pnl: cumulative sim PnL
        - live_pnl: cumulative live PnL
        Returns True if new entries are ALLOWED, False if drawdown > threshold.
        """
        equity = SIM_ACCOUNT_BALANCE + sim_pnl + live_pnl
        # update peak if we hit a new high
        if equity > self.peak_equity:
            self.peak_equity = equity
            self._set_metric('peak_equity', equity)
        # compute drawdown
        drawdown = (self.peak_equity - equity) / self.peak_equity
        print(f"🔒 Equity: {equity:.2f}, Peak: {self.peak_equity:.2f}, Drawdown: {drawdown:.2%}")
        # allow entries only if drawdown <= threshold
        if drawdown > CIRCUIT_BREAKER_THRESHOLD:
            print(f"⛔ Drawdown {drawdown:.2%} > {CIRCUIT_BREAKER_THRESHOLD:.2%}, pausing new entries")
            return False
        return True
