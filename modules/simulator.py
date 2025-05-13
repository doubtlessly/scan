# modules/simulator.py
import sqlite3
import time
import json
from .config import DB_PATH
from .exchange import get_order_book

class Simulator:
    def __init__(self):
        # Connect to the SQLite database (creates file if necessary)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        """
        Create the `simulated_trades` table if it doesn't exist,
        and migrate by adding the `trained` column if missing.
        """
        # 1) Create base table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS simulated_trades (
            id INTEGER PRIMARY KEY,
            ts INTEGER,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            confidence REAL,
            params TEXT,
            exit_ts INTEGER,
            exit_price REAL,
            pnl REAL
        )""")
        # 2) Add 'trained' column if not present
        cur = self.conn.execute("PRAGMA table_info(simulated_trades)")
        cols = [row[1] for row in cur.fetchall()]
        if 'trained' not in cols:
            self.conn.execute("ALTER TABLE simulated_trades ADD COLUMN trained INTEGER DEFAULT 0")
        self.conn.commit()

    def log_signals(self, signals: list[dict]):
        """
        Insert new simulated trades for each signal.
        Fields: timestamp, symbol, side, entry_price, confidence, params (feature_vector JSON).
        """
        now = int(time.time())
        for s in signals:
            self.conn.execute(
                "INSERT INTO simulated_trades "
                "(ts, symbol, side, entry_price, confidence, params) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    now,
                    s['symbol'],
                    s['side'],
                    s['feature_vector']['mid_price'],
                    s['probability'],
                    json.dumps(s['feature_vector'])
                )
            )
        self.conn.commit()

    async def update_outcomes(self, horizon_s: int):
        """
        For any trades older than `horizon_s` seconds and not yet closed,
        fetch the current mid-price and compute PnL, updating exit_ts, exit_price, and pnl.
        """
        cutoff = int(time.time()) - horizon_s
        cur = self.conn.execute(
            "SELECT id, symbol, side, entry_price "
            "FROM simulated_trades "
            "WHERE exit_ts IS NULL AND ts <= ?",
            (cutoff,)
        )
        rows = cur.fetchall()
        for tid, sym, side, entry in rows:
            ob = await get_order_book(sym, 1)
            bid, ask = ob['bids'][0][0], ob['asks'][0][0]
            exit_p = (bid + ask) / 2
            pnl    = (exit_p - entry) if side == 'buy' else (entry - exit_p)
            self.conn.execute(
                "UPDATE simulated_trades "
                "SET exit_ts = ?, exit_price = ?, pnl = ? "
                "WHERE id = ?",
                (int(time.time()), exit_p, pnl, tid)
            )
        self.conn.commit()

    def get_untrained_trades(self) -> list[tuple[int, dict, float]]:
        """
        Return all closed trades (exit_ts not null) that have trained=0.
        Each tuple is (trade_id, feature_vector dict, pnl).
        """
        cur = self.conn.execute("""
            SELECT id, params, pnl
            FROM simulated_trades
            WHERE exit_ts IS NOT NULL AND trained = 0
        """)
        rows = []
        for tid, params_json, pnl in cur.fetchall():
            fv = json.loads(params_json)
            rows.append((tid, fv, pnl))
        return rows

    def mark_trained(self, trade_id: int):
        """
        Mark a closed trade as used for training (trained=1).
        """
        self.conn.execute(
            "UPDATE simulated_trades SET trained = 1 WHERE id = ?",
            (trade_id,)
        )
        self.conn.commit()

    def summary(self) -> dict:
        """
        Return summary statistics over all simulated trades:
          - total trades
          - average PnL per trade
          - win rate (fraction of profitable trades)
        """
        cur = self.conn.execute(
            "SELECT COUNT(*), AVG(pnl), SUM(pnl > 0)*1.0/COUNT(*) "
            "FROM simulated_trades"
        )
        total, avg_pnl, win_rate = cur.fetchone()
        return {'total': total, 'avg_pnl': avg_pnl, 'win_rate': win_rate}

    def summary_since(self, lookback_trades: int) -> dict:
        """
        Return summary statistics over only the last `lookback_trades` closed trades.
          - total trades
          - sum of PnL
          - average PnL
          - win rate
        """
        cur = self.conn.execute("""
          SELECT pnl FROM simulated_trades
          WHERE exit_ts IS NOT NULL
          ORDER BY id DESC
          LIMIT ?
        """, (lookback_trades,))
        pnls = [row[0] for row in cur.fetchall()]
        total   = len(pnls)
        sum_pnl = sum(pnls) if pnls else 0.0
        avg_pnl = sum_pnl / total if total else 0.0
        win_rate= sum(1 for p in pnls if p > 0) / total if total else 0.0
        return {'total': total, 'sum_pnl': sum_pnl, 'avg_pnl': avg_pnl, 'win_rate': win_rate}
