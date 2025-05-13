# modules/param_tuner.py
import optuna
import datetime
import sqlite3
from .simulator import Simulator
from .signals   import SignalGenerator
from .portfolio import PortfolioOptimizer
from .config    import (
    DB_PATH,
    PARAM_TUNER_LOOKBACK_DAYS,
    PARAM_TUNER_MARGIN
)

class ParamTuner:
    def __init__(self):
        self.sim = Simulator()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._ensure_table()

    def _ensure_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS param_trials (
          ts INTEGER,
          params TEXT,
          win_rate REAL,
          avg_pnl REAL
        )""")
        self.conn.commit()

    def _objective(self, trial):
        # sample parameters
        entry = trial.suggest_float("entry", 0.4, 0.8)
        exit  = trial.suggest_float("exit", 0.1, 0.5)
        atr_m = trial.suggest_float("atr_mult", 0.5, 2.0)
        # update config in-memory
        from .config import SIGNAL_THRESHOLDS, ATR_MULTIPLIER
        SIGNAL_THRESHOLDS['entry'], SIGNAL_THRESHOLDS['exit'] = entry, exit
        ATR_MULTIPLIER = atr_m
        # run a quick backtest (last N days)
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=PARAM_TUNER_LOOKBACK_DAYS)
        # you’ll need a backtest API that accepts start/end; assume sim.backtest()
        summary = self.sim.backtest(start, end)
        # objective: maximize win_rate * avg_pnl
        score = summary['win_rate'] * summary['avg_pnl']
        # log trial
        self.conn.execute(
            "INSERT INTO param_trials(ts,params,win_rate,avg_pnl) VALUES (?,?,?,?)",
            (int(time.time()), str(trial.params), summary['win_rate'], summary['avg_pnl'])
        )
        self.conn.commit()
        return score

    def tune(self, n_trials: int = 50):
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(self._objective, n_trials=n_trials)
        best = study.best_params
        # apply only if it exceeds margin over current
        current_summary = self.sim.summary_since(100)
        best_summary = self.sim.backtest(
            datetime.datetime.utcnow() - datetime.timedelta(days=PARAM_TUNER_LOOKBACK_DAYS),
            datetime.datetime.utcnow()
        )
        if (best_summary['win_rate'] - current_summary['win_rate'] > PARAM_TUNER_MARGIN):
            from .config import SIGNAL_THRESHOLDS, ATR_MULTIPLIER
            SIGNAL_THRESHOLDS['entry'] = best['entry']
            SIGNAL_THRESHOLDS['exit']  = best['exit']
            ATR_MULTIPLIER = best['atr_mult']
        return best
