# modules/config.py

import os

# ── Default symbol & order‐book settings ──────────────────────────────────────
SYMBOL                    = 'BTC/USDT:USDT'
ORDER_BOOK_LIMIT          = 50

# ── Exchange & universe ──────────────────────────────────────────────────────
EXCHANGE_ID               = 'mexc'
EXCHANGE_PARAMS           = {
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
}
UNIVERSE_SIZE             = 50
MIN_VOLUME                = 2_000_000

# ── Default signal thresholds (neutral mode) ────────────────────────────────
SIGNAL_THRESHOLDS         = {'entry': 0.5, 'exit': 0.3}

# ── Regime filter ─────────────────────────────────────────────────────────────
ENABLE_REGIME_FILTER      = True
REGIME_PARAMS             = {
    'symbol': SYMBOL,
    'momentum_period': 60,
    'momentum_threshold': 0.0,
}
REGIME_THRESHOLDS         = {
    'bull':    {'entry': 0.6, 'exit': 0.3},
    'neutral': SIGNAL_THRESHOLDS,
    'bear':    {'entry': 0.8, 'exit': 0.4},
}

# ── Funding‐flow feature ──────────────────────────────────────────────────────
ENABLE_FUNDING_FEATURE    = True
FUNDING_WEIGHT            = 0.3

# ── Sentiment feature ─────────────────────────────────────────────────────────
ENABLE_SENTIMENT          = True
SENTIMENT_SOURCE_URL      = 'https://api.alternative.me/fng/'
SENTIMENT_GREED_THRESH    = 0.8
SENTIMENT_WEIGHT          = -0.3

# ── Multi-Timeframe Features ──────────────────────────────────────────────────
ENABLE_MULTITF            = True
MTF_TIMEFRAMES            = ['5m', '15m', '1h']
MTF_LIMIT                 = 100
MTF_WEIGHTS               = {'5m':0.10, '15m':0.05, '1h':0.02}

# ── RSI & MACD toggles ────────────────────────────────────────────────────────
ENABLE_RSI                = True
RSI_PERIOD                = 14
ENABLE_MACD               = True
MACD_FAST                 = 12
MACD_SLOW                 = 26
MACD_SIGNAL               = 9

# ── Multi‐Phase Scanning ─────────────────────────────────────────────────────
ENABLE_MULTI_PHASE        = True
PHASE1_TIMEFRAMES         = ['5m', '15m']
PHASE1_LIMIT              = 50
PHASE1_THRESHOLD          = 0.0

# ── Position sizing & leverage ────────────────────────────────────────────────
SIM_ACCOUNT_BALANCE       = 100.0

# ── ML ensemble ───────────────────────────────────────────────────────────────
ENABLE_ML_MODEL           = True
MODEL_PATH                = 'models/lgbm_model.pkl'
ENSEMBLE_WEIGHT           = 0.5

# ── Online incremental learning ───────────────────────────────────────────────
ENABLE_ONLINE_LEARNING    = True
MODEL_ONLINE_PATH         = 'models/online_model.pkl'

# ── Portfolio optimization ────────────────────────────────────────────────────
ENABLE_PORTFOLIO_OPT      = True
PORTFOLIO_METHOD          = 'risk_parity'
CORRELATION_LOOKBACK      = 100

# ── Circuit‐breaker ───────────────────────────────────────────────────────────
CIRCUIT_BREAKER_LOOKBACK_TRADES = 1
CIRCUIT_BREAKER_THRESHOLD = 1  # 5% drawdown

# ── Exit manager (ATR stops & targets) ────────────────────────────────────────
EXIT_ATR_MULTIPLIER       = 1.0
PROFIT_TARGET_ATR         = 1.0
EXIT_CHECK_INTERVAL       = 5

# ── Live trading toggle & credentials ─────────────────────────────────────────
ENABLE_LIVE_TRADING       = False
LIVE_EXCHANGE_ID          = 'kucoinfutures'
LIVE_EXCHANGE_PARAMS      = {
    'apiKey':    os.getenv("KUCOIN_API_KEY"),
    'secret':    os.getenv("KUCOIN_API_SECRET"),
    'password':  os.getenv("KUCOIN_PASSWORD"),
    'enableRateLimit': True,
}

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID          = os.getenv("TELEGRAM_CHAT_ID")

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH                   = 'data/trades.db'

# ── OpenAI Integration ────────────────────────────────────────────────────────
OPENAI_API_KEY            = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_CHAT         = os.getenv("OPENAI_MODEL_CHAT", "gpt-3.5-turbo")
OPENAI_MODEL_DEEP         = os.getenv("OPENAI_MODEL_DEEP", "gpt-4")
OPENAI_TIMEOUT_SEC        = 15
OPENAI_RETRY_ATTEMPTS     = 2
OPENAI_RETRY_BACKOFF      = 1.0

# ── Pre‐scan ChatGPT feedback toggle & history size ─────────────────────────────
ENABLE_PRE_SCAN_FEEDBACK  = True
FEEDBACK_LOOKBACK_TRADES  = 200
FEEDBACK_CACHE_FILE       = 'data/last_feedback.json'

# ── Feature-Importance & Auto-Prune ───────────────────────────────────────────
FEATURE_DRIFT_THRESHOLD    = 0.2   # flag >20% drift from long‐term mean
MIN_IMPORTANCE_THRESHOLD   = 0.02  # drop features whose avg importance <2%
