import os


def _require_env(name: str) -> str:
    """Raise ValueError at import time if a required environment variable is not set."""
    value = os.getenv(name)
    if value is None:
        raise ValueError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


# ======================== IBKR CONFIG ===========================
# IBKR doesn't use API keys. It connects via host, port, and client ID.
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")           # Localhost (TWS or IB Gateway)
IB_PORT = int(os.getenv("IB_PORT", "7497"))           # 7497 = TWS paper/live, 4002 = IB Gateway paper
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))    # Arbitrary unique client ID per app instance
IB_CLIENT_ID_TRADE = int(os.getenv("IB_CLIENT_ID_TRADE", "2"))
IB_CLIENT_ID_DATA = int(os.getenv("IB_CLIENT_ID_DATA", "3"))

# ======================== DATABASE CONFIG ========================
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = _require_env("DB_PASSWORD")           # required — no default
DB_HOST = _require_env("DB_HOST")                   # required — no default
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# =================== Trading & Modeling Parameters ===================
ATR_WINDOW = 14
NOTIONAL = 100          # Amount in dollars to spend on each trade
MODEL_LOOKBACK = 50       # Historical data window for features
FEATURE_LAGS = 2          # Lag features for model input
SHORT_MA = 9              # Short moving average
LONG_MA = 21              # Long moving average
RSI_PERIOD = 14           # RSI calculation period
RETRAIN_FREQUENCY = 20     # Retrain model every N minutes
DOJI_THRESHOLD = 0.001    # Doji pattern threshold

# =================== Historical Data Range ===================
START_DATE = "2005-01-01"
STOP_LOSS_PCT = 0.02      # 2% stop loss

TARGET_THRESHOLD = 0.003  # 0.3% upward move as target label

# =================== Market Hours & Timezone ===================
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 17
MARKET_CLOSE_MINUTE = 0
TIMEZONE_NAME = "US/Eastern"

# =================== Trading Settings ===================
ALLOW_AFTER_HOURS_TRADING = False  # Only trade during market hours (real-time prices required for valid entries)

# Exit management constants (Phase 2) — overridable via environment variables
HARD_STOP_ATR_MULT   = float(os.getenv("HARD_STOP_ATR_MULT",   "2.0"))   # EXIT-01: stop = entry ± 2*ATR
TAKE_PROFIT_ATR_MULT = float(os.getenv("TAKE_PROFIT_ATR_MULT", "4.0"))   # EXIT-02: target = entry ± 4*ATR
TRAILING_ATR_MULT    = float(os.getenv("TRAILING_ATR_MULT",    "1.5"))   # EXIT-03: trail = high - 1.5*ATR
TRAILING_TRIGGER_ATR = float(os.getenv("TRAILING_TRIGGER_ATR", "1.0"))   # EXIT-03: activate after 1x ATR profit
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.02"))  # EXIT-05: 2% NAV daily loss limit
