import os

# ======================== IBKR CONFIG ===========================
# IBKR doesn't use API keys. It connects via host, port, and client ID.
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")           # Localhost (TWS or IB Gateway)
IB_PORT = int(os.getenv("IB_PORT", "7497"))           # 7497 = TWS paper/live, 4002 = IB Gateway paper
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))    # Arbitrary unique client ID per app instance
IB_CLIENT_ID_TRADE = int(os.getenv("IB_CLIENT_ID_TRADE", "2"))
IB_CLIENT_ID_DATA = int(os.getenv("IB_CLIENT_ID_DATA", "3"))

# ======================== DATABASE CONFIG ========================
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Stocks123")
DB_HOST = os.getenv("DB_HOST", "database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com")
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
ALLOW_AFTER_HOURS_TRADING = True  # Allow trading when market is closed
