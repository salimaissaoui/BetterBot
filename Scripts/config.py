import os

# ======================== CONFIG & CONSTANTS ===========================
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "PK023FOQLSDB6VR3UDRV")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "2CalVxgSuOnObttDwgBRcpQAbBMTXR4XkBa7qXdR")
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Stocks123")
DB_HOST = os.getenv("DB_HOST", "database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# ------------------- Trading & Modeling Parameters ---------------------
NOTIONAL = 1000           # Amount in dollars to spend on each trade
MODEL_LOOKBACK = 50       # How many data points back you use to train or generate features
FEATURE_LAGS = 2          # How many lagged returns or indicator values to include as features
SHORT_MA = 9              # Typical short moving average period
LONG_MA = 21             # Typical long moving average period
RSI_PERIOD = 14           # Classic RSI period
RETRAIN_FREQUENCY = 5   # in minutes (e.g., retrain every 2 hours)

# ------------------- Market Hours & Timezone --------------------------
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

TIMEZONE_NAME = "US/Eastern"
