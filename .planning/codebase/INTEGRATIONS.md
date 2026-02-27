# External Integrations

**Analysis Date:** 2026-02-27

## APIs & External Services

**Broker APIs:**
- Interactive Brokers (IBKR) - Primary trading execution and real-time market data
  - SDK/Client: `ib_insync` 3.2.0
  - Connection: IB_HOST, IB_PORT, IB_CLIENT_ID, IB_CLIENT_ID_TRADE, IB_CLIENT_ID_DATA (configured in `Scripts/config.py`)
  - Operations: Contract qualification, historical data fetching, order placement, position management

- Alpaca Markets - Secondary/fallback trading API
  - SDK/Client: `alpaca-trade-api` 3.2.0
  - Configuration: API key via environment variables (likely APCA_API_KEY_ID, APCA_API_SECRET_KEY)
  - Status: Imported in requirements but integration status not fully evident in main flow

**News & Sentiment:**
- NewsAPI - News articles for sentiment analysis
  - API Endpoint: https://newsapi.org/v2/everything
  - Auth: NEWSAPI_KEY environment variable
  - Implementation: `Scripts/news.py`
  - Usage: fetch_articles_for_symbol() fetches articles for specified symbols; basic keyword-based sentiment scoring applied

**Financial Data:**
- yfinance - Fallback financial data source (optional/conditional import)
  - Status: Optional - imported with try/except in `Scripts/advanced_features.py`
  - Used for: Alternative market data if primary sources unavailable

## Data Storage

**Databases:**
- PostgreSQL - Primary persistent data store for stock prices and portfolio state
  - Connection: `postgresql+psycopg2://[DB_USER]:[DB_PASSWORD]@[DB_HOST]:[DB_PORT]/[DB_NAME]`
  - Configured in `Scripts/config.py` with AWS RDS defaults: `database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com`
  - Client: SQLAlchemy 2.0.36 ORM with psycopg2 drivers
  - Tables:
    - `stock_data` - Historical OHLCV bars (columns: symbol, timestamp, open, high, low, close, volume)
    - `portfolio` - Holdings and cost basis tracking (columns: symbol, shares, cost_basis)

**File Storage:**
- Local filesystem - Model and data files
  - `ensemble_model.joblib` - Serialized ML ensemble model (XGBoost + LightGBM voting classifier)
  - `sp500.csv` - S&P 500 stock symbols list (columns: Symbol, industry, etc.)

**Caching:**
- In-memory Python dictionaries - Runtime caches
  - `active_positions` dict (trade.py) - Holds current holdings
  - `latest_indicators_dict` dict (trade.py) - Technical indicator cache per symbol

## Authentication & Identity

**Auth Provider:**
- Custom / Broker-Native
  - Interactive Brokers: Connection-based auth via client IDs (IB_CLIENT_ID, IB_CLIENT_ID_TRADE, IB_CLIENT_ID_DATA)
  - Alpaca: API key/secret environment variables (implementation details in alpaca-trade-api)
  - NewsAPI: API key via NEWSAPI_KEY environment variable

**Security:**
- Environment variable-based secrets (config.py reads from os.getenv())
- No oauth/external identity provider
- Hardcoded defaults in config.py for development (CRITICAL: DB credentials exposed in config defaults)

## Monitoring & Observability

**Error Tracking:**
- Python logging module - All modules configured with basicConfig()
- Log level: INFO
- Format: `%(asctime)s:%(levelname)s:%(message)s`

**Logs:**
- Console output via logging.basicConfig()
- Log files: None configured; all output to stdout
- Key log points: Data fetch, database operations, model training/retraining, trade execution, market hours checks

**Performance Logging:**
- Model performance tracking via `Scripts/model_performance.py` (log_trade_performance, log_portfolio_performance, log_model_prediction functions)
- A/B testing framework: setup_ab_testing() tracks model performance allocation

## CI/CD & Deployment

**Hosting:**
- Self-hosted/Local deployment
- Primary deployment artifact: `Scripts/main.spec` (PyInstaller spec for executable)
- No detected cloud hosting configuration (may run on local machine or VPS)

**CI Pipeline:**
- None detected - No GitHub Actions, Jenkins, or other CI/CD configuration

**Version Control:**
- Git repository present (.git directory)
- Recent commits indicate active development (main.py updates, merge conflicts in main.py suggesting branch work)

## Environment Configuration

**Required env vars:**
- IB_HOST (default: 127.0.0.1) - Interactive Brokers connection host
- IB_PORT (default: 7497) - IBKR connection port
- IB_CLIENT_ID (default: 1) - IBKR primary client ID
- IB_CLIENT_ID_TRADE (default: 2) - IBKR trading client ID
- IB_CLIENT_ID_DATA (default: 3) - IBKR data client ID
- DB_USER (default: postgres) - PostgreSQL username
- DB_PASSWORD (default: Stocks123) - PostgreSQL password
- DB_HOST (default: database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com) - PostgreSQL host
- DB_PORT (default: 5432) - PostgreSQL port
- DB_NAME (default: postgres) - PostgreSQL database name
- NEWSAPI_KEY (required for news sentiment) - NewsAPI key
- APCA_API_KEY_ID (optional for Alpaca) - Alpaca API key
- APCA_API_SECRET_KEY (optional for Alpaca) - Alpaca API secret

**Secrets location:**
- Environment variables (os.getenv() calls in config.py)
- No .env file detected; configuration hardcoded with defaults in `Scripts/config.py`
- CRITICAL SECURITY ISSUE: Database credentials embedded as defaults in source code

## Webhooks & Callbacks

**Incoming:**
- None detected - No webhook handlers configured for external event notification

**Outgoing:**
- Order execution callbacks via IBKR IB object
- No explicit webhook dispatch to external services

**Real-time Data Streams:**
- IB WebSocket/socket connection for real-time bars (on_bar callback in trade.py)
- Real-time position updates via ib.positions() polling

---

*Integration audit: 2026-02-27*
