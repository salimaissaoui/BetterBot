# Technology Stack

**Analysis Date:** 2026-02-27

## Languages

**Primary:**
- Python 3.x - Core application for trading bot, data processing, and ML modeling

## Runtime

**Environment:**
- Python 3.x runtime
- Virtual environment: `.venv/` (present in project)

**Package Manager:**
- pip - Python dependency management
- Lockfile: `requirements.txt` (present)

## Frameworks

**Core Trading:**
- ib_insync 3.2.0 - Interactive Brokers API client for trading and real-time data
- alpaca-trade-api 3.2.0 - Alpaca Markets trading API integration

**Machine Learning & Modeling:**
- scikit-learn 1.2.0 - Core ML algorithms and utilities
- XGBoost 1.6.2 - Gradient boosting classifier for predictions
- LightGBM 3.3.5 - Light gradient boosting for ensemble modeling
- joblib 1.4.2 - Model persistence (serialization/deserialization)

**Data Processing:**
- pandas 2.2.3 - Data manipulation and analysis
- numpy 1.23.5 - Numerical computing
- scipy 1.14.1 - Scientific computing (statistics, signal processing)

**Database:**
- SQLAlchemy 2.0.36 - ORM for database operations
- psycopg2 2.9.10 & psycopg2-binary 2.9.10 - PostgreSQL database driver

**Scheduling & Async:**
- APScheduler 3.11.0 - Background task scheduling for periodic retraining
- asyncio 3.4.3 - Asynchronous I/O operations

**Web & HTTP:**
- aiohttp 3.11.10 - Async HTTP client
- requests 2.32.3 - HTTP requests library
- websocket-client 1.8.0 - WebSocket support
- websockets 10.4 - WebSocket protocol

**Utilities:**
- Beautiful Soup 4.12.3 - Web scraping (potential news/data extraction)
- python-dateutil 2.9.0.post0 - Date/time utilities
- pytz 2024.2 - Timezone handling
- PyYAML 6.0.1 - YAML parsing for configuration

**Development/Build:**
- Cython 3.0.11 - C extensions
- ta 0.11.0 - Technical analysis indicators library

## Key Dependencies

**Critical:**
- ib_insync - Enables connection to Interactive Brokers TWS/Gateway for live trading
- SQLAlchemy + psycopg2 - Persists stock data and portfolio state to PostgreSQL
- scikit-learn, XGBoost, LightGBM - Core ML ensemble for price prediction signals
- pandas, numpy - Data transformation and numerical operations

**Infrastructure:**
- APScheduler - Enables periodic model retraining background job
- aiohttp, requests - Network communication for data fetching and API calls
- joblib - Model serialization/deserialization (ensemble_model.joblib observed in repo)

## Configuration

**Environment:**
- Configuration via environment variables defined in `Scripts/config.py`
- Key configs: IB connection parameters (host, port, client IDs), database credentials, trading parameters (ATR window, position sizing, stop loss %), market hours

**Build:**
- PyInstaller spec file present: `Scripts/main.spec` - Used to build executable from main.py

## Platform Requirements

**Development:**
- Python 3.x with pip
- Virtual environment isolated dependencies in `.venv/`
- PostgreSQL database connectivity (AWS RDS or local instance)
- Interactive Brokers TWS or IB Gateway running for trading operations

**Production:**
- Deployment target: Standalone Python application
- Requires: PostgreSQL database, IBKR connection, Alpaca API (optional fallback)
- Persistent storage: `ensemble_model.joblib` (cached ML model), `sp500.csv` (stock symbols list)

---

*Stack analysis: 2026-02-27*
