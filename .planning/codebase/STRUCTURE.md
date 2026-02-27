# Codebase Structure

**Analysis Date:** 2026-02-27

## Directory Layout

```
BetterBot/
├── .git/                   # Git repository
├── .claude/                # Claude CLI configuration
├── .planning/              # GSD planning output
│   └── codebase/          # This document
├── .venv/                  # Python virtual environment (dependencies)
├── Scripts/                # Core application code
│   ├── __init__.py
│   ├── main.py            # Entry point: bot orchestration, trading loop, scheduler
│   ├── config.py          # Configuration: IBKR, DB, trading parameters
│   ├── database.py        # SQLAlchemy ORM: StockData, Portfolio models, session management
│   ├── data_fetch.py      # IBKR data fetching: symbol loading, historical data
│   ├── indicators.py      # Technical indicators: SMA, RSI, MACD, Bollinger Bands, etc.
│   ├── modeling.py        # Basic ensemble: XGBoost + LightGBM with TimeSeriesSplit CV
│   ├── advanced_modeling.py # Advanced models: RL environments, deep learning, GBM ensembles
│   ├── advanced_features.py # Feature engineering: market regime detection, pattern detection
│   ├── trade.py           # Trading execution: order placement, position management
│   ├── position_sizing.py # ML-based position sizing: RandomForestRegressor
│   ├── model_performance.py # Performance tracking: trades, portfolio, predictions, A/B test
│   ├── utils.py           # Utility functions: XGB/LGB wrappers, market hours check
│   └── news.py            # News fetching: NewsAPI integration (optional)
├── ensemble_model.joblib   # Persisted basic ensemble model (binary)
├── sp500.csv              # S&P 500 symbol list (CSV format)
├── requirements.txt        # Python dependencies
├── README.md              # Basic readme
├── .gitignore            # Git ignore patterns
└── .gitattributes        # Git attributes

```

## Directory Purposes

**`Scripts/`:**
- Purpose: Core trading bot application code
- Contains: 14 Python modules covering data ingestion, ML pipelines, trading, monitoring
- Key files: `main.py` (entry), `modeling.py` (training), `trade.py` (execution), `database.py` (persistence)

**`.venv/`:**
- Purpose: Isolated Python environment with dependencies
- Contains: Installed packages (ib_insync, sqlalchemy, sklearn, xgboost, lightgbm, torch, etc.)
- Generated: Yes (ignored in git)
- Committed: No

**`.planning/codebase/`:**
- Purpose: GSD documentation output (this directory)
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md (as generated)
- Generated: Yes
- Committed: Yes (via `.planning/` in git)

## Key File Locations

**Entry Points:**

- `Scripts/main.py` (line 642-691): Main event loop
  - `main()`: Connect to IBKR, initialize bot, run trading loop + scheduler
  - `initialize_bot()`: Fetch symbols, load/train model, setup A/B testing
  - `trading_loop()`: 30-sec cycle processing 5 symbols
  - `scheduled_retrain()`: Background job for model retraining
  - `hourly_portfolio_scan()`: Background job for position analysis
  - `generate_portfolio_report()`: Daily reporting

**Configuration:**

- `Scripts/config.py`: All environment-driven settings
  - IBKR: `IB_HOST`, `IB_PORT`, `IB_CLIENT_ID`, `IB_CLIENT_ID_TRADE`, `IB_CLIENT_ID_DATA`
  - Database: `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`
  - Trading: `NOTIONAL`, `RETRAIN_FREQUENCY`, `STOP_LOSS_PCT`, `TARGET_THRESHOLD`, `ALLOW_AFTER_HOURS_TRADING`
  - Indicators: `SHORT_MA`, `LONG_MA`, `RSI_PERIOD`, `FEATURE_LAGS`, `ATR_WINDOW`, `DOJI_THRESHOLD`
  - Market hours: `MARKET_OPEN_HOUR`, `MARKET_CLOSE_HOUR`, `TIMEZONE_NAME`

**Core Logic:**

- `Scripts/modeling.py` (line 26-159): `train_model()` function
  - Ensemble with XGBoost + LightGBM
  - TimeSeriesSplit CV with RandomizedSearchCV
  - Early stopping and final ensemble fit
  - Saves to `ensemble_model.joblib`

- `Scripts/trade.py`: Trading execution
  - `on_bar()`: Main bar event handler (incomplete in file)
  - `submit_ml_sized_order()`: Execute order with position sizing
  - `close_position()`: Close out holdings
  - `short_position()`: Short sell
  - `initialize_advanced_model()`: Setup advanced model system

- `Scripts/database.py`: ORM and persistence
  - `StockData` model: id, symbol, timestamp, open, high, low, close, volume
  - `Portfolio` model: id, symbol, shares, cost_basis
  - `get_session()`: Context manager for transactions
  - `insert_stock_data()`: Single bar insert
  - `insert_historical_data()`: Bulk historical upsert

**Testing:**

- No dedicated test directory found
- Testing patterns not established

**Data Files:**

- `sp500.csv`: Symbol list (CSV with 'Symbol' column)
- `ensemble_model.joblib`: Persisted trained model (binary joblib format)

## Naming Conventions

**Files:**

- `snake_case.py`: All Python modules use snake_case
- Examples: `data_fetch.py`, `model_performance.py`, `advanced_modeling.py`
- Binary artifacts: `ensemble_model.joblib`
- Config files: `config.py` (single config module)

**Directories:**

- `lowercase`: All directories use lowercase (Scripts, .venv, .planning)
- No nested directories within `Scripts/` (flat module layout)

**Functions:**

- `snake_case`: All functions use snake_case
- Examples: `fetch_historical_data()`, `compute_technical_indicators()`, `submit_ml_sized_order()`
- Private: Prefixed with `_` (not widely used in codebase)
- Classes: `PascalCase`
  - Examples: `StockData`, `Portfolio`, `MLPositionSizer`, `TradingPerformanceMetrics`, `MarketRegimeDetector`

**Variables:**

- `snake_case`: Local and module-level variables use snake_case
- Examples: `polling_symbols`, `bar_count_since_last_train`, `active_positions`
- Constants: `UPPER_CASE`
  - Examples: `IB_HOST`, `NOTIONAL`, `RETRAIN_FREQUENCY`, `TARGET_THRESHOLD`
- Global state: Prefixed with module import path
  - Examples: `from .config import IB_HOST` (not imported to global namespace pollution)

**Types/Classes:**

- `PascalCase`: All class names
- Model wrappers: `XGBClassifierWrapper`, `LGBMClassifierWrapper` (suffix descriptive)
- Tracking classes: `TradingPerformanceMetrics`, `MLPositionSizer`, `MarketRegimeDetector`
- ORM models: `StockData`, `Portfolio`

## Where to Add New Code

**New Feature (Trading Signal or Analysis):**
- Primary code: `Scripts/[feature_name].py` (new module in Scripts)
  - Example: `Scripts/sentiment_analysis.py` for news sentiment scoring
- Integration point: Import + call from `Scripts/trade.py` in `on_bar()` or from `Scripts/main.py`
- Tests: `Scripts/test_[feature_name].py` (not currently established; see Testing section)

**New Indicator/Feature:**
- If basic: Add function to `Scripts/indicators.py` (line 20+)
  - Follow naming: `compute_[indicator_name]()`
  - Add to `compute_technical_indicators()` pipeline
  - Include in `prepare_features()` return
- If advanced: Add class/function to `Scripts/advanced_features.py`
  - Example: `MarketRegimeDetector.calculate_regime_features()` (line 43+)

**New Model Type (beyond ensemble):**
- Create `Scripts/[model_type]_modeling.py` (new file)
  - Example: `lstm_modeling.py` for time-series neural network
  - Export training function: `def train_[model_type]_model(X, y) -> Model`
  - Implement `predict()` and `predict_proba()` methods
- Update `Scripts/trade.py`: `initialize_advanced_model()` to load new model
- Update selection logic in `Scripts/main.py`: `get_model_for_prediction()`

**New Data Source/Integration:**
- Create `Scripts/[source]_fetch.py` (new file)
  - Example: `crypto_fetch.py` for cryptocurrency prices
  - Export fetch function: `def fetch_[source]_data(symbol, ...) -> pd.DataFrame`
- Store in DB: Add ORM model to `Scripts/database.py` if new asset class
- Call from `Scripts/main.py` `initialize_bot()` as additional data load step

**Portfolio/Risk Module:**
- Create `Scripts/risk_management.py` (new file)
  - Classes: `PortfolioRiskAnalyzer`, `RiskLimitEnforcer`
  - Call from `Scripts/main.py` `hourly_portfolio_scan()` before trade execution
  - Update decision logic in position_analysis loop

**Utilities/Helpers:**
- If cross-module: Add to `Scripts/utils.py` (line 48+)
  - Examples: `is_market_open()`, ML wrapper classes
- If specific to module: Keep inline in that module

## Special Directories

**`.venv/`:**
- Purpose: Python virtual environment (isolated dependencies)
- Generated: Yes (created via `python -m venv .venv`)
- Committed: No (in .gitignore)
- Activation: `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows)
- Packages: Installed via `pip install -r requirements.txt`

**`.git/`:**
- Purpose: Git version control
- Generated: Yes (via `git init`)
- Committed: Yes (all Git metadata)

**`.claude/`:**
- Purpose: Claude IDE configuration
- Generated: Automatically by Claude IDE
- Committed: Partially (settings.local.json if included)

**`.planning/`:**
- Purpose: GSD orchestrator output
- Generated: Yes (created by `/gsd:map-codebase` commands)
- Committed: Yes (version tracked for reference)

## Recommended Module Organization for Extensions

**If adding sub-modules for complexity (future refactor):**

```
Scripts/
├── core/                   # Core bot loop and IBKR integration
│   ├── main.py            # Entry point
│   ├── scheduler.py       # Background job management
│   └── ib_manager.py      # IBKR connection wrapper
├── models/                 # All ML model code
│   ├── basic/
│   │   └── ensemble.py    # XGB + LGB ensemble
│   ├── advanced/
│   │   ├── reinforcement_learning.py
│   │   ├── deep_learning.py
│   │   └── market_regime.py
│   └── training.py        # Shared training utilities
├── features/               # Feature engineering
│   ├── technical.py       # Technical indicators
│   ├── advanced.py        # Market regime, patterns
│   └── sentiment.py       # News/social sentiment
├── trading/                # Execution and management
│   ├── executor.py        # Order placement
│   ├── position_sizer.py  # ML position sizing
│   └── risk_manager.py    # Risk checks
├── data/                   # Data layer
│   ├── fetch.py           # IBKR, external APIs
│   ├── storage.py         # Database ORM
│   └── loaders.py         # Symbol/historical loaders
├── monitoring/             # Tracking and reporting
│   ├── performance.py     # Trade/prediction tracking
│   └── reporting.py       # Portfolio reports
├── config.py              # Configuration
└── utils.py               # Common utilities
```

**Current flat structure trade-off:**
- ✓ Simple navigation, low cognitive load
- ✓ Fast module discovery
- ✗ 14 modules becoming unwieldy as codebase grows
- ✗ Mixing concerns (e.g., `modeling.py` + `advanced_modeling.py` redundancy)

---

*Structure analysis: 2026-02-27*
