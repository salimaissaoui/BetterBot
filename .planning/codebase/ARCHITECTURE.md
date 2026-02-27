# Architecture

**Analysis Date:** 2026-02-27

## Pattern Overview

**Overall:** Layered Event-Driven Trading Bot with ML Ensemble Pipelines

**Key Characteristics:**
- **Modular layer separation:** Data → Features → ML Models → Trading → Portfolio Tracking
- **Real-time event processing:** Market data via IBKR triggers trading decisions through scheduled jobs
- **Ensemble approach:** Multiple ML models (XGBoost, LightGBM) vote on predictions with soft voting
- **Asynchronous background scheduling:** Periodic model retraining, portfolio scanning, and reporting
- **Hybrid basic + advanced models:** Baseline ensemble with optional advanced features and reinforcement learning
- **A/B testing framework:** Dual model allocation with performance comparison (30% basic, 70% advanced)

## Layers

**Entry Point & Orchestration:**
- Purpose: Main event loop, IB connection management, scheduler setup
- Location: `Scripts/main.py`
- Contains: `main()`, `initialize_bot()`, trading loop, scheduler setup
- Depends on: All other modules
- Used by: None (top-level)

**Data Ingestion Layer:**
- Purpose: Fetch historical and real-time market data from Interactive Brokers
- Location: `Scripts/data_fetch.py`
- Contains: Symbol loading from CSV, historical data fetching, contract qualification
- Depends on: IB API, `Scripts/database.py`
- Used by: `Scripts/main.py`, `Scripts/modeling.py`

**Data Storage & ORM:**
- Purpose: Manage PostgreSQL persistence for stock data and portfolio state
- Location: `Scripts/database.py`
- Contains: SQLAlchemy ORM models (`StockData`, `Portfolio`), session management, insert/upsert operations
- Depends on: PostgreSQL database, SQLAlchemy
- Used by: All analytical modules

**Feature Engineering Layer:**
- Purpose: Transform raw OHLCV data into machine-learning-ready features
- Location: `Scripts/indicators.py` (basic indicators), `Scripts/advanced_features.py` (advanced)
- Contains: Technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR), lagged features, pattern detection, market regime detection
- Depends on: Pandas, NumPy, SciPy
- Used by: `Scripts/modeling.py`, `Scripts/advanced_modeling.py`, `Scripts/trade.py`

**Model Training & Prediction:**
- Purpose: Train, persist, and load predictive models
- Location: `Scripts/modeling.py` (basic), `Scripts/advanced_modeling.py` (advanced)
- Contains:
  - Basic: VotingClassifier with XGBoost + LightGBM, TimeSeriesSplit CV, hyperparameter tuning
  - Advanced: Reinforcement learning environments, deep learning with PyTorch, ensemble with Random Forest & Gradient Boosting
- Depends on: scikit-learn, XGBoost, LightGBM, PyTorch (optional), stable_baselines3 (optional)
- Used by: `Scripts/trade.py`, `Scripts/main.py`

**Trading Execution:**
- Purpose: Evaluate signals and execute buy/sell orders via IBKR
- Location: `Scripts/trade.py`
- Contains: `on_bar()` handler, order placement, position management, market hours checks, advanced model initialization
- Depends on: IB API, models, indicators
- Used by: `Scripts/main.py` trading loop

**Position Sizing:**
- Purpose: ML-driven calculation of optimal trade sizes based on risk/account state
- Location: `Scripts/position_sizing.py`
- Contains: `MLPositionSizer` class with RandomForestRegressor, account metrics, risk budgets
- Depends on: scikit-learn, database
- Used by: `Scripts/trade.py`

**Performance Monitoring:**
- Purpose: Track trades, portfolio snapshots, model accuracy, and A/B test results
- Location: `Scripts/model_performance.py`
- Contains: `TradingPerformanceMetrics` class, trade logging, portfolio history, prediction evaluation
- Depends on: scikit-learn metrics, database
- Used by: `Scripts/main.py`, `Scripts/trade.py`

**Configuration & Utilities:**
- Purpose: Environment-based config, helper functions, market hours checks
- Location: `Scripts/config.py`, `Scripts/utils.py`
- Contains: IBKR credentials, DB config, trading parameters, custom ML wrappers, timezone handling
- Depends on: os, pytz
- Used by: All modules

**News/Sentiment (Optional):**
- Purpose: Fetch news articles for sentiment analysis
- Location: `Scripts/news.py`
- Contains: `fetch_articles_for_symbol()` using NewsAPI
- Depends on: requests, NewsAPI
- Used by: None (currently unused but available)

## Data Flow

**Daily Bot Initialization:**

1. `main()` connects to IBKR (IB_HOST:IB_PORT)
2. `initialize_bot()` calls `fetch_and_load_symbols()` → reads `sp500.csv` → qualifies contracts via IBKR
3. `fetch_historical_data()` retrieves 1 year of daily OHLCV data per symbol → `insert_historical_data()` upserts to PostgreSQL
4. `load_existing_model()` attempts to load `ensemble_model.joblib`
5. If no model exists: `retrain_model()` pulls all symbols from DB → computes indicators → trains ensemble
6. `initialize_advanced_model()` initializes advanced model (if reinforcement learning enabled)
7. `setup_scheduler()` configures 3 background jobs:
   - Model retraining every N minutes (RETRAIN_FREQUENCY)
   - Portfolio scan every hour
   - Daily report at 6 PM ET

**Real-Time Trading Loop (30-second cycles):**

1. `trading_loop()` processes 5 symbols per cycle from `polling_symbols`
2. For each symbol: request 2 days of 5-min delayed bars via IBKR
3. Latest bar → `on_bar(bar, model)` → computes indicators → generates prediction
4. Prediction confidence + current position size → `ml_position_sizer.calculate_position_size()`
5. If buy signal: `submit_ml_sized_order('BUY', confidence)` → IBKR market order (optionally outside RTH)
6. Log trade/portfolio snapshot → `log_portfolio_performance()`
7. Rotate symbols list for next cycle

**Hourly Portfolio Scan:**

1. Query account summary (NetLiquidation, AvailableFunds)
2. Get all current positions
3. For each position:
   - Fetch recent 100 bars from DB
   - Compute indicators + advanced features
   - Run prediction on active model (advanced if trained, else basic)
   - Decision logic: BUY_MORE (prob>0.65), SELL (prob<0.35), HOLD, STOP_LOSS, TAKE_PROFIT
   - Execute recommended action
4. Scan watchlist (top 10 symbols) for new opportunities
5. Enter new positions if probability > 0.70 and confidence > 0.40
6. Log comprehensive portfolio snapshot

**Model Retraining (Background Job):**

1. If market is closed OR allow_after_hours_trading is True:
   - Query all distinct symbols from DB
   - For each symbol: fetch all records, compute indicators
   - Concatenate all processed data
   - Extract features via `prepare_features()` → target is forward-looking 0.3% return
   - Train ensemble with TimeSeriesSplit CV
   - Save to `ensemble_model.joblib`

**Model Selection & Prediction:**

- **Active Model Selection Logic:**
  - If advanced model is initialized AND trained: use advanced model
  - Else: fall back to basic ensemble

- **Prediction Pipeline:**
  - Get recent 50-100 bars for symbol
  - Compute technical indicators (SMA, RSI, MACD, Bollinger Bands, ATR, OBV)
  - Advanced path: create_advanced_features() → market regime detection, volatility clustering, trend features
  - Extract feature subset → standardize
  - `predict_proba()` → get probability of upward movement
  - Confidence = |probability - 0.5| * 2 (scaled 0-1)

**State Management:**

- **Global State in `Scripts/main.py`:**
  - `model`: Current ensemble model (reloaded on each retrain)
  - `ib`: Global IB connection instance
  - `polling_symbols`: List of symbols for trading rotation
  - `advanced_model`: Global reference to advanced model (if initialized)

- **Database State:**
  - `StockData` table: Historical prices (symbol, timestamp, OHLCV)
  - `Portfolio` table: Current holdings (symbol, shares, cost_basis) — primarily updated via IBKR positions

- **Performance Cache:**
  - `performance_tracker` in `Scripts/model_performance.py` maintains in-memory trade history, portfolio history, predictions for 30-day lookback

## Key Abstractions

**Ensemble Classifier (Basic):**
- Purpose: Vote-based meta-model combining XGBoost + LightGBM
- Examples: `Scripts/modeling.py` line 54-57
- Pattern: Soft voting (probability averaging) for 0-1 binary classification (down/up)
- Rationale: Reduce overfitting, capture complementary strengths

**Advanced Trading Environment:**
- Purpose: Gymnasium-compatible environment for RL training
- Examples: `Scripts/advanced_modeling.py` line 53+
- Pattern: Simulated order execution, reward calculation, state management
- Rationale: Enable stable_baselines3 algorithms (PPO, A2C, DQN)

**MLPositionSizer:**
- Purpose: ML-driven Kelly criterion-like position sizing
- Examples: `Scripts/position_sizing.py` line 17+
- Pattern: RandomForestRegressor predicts optimal share count based on account state + volatility + confidence
- Rationale: Dynamic position sizing adapts to market conditions and portfolio concentration

**TradingPerformanceMetrics:**
- Purpose: Centralized tracking of trading outcomes, model accuracy
- Examples: `Scripts/model_performance.py` line 24+
- Pattern: Append-only event logs with 30-day rolling window analysis
- Rationale: Enable A/B testing comparisons, performance-based model switching

**MarketRegimeDetector:**
- Purpose: Unsupervised classification of market conditions (bull, bear, sideways, volatile)
- Examples: `Scripts/advanced_features.py` line 31+
- Pattern: KMeans clustering on rolling volatility/return features
- Rationale: Adapt model parameters and position sizing to regime

## Entry Points

**Main Event Loop:**
- Location: `Scripts/main.py` line 642-691
- Triggers: Script execution (`python -m Scripts.main`)
- Responsibilities:
  - Initialize IBKR connection and bot
  - Start background scheduler
  - Loop every 30s: execute `trading_loop()`
  - Handle keyboard interrupt gracefully

**Scheduled Jobs:**
- `scheduled_retrain()`: Every RETRAIN_FREQUENCY minutes → calls `retrain_model()`
- `hourly_portfolio_scan()`: Every hour → comprehensive position analysis + new opportunity detection
- `generate_portfolio_report()`: Daily at 6 PM ET → performance metrics + risk analysis

**Bar Event Handler:**
- Location: `Scripts/trade.py` line 150+ (incomplete in excerpt)
- Triggers: Every 5-min bar received in `trading_loop()`
- Responsibilities: Prediction + order placement

## Error Handling

**Strategy:** Try-except blocks with logging at module level; graceful degradation

**Patterns:**

1. **Data Fetch Failures:**
   - IB API call → catch exception → log warning → return empty DataFrame or retry
   - Example: `Scripts/data_fetch.py` line 51-68 (max_retries loop)

2. **DB Connection:**
   - Session context manager in `Scripts/database.py` line 68-79 → auto-rollback on error
   - Duplicate inserts caught via `on_conflict_do_nothing()` in Postgres

3. **Model Training:**
   - Empty/insufficient data → return None
   - Exception during fit → log error + traceback → return None
   - Fallback: Use previously loaded model

4. **Trading Execution:**
   - Order placement exceptions logged → trade skipped but loop continues
   - Portfolio scan errors don't halt bot (individual symbol errors caught)

5. **Advanced Model Failures:**
   - If advanced model init fails → log warning → use basic model only
   - If advanced prediction fails → fall back to basic model prediction

## Cross-Cutting Concerns

**Logging:**
- `logging.basicConfig()` in each module (level=INFO)
- Format: `%(asctime)s:%(levelname)s:%(message)s`
- Logs market data fetches, model training, trades, portfolio snapshots

**Validation:**
- Symbol sanitization: `Scripts/data_fetch.py` line 11-17 filters booleans + normalizes to uppercase
- Data integrity: Required columns checked in `Scripts/indicators.py` line 29-32
- Feature availability: Check non-NaN feature counts before prediction

**Authentication:**
- IBKR: Host/port/client_id from `Scripts/config.py` (env vars or defaults)
- PostgreSQL: User/password/host/port from `Scripts/config.py` (env vars or defaults)
- NewsAPI: Optional NEWSAPI_KEY from env in `Scripts/news.py`

**Market Hours:**
- `Scripts/utils.py` line 48-61: `is_market_open()` checks US Eastern timezone, weekday, 9:30-17:00
- Used to decide: retrain outside market hours, allow after-hours trading flag

**Transaction Safety:**
- PostgreSQL transactions via context manager
- IBKR orders placed immediately (market orders, no pending order checks)
- No locking mechanism for race conditions (single-threaded bot)

---

*Architecture analysis: 2026-02-27*
