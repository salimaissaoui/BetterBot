# Codebase Concerns

**Analysis Date:** 2026-02-27

## Critical Issues

### Git Merge Conflict in Main Entry Point

**Issue:** Unresolved git merge conflict in `main.py` blocking execution
- Files: `Scripts/main.py` (lines 11-17)
- Impact: Code will not execute or import correctly
- Conflict markers:
  ```
  <<<<<<< Updated upstream
  from .database import engine
  from .data_fetch import fetch_and_load_symbols, fetch_historical_data, insert_historical_data
  =======
  from .database import engine, get_session, StockData
  from Scripts.data_fetch import fetch_and_load_symbols, fetch_historical_data, insert_historical_data
  >>>>>>> Stashed changes
  ```
- Fix approach: Resolve conflict by choosing correct import paths and removing merge markers. The stashed changes use incorrect relative imports (`Scripts.data_fetch` instead of `.data_fetch`).

## Tech Debt

### Hardcoded Database Credentials in Config

**Issue:** Database password is hardcoded in plain text
- Files: `Scripts/config.py` (line 13)
- Impact: Security vulnerability; credentials exposed in version control
- Current state: `DB_PASSWORD = os.getenv("DB_PASSWORD", "Stocks123")`
- Fix approach:
  1. Remove all default password values
  2. Require `DB_PASSWORD` env var to be explicitly set
  3. Use `.env` file (in `.gitignore`) for local development
  4. Store credentials in secure vault for production

### Hardcoded AWS RDS Connection String

**Issue:** Full RDS hostname exposed in default config
- Files: `Scripts/config.py` (line 14)
- Impact: AWS infrastructure details are public; potential attack surface
- Current state: `DB_HOST = os.getenv("DB_HOST", "database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com")`
- Fix approach: Only allow env var override; remove default RDS hostname entirely

### Multiple Global State Variables

**Issue:** Extensive use of module-level globals for state management
- Files: `Scripts/main.py` (lines 28-30), `Scripts/trade.py` (lines 31-38)
- Impact: Makes testing difficult; potential race conditions in concurrent scenarios
- Variables affected:
  - `model` - Global model instance in main.py
  - `ib` - Global IB connection in trade.py
  - `active_positions`, `latest_indicators_dict`, `previous_obv`, `bar_count_since_last_train`, `advanced_model`, `use_advanced_features` in trade.py
- Fix approach: Encapsulate into singleton or context manager classes

### Duplicate IB Connection Instances

**Issue:** Multiple separate IB connections created for different purposes
- Files: `Scripts/main.py` (line 29), `Scripts/trade.py` (line 31)
- Impact: Resource leak; connection pooling not utilized; potential port conflicts
- Current state: Two separate `IB()` instances, each connecting independently
- Fix approach: Create single shared IB connection manager; reuse across modules

### Fallback Position Sizing with Hardcoded Assumptions

**Issue:** Position sizing falls back to fixed $100/share assumption
- Files: `Scripts/trade.py` (lines 118, 343)
- Impact: Incorrect share quantities for stocks <$100 or >$100; losses on cheap/expensive stocks
- Current code: `estimated_price = 100.0` and `dollar_amount / 100`
- Fix approach: Fetch actual current price via market data before fallback calculation

### Inconsistent Data Type Handling for Symbols

**Issue:** Symbols stored inconsistently; bool values can slip through
- Files: `Scripts/database.py` (lines 91-93, 140-142), `Scripts/data_fetch.py` (lines 13-17)
- Impact: Type mismatches; query failures; silent data corruption
- Current sanitization is fragile:
  ```python
  if isinstance(symbol, bool):
      return None
  symbol = str(symbol).upper().strip()
  ```
- Fix approach: Use typed data classes; validate symbols at entry points with regex pattern

## Known Bugs

### Model Not Returned from execute_trade

**Issue:** `execute_trade()` modifies global state but returns nothing
- Files: `Scripts/trade.py` (lines 141-339)
- Impact: Unused return value pattern; potential refactoring risks
- Current state: `execute_trade()` called but return value ignored
- Trigger: Every bar processing in `on_bar()`
- Fix approach: Return trade execution summary; improve function signature consistency

### Potential NaN Values in Technical Indicators

**Issue:** Division by zero in RSI calculation not fully protected
- Files: `Scripts/indicators.py` (lines 58-66)
- Impact: NaN propagation through features; model predictions with invalid inputs
- Current code: `rs = avg_gain / avg_loss.replace(0, np.nan)` - replaces zeros with NaN instead of using clip
- Fix approach: Use safe division: `rs = np.divide(avg_gain, avg_loss, where=avg_loss!=0, out=np.zeros_like(avg_gain))`

### Silent Database Error Handling

**Issue:** Database errors in bar processing suppress exceptions
- Files: `Scripts/trade.py` (lines 395-398)
- Impact: Data inconsistency; lost trades; silent failures
- Current code:
  ```python
  except Exception as db_error:
      logging.warning(f"Database error for {symbol}: {db_error}")
      pass
  ```
- Fix approach: Log full stacktrace; implement retry logic; queue failed writes for later

## Security Considerations

### No Input Validation on Market Data

**Issue:** Bar data from IBKR not validated before database storage
- Files: `Scripts/trade.py` (lines 374-391)
- Impact: Malformed OHLCV data could corrupt training sets; model performance degrades silently
- Current state: Minimal NaN checking; no range validation
- Fix approach:
  1. Validate price ranges (>0, reasonable for symbol)
  2. Check volume >= 0
  3. Verify high >= low >= close >= 0

### Unprotected Market Data Requests

**Issue:** No rate limiting or retry backoff on IBKR market data requests
- Files: `Scripts/trade.py` (lines 217-220, 249-252), `Scripts/main.py` (lines 140-144)
- Impact: Connection throttling; banned IPs; trading loop halts
- Current state: Immediate requests without pause; hardcoded `ib.sleep(1)` or `ib.sleep(2)`
- Fix approach: Implement exponential backoff; track request rate; throttle to API limits

### No Order Validation Before Placement

**Issue:** Orders placed without pre-flight checks
- Files: `Scripts/trade.py` (lines 84-110, 112-134)
- Impact: Potential to place duplicate orders; exceed position limits; crash account
- Current state: No balance check before order; no duplicate prevention
- Fix approach:
  1. Verify available funds >= order cost
  2. Check position already exists before buy
  3. Prevent concurrent duplicate orders same symbol

### Prediction Confidence Not Validated

**Issue:** Model predictions used directly without sanity checks
- Files: `Scripts/trade.py` (lines 156-163, 199-202)
- Impact: Extreme confidence values (0.0, 1.0) bypass logic; poor trades
- Current state: No clamping; no NaN checks
- Fix approach: Assert 0 <= prediction_prob <= 1; handle NaN as 0.5 (neutral)

## Performance Bottlenecks

### Inefficient Database Queries in Tight Loop

**Issue:** Fetches last 100-200 records on every bar
- Files: `Scripts/trade.py` (lines 402-405), `Scripts/main.py` (lines 159-161, 278-280)
- Impact: Slow prediction latency; database connection pool exhaustion; missed trading opportunities
- Current approach: `session.query(StockData).filter().order_by().limit(200)` called every bar (5-30 times/min)
- Profiling expected: Database becomes bottleneck at >10 symbols
- Fix approach:
  1. Cache recent data in memory; update on append
  2. Use materialized views for 50/100-bar windows
  3. Add database indexing on (symbol, timestamp)

### Repeated Technical Indicator Computation

**Issue:** Indicators recomputed from scratch for overlapping data windows
- Files: `Scripts/trade.py` (lines 478), `Scripts/main.py` (lines 171)
- Impact: O(n) work repeated on 90% old data; CPU waste
- Current approach: `compute_technical_indicators(df)` on full 200-bar window every bar
- Fix approach:
  1. Incremental indicator updates (append new bar's indicators)
  2. Rolling window buffer in memory
  3. Cache computed indicators keyed by (symbol, timestamp)

### Model Prediction Overhead Not Optimized

**Issue:** A/B testing overhead in prediction path
- Files: `Scripts/trade.py` (lines 440-445)
- Impact: Extra model calls; slower decision latency
- Current state: `get_model_for_prediction()` adds allocation logic per prediction
- Fix approach: Cache selected variant for batch duration; switch only on schedule

### Portfolio Scan Synchronous

**Issue:** `hourly_portfolio_scan()` blocks trading loop
- Files: `Scripts/main.py` (lines 103-369)
- Impact: Heavy computation (predictions for 10+ symbols) delays trading responses
- Current approach: Scheduled job runs synchronously in background thread
- Fix approach: Offload to thread pool; use async/await for I/O; timeout after 55 minutes

## Fragile Areas

### Advanced Model Fallback Logic

**Issue:** Complex conditional paths for basic vs advanced model create maintenance burden
- Files: `Scripts/trade.py` (lines 351, 422-475)
- Impact: Easy to introduce bugs when modifying prediction logic; hard to test all paths
- Fragility: Three-layer fallback (advanced → basic → skip)
  ```python
  active_model = advanced_model if advanced_model and advanced_model.is_trained else model
  if use_advanced_features and advanced_model and advanced_model.is_trained:
      # ... advanced path with try/except fallback to basic
  ```
- Safe modification: Extract into dedicated model selector class with unit tests

### Feature Preparation Pipeline

**Issue:** Feature column filtering fragile to schema changes
- Files: `Scripts/trade.py` (lines 433-434, 301-302)
- Impact: Column name typos silently remove features; model gets wrong inputs
- Current code: `feature_cols = [col for col in enhanced_df.columns if col not in ['timestamp', 'target', 'future_return', 'symbol']]`
- Test coverage: No unit tests for feature pipeline
- Safe modification:
  1. Define feature schema as class constant
  2. Unit test with sample data
  3. Validate actual columns match schema at runtime

### Position State Management

**Issue:** `active_positions` dict can become inconsistent with actual IBKR positions
- Files: `Scripts/trade.py` (lines 33)
- Impact: Position accounting errors; double-selling; orphaned short positions
- Fragility: Updated locally but not synced with broker on crashes/restarts
- Safe modification:
  1. Sync with broker positions on startup
  2. Add reconciliation job
  3. Prefer broker position data as source-of-truth

### Signal Threshold Tuning

**Issue:** Hardcoded buy/sell thresholds scattered across files
- Files: `Scripts/trade.py` (lines 152-155, 188-194, 316), `Scripts/main.py` (lines 200, 204, 210, 211)
- Impact: Impossible to tune signals systematically; magic numbers everywhere
- Current state:
  - `BUY_THRESHOLD = 0.51` (default)
  - `0.65` (main.py hourly scan)
  - `0.70` (opportunity detection)
  - Regime-adjusted values in trade.py
- Safe modification: Extract to parameterized trading strategy class

## Test Coverage Gaps

### No Integration Tests

**Issue:** No tests for end-to-end trading flow
- What's not tested: Bar → Prediction → Order → Execution cycle
- Files: Entire flow spans `trade.py`, `modeling.py`, `database.py`, `main.py`
- Risk: Untested failure modes (order rejection, partial fills, connection loss)
- Priority: High - core trading logic

### No Model Performance Baseline Tests

**Issue:** Model accuracy not validated against historical data
- What's not tested: Prediction accuracy; ROC-AUC; directional correctness
- Files: `Scripts/modeling.py`, `Scripts/indicators.py` have no test coverage
- Risk: Model degrades silently; trades based on poor predictions
- Priority: High - model is core to strategy

### No Position Sizing Tests

**Issue:** No unit tests for position sizing logic
- What's not tested: ML position sizer outputs; fallback calculations; edge cases (zero balance, extreme confidence)
- Files: `Scripts/position_sizing.py` (lines 295-329)
- Risk: Position sizing errors cascade (over-leveraged trades, insufficient capital)
- Priority: High - capital management

### No Database Schema Migration Tests

**Issue:** No tests for UPSERT logic
- What's not tested: Duplicate handling; constraint violations; concurrent inserts
- Files: `Scripts/database.py` (lines 165-169)
- Risk: Data corruption; lost bars; training data inconsistencies
- Priority: Medium

### No Error Recovery Tests

**Issue:** No tests for graceful degradation
- What's not tested: IBKR connection loss; database unavailable; model load failure
- Files: Exception handlers in `Scripts/trade.py`, `Scripts/main.py`
- Risk: Bot crashes on first real issue
- Priority: Medium

## Missing Critical Features

### No Circuit Breaker

**Issue:** No mechanism to halt trading on systemic failure
- Problem: Bot will keep trading even if model accuracy drops to 45%, database is corrupted, or risk limits breached
- Files: Not implemented anywhere; would need `Scripts/risk_management.py`
- Blocks: Cannot safely deploy to production

### No Position Reconciliation

**Issue:** No periodic sync between internal state and IBKR broker
- Problem: Crash recovery impossible; position accounting diverges from reality
- Files: Not implemented; would need weekly reconciliation routine
- Blocks: Multi-day operation unreliable

### No Drawdown Protection

**Issue:** No stop trading on max drawdown
- Problem: Market crash could wipe entire account while bot continues
- Files: Mentioned in alerts (`main.py` line 507) but not enforced
- Blocks: Portfolio protection missing

### No Trade Logging to Disk

**Issue:** Trade history only in memory; lost on restart
- Problem: Cannot analyze performance; cannot verify backtest matches live
- Files: `Scripts/model_performance.py` logs in-memory only
- Blocks: Post-trade analysis impossible

## Scaling Limits

### Single-Symbol Throughput Bottleneck

**Current capacity:** ~50 bars/minute for 5 symbols (limited by IBKR data feed)
**Limit:** Prediction latency >5 seconds when processing 10+ symbols
**Scaling path:**
1. Move to AsyncIO for I/O operations
2. Use thread pool for indicator computation
3. Cache model predictions; batch predictions per symbol

### Database Connection Pool Exhaustion

**Current capacity:** 20 connections (configured in `Scripts/database.py` line 23)
**Limit:** At ~100 bar updates/minute + portfolio scans = 200 queries/min, pool will timeout
**Scaling path:**
1. Increase `pool_size` to 50
2. Implement connection pooling per thread
3. Cache frequent queries (last N bars per symbol)

### Memory Usage with Advanced Models

**Current capacity:** ~500MB for ensemble + advanced models + data cache
**Limit:** On large portfolios (100+ symbols), historical data in memory will exceed 1GB
**Scaling path:**
1. Lazy load symbol data on demand
2. Implement LRU cache for indicators
3. Move advanced model to separate process

## Dependencies at Risk

### scikit-learn Version Lock

**Risk:** Code uses deprecated RandomizedSearchCV patterns
- Files: `Scripts/modeling.py` (line 86)
- Impact: scikit-learn 1.5+ may break `error_score='raise'` behavior
- Migration plan: Use `error_score=np.nan` and filter; test on newer versions

### XGBoost/LightGBM Compatibility

**Risk:** Wrapper classes assume specific sklearn tag behavior
- Files: `Scripts/utils.py` (lines 8-45)
- Impact: sklearn 1.7+ may change `__sklearn_tags__` interface
- Migration plan: Abstract wrapper behind version check; test with new releases

### PyTorch/RL Library Optional Dependencies

**Risk:** optional imports silently disabled but code assumes availability
- Files: `Scripts/advanced_modeling.py` (lines 22-42)
- Impact: Code paths exist for RL trading but will fail if torch not installed
- Migration plan: Explicit feature flag at startup; fail early if RL enabled but unavailable

## Operational Concerns

### No Graceful Shutdown

**Issue:** Bot cannot clean up positions on stop
- Files: `Scripts/main.py` (lines 687-691)
- Impact: Unexited positions over weekend; forced liquidation at market open
- Fix approach: On SIGTERM, close all open positions before exit

### No Monitoring/Alerting

**Issue:** No system for detecting bot failures in production
- Files: Logging is local file only; no webhook/email alerts
- Impact: Bot could fail silently; losses mount before detection
- Fix approach: Push logs to CloudWatch/Datadog; alert on error rate threshold

### Hardcoded Timezone Handling

**Issue:** US/Eastern timezone hardcoded; no DST handling explicit
- Files: `Scripts/config.py` (line 40), `Scripts/utils.py` (lines 50-51)
- Impact: Market hours wrong for 4 weeks during DST transition
- Fix approach: Use pytz with explicit DST-aware checks

---

*Concerns audit: 2026-02-27*
