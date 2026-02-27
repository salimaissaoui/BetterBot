# Architecture Research

**Domain:** Event-driven algorithmic trading bot (Python / Interactive Brokers)
**Researched:** 2026-02-27
**Confidence:** HIGH — based on direct codebase analysis of all Scripts/ modules

---

## Standard Architecture

### System Overview (Existing + New Components)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATION LAYER                                 │
│  Scripts/main.py                                                             │
│  main() → initialize_bot() → trading_loop() [30s]                           │
│  Background: scheduled_retrain | hourly_portfolio_scan | daily_report        │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │ calls
┌──────────────────────────────▼──────────────────────────────────────────────┐
│                          TRADING EXECUTION LAYER                             │
│  Scripts/trade.py                                                            │
│                                                                              │
│  on_bar(bar, model)                                                          │
│    │                                                                         │
│    ├── [1] DB upsert of incoming bar                                         │
│    ├── [2] Feature computation (basic or advanced path)                      │
│    ├── [3] ML prediction (A/B test model selection)                          │
│    ├── [4] ── NEW: sentiment_gating() ───────────────────┐                  │
│    ├── [5] ── NEW: regime_gate() ───────────────────────┐│                  │
│    ├── [6] execute_trade([(symbol, prob)])              ││                  │
│    │         └── NEW: ExitManager.check_exits()         ││                  │
│    └── [7] log_model_prediction()                       ││                  │
│                                                         ││                  │
│  execute_trade()                                        ││                  │
│    ├── detect_market_regime() [existing, active]        ││                  │
│    ├── threshold adjustment by regime                   ││                  │
│    ├── BUY / SELL / SHORT / COVER logic                 ││                  │
│    └── log_trade_performance()                          ││                  │
└─────────────────────────────────────────────────────────┘│                  │
                                                            │                  │
┌───────────────────────────────────────────────────────────▼──────────────┐  │
│                     NEW: EXIT MANAGEMENT COMPONENT                        │  │
│  Scripts/exit_manager.py                                                  │  │
│                                                                           │  │
│  class ExitManager:                                                       │  │
│    position_registry: {symbol -> EntryRecord}                             │  │
│      EntryRecord: {entry_price, entry_time, atr_at_entry, stop_price,    │  │
│                    target_price, trailing_stop_price, quantity}           │  │
│                                                                           │  │
│    check_exits(symbol, current_price, current_bar) -> ExitSignal | None  │  │
│      ├── hard_stop_loss (entry_price * (1 - STOP_LOSS_PCT))              │  │
│      ├── trailing_stop  (peak_price * (1 - TRAILING_STOP_PCT))           │  │
│      └── take_profit    (entry_price * (1 + TARGET_PCT))                 │  │
│                                                                           │  │
│    register_entry(symbol, entry_price, atr, qty)                         │  │
│    clear_position(symbol)                                                 │  │
│    update_trailing_stop(symbol, current_price)                            │  │
└───────────────────────────────────────────────────────────────────────────┘  │
                                                                                │
┌───────────────────────────────────────────────────────────────────────────────▼┐
│                     NEW: SENTIMENT PIPELINE COMPONENT                          │
│  Scripts/sentiment.py                                                          │
│                                                                                │
│  class SentimentPipeline:                                                      │
│    cache: {symbol -> SentimentScore, fetched_at}  # TTL: 15 minutes           │
│                                                                                │
│    get_sentiment(symbol) -> float  # -1.0 (very bearish) to +1.0 (very bull.) │
│      ├── yahoo_rss_sentiment(symbol)     # yfinance / Yahoo Finance RSS        │
│      ├── reddit_wsb_sentiment(symbol)    # PRAW / Reddit pushshift             │
│      └── weighted_average([rss, reddit])                                       │
│                                                                                │
│    is_sentiment_blocking(symbol, direction) -> bool                            │
│      # Returns True if sentiment opposes the proposed trade direction          │
│      # Suppresses BUY when sentiment < -0.3, suppresses SHORT when > +0.3     │
└────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                  EXISTING: REGIME DETECTION COMPONENT                        │
│  Scripts/advanced_features.py — MarketRegimeDetector                         │
│                                                                              │
│  CURRENT STATE: detect_market_regime(df) is called in execute_trade() and   │
│  adjusts BUY/SHORT thresholds, but MarketRegimeDetector.fit_regime_detector()│
│  is NEVER called — so regime_model is always None, returns 'unknown'.        │
│                                                                              │
│  WHAT NEEDS TO HAPPEN:                                                       │
│    1. Call fit_regime_detector() during initialize_bot() using DB data       │
│    2. Persist fitted KMeans model to disk (regime_model.joblib)              │
│    3. Add regime-aware GATE in execute_trade(): suppress ALL new entries     │
│       when regime == 'volatile' or 'bearish' unless signal is very strong    │
│    4. Expose current_regime as module-level state for logging                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE ENGINEERING LAYER                           │
│  Scripts/indicators.py          — basic technical indicators                 │
│  Scripts/advanced_features.py   — regime detection, patterns, microstructure │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          ML MODEL LAYER                                      │
│  Scripts/modeling.py            — XGBoost + LightGBM VotingClassifier       │
│  Scripts/advanced_modeling.py   — RL environments, deep learning ensemble   │
│  Scripts/position_sizing.py     — RandomForestRegressor for position size   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA & PERSISTENCE LAYER                            │
│  Scripts/data_fetch.py          — IBKR historical and real-time data fetch  │
│  Scripts/database.py            — SQLAlchemy ORM: StockData, Portfolio      │
│  Scripts/model_performance.py   — In-memory trade/portfolio/prediction logs │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| `Scripts/main.py` | Bot lifecycle, scheduler, trading loop | All modules |
| `Scripts/trade.py` — `on_bar()` | Bar event handler: features → prediction → trade decision | `indicators.py`, `advanced_features.py`, `modeling.py`, `execute_trade()`, NEW: `exit_manager.py`, `sentiment.py` |
| `Scripts/trade.py` — `execute_trade()` | Signal thresholding, regime gating, order execution | `advanced_features.detect_market_regime()`, `position_sizing.py`, IBKR via `ib_insync`, `model_performance.py` |
| NEW `Scripts/exit_manager.py` | Per-position exit rules: hard stop, trailing stop, take profit | Called from `execute_trade()` and `hourly_portfolio_scan()` |
| NEW `Scripts/sentiment.py` | Fetch + score news/social sentiment; block opposing-direction entries | Called from `on_bar()` before `execute_trade()` |
| `Scripts/advanced_features.py` — `MarketRegimeDetector` | Regime classification (bull/bear/sideways/volatile) via KMeans | Called from `execute_trade()` and `initialize_bot()` |
| `Scripts/indicators.py` | Technical indicator computation (SMA, RSI, MACD, BB, ATR, OBV) | Used by `trade.py`, `modeling.py` |
| `Scripts/modeling.py` | Basic XGBoost + LightGBM ensemble training and loading | Called by `main.py` scheduler and `trade.py` |
| `Scripts/advanced_modeling.py` | Advanced model: RL, deep learning, GBM ensemble | Called by `trade.py` `initialize_advanced_model()` |
| `Scripts/position_sizing.py` | ML-driven position size via RandomForestRegressor | Called by `trade.py` `submit_ml_sized_order()` |
| `Scripts/model_performance.py` | Trade, portfolio, prediction logging; A/B testing; adaptive learning | Called by `trade.py`, `main.py` |
| `Scripts/database.py` | PostgreSQL ORM (StockData, Portfolio), session management | Used by all data-querying modules |
| `Scripts/data_fetch.py` | IBKR data fetching, symbol loading | Called by `main.py` initialize and `trade.py` `on_bar()` |
| `Scripts/config.py` | Environment-driven constants | Imported by all modules |
| `Scripts/news.py` (existing) | NewsAPI fetch + keyword sentiment — CURRENTLY UNUSED | Will be replaced by new `sentiment.py` |

---

## Data Flow

### Bar Event Flow (30-second cycle, primary hot path)

```
IBKR reqHistoricalData (2D, 5min bars)
    │
    ▼
trading_loop() [Scripts/main.py]
    │  latest_bar → setattr(bar, "symbol", symbol)
    ▼
on_bar(bar, model) [Scripts/trade.py]
    │
    ├─[1] DB upsert: StockData INSERT or UPDATE
    │
    ├─[2] Load recent 200 bars from DB
    │
    ├─[3] Feature path (advanced or basic):
    │       Advanced: create_advanced_features(df, symbol)
    │                 → ~100+ features incl. regime one-hots
    │       Basic:    compute_technical_indicators(df)
    │                 → SMA, RSI, MACD, BB, ATR, OBV
    │
    ├─[4] NEW: sentiment gate
    │       sentiment = SentimentPipeline.get_sentiment(symbol)  [cached 15min]
    │       if sentiment blocks proposed direction → return early, no trade
    │
    ├─[5] ML prediction: predict_proba(latest_features)[0][1] → probability 0-1
    │       A/B test: get_model_for_prediction() selects basic or advanced
    │
    ├─[6] execute_trade([(symbol, prob)]) [Scripts/trade.py]
    │       │
    │       ├─ detect_market_regime(df) → 'bullish'|'bearish'|'sideways'|'volatile'
    │       │    [Scripts/advanced_features.py]
    │       │    Regime → adjust BUY_THRESHOLD / SHORT_THRESHOLD
    │       │
    │       ├─ NEW: ExitManager.check_exits(symbol, current_price)
    │       │    → hard stop / trailing stop / take profit signals
    │       │    → if exit triggered: close_position(symbol) → IBKR market order
    │       │
    │       ├─ Entry decision: prob > BUY_THRESHOLD AND regime not blocking
    │       │    → submit_ml_sized_order() → MLPositionSizer → IBKR market order
    │       │    → ExitManager.register_entry(symbol, entry_price, atr, qty)
    │       │
    │       └─ log_trade_performance() [Scripts/model_performance.py]
    │
    └─[7] log_model_prediction() [Scripts/model_performance.py]
```

### Sentiment Pipeline Data Flow

```
Fetch trigger: on_bar() calls get_sentiment(symbol) once per bar
    │
    ▼
SentimentPipeline.get_sentiment(symbol)
    │
    ├─ Check cache: if cached_at within 15min → return cached score
    │
    ├─ yahoo_rss_sentiment(symbol)
    │    └─ yfinance.Ticker(symbol).news → title + summary → VADER/keyword score
    │
    ├─ reddit_wsb_sentiment(symbol)
    │    └─ PRAW Reddit.subreddit("wallstreetbets").search(symbol, limit=25)
    │       → title + selftext → VADER/keyword score
    │
    ├─ weighted_average([rss_score * 0.6, reddit_score * 0.4])
    │
    ├─ Cache result with timestamp
    │
    └─ Return float in [-1.0, +1.0]

Gating logic in on_bar() (before execute_trade):
    sentiment = get_sentiment(symbol)     # -1 very bearish, +1 very bullish
    if prob > BUY_THRESHOLD and sentiment < -0.3:
        skip entry  # ML says buy but sentiment strongly negative
    if prob < SHORT_THRESHOLD and sentiment > +0.3:
        skip entry  # ML says short but sentiment strongly positive
```

### Regime Detection Data Flow

```
initialize_bot() [Scripts/main.py]
    │
    ├─ Query all symbols + historical data from DB
    │
    ├─ MarketRegimeDetector.fit_regime_detector({symbol: df, ...})
    │    └─ calculate_regime_features() → rolling stats + momentum + vol features
    │    └─ StandardScaler → PCA (10 components) → KMeans(n_clusters=4)
    │    └─ Assigns: {0: 'bearish', 1: 'sideways', 2: 'bullish', 3: 'volatile'}
    │
    └─ Save fitted model to regime_model.joblib

Per-bar (in execute_trade):
    detect_market_regime(df_recent_100_bars)
        └─ regime_detector.predict_regime(df)
             └─ calculate_regime_features() → scale → PCA → KMeans.predict()
             └─ return regime string

Risk gate in execute_trade():
    if regime == 'volatile':
        suppress ALL new entries (no BUY or SHORT)
    elif regime == 'bearish':
        suppress new BUY entries (only allow SHORT or HOLD)
    elif regime == 'bullish':
        suppress new SHORT entries (only allow BUY or HOLD)
```

### Exit Manager Data Flow

```
On BUY execution (in execute_trade):
    ExitManager.register_entry(symbol, entry_price, atr, quantity)
        └─ calculate: stop_price    = entry_price * (1 - STOP_LOSS_PCT)
                      trailing_high = entry_price
                      target_price  = entry_price * (1 + TARGET_PCT)
        └─ store in position_registry[symbol]

On every bar for held positions (in execute_trade or on_bar):
    ExitManager.check_exits(symbol, current_price, bar)
        ├─ update_trailing_stop(symbol, current_price)
        │    └─ if current_price > trailing_high: update trailing_high
        │    └─ trailing_stop = trailing_high * (1 - TRAILING_STOP_PCT)
        │
        ├─ if current_price <= stop_price        → ExitSignal('HARD_STOP_LOSS')
        ├─ if current_price <= trailing_stop      → ExitSignal('TRAILING_STOP')
        └─ if current_price >= target_price       → ExitSignal('TAKE_PROFIT')
        → None if no exit triggered

On exit:
    close_position(symbol) → IBKR market order
    ExitManager.clear_position(symbol)
    log_trade_performance(action='STOP_LOSS'|'TRAILING_STOP'|'TAKE_PROFIT', ...)
```

---

## Recommended Project Structure

The flat `Scripts/` layout is preserved (no refactor needed for this milestone). New files slot in alongside existing modules:

```
Scripts/
├── main.py                  # MODIFY: call fit_regime_detector() in initialize_bot()
├── config.py                # MODIFY: add TRAILING_STOP_PCT, SENTIMENT_BLOCK_THRESHOLD
├── trade.py                 # MODIFY: integrate ExitManager + sentiment gate in on_bar()
├── advanced_features.py     # MODIFY: expose global regime_detector; add fit/persist
│
├── exit_manager.py          # NEW: ExitManager class — hard stop, trailing, take profit
├── sentiment.py             # NEW: SentimentPipeline class — Yahoo RSS + Reddit/WSB
│
├── database.py              # unchanged
├── data_fetch.py            # unchanged
├── indicators.py            # unchanged
├── modeling.py              # unchanged
├── advanced_modeling.py     # unchanged
├── position_sizing.py       # unchanged
├── model_performance.py     # MODIFY: add exit_reason field to log_trade()
├── utils.py                 # unchanged
└── news.py                  # DEPRECATE: replaced by sentiment.py (keep for compatibility)
```

### Structure Rationale

- **`exit_manager.py` as separate module:** Exit logic owns position state (entry prices, trailing highs, stop levels) that must persist across bars. It cannot be inline in `execute_trade()` because that function is stateless per call. It communicates with `trade.py` via import.
- **`sentiment.py` as separate module:** Sentiment fetches are slow (HTTP), must be cached, and are symbol-scoped with a TTL. Separating into its own module allows easy mocking in tests and clean cache management.
- **Regime detector stays in `advanced_features.py`:** The class already exists and is partially wired. The fix is behavioral (call `fit_regime_detector()` at startup), not structural.
- **No subdirectory refactor yet:** The flat structure adds 2 files and modifies 4. A subdirectory refactor is a future concern; doing it now would introduce merge risk while blocking bugs are open.

---

## Architectural Patterns

### Pattern 1: Stateful Singleton Per Component

**What:** Each new component (ExitManager, SentimentPipeline) is instantiated once at module level as a global singleton, matching the existing pattern (`ml_position_sizer`, `advanced_features`, `performance_tracker`).

**When to use:** When state must persist across the 30-second bar loop — exit levels must survive between calls, sentiment cache must not re-fetch on every bar.

**Trade-offs:** Global mutable state is a test hazard. Acceptable here because the bot is single-threaded and tests can reset singletons directly.

**Example:**
```python
# Scripts/exit_manager.py
exit_manager = ExitManager()  # module-level singleton, matches existing pattern

# Scripts/trade.py
from .exit_manager import exit_manager

def execute_trade(pred_results, model):
    for symbol, prob in pred_results:
        # Check exits first (before checking new entry signals)
        exit_signal = exit_manager.check_exits(symbol, current_price, current_bar)
        if exit_signal:
            close_position(symbol)
            exit_manager.clear_position(symbol)
            log_trade_performance({..., 'exit_reason': exit_signal.reason})
            continue

        if want_to_buy and symbol not in active_positions:
            qty = submit_ml_sized_order(symbol, 'buy', prob, entry_price)
            if qty > 0:
                exit_manager.register_entry(symbol, entry_price, atr_value, qty)
```

### Pattern 2: Gate Chain in on_bar() Before execute_trade()

**What:** Layer sentiment and regime as suppression gates between prediction and execution. Each gate can veto a trade without knowing about the others. This avoids nesting sentiment logic inside execute_trade's regime logic.

**When to use:** When adding signals that suppress (not replace) trading decisions.

**Trade-offs:** Each gate adds latency to the hot path. Sentiment fetch is cached so it adds ~0ms on cache hit. Regime predict runs KMeans inference — fast enough at 100 bars.

**Example:**
```python
# Scripts/trade.py — in on_bar(), after ML prediction
prediction_prob = selected_model.predict_proba(latest_features)[0][1]

# Gate 1: sentiment
from .sentiment import sentiment_pipeline
sentiment_score = sentiment_pipeline.get_sentiment(symbol)
if sentiment_blocks_entry(sentiment_score, prediction_prob):
    log_model_prediction(symbol, prediction_prob, confidence=confidence)
    return model  # skip execute_trade

# Gate 2: regime (already inside execute_trade — handled there)
execute_trade([(symbol, prediction_prob)], active_model)
```

### Pattern 3: Position Registry for Exit Tracking

**What:** ExitManager maintains a dict keyed by symbol containing entry metadata needed to compute exit levels. This is the canonical source of truth for "is this bot in a position and at what levels."

**When to use:** Whenever exit prices (stops, targets) depend on entry price — which is always the case for stop loss and take profit.

**Trade-offs:** The existing `active_positions` dict in `trade.py` tracks position direction (`'long'/'short'`) but NOT entry price. ExitManager's registry supplements this rather than replacing it, avoiding a large refactor of execute_trade.

**Example:**
```python
# Scripts/exit_manager.py
@dataclass
class EntryRecord:
    symbol: str
    entry_price: float
    entry_time: datetime
    atr_at_entry: float
    quantity: int
    direction: str          # 'long' or 'short'
    stop_price: float       # hard stop
    trailing_high: float    # for long: highest price since entry
    trailing_stop: float    # trailing stop level
    target_price: float     # take profit level

class ExitManager:
    def __init__(self):
        self.positions: Dict[str, EntryRecord] = {}
```

---

## Integration Points

### Where New Components Connect to Existing Code

| Boundary | Where to Hook | Call Direction | Notes |
|----------|--------------|----------------|-------|
| ExitManager ↔ execute_trade() | Top of BUY/SELL branch in `execute_trade()` | trade.py → exit_manager.py | Check exits before checking new entry. Register on successful BUY. Clear on any close. |
| ExitManager ↔ hourly_portfolio_scan() | After position prediction, before action execution | main.py → exit_manager.py | Hourly scan already has stop/take-profit at -15%/+25% hardcoded; replace with ExitManager call |
| SentimentPipeline ↔ on_bar() | After ML prediction, before execute_trade() call | trade.py → sentiment.py | One call per symbol per bar; cache TTL prevents rate limits |
| MarketRegimeDetector.fit ↔ initialize_bot() | At end of initialize_bot(), after DB is populated | main.py → advanced_features.py | Needs DB data; fits once at startup, loaded from disk on subsequent runs |
| Regime gate ↔ execute_trade() | Already partially present (threshold adjustment); add suppression gate | trade.py → advanced_features.py | Extend existing detect_market_regime() call to also suppress entries |
| ExitManager ↔ log_trade_performance() | At exit execution point | trade.py → model_performance.py | Add `exit_reason` field: 'HARD_STOP', 'TRAILING_STOP', 'TAKE_PROFIT', 'SIGNAL' |

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Yahoo Finance (yfinance) | `yf.Ticker(symbol).news` + RSS | Already imported in `advanced_features.py`; free, no key |
| Reddit (PRAW) | `praw.Reddit(client_id=...)` subreddit search | Requires Reddit app credentials (free); rate limit: 60 req/min |
| IBKR (ib_insync) | Market orders via `ib.placeOrder()` | All exit orders use existing `close_position()` |
| PostgreSQL | SQLAlchemy session via `get_session()` | No new tables needed; exit context goes in trade log dict |

---

## Suggested Build Order

Build order is driven by two rules: (1) fix blocking bugs before adding intelligence, (2) exits before entries.

### Phase A: Unblock (prerequisite for everything)

1. Resolve git merge conflict in `Scripts/main.py` (lines 11-17)
2. Move hardcoded DB credentials in `Scripts/config.py` to `.env` / environment variables
3. Verify bot reaches `trading_loop()` without crashing

These are not architecture — they are preconditions. Nothing below works until Phase A is complete.

### Phase B: Exit Manager (build first among new features)

Build `Scripts/exit_manager.py` before sentiment or regime because:
- It has zero external dependencies (no HTTP, no ML training)
- It produces immediate, observable behavior (positions will actually close)
- It is the highest-priority requirement per PROJECT.md ("exits matter more than entries")
- Regime and sentiment gates reduce entry frequency; ExitManager reduces hold time; both are needed but ExitManager has higher failure cost if absent

Integration order within Phase B:
1. Implement `EntryRecord` dataclass and `ExitManager` class
2. Wire `register_entry()` into the BUY branch of `execute_trade()`
3. Wire `check_exits()` at the top of the per-symbol loop in `execute_trade()`
4. Replace hardcoded -15%/+25% thresholds in `hourly_portfolio_scan()` with `ExitManager.check_exits()`
5. Add `exit_reason` field to `log_trade_performance()` calls

### Phase C: Regime Detection Activation (build second)

The `MarketRegimeDetector` class already exists and `detect_market_regime()` is already called. The gap is that `fit_regime_detector()` is never invoked, so the model is always None and every regime returns 'unknown'.

Integration order within Phase C:
1. Add `fit_regime_detector()` call in `initialize_bot()` after DB population
2. Add joblib persist/load for the fitted KMeans model (avoid refitting on every restart)
3. Extend regime gate in `execute_trade()` from threshold adjustment to entry suppression
4. Add logging of current regime per trade for post-hoc analysis

Phase C before sentiment because: regime suppression is a hard gate (no trades in volatile regime), whereas sentiment is a soft filter. Hard gates have more impact and are simpler to implement.

### Phase D: Sentiment Pipeline (build third)

Build `Scripts/sentiment.py` after regime is working because:
- Regime is already partially coded and just needs activation; sentiment needs net-new HTTP fetch logic
- Sentiment adds external dependencies (PRAW, rate limits) that complicate testing
- Sentiment signals are soft filters; they should only block trades that pass both ML threshold and regime gate

Integration order within Phase D:
1. Implement `yahoo_rss_sentiment()` using yfinance (already in `.venv`, no new dep)
2. Implement in-memory TTL cache (15-minute expiry per symbol)
3. Wire `get_sentiment()` into `on_bar()` after prediction, before `execute_trade()`
4. Implement `reddit_wsb_sentiment()` with PRAW (new dependency, new credentials)
5. Add weighted combination and tunable block threshold in `config.py`

---

## Anti-Patterns

### Anti-Pattern 1: Exit Logic Inline in execute_trade()

**What people do:** Add stop-loss checks as if-else branches inside `execute_trade()` using local variables for entry price.

**Why it's wrong:** `execute_trade()` is called with fresh probability results every 30 seconds. It has no memory of entry price between calls. Any inline stop calculation will use the wrong reference price unless entry prices are stored somewhere persistent. The existing `active_positions` dict only stores `'long'/'short'`, not prices.

**Do this instead:** Use `ExitManager.position_registry` as the authoritative entry price store. Check exits as the first action per symbol in `execute_trade()`, before evaluating any new entry signals.

### Anti-Pattern 2: Fetching Sentiment on Every Bar Without Caching

**What people do:** Call the sentiment fetch function directly inside the 30-second trading loop for every symbol processed.

**Why it's wrong:** With 5 symbols per cycle, that is 5 HTTP requests every 30 seconds to Yahoo Finance and Reddit. Yahoo Finance will rate-limit within minutes. Reddit PRAW allows 60 requests/minute but that budget is consumed by the main data fetches.

**Do this instead:** Cache sentiment per symbol with a 15-minute TTL. News does not change meaningfully in 30 seconds. On cache hit, return immediately at ~0ms latency.

### Anti-Pattern 3: Fitting Regime Model on Every Bot Start

**What people do:** Call `fit_regime_detector()` inside `initialize_bot()` without checking for a persisted model first.

**Why it's wrong:** `fit_regime_detector()` runs KMeans on all historical data across all symbols. With 500 S&P symbols and 1 year of data, this takes 30-120 seconds and blocks bot startup on every run.

**Do this instead:** Persist the fitted `KMeans`, `StandardScaler`, and `PCA` objects to `regime_model.joblib`. On startup: try to load. If missing or stale (older than N days), refit and save. The background `scheduled_retrain()` job is the right place to refresh the regime model periodically.

### Anti-Pattern 4: Regime Gate That Suppresses Exits

**What people do:** Apply regime-based suppression to all trade actions, including closing positions.

**Why it's wrong:** If regime is 'volatile' and the bot is holding a losing position, suppressing the SELL exit means the bot continues to hold — the original failure mode. Regime gating must only apply to NEW ENTRY decisions, never to exits.

**Do this instead:** ExitManager exits are unconditional — they fire regardless of regime. Regime gate applies only before the BUY/SHORT entry branches in `execute_trade()`.

### Anti-Pattern 5: Using news.py's NewsAPI Instead of Free Sources

**What people do:** Extend the existing `Scripts/news.py` to add sentiment capabilities.

**Why it's wrong:** `news.py` uses NewsAPI, which requires a paid subscription for access beyond 30 days or for real-time news. PROJECT.md explicitly constrains to free sources.

**Do this instead:** Build `Scripts/sentiment.py` fresh using yfinance (Yahoo Finance news, free) and PRAW (Reddit, free with app registration). Keep `news.py` for backward compatibility but do not invest further in it.

---

## Scalability Considerations

This bot is single-symbol-per-bar with a rotating 5-symbol-per-cycle processing queue. Scaling concerns are not about user count but about symbol universe size:

| Symbol Universe | Architecture Adjustments |
|-----------------|--------------------------|
| 5-50 symbols | Current architecture is fine. Sentiment fetch on every bar is feasible with caching. |
| 50-500 symbols | Sentiment cache TTL becomes critical. Pre-warm cache in a background job at startup rather than on-demand. Regime model fits fast enough. |
| 500+ symbols (full S&P 500) | Batch sentiment fetch during market-closed hours. Regime model fitting needs symbol sampling or incremental update. Consider asyncio for parallel IBKR data requests. |

### First Bottleneck

Sentiment HTTP fetches. Mitigated by 15-minute TTL cache. Pre-warming the cache for all symbols during the 9:25-9:30 AM pre-market window is the next optimization if latency becomes visible.

### Second Bottleneck

Regime model refit time during `scheduled_retrain()`. Mitigated by persisting the fitted model and only refitting daily outside market hours.

---

## Sources

- Direct codebase analysis: `Scripts/trade.py`, `Scripts/main.py`, `Scripts/advanced_features.py`, `Scripts/model_performance.py`, `Scripts/config.py`, `Scripts/news.py` (2026-02-27)
- Architecture derived from observed code behavior, not documentation
- Confidence: HIGH — all integration points verified against actual code, not inferred

---

*Architecture research for: BetterBot — event-driven Python trading bot (subsequent milestone)*
*Researched: 2026-02-27*
