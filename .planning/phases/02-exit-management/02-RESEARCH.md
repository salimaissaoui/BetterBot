# Phase 2: Exit Management - Research

**Researched:** 2026-02-27
**Domain:** Per-position exit rule enforcement with PostgreSQL persistence and IBKR reconciliation
**Confidence:** HIGH — based on direct codebase analysis of all relevant source files

---

## Summary

Phase 2 introduces the most critical safety feature in the bot: every position entered gets a hard stop loss, take-profit target, and eventually a trailing stop — all persisted to PostgreSQL so they survive restarts. The existing codebase has zero exit enforcement. The `active_positions` dict in `Scripts/trade.py` stores only `'long'` or `'short'` strings — no entry price, no stop level, no target. The `hourly_portfolio_scan()` in `Scripts/main.py` has hardcoded -15%/+25% thresholds that must be replaced. There is no position reconciliation on startup.

The core deliverable is a new `Scripts/exit_manager.py` module with an `ExitManager` class that owns a position registry, a new `position_registry` PostgreSQL table for persistence, and startup reconciliation wired into `initialize_bot()`. Exit checks must fire before any new entry evaluation in `execute_trade()`. The daily P&L circuit breaker (EXIT-05) and structured trade logging (OBS-01) round out the phase.

No new library dependencies are required. ATR is already computed by `Scripts/indicators.py`. IBKR order placement uses the existing `close_position()` function. SQLAlchemy is already the ORM. The entire phase is pure Python state management plus two targeted modifications to existing files (`trade.py`, `main.py`) and one new ORM model in `database.py`.

**Primary recommendation:** Build `ExitManager` as a new module with a PostgreSQL-backed registry, wire it into `execute_trade()` with exits-before-entries ordering, and add startup reconciliation in `initialize_bot()`. Replace the hardcoded stop/take-profit in `hourly_portfolio_scan()` with ExitManager calls.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EXIT-01 | Hard stop loss on every position — `entry_price - (2 * ATR)` at time of entry | ATR already in `indicators.py`; stop computed at register_entry(); stored in position_registry table |
| EXIT-02 | Take profit target — minimum 2:1 R:R = `entry_price + (4 * ATR)` | Computed alongside stop at register_entry(); same registry row |
| EXIT-03 | Trailing stop after 1x ATR profit — trails at 1.5x ATR below running high-water mark | ExitManager.check_exits() updates trailing_high per bar; trailing stop activates when profit >= 1x ATR |
| EXIT-04 | Position registry persisted to PostgreSQL; reconciled against IBKR on startup | New `position_registry` table via SQLAlchemy ORM; reconcile_from_ibkr() called in initialize_bot() |
| EXIT-05 | Halt new entries if daily P&L < -2% of account NAV; circuit breaker resets next day | daily_pnl_gate flag in ExitManager; checked in execute_trade() before entry branches only |
| OBS-01 | Structured DB record per trade: entry reason, exit reason, signal values, regime, sentiment at decision time | New `trade_log` table or extended dict passed to log_trade_performance(); exit_reason field added |
</phase_requirements>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | 2.x (already installed) | ORM for new position_registry and trade_log tables | Already the project ORM; `Base.metadata.create_all(engine)` pattern established in database.py |
| psycopg2 | already installed | PostgreSQL driver | Already in use; no change |
| dataclasses | stdlib | EntryRecord data structure | Zero dependency; clean typed container; matches project's use of plain Python data objects |
| datetime | stdlib | Entry time, trailing stop timestamps | Already used throughout codebase |
| ta (0.11.0) | already installed | ATR retrieval from computed indicator df | Already used in indicators.py for ATR computation |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ib_insync | already installed | close_position() market order for exit execution | Existing close_position() function reused unchanged for all exits |
| logging | stdlib | Structured exit reason logging | Already used project-wide; format consistent |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PostgreSQL persistence for registry | In-memory dict only | In-memory loses state on restart — Pitfall #3. Postgres is mandatory for EXIT-04. |
| SQLAlchemy ORM for position_registry | Raw SQL / psycopg2 direct | ORM is already the project pattern; consistency over micro-optimization |
| Python-side trailing stop management | IBKR trailing stop orders | IBKR trailing stops have known edge cases on paper accounts (confirmed in STACK.md); Python-side is more reliable and testable |

**Installation:** No new packages required. All dependencies are already present in the environment.

---

## Architecture Patterns

### Recommended Project Structure

```
Scripts/
├── exit_manager.py          # NEW: ExitManager class — hard stop, trailing, take profit, circuit breaker
├── trade.py                 # MODIFY: wire ExitManager into execute_trade(); exits before entries
├── database.py              # MODIFY: add PositionRegistry + TradeLog ORM models
├── main.py                  # MODIFY: call reconcile_from_ibkr() in initialize_bot()
└── config.py                # MODIFY: add HARD_STOP_ATR_MULT, TAKE_PROFIT_ATR_MULT, TRAILING_ATR_MULT, DAILY_LOSS_LIMIT_PCT
```

### Pattern 1: ExitManager as Stateful Singleton

**What:** `ExitManager` is instantiated once at module level in `exit_manager.py`. It owns the in-memory position registry (dict keyed by symbol) and syncs reads/writes to PostgreSQL. Matches the existing singleton pattern for `ml_position_sizer`, `performance_tracker`.

**When to use:** State must persist across the 30-second bar loop. Exit levels must survive between calls without being recomputed.

**Example:**
```python
# Scripts/exit_manager.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict

@dataclass
class EntryRecord:
    symbol: str
    direction: str          # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    atr_at_entry: float
    quantity: int
    stop_price: float       # entry_price - (2 * ATR)   [EXIT-01]
    target_price: float     # entry_price + (4 * ATR)   [EXIT-02]
    trailing_high: float    # highest close since entry (starts at entry_price)
    trailing_stop: Optional[float]  # None until 1x ATR profit reached [EXIT-03]

class ExitManager:
    def __init__(self):
        self.positions: Dict[str, EntryRecord] = {}
        self._daily_pnl: float = 0.0
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_date: Optional[str] = None

    def register_entry(self, symbol, direction, entry_price, atr, quantity):
        """Called immediately after a BUY/SHORT order fills."""
        record = EntryRecord(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(),
            atr_at_entry=atr,
            quantity=quantity,
            stop_price=entry_price - (2.0 * atr),   # EXIT-01
            target_price=entry_price + (4.0 * atr),  # EXIT-02
            trailing_high=entry_price,
            trailing_stop=None,
        )
        self.positions[symbol] = record
        self._persist_to_db(record)

    def check_exits(self, symbol, current_price) -> Optional[str]:
        """
        Returns exit reason string or None.
        Must be called BEFORE new entry evaluation for this symbol.
        """
        if symbol not in self.positions:
            return None
        rec = self.positions[symbol]

        # Update trailing high-water mark
        if current_price > rec.trailing_high:
            rec.trailing_high = current_price
            self._update_db_trailing_high(symbol, current_price)

        # Activate trailing stop once 1x ATR profit reached [EXIT-03]
        profit = current_price - rec.entry_price
        if rec.trailing_stop is None and profit >= rec.atr_at_entry:
            rec.trailing_stop = rec.trailing_high - (1.5 * rec.atr_at_entry)

        # Update trailing stop level
        if rec.trailing_stop is not None:
            new_trail = rec.trailing_high - (1.5 * rec.atr_at_entry)
            if new_trail > rec.trailing_stop:
                rec.trailing_stop = new_trail

        # Check exit conditions (hard stop takes priority)
        if current_price <= rec.stop_price:
            return 'HARD_STOP_LOSS'
        if rec.trailing_stop is not None and current_price <= rec.trailing_stop:
            return 'TRAILING_STOP'
        if current_price >= rec.target_price:
            return 'TAKE_PROFIT'
        return None

    def clear_position(self, symbol):
        """Called after close_position() executes."""
        if symbol in self.positions:
            del self.positions[symbol]
            self._delete_from_db(symbol)

    def is_circuit_breaker_active(self) -> bool:
        """Returns True if daily P&L has breached -2% NAV. [EXIT-05]"""
        today = datetime.now().strftime('%Y-%m-%d')
        if self._circuit_breaker_date != today:
            # New day — reset
            self._circuit_breaker_active = False
            self._daily_pnl = 0.0
            self._circuit_breaker_date = today
        return self._circuit_breaker_active

    def record_trade_pnl(self, pnl: float, account_nav: float):
        """Update daily P&L and trigger circuit breaker if needed."""
        self._daily_pnl += pnl
        if account_nav > 0 and (self._daily_pnl / account_nav) < -0.02:
            self._circuit_breaker_active = True
            logging.warning(f"Circuit breaker activated: daily P&L {self._daily_pnl:.2f} "
                            f"= {self._daily_pnl/account_nav:.2%} of NAV")

    def reconcile_from_ibkr(self, ib_positions, account_nav):
        """
        Startup reconciliation. [EXIT-04]
        Called in initialize_bot() after IBKR connect.
        Loads DB registry, then cross-checks against live IBKR positions.
        Drops DB records for positions IBKR no longer holds.
        Creates stub records (no stop/target) for IBKR positions not in DB.
        """
        ...

# Module-level singleton
exit_manager = ExitManager()
```

### Pattern 2: Exits-Before-Entries Ordering in execute_trade()

**What:** At the top of the per-symbol loop in `execute_trade()`, check `exit_manager.check_exits()` before evaluating `want_to_buy` / `want_to_short`. If an exit fires, execute it and `continue` to the next symbol. This ensures regime gates never suppress exits.

**When to use:** Every bar, for every symbol currently in `active_positions`.

**Example:**
```python
# Scripts/trade.py — in execute_trade(), inside the for symbol, prob loop
for symbol, prob in pred_results:
    # --- EXIT CHECK (unconditional — fires regardless of regime or circuit breaker) ---
    if symbol in active_positions:
        try:
            contract = Stock(symbol, 'SMART', 'USD')
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(1)
            current_price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
            ib.cancelMktData(contract)
        except Exception as e:
            logging.warning(f"Error getting price for exit check {symbol}: {e}")
            current_price = None

        if current_price:
            exit_signal = exit_manager.check_exits(symbol, current_price)
            if exit_signal:
                close_position(symbol)
                exit_manager.clear_position(symbol)
                del active_positions[symbol]
                log_trade_performance({
                    'symbol': symbol,
                    'action': exit_signal,        # 'HARD_STOP_LOSS', 'TRAILING_STOP', 'TAKE_PROFIT'
                    'exit_price': current_price,
                    'exit_time': datetime.now(),
                    'exit_reason': exit_signal,   # OBS-01
                })
                continue  # Skip entry evaluation for this symbol this bar

    # --- CIRCUIT BREAKER CHECK (entries only) --- [EXIT-05]
    if exit_manager.is_circuit_breaker_active():
        logging.info(f"Circuit breaker active — suppressing new entry for {symbol}")
        continue

    # --- REGIME + ENTRY LOGIC (unchanged from current code) ---
    ...
    if want_to_buy and symbol not in active_positions:
        qty = submit_ml_sized_order(symbol, 'buy', prob, entry_price)
        if qty > 0:
            active_positions[symbol] = 'long'
            exit_manager.register_entry(symbol, 'long', entry_price, atr_value, qty)
            log_trade_performance({..., 'entry_reason': f'prob={prob:.3f}', 'regime': market_regime})
```

### Pattern 3: PostgreSQL Position Registry Table

**What:** A new `PositionRegistry` SQLAlchemy model in `database.py`. One row per open position. Upserted on `register_entry()`, deleted on `clear_position()`. Loaded into memory on startup.

**Example:**
```python
# Scripts/database.py — new ORM model to add
class PositionRegistry(Base):
    __tablename__ = 'position_registry'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, unique=True)
    direction = Column(String(5), nullable=False)       # 'long' or 'short'
    entry_price = Column(Float, nullable=False)
    entry_time = Column(TIMESTAMP, nullable=False)
    atr_at_entry = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    stop_price = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    trailing_high = Column(Float, nullable=False)
    trailing_stop = Column(Float, nullable=True)        # NULL until activated
```

### Pattern 4: Structured Trade Log for OBS-01

**What:** `log_trade_performance()` already accepts a dict. Phase 2 adds required fields to every call: `exit_reason`, `entry_reason`, `regime_at_entry`, `sentiment_at_entry` (None for Phase 2 — sentiment comes in Phase 4), `signal_prob`. No new table needed for Phase 2 — fields are added to the existing dict that `TradingPerformanceMetrics` stores. A `TradeLog` DB table can be a separate ORM model persisted alongside.

**Example:**
```python
# Extended trade logging dict — passes to log_trade_performance()
log_trade_performance({
    'symbol': symbol,
    'action': 'BUY',
    'quantity': qty,
    'entry_price': entry_price,
    'entry_time': datetime.now(),
    'entry_reason': f'ml_prob={prob:.3f}',       # OBS-01
    'regime': market_regime,                       # OBS-01
    'sentiment_score': None,                       # Phase 4 fills this
    'prediction_confidence': prob,
    'stop_price': entry_record.stop_price,
    'target_price': entry_record.target_price,
})

# On exit:
log_trade_performance({
    'symbol': symbol,
    'action': exit_signal,
    'exit_price': current_price,
    'exit_time': datetime.now(),
    'exit_reason': exit_signal,                    # OBS-01: 'HARD_STOP_LOSS' | 'TRAILING_STOP' | 'TAKE_PROFIT' | 'SIGNAL'
})
```

### Pattern 5: Startup Reconciliation in initialize_bot()

**What:** After IBKR connect and symbol load, call `exit_manager.reconcile_from_ibkr()`. This loads DB registry rows into memory, then queries live IBKR positions to find divergence.

**Example:**
```python
# Scripts/main.py — in initialize_bot(), after fetch_and_load_symbols()
from .exit_manager import exit_manager

def initialize_bot():
    symbols = fetch_and_load_symbols(ib)
    ...
    # Reconcile exit registry against live IBKR positions [EXIT-04]
    logging.info("Reconciling position registry with IBKR...")
    account_summary = ib.accountSummary()
    account_nav = next((float(x.value) for x in account_summary if x.tag == 'NetLiquidation'), 0.0)
    live_positions = ib.positions()
    exit_manager.reconcile_from_ibkr(live_positions, account_nav)
    logging.info(f"Reconciliation complete: {len(exit_manager.positions)} positions tracked")
```

### Anti-Patterns to Avoid

- **Inline stop logic in execute_trade():** `execute_trade()` is stateless per call. It has no memory of entry prices between calls. Any stop check written inline using local variables will silently use wrong prices. Use `ExitManager.position_registry` as the only source of truth for entry prices.
- **Regime gate applied to exits:** If the regime is 'volatile' and the bot holds a losing position, suppressing the SELL means the original failure mode returns. Regime gate applies ONLY to entry branches. Exits are unconditional.
- **Circuit breaker suppressing exits:** EXIT-05 halts new entries only. A position with a triggered stop loss must still be closed regardless of the circuit breaker state.
- **Not persisting the registry:** Position state held only in memory is lost on restart. If the bot crashes during market hours and restarts, it will re-enter positions it is already holding. EXIT-04 is not optional.
- **ATR fetched at exit time instead of entry time:** Trailing stop multipliers must use `atr_at_entry`, not the current ATR, so the initial stop distance is stable. Current ATR can shrink after a volatile move, making the stop appear to widen — use the frozen entry-time ATR.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ATR calculation | Custom rolling ATR | Existing `compute_technical_indicators(df)` returns ATR column | Already computed and tested; `ta` library handles edge cases |
| IBKR order for exit | Custom order type | Existing `close_position(symbol)` function | Already handles SELL vs BUY side, after-hours flag, logging |
| DB session management | Direct psycopg2 | Existing `get_session()` context manager | Auto-commit, rollback, close already wired |
| ORM table creation | Manual DDL | `Base.metadata.create_all(engine)` already called at module load | New models auto-create on first import |
| Trailing stop formula | Custom algorithm | Chandelier Exit (trailing_high - 1.5*ATR) as documented in STACK.md | Industry standard; ATR multiplier 1.5 appropriate for intraday |

**Key insight:** Every infrastructure concern (DB sessions, IBKR orders, ATR, logging) is already solved. Phase 2 is purely application logic wiring on top of existing infrastructure.

---

## Common Pitfalls

### Pitfall 1: Exit Check Fires After Entry Evaluation

**What goes wrong:** New entry signal executes (BUY), then on the same bar the exit check fires for that same symbol, immediately closing the just-opened position.

**Why it happens:** Wrong ordering — exit check placed after entry logic instead of before.

**How to avoid:** Exit check must be the FIRST action per symbol in the `for symbol, prob in pred_results` loop. Use `continue` after closing to skip all entry evaluation for that symbol on that bar.

**Warning signs:** Log shows BUY immediately followed by HARD_STOP_LOSS for the same symbol on consecutive lines.

### Pitfall 2: ATR Is NaN at Entry Time

**What goes wrong:** `register_entry()` receives `atr=NaN`, so `stop_price = entry_price - (2 * NaN) = NaN`. The stop check `current_price <= NaN` always evaluates False. Position is never stopped out.

**Why it happens:** ATR requires 14 bars of data. If fewer than 14 bars exist, `ta` returns NaN for early rows. The latest row may still be NaN if data is sparse.

**How to avoid:** Guard in `register_entry()`: `if atr is None or pd.isna(atr) or atr <= 0: log warning and use fallback (e.g., 0.5% of entry_price)`. Never store a NaN stop.

**Warning signs:** `stop_price` column in `position_registry` table contains NULL or NaN values.

### Pitfall 3: Position State Lost on Restart (the Core EXIT-04 Concern)

**What goes wrong:** Bot restarts during market hours with open positions. `active_positions` dict is empty. `exit_manager.positions` is empty. Bot sees no existing positions and re-enters them, doubling exposure. No stops are active for the original positions.

**Why it happens:** In-memory state only. `active_positions` is a module-level dict that does not survive process restart.

**How to avoid:** On startup, `reconcile_from_ibkr()` must: (1) load all rows from `position_registry` table into memory, (2) query live IBKR positions, (3) for each IBKR position not in DB registry, create a stub record with conservative stops (current_price * 0.98 as stop, current_price * 1.04 as target), (4) for each DB record where IBKR shows no position, delete the stale record.

**Warning signs:** Bot logs show BUY for a symbol that is already visible in IBKR positions list.

### Pitfall 4: hourly_portfolio_scan() Has Its Own Hardcoded Stops

**What goes wrong:** Even after ExitManager is wired into `execute_trade()`, the hourly scan still uses its own hardcoded -15%/+25% logic (main.py lines 205-210). Two competing exit systems exist — they can race, log duplicates, and leave `position_registry` out of sync.

**Why it happens:** The hourly scan was written independently with inline stop logic before ExitManager existed.

**How to avoid:** Replace the hardcoded stop/take-profit block in `hourly_portfolio_scan()` with `exit_manager.check_exits(symbol, current_price)`. The scan becomes a secondary enforcement layer, not an independent system.

**Warning signs:** Trade log shows SELL exits with `exit_reason=None` or action='SELL' from the hourly scan that bypass ExitManager.

### Pitfall 5: Circuit Breaker Checked Before Exit — Suppresses Stops

**What goes wrong:** Circuit breaker fires, then later in the same bar the bot hits the exit check. Because of wrong ordering, the circuit breaker `continue` happens before the exit check, and a position that should be stopped out is held overnight.

**Why it happens:** Putting circuit breaker check too early in the per-symbol loop.

**How to avoid:** Ordering within the per-symbol loop must be strictly: (1) exit check → (2) circuit breaker check → (3) entry logic. Circuit breaker only affects step (3).

### Pitfall 6: Daily P&L Not Reset Between Days

**What goes wrong:** Circuit breaker triggers on Day 1. Bot never recovers because `_daily_pnl` is never reset and `_circuit_breaker_active` stays True indefinitely.

**Why it happens:** No date-based reset logic in circuit breaker state.

**How to avoid:** `is_circuit_breaker_active()` compares `_circuit_breaker_date` to today's date string. If date differs, reset both `_daily_pnl` and `_circuit_breaker_active` before returning. This is the simplest correct implementation.

---

## Code Examples

### ATR Extraction from Existing Indicators

```python
# Source: direct analysis of Scripts/indicators.py + Scripts/trade.py pattern
# ATR is already in the indicators df computed by compute_technical_indicators()
# Extract it at the point of entry in execute_trade()

with get_session() as session:
    recent_data = session.query(StockData).filter(
        StockData.symbol == symbol
    ).order_by(StockData.timestamp.desc()).limit(100).all()
    df = pd.DataFrame([...])

indicators = compute_technical_indicators(df)
atr_value = indicators['atr'].iloc[-1]  # float; may be NaN for sparse data

# Guard against NaN before registering entry
if pd.isna(atr_value) or atr_value <= 0:
    atr_value = entry_price * 0.005  # fallback: 0.5% of price as ATR estimate
    logging.warning(f"ATR unavailable for {symbol}, using fallback: {atr_value:.4f}")

exit_manager.register_entry(symbol, 'long', entry_price, atr_value, qty)
```

### PositionRegistry DB Persist / Load Pattern

```python
# Source: pattern from existing database.py insert_historical_data()

def _persist_to_db(self, record: EntryRecord):
    with get_session() as session:
        stmt = pg_insert(PositionRegistry).values(
            symbol=record.symbol,
            direction=record.direction,
            entry_price=record.entry_price,
            entry_time=record.entry_time,
            atr_at_entry=record.atr_at_entry,
            quantity=record.quantity,
            stop_price=record.stop_price,
            target_price=record.target_price,
            trailing_high=record.trailing_high,
            trailing_stop=record.trailing_stop,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['symbol'],
            set_={
                'stop_price': stmt.excluded.stop_price,
                'target_price': stmt.excluded.target_price,
                'trailing_high': stmt.excluded.trailing_high,
                'trailing_stop': stmt.excluded.trailing_stop,
            }
        )
        session.execute(stmt)

def _load_from_db(self):
    """Called once at startup before reconcile."""
    with get_session() as session:
        rows = session.query(PositionRegistry).all()
        for row in rows:
            self.positions[row.symbol] = EntryRecord(
                symbol=row.symbol,
                direction=row.direction,
                entry_price=row.entry_price,
                entry_time=row.entry_time,
                atr_at_entry=row.atr_at_entry,
                quantity=row.quantity,
                stop_price=row.stop_price,
                target_price=row.target_price,
                trailing_high=row.trailing_high,
                trailing_stop=row.trailing_stop,
            )
```

### Config Constants to Add

```python
# Scripts/config.py — new constants for Phase 2
HARD_STOP_ATR_MULT   = float(os.getenv("HARD_STOP_ATR_MULT", "2.0"))    # EXIT-01: stop = entry - 2*ATR
TAKE_PROFIT_ATR_MULT = float(os.getenv("TAKE_PROFIT_ATR_MULT", "4.0"))  # EXIT-02: target = entry + 4*ATR
TRAILING_ATR_MULT    = float(os.getenv("TRAILING_ATR_MULT", "1.5"))     # EXIT-03: trail = high - 1.5*ATR
TRAILING_TRIGGER_ATR = float(os.getenv("TRAILING_TRIGGER_ATR", "1.0"))  # EXIT-03: activate after 1x ATR profit
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.02")) # EXIT-05: 2% NAV daily loss limit
```

---

## Key Integration Points (What Must Change in Existing Files)

### Scripts/trade.py — execute_trade()

Current state: No exit logic. `active_positions` stores `'long'`/`'short'` only. No ATR at entry. No entry_reason in log.

Required changes:
1. Import `exit_manager` from `.exit_manager`
2. At top of per-symbol loop: add exit check block (exits-before-entries)
3. After circuit breaker check: add circuit breaker gate (entries only)
4. In BUY branch: extract `atr_value` from indicators, call `exit_manager.register_entry()`
5. In SHORT branch: same — extract ATR, register entry with direction='short'
6. In SELL/COVER branches: call `exit_manager.clear_position(symbol)` after `close_position()`
7. Add `entry_reason` and `exit_reason` fields to all `log_trade_performance()` calls

### Scripts/main.py — initialize_bot()

Current state: No reconciliation. No ExitManager awareness.

Required changes:
1. Import `exit_manager` from `.exit_manager`
2. After `fetch_and_load_symbols()`: call `exit_manager.reconcile_from_ibkr(ib.positions(), nav)`
3. Wire `active_positions` (from trade.py) sync so it reflects reconciled positions

### Scripts/main.py — hourly_portfolio_scan()

Current state: Lines 205-210 have hardcoded `position_return < -0.15` and `position_return > 0.25` thresholds.

Required changes:
1. Replace hardcoded stop/take-profit logic with `exit_manager.check_exits(symbol, current_price)`
2. When exit fires from scan: call `exit_manager.clear_position(symbol)`, pass `exit_reason` to log

### Scripts/database.py

Current state: Two ORM models — `StockData`, `Portfolio`. `Base.metadata.create_all(engine)` at module load.

Required changes:
1. Add `PositionRegistry` ORM model (unique on symbol)
2. Optionally add `TradeLog` ORM model for OBS-01 structured records
3. No migration needed — `create_all()` adds new tables automatically on next run

### Scripts/config.py

Required changes: Add 5 new constants (HARD_STOP_ATR_MULT, TAKE_PROFIT_ATR_MULT, TRAILING_ATR_MULT, TRAILING_TRIGGER_ATR, DAILY_LOSS_LIMIT_PCT) with env var overrides and sensible defaults.

---

## Open Questions

1. **ATR column name in indicator output**
   - What we know: `compute_technical_indicators()` in `indicators.py` computes ATR using the `ta` library
   - What's unclear: The exact column name in the returned DataFrame (likely `'atr'` but must be verified against `indicators.py` output)
   - Recommendation: Read `Scripts/indicators.py` during planning to confirm column name before writing ExitManager code

2. **active_positions dict sync with reconciled positions**
   - What we know: `active_positions` in `trade.py` is a module-level global dict. `reconcile_from_ibkr()` rebuilds `exit_manager.positions` from DB + IBKR
   - What's unclear: Whether `active_positions` also needs rebuilding on startup, and how to do it cleanly given it lives in a different module
   - Recommendation: During reconciliation, call `trade.active_positions[symbol] = direction` for each reconciled position to keep both dicts in sync. Import `active_positions` directly or expose a setter function

3. **account_nav for circuit breaker**
   - What we know: NAV is available via `ib.accountSummary()` tag `'NetLiquidation'`
   - What's unclear: Whether to fetch NAV fresh on each trade close (expensive) or cache it from the last `trading_loop()` poll
   - Recommendation: Cache the NAV in ExitManager, refreshed when `record_trade_pnl()` is called with the value passed in from execute_trade / hourly scan

4. **Short position trailing stop direction**
   - What we know: EXIT-01 through EXIT-03 are described in terms of long positions (subtract ATR from entry)
   - What's unclear: Whether shorts should use a mirrored formula (stop = entry + 2*ATR, target = entry - 4*ATR, trailing low tracks low-water mark)
   - Recommendation: Yes — implement mirrored short logic in check_exits(). The direction field on EntryRecord drives which formula applies.

---

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis: `Scripts/trade.py` — full source read 2026-02-27 (active_positions structure, execute_trade loop, log_trade_performance call sites)
- Direct codebase analysis: `Scripts/database.py` — full source read 2026-02-27 (ORM models, Base pattern, get_session, pg_insert upsert pattern)
- Direct codebase analysis: `Scripts/main.py` — full source read 2026-02-27 (initialize_bot flow, hourly_portfolio_scan hardcoded stops at lines 205-210, reconciliation gap)
- `.planning/research/ARCHITECTURE.md` — ExitManager design and integration points (2026-02-27)
- `.planning/research/STACK.md` — ATR exit formula, IBKR bracket order vs Python-side trailing stop tradeoff (2026-02-27)
- `.planning/REQUIREMENTS.md` — EXIT-01 through EXIT-05, OBS-01 exact specifications (2026-02-27)

### Secondary (MEDIUM confidence)
- STACK.md sourced: Chandelier Exit formula (trailing_high - ATR * multiplier) — multiple trading algo sources (2026-02-27)
- STACK.md sourced: IBKR trailing stop paper account edge cases — confirmed via community report

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new dependencies; all libraries already installed and in use
- Architecture: HIGH — all integration points verified against actual source code line-by-line
- Pitfalls: HIGH — Pitfalls 1-4 derived directly from reading the existing code; Pitfalls 5-6 are logic-derived from the requirements

**Research date:** 2026-02-27
**Valid until:** 2026-03-29 (stable domain — no external API or library version changes affect this phase)
