---
phase: 02-exit-management
plan: "01"
subsystem: database
tags: [sqlalchemy, postgresql, exit-management, position-registry, atr, circuit-breaker]

requires:
  - phase: 01-infrastructure-unblock
    provides: Working database connection and ORM base

provides:
  - ExitManager class with register_entry, check_exits, clear_position, record_trade_pnl, reconcile_from_ibkr
  - EntryRecord dataclass for in-memory position state
  - PositionRegistry ORM model (position_registry table)
  - TradeLog ORM model (trade_log table)
  - Five ATR-multiplier config constants (HARD_STOP_ATR_MULT, TAKE_PROFIT_ATR_MULT, TRAILING_ATR_MULT, TRAILING_TRIGGER_ATR, DAILY_LOSS_LIMIT_PCT)
  - Startup reconciliation wired into initialize_bot()

affects: [02-02-PLAN, 02-03-PLAN, 02-04-PLAN, trade.py, main.py]

tech-stack:
  added: []
  patterns:
    - "Module-level singleton: exit_manager = ExitManager() imported everywhere"
    - "Local function imports inside initialize_bot() to avoid circular imports"
    - "PostgreSQL upsert via on_conflict_do_update on symbol column"
    - "ATR NaN guard: fallback to 0.5% of entry_price when ATR unavailable"
    - "Exits-before-entries rule: check_exits() called before entry evaluation per bar"

key-files:
  created:
    - Scripts/exit_manager.py
  modified:
    - Scripts/database.py
    - Scripts/config.py
    - Scripts/main.py

key-decisions:
  - "Created Scripts/exit_manager.py as a dedicated module for position registry and exit rules."
  - "Used a module-level singleton exit_manager for centralized state management."
  - "Added PositionRegistry and TradeLog tables for persistence and observability."
  - "Implemented startup reconciliation to recover state from IBKR live positions."
  - "ATR frozen at entry time for all trailing/stop math — not recalculated each bar."

patterns-established:
  - "Pattern 1: exit_manager singleton — import from Scripts.exit_manager in all trade execution code"
  - "Pattern 2: check_exits() before entry — always call check_exits() first in per-bar logic"
  - "Pattern 3: clear_position() after close — always call after close_position() to clean registry"

requirements-completed: [EXIT-01, EXIT-02, EXIT-03, EXIT-04, EXIT-05, OBS-01]

duration: 3min
completed: 2026-02-27
---

# Phase 02 Plan 01: ExitManager Foundation Summary

**ExitManager singleton with ATR-based hard stop, take profit, and trailing stop; PositionRegistry + TradeLog ORM tables; startup IBKR reconciliation wired into initialize_bot()**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-27T00:00:00Z
- **Completed:** 2026-02-27T00:03:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created `Scripts/exit_manager.py` with `ExitManager` class, `EntryRecord` dataclass, and `exit_manager` singleton
- Implemented all exit rules: hard stop at 2x ATR (EXIT-01), take profit at 4x ATR (EXIT-02), trailing stop activating at 1x ATR profit ratcheting at 1.5x ATR (EXIT-03)
- Added `PositionRegistry` and `TradeLog` ORM models to `Scripts/database.py` with auto-create via `Base.metadata.create_all`
- Added 5 env-overridable ATR multiplier constants to `Scripts/config.py`
- Wired `reconcile_from_ibkr()` into `initialize_bot()` in `Scripts/main.py` with active_positions sync

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Scripts/exit_manager.py** - (feat: ExitManager class with ATR exits and DB persistence)
2. **Task 2: DB models + config constants + main.py wiring** - (feat: PositionRegistry/TradeLog ORM, config constants, reconciliation)

## Files Created/Modified
- `Scripts/exit_manager.py` - ExitManager class, EntryRecord dataclass, module-level singleton
- `Scripts/database.py` - Added PositionRegistry and TradeLog ORM models
- `Scripts/config.py` - Added HARD_STOP_ATR_MULT, TAKE_PROFIT_ATR_MULT, TRAILING_ATR_MULT, TRAILING_TRIGGER_ATR, DAILY_LOSS_LIMIT_PCT
- `Scripts/main.py` - Reconciliation block in initialize_bot() syncing active_positions

## Decisions Made
- Used module-level singleton `exit_manager` so all modules share one instance without passing it as parameter
- ATR frozen at entry time (not recalculated) ensures consistent stop/target math throughout position lifetime
- Local import of exit_manager inside initialize_bot() avoids circular import at module load time
- Upsert (on_conflict_do_update on symbol) means re-registering an existing symbol overwrites rather than errors

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required. DB tables auto-created by SQLAlchemy on next bot start.

## Next Phase Readiness
- ExitManager foundation complete; plans 02-02 through 02-04 can call `register_entry`, `check_exits`, `clear_position`, `record_trade_pnl`
- `trade_log` table ready for structured trade logging (OBS-01) in 02-03
- No blockers for subsequent plans

## Self-Check: PASSED
- `Scripts/exit_manager.py` confirmed present
- `Scripts/database.py` contains PositionRegistry (line 61) and TradeLog (line 78) classes
- `Scripts/config.py` contains all 5 constants with correct defaults (2.0, 4.0, 1.5, 1.0, 0.02) - verified by assertion
- `Scripts/main.py` contains reconciliation block (lines 96-109)

---
*Phase: 02-exit-management*
*Completed: 2026-02-27*
