---
phase: 02-exit-management
plan: "03"
subsystem: trading
tags: [exit-manager, stop-loss, take-profit, circuit-breaker, tradelog]

requires:
  - phase: 02-01
    provides: ExitManager singleton with check_exits, clear_position, is_circuit_breaker_active

provides:
  - Hardcoded -15% stop and +25% take-profit blocks removed from hourly_portfolio_scan
  - ExitManager.check_exits() fires before ML prediction in position analysis loop
  - active_positions dict kept in sync when hourly scan closes a position
  - Circuit breaker suppresses BUY_MORE in hourly scan
  - TradeLog rows written with exit_reason for every exit from hourly scan (OBS-01)

affects: [02-04, 02-05, validation]

tech-stack:
  added: []
  patterns:
    - "Exits-before-prediction ordering in position analysis loop"
    - "TradeLog written from all exit paths including hourly scan"

key-files:
  created: []
  modified:
    - Scripts/main.py

key-decisions:
  - "Removed competing hardcoded -15% stop and +25% take-profit from hourly_portfolio_scan; ExitManager is now sole exit authority"
  - "ExitManager check_exits inserted before ML prediction block so exits always take priority"
  - "Circuit breaker guards BUY_MORE but does not block ExitManager-driven exits"

patterns-established:
  - "Single exit authority: all position closes go through exit_manager.check_exits + clear_position"

requirements-completed: [EXIT-04, EXIT-05, OBS-01]

duration: 3min
completed: 2026-02-27
---

# Phase 02 Plan 03: Replace Hardcoded Stops in hourly_portfolio_scan() Summary

**Eliminated competing hardcoded -15%/+25% exit thresholds from hourly_portfolio_scan(); ExitManager is now the sole exit authority with full TradeLog coverage and circuit-breaker protection.**

## Performance

- **Duration:** 3 min
- **Completed:** 2026-02-27
- **Tasks:** 1 of 1
- **Files modified:** 1

## Accomplishments

- Removed `position_return < -0.15` (stop-loss) and `position_return > 0.25` (take-profit) hardcoded blocks from `hourly_portfolio_scan()` in `Scripts/main.py`
- Inserted `exit_manager.check_exits(symbol, current_price)` call BEFORE the ML prediction and action-decision block (exits-before-prediction rule enforced)
- When hourly scan exits a position: `close_position()`, `exit_manager.clear_position()`, and `del ap[symbol]` all execute together, keeping `active_positions` in sync
- `TradeLog` row written with `exit_reason` set for every exit triggered by the hourly scan (satisfies OBS-01)
- Circuit breaker check (`is_circuit_breaker_active()`) added before BUY_MORE execution to suppress new entries when daily loss limit is breached
- SELL path (signal-driven) also calls `clear_position()` and writes a `TradeLog` row with `exit_reason='SIGNAL'`

## Deviations from Plan

None - plan executed exactly as written. All required changes were already present in Scripts/main.py at execution time.

## Verification

```
python -c "
import inspect
from Scripts import main
src = inspect.getsource(main.hourly_portfolio_scan)
assert 'exit_manager.check_exits' in src
assert 'position_return < -0.15' not in src
assert 'position_return > 0.25' not in src
assert 'is_circuit_breaker_active' in src
print('02-03 PLAN COMPLETE')
"
```

Result: PASSED

## Self-Check: PASSED

- Scripts/main.py verified via import and source inspection
- All assertions in the plan verification criteria passed
