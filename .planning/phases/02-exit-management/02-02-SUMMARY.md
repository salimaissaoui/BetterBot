---
phase: "02"
plan: "02"
subsystem: exit-management
tags: [exit-manager, circuit-breaker, trade-registration, obs-01, tradelog]
dependency_graph:
  requires: [02-01]
  provides: [EXIT-01, EXIT-02, EXIT-03, EXIT-04, EXIT-05, OBS-01]
  affects: [Scripts/trade.py]
tech_stack:
  added: []
  patterns: [exits-before-entries, circuit-breaker-gate, atr-at-entry-frozen]
key_files:
  modified: [Scripts/trade.py]
decisions:
  - "Exits-before-entries enforced: exit check block placed as FIRST action in per-symbol loop, before threshold reset or regime detection"
  - "Circuit breaker uses continue (not return) — only suppresses new entries, never existing positions"
  - "ATR extracted from DB + compute_technical_indicators() at entry time and frozen in EntryRecord — not recomputed per bar"
  - "Rule 1 auto-fix: SHORT branch register_entry/log calls were incorrectly inside except block; moved to unconditional post-try scope"
metrics:
  duration: "4 min"
  completed: "2026-02-27"
  tasks_completed: 1
  files_modified: 1
---

# Phase 02 Plan 02: Wire ExitManager into execute_trade() Summary

ExitManager fully wired into execute_trade() with ATR-based stop/target registration, exits-before-entries ordering, circuit breaker entry gating, and TradeLog DB writes for OBS-01.

## Tasks Completed

| Task | Description | Commit | Status |
|------|-------------|--------|--------|
| 02-02-T1 | Add exit check, circuit breaker, entry registration, and TradeLog logging | 67067b0 | Complete |

## What Was Built

`Scripts/trade.py` `execute_trade()` now enforces:

1. **Exit check (first in loop):** If symbol is in `active_positions`, fetch current price, call `exit_manager.check_exits()`. On signal: `close_position()` -> `clear_position()` -> `record_trade_pnl()` -> `_log_exit_to_db()` -> `del active_positions[symbol]` -> `continue`. Fires regardless of regime or circuit breaker.

2. **Circuit breaker gate (second in loop):** `exit_manager.is_circuit_breaker_active()` — if True, log and `continue` for the symbol. Does NOT affect exits.

3. **BUY branch entry registration:** After `qty > 0` confirmed, extracts ATR from DB rows via `compute_technical_indicators()`, calls `exit_manager.register_entry(symbol, 'long', entry_price, _atr_val, qty)` and `_log_entry_to_db()`.

4. **SHORT branch entry registration:** Identical pattern to BUY but with `direction='short'`.

5. **SELL and COVER branches:** Both call `exit_manager.clear_position(symbol)` and include `'exit_reason': 'SIGNAL'` in `log_trade_performance()`.

6. **Helper functions:** `_log_exit_to_db()` and `_log_entry_to_db()` write structured rows to the `TradeLog` DB table (OBS-01).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] SHORT branch register_entry called inside except block**
- **Found during:** Task 02-02-T1 (code review of existing implementation)
- **Issue:** `exit_manager.register_entry()`, `_log_entry_to_db()`, and `log_trade_performance()` in the SHORT branch were indented under `except Exception as _atr_err:` — meaning they only executed when ATR extraction raised an exception. On successful ATR extraction, the SHORT entry was never registered with ExitManager, leaving no stop/target computed and no TradeLog row written.
- **Fix:** Dedented the three call blocks one level — now at the same indentation as the `try/except`, executing unconditionally after ATR extraction (whether or not it succeeded).
- **Files modified:** Scripts/trade.py (lines 458-474)
- **Commit:** 67067b0

## Self-Check: PASSED

- Scripts/trade.py: modified and committed
- Commit 67067b0: verified present
- All plan must_haves met:
  - Exit check block appears BEFORE threshold reset: confirmed (lines 225-258 vs reset at line 278)
  - Circuit breaker uses continue, does not close positions: confirmed (lines 263-265)
  - register_entry called in BUY branch after qty > 0: confirmed (line 373)
  - register_entry called in SHORT branch with direction='short': confirmed (line 461, fixed)
  - SELL and COVER call clear_position: confirmed (lines 406, 493)
  - TradeLog rows written on every entry and exit: confirmed via _log_entry_to_db and _log_exit_to_db
  - ATR extracted from DB+indicators using 'atr' column: confirmed (lines 355-371, 443-457)
