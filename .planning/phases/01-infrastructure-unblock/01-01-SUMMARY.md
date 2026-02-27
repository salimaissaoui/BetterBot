---
phase: 01-infrastructure-unblock
plan: 01
subsystem: infra
tags: [python, imports, git-conflict]

# Dependency graph
requires: []
provides:
  - Scripts/main.py is conflict-free and syntactically valid Python
  - Both get_session/StockData and relative .data_fetch import are present
affects:
  - 01-02 (BUG-02 credentials fix can now import main.py safely)
  - all subsequent phases that depend on the bot starting

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Relative imports only inside Scripts package (from .module import ..., never from Scripts.module import ...)"

key-files:
  created: []
  modified:
    - Scripts/main.py

key-decisions:
  - "Kept get_session and StockData in database import — both used in hourly_portfolio_scan lines ~158 and ~277"
  - "Used .data_fetch (relative) not Scripts.data_fetch (absolute) — absolute import fails with ModuleNotFoundError when running as package"

patterns-established:
  - "Scripts package must use relative imports (from .module import ...) throughout"

requirements-completed: [BUG-01]

# Metrics
duration: 1min
completed: 2026-02-27
---

# Phase 1 Plan 01: Resolve Merge Conflict in Scripts/main.py Summary

**Removed 7-line git merge conflict block in Scripts/main.py lines 11-17, replacing it with two correct relative import lines that include get_session, StockData, and relative .data_fetch**

## Performance

- **Duration:** 1 min
- **Started:** 2026-02-27T21:56:33Z
- **Completed:** 2026-02-27T21:57:40Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Removed all three conflict marker lines (`<<<<<<< Updated upstream`, `=======`, `>>>>>>> Stashed changes`) from Scripts/main.py
- Resolved import to keep `get_session` and `StockData` from the stashed side (both used inside `hourly_portfolio_scan`)
- Resolved import to use `.data_fetch` relative syntax from the upstream side (avoids `ModuleNotFoundError` when run as a package)
- File now passes `ast.parse()` with no SyntaxError

## Task Commits

Each task was committed atomically:

1. **Task 1: Resolve merge conflict in Scripts/main.py** - `8ae518f` (fix)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `Scripts/main.py` - Conflict markers removed; import block at lines 11-12 now reads the correct two relative import lines

## Decisions Made

- Kept `get_session` and `StockData` in the `.database` import: both names are referenced in `hourly_portfolio_scan` (lines ~158, ~277) and dropping them would cause a `NameError` at runtime.
- Used `.data_fetch` (relative) rather than `Scripts.data_fetch` (absolute): the absolute form fails with `ModuleNotFoundError` when the bot is executed as a Python package module.

## Deviations from Plan

None - plan executed exactly as written.

Note: The plan's verification command `python -c "import ast; ast.parse(open('Scripts/main.py').read())"` produces a `UnicodeDecodeError` on Windows when the system locale is cp1252, because the file contains UTF-8 bytes for emoji characters (`⚠️`) on lines 490 and 492. This is a pre-existing condition unrelated to the conflict fix. The file parses cleanly when the encoding is specified explicitly: `python -c "import ast; ast.parse(open('Scripts/main.py', encoding='utf-8').read()); print('PASS')"`. This does not affect runtime — Python reads source files as UTF-8 by default.

## Issues Encountered

- Windows default locale (cp1252) prevented the bare `open('Scripts/main.py')` syntax in the ast.parse verification command from reading the file. Resolved by adding `encoding='utf-8'`. The underlying syntax of main.py is valid; this is a Windows shell encoding edge case only.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- BUG-01 is resolved: `Scripts/main.py` has no merge conflict markers and is syntactically valid Python.
- BUG-02 (credentials / config.py hardcoded password) can now proceed — plan 01-02 is unblocked.
- The CRITICAL blocker noted in STATE.md (AWS RDS password in git history) should still be addressed before any new commits that touch config.py.

---
*Phase: 01-infrastructure-unblock*
*Completed: 2026-02-27*
