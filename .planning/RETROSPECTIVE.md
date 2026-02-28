# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

---

## Milestone: v1.0 — MVP

**Shipped:** 2026-02-27
**Phases:** 5 | **Plans:** 12 | **Sessions:** 1 (1-day sprint)

### What Was Built
- `ExitManager` with hard stop (2×ATR), trailing stop (1.5×ATR below HWM, activates at 1×ATR profit), take-profit (4×ATR), circuit breaker (-2% NAV/day), and full PostgreSQL persistence via `PositionRegistry`/`TradeLog`
- KMeans regime detector wired at startup with `joblib` persistence + VIX-based position-size scaling (100%/50%/25%)
- `SentimentAnalyzer` using Yahoo Finance RSS + VADER with 15-min TTL cache, wired as entry gate (suppress if score < -0.05)
- `vix_at_decision` column added to `TradeLog`; structured SQL queries prepared for 14-day validation
- Resolved merge conflict in `main.py` and moved hardcoded AWS RDS credentials to env vars

### What Worked
- **Phase ordering was correct**: fixing the merge conflict first (Phase 1) unblocked all subsequent work without interference
- **Singleton pattern for ExitManager**: module-level `exit_manager` singleton made wiring clean — no dependency injection needed across `trade.py`, `main.py`
- **Offline smoke tests**: `verify_phase2.py`, `verify_phase3_01.py` etc. caught logic issues (notably: `record_trade_pnl()` circuit breaker date initialization) before live testing
- **Additive phases**: each phase extended the system without breaking prior work — sentiment and VIX gating layered on cleanly

### What Was Inefficient
- **REQUIREMENTS.md not updated after Phase 2**: RISK-01/02, SENT-01/02, VAL-01 remained unchecked even after implementation — created false gap at milestone close
- **STATE.md accumulated stale pending todos**: "Phase 3: Wire fit_regime_detector()" remained after Phase 3 completed, muddying the state
- **02-05 SUMMARY.md never created**: Human verification plan had no summary, causing the GSD tools to report phase 2 as incomplete even though all code was shipped in the same commit

### Patterns Established
- `Scripts/` uses relative imports throughout (`from .module import ...`) — absolute imports fail when running as a package
- All new entry gates follow the pattern: fetch → cache → gate → log suppression reason
- Verification scripts (`verify_phase*.py`) at root level serve as regression tests for each phase

### Key Lessons
1. **Update REQUIREMENTS.md in the same commit as the phase code** — don't defer to planning docs; stale requirements create noise at milestone close
2. **Create SUMMARY.md for human verification plans** — even a 3-line "verified by user" summary prevents false incomplete signals in GSD tools
3. **Smoke tests before wiring** — writing `verify_phase*.py` before wiring to `execute_trade()` caught the circuit breaker initialization bug that would have silently reset same-day limits

### Cost Observations
- Model: Sonnet 4.6 throughout
- Sessions: 1 (executed all 5 phases in a single session from a cold start)
- Notable: All implementation fit in one session due to tight phase scoping — no context resets needed mid-milestone

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 MVP | 1 | 5 | First milestone — established patterns |

### Cumulative Quality

| Milestone | Smoke Tests | Coverage | New Modules |
|-----------|-------------|----------|-------------|
| v1.0 | 5 offline verify scripts | Manual | exit_manager.py, news.py |

### Top Lessons (Verified Across Milestones)

1. Update tracking docs (REQUIREMENTS.md, STATE.md) in the same commit as code — deferred updates cause milestone-close noise
2. Write smoke tests per phase before wiring to production path — catches logic bugs before integration
