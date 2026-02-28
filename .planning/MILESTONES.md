# Milestones

## v1.0 MVP (Shipped: 2026-02-28)

**Phases completed:** 5 phases, 12 plans
**Git range:** fix(01-01) → docs(02) save phase 2 execution state
**Python LOC:** ~5,538 | **Files changed:** ~65

**Key accomplishments:**
1. Resolved git merge conflict in `Scripts/main.py` and removed hardcoded AWS RDS credentials from `config.py` — bot can now start cleanly
2. Built `ExitManager` with hard stop (2×ATR), take-profit (4×ATR), trailing stop (1.5×ATR), circuit breaker (-2% NAV/day), and `PositionRegistry`/`TradeLog` PostgreSQL persistence
3. Replaced all hardcoded stop/TP blocks in `hourly_portfolio_scan()` with `ExitManager` calls; added startup reconciliation against live IBKR positions
4. Wired KMeans regime detector at startup with `joblib` persistence and VIX-based position-size scaling (100%/50%/25% for bullish/neutral/bearish)
5. Added `SentimentAnalyzer` (Yahoo Finance RSS + VADER, 15-min TTL cache) as entry gate — entries suppressed when composite score < -0.05
6. Added `vix_at_decision` to `TradeLog` schema; SQL validation queries prepared for 14-day paper trading session

### Known Gaps

- **VAL-01**: 14-day paper trading validation not yet completed — code is ready, live session pending. Success criteria: net-positive P&L on ≥10/14 days, no null exit reasons in TradeLog.

---

