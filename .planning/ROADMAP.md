# Roadmap: BetterBot

## Overview

BetterBot has a working ML entry pipeline but no exit logic — the primary failure mode confirmed in production. This milestone delivers the missing half of the trade lifecycle: every position gets a defined exit on entry, the market regime detector gets wired, a lightweight sentiment filter gets added as a confirming gate, and the whole system is validated in paper trading for two weeks. The path is strictly ordered: fix the blocking bugs so the bot can start, then build exits, then add gating intelligence, then validate.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Infrastructure Unblock** - Resolve the merge conflict and remove hardcoded credentials so the bot can start
- [ ] **Phase 2: Exit Management** - Give every position a hard stop, trailing stop, take-profit target, and circuit breaker
- [ ] **Phase 3: Risk Gating** - Wire the existing regime detector and add VIX-based position-size scaling
- [ ] **Phase 4: Sentiment Pipeline** - Add Yahoo Finance RSS + VADER as a lightweight confirming entry filter
- [ ] **Phase 5: Validation** - Demonstrate consistent paper trading profitability over 14 consecutive days

## Phase Details

### Phase 1: Infrastructure Unblock
**Goal**: The bot starts, connects to IBKR, and reaches the trading loop without errors or credential exposure
**Depends on**: Nothing (first phase)
**Requirements**: BUG-01, BUG-02
**Success Criteria** (what must be TRUE):
  1. Running `python Scripts/main.py` produces no merge-conflict syntax errors and reaches the trading loop
  2. Database credentials are read exclusively from environment variables — no literals in `config.py` or git history
  3. Starting the bot with missing environment variables fails with a clear error message rather than silently using wrong credentials
**Plans**: TBD

Plans:
- [x] 01-01: Resolve git merge conflict in Scripts/main.py and verify startup reaches trading_loop()
- [x] 01-02: Move hardcoded AWS RDS credentials out of Scripts/config.py into environment variables

### Phase 2: Exit Management
**Goal**: Every position entered by the bot has a defined exit — hard stop, trailing stop, take-profit target — persisted to the database and enforced on every bar
**Depends on**: Phase 1
**Requirements**: EXIT-01, EXIT-02, EXIT-03, EXIT-04, EXIT-05, OBS-01
**Success Criteria** (what must be TRUE):
  1. After a BUY executes, a hard stop loss and take-profit target are immediately computed and stored in PostgreSQL for that position
  2. After the position gains 1x ATR profit, the bot activates a trailing stop that moves up with the high-water mark
  3. If the daily P&L drops below -2% of account NAV, the bot halts new entry signals for the rest of the trading day
  4. After a position closes, the database contains a structured record of why it was entered and why it was exited (signal values, regime state, exit type)
  5. On bot restart, open positions are reconciled from IBKR and the exit registry is rebuilt — no position is silently orphaned
**Plans**: TBD

Plans:
- [ ] 02-01: Implement ExitManager class with EntryRecord persistence and IBKR reconciliation on startup
- [ ] 02-02: Implement hard stop loss (2x ATR) and take-profit target (4x ATR) placed at entry
- [ ] 02-03: Implement trailing stop activation after 1x ATR gain, trailing at 1.5x ATR below high-water mark
- [ ] 02-04: Implement daily P&L circuit breaker that halts new entries below -2% NAV
- [ ] 02-05: Extend trade logging to capture entry reason, exit reason, and signal state as structured DB records

### Phase 3: Risk Gating
**Goal**: The existing KMeans market regime detector is wired and active, and VIX levels modify position sizing so the bot trades smaller in hostile conditions rather than going silent
**Depends on**: Phase 2
**Requirements**: RISK-01, RISK-02
**Success Criteria** (what must be TRUE):
  1. On startup, the regime detector is fitted and its model is persisted to disk — regime labels are never 'unknown' during a trading session
  2. When VIX exceeds 30, new position sizes are reduced (not halted) — the bot continues to trade but smaller
  3. Regime state (bullish / neutral / bearish) is visible in the trade log for every entry decision, with the position-size multiplier that was applied
**Plans**: TBD

Plans:
- [ ] 03-01: Wire fit_regime_detector() at bot startup and persist the fitted model to regime_model.joblib
- [ ] 03-02: Fetch VIX via yfinance on each entry decision and apply position-size multiplier (100% / 50% / 25%) based on regime + VIX level

### Phase 4: Sentiment Pipeline
**Goal**: Yahoo Finance RSS headlines are fetched and VADER-scored per ticker, and entries are suppressed when sentiment is clearly negative — acting as a lightweight confirming filter on top of ML signals
**Depends on**: Phase 3
**Requirements**: SENT-01, SENT-02
**Success Criteria** (what must be TRUE):
  1. For each candidate ticker before an entry decision, the bot fetches Yahoo Finance RSS headlines using feedparser without requiring an API key
  2. When the VADER composite sentiment score for a ticker's headlines is below -0.05, the bot skips the entry and logs the suppression reason
  3. Sentiment scores are cached per ticker for 15 minutes — the bot does not make a new HTTP request on every 30-second bar
**Plans**: TBD

Plans:
- [ ] 04-01: Implement SentimentPipeline class with feedparser RSS fetch, VADER scoring, and 15-minute TTL cache
- [ ] 04-02: Wire sentiment gate into on_bar() entry chain — suppress entry and log reason when sentiment score < -0.05

### Phase 5: Validation
**Goal**: The bot demonstrates consistent paper trading profitability over a 14-day window, proving the exit strategy and gating logic work together under real market conditions
**Depends on**: Phase 4
**Requirements**: VAL-01
**Success Criteria** (what must be TRUE):
  1. The bot runs continuously in paper trading mode for 14 consecutive trading days without crashing or orphaning positions
  2. Trade logs in PostgreSQL show net-positive daily P&L on at least 10 of the 14 days
  3. Every closed position in the database has a recorded exit reason — no positions close with a missing or null exit type
**Plans**: TBD

Plans:
- [ ] 05-01: Run 14-day paper trading session and query PostgreSQL for daily P&L and exit-reason coverage

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure Unblock | 2/2 | Complete | 2026-02-27 |
| 2. Exit Management | 0/5 | Not started | - |
| 3. Risk Gating | 0/2 | Not started | - |
| 4. Sentiment Pipeline | 0/2 | Not started | - |
| 5. Validation | 0/1 | Not started | - |
