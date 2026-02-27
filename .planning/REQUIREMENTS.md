# Requirements: BetterBot

**Defined:** 2026-02-27
**Core Value:** Every trade entered must have a defined exit — no position held indefinitely.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Bug Fixes

- [ ] **BUG-01**: Bot can start without errors — git merge conflict in `Scripts/main.py` resolved
- [ ] **BUG-02**: Database credentials loaded from environment variables — hardcoded AWS RDS password and hostname removed from `Scripts/config.py`

### Exit Strategy

- [ ] **EXIT-01**: Bot places a hard stop loss on every position entered — calculated as `entry_price - (2 * ATR)` at time of entry
- [ ] **EXIT-02**: Bot places a take profit target on every position entered — minimum 2:1 reward-to-risk ratio (`entry_price + (4 * ATR)`)
- [ ] **EXIT-03**: Bot activates a trailing stop after 1× ATR profit — trails at 1.5× ATR below the running high-water mark
- [ ] **EXIT-04**: Position registry is persisted to PostgreSQL — entry price, stop price, take profit, trailing high, entry time stored per position; reconciled against IBKR on startup
- [ ] **EXIT-05**: Bot halts new entries if daily P&L falls below -2% of account NAV — circuit breaker resets next trading day

### Observability

- [ ] **OBS-01**: Bot logs entry reason and exit reason for every trade as structured records in the database — includes signal values, regime state, and sentiment score at time of decision

### Risk Gating

- [ ] **RISK-01**: Bot suppresses new entries when VIX > 30 — fetched via yfinance `^VIX` ticker before each entry decision
- [ ] **RISK-02**: Existing KMeans market regime detector is activated at startup and used as a position-size scaler — 100% size in bullish regime, 50% in neutral, 25% in bearish/volatile

### Sentiment

- [ ] **SENT-01**: Bot fetches Yahoo Finance RSS headlines for each candidate ticker using feedparser (free, no API key required)
- [ ] **SENT-02**: Bot suppresses new entries when VADER sentiment score for ticker headlines is negative (composite score < -0.05)

### Validation

- [ ] **VAL-01**: Bot achieves net-positive daily P&L for 10 out of 14 consecutive paper trading days — verified from trade logs in PostgreSQL

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Advanced Sentiment

- **SENT-03**: End-of-day FinBERT batch scoring of full article text for next-day signal bias
- **SENT-04**: Reddit/WSB monitoring as contra-indicator — suppress entries in heavily mentioned tickers

### Advanced Exits

- **EXIT-06**: Partial profit-taking — close 50% of position at 1:1 R:R, trail remainder
- **EXIT-07**: Session-close flat — force exit all intraday positions before 3:55 PM ET

### Regime Detection

- **RISK-03**: HMM-based regime states (hmmlearn) as supplementary layer alongside existing KMeans
- **RISK-04**: Max portfolio drawdown halt — pause trading if account drops more than 5% from high-water mark

### Infrastructure

- **INFRA-01**: Migrate from archived `ib_insync` to `ib_async` (drop-in replacement)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Live trading with real money | Validate in paper mode first — 2+ weeks of consistent profit required |
| Paid news APIs (Benzinga, Polygon.io) | Free sources (Yahoo Finance RSS) sufficient for v1; cost not justified yet |
| Reddit/WSB as primary entry signal | Research shows negative returns for retail buyers at peak WSB attention — contra-indicator only |
| Scalping / HFT | Latency and broker fees make this impractical at retail scale |
| Options, futures, crypto | Stocks only — scope creep |
| Mobile / web dashboard | Command-line and database logs sufficient for paper trading phase |
| RL-based exit policies | Requires labeled exit data that does not exist until rule-based exits are running |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUG-01 | Phase 1 | Pending |
| BUG-02 | Phase 1 | Pending |
| EXIT-01 | Phase 2 | Pending |
| EXIT-02 | Phase 2 | Pending |
| EXIT-03 | Phase 2 | Pending |
| EXIT-04 | Phase 2 | Pending |
| EXIT-05 | Phase 2 | Pending |
| OBS-01 | Phase 2 | Pending |
| RISK-01 | Phase 3 | Pending |
| RISK-02 | Phase 3 | Pending |
| SENT-01 | Phase 4 | Pending |
| SENT-02 | Phase 4 | Pending |
| VAL-01 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-27*
*Last updated: 2026-02-27 after initial definition*
