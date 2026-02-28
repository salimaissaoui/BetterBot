# BetterBot

## What This Is

An automated stock trading bot built in Python that connects to Interactive Brokers for live/paper order execution. It uses an ML ensemble (XGBoost + LightGBM) to generate entry signals from technical indicators, enforces defined exits (hard stop, trailing stop, take-profit) on every position, gates entries against market regime state and VIX level, and filters entries with a VADER sentiment score from Yahoo Finance RSS headlines. The bot is fully implemented for paper trading and awaiting a 14-day validation session.

## Core Value

Every trade entered must have a defined exit — no position held indefinitely. Exits matter more than entries.

## Requirements

### Validated

- ✓ Interactive Brokers (IBKR) API integration for order execution and real-time market data — existing
- ✓ XGBoost + LightGBM voting ensemble for trade entry signals — existing
- ✓ Technical indicators: SMA, RSI, MACD, Bollinger Bands, ATR — existing
- ✓ Advanced feature engineering: market regime detection scaffolding, lagged features, pattern detection — existing
- ✓ PostgreSQL persistence for stock data and portfolio state — existing
- ✓ ML-driven position sizing (RandomForestRegressor) — existing
- ✓ A/B testing framework (30% basic / 70% advanced model allocation) — existing
- ✓ Scheduled model retraining via APScheduler — existing
- ✓ Portfolio tracking and performance metrics — existing
- ✓ Alpaca Markets API as execution fallback — existing
- ✓ BUG-01: Bot starts without merge-conflict errors — v1.0
- ✓ BUG-02: Database credentials loaded from environment variables — v1.0
- ✓ EXIT-01: Hard stop loss at 2×ATR on every position — v1.0
- ✓ EXIT-02: Take-profit target at 4×ATR on every position — v1.0
- ✓ EXIT-03: Trailing stop activates at 1×ATR profit, trails at 1.5×ATR below HWM — v1.0
- ✓ EXIT-04: Position registry persisted to PostgreSQL, reconciled against IBKR on restart — v1.0
- ✓ EXIT-05: Circuit breaker halts new entries when daily P&L < -2% NAV — v1.0
- ✓ OBS-01: TradeLog records entry reason, exit reason, regime state, sentiment score, VIX at decision — v1.0
- ✓ RISK-01: VIX fetched via yfinance; position sizing scaled by VIX level (100%/50%/25%) — v1.0
- ✓ RISK-02: KMeans regime detector fitted at startup, model persisted to `regime_model.joblib` — v1.0
- ✓ SENT-01: Yahoo Finance RSS headlines fetched per ticker via feedparser (no API key) — v1.0
- ✓ SENT-02: Entry suppressed when VADER composite score < -0.05; suppression logged — v1.0

### Active

- [ ] **VAL-01**: Bot achieves net-positive daily P&L for 10 of 14 consecutive paper trading days — verified from TradeLog in PostgreSQL

### Out of Scope

- Live trading with real money — validate in paper first
- Paid data APIs (Benzinga, Polygon.io) — free sources only for now
- Scalping or HFT — latency and broker fees make this impractical
- Options, futures, or crypto — stocks only
- Mobile/web dashboard — command-line and logs sufficient for now
- RL-based exit policies — requires labeled exit data that doesn't exist until rule-based exits are running
- INFRA-01: Migrate from `ib_insync` to `ib_async` — deferred; not blocking paper trading

## Context

- **v1.0 shipped 2026-02-27** — exits, risk gating, sentiment pipeline all implemented and internally verified
- Python codebase in `Scripts/` (~5,538 LOC): `exit_manager.py`, `news.py`, `utils.py` (VIX), `trade.py`, `main.py`, `advanced_features.py`, `database.py`
- Regime model persisted to `regime_model.joblib` on disk
- PostgreSQL on AWS RDS (credentials via `.env`)
- Paper trading account on IBKR — TWS/Gateway must be running in paper mode
- 14-day validation session is the only remaining milestone gate

## Constraints

- **Tech Stack**: Python — no rewriting from scratch, extend existing code
- **Budget**: Free data sources only — no paid news/sentiment APIs
- **Safety**: Paper trading only until consistent profitability demonstrated (VAL-01)
- **Broker**: Interactive Brokers (TWS or IB Gateway must be running)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix exit strategy first | Identified as the #1 failure mode — bot held positions indefinitely | ✓ ExitManager built; all positions now have defined exits |
| Free sentiment sources | Yahoo Finance RSS + VADER before paying for Benzinga/Polygon | ✓ feedparser + vaderSentiment; no API key required |
| Paper trading only | Validate performance before risking real capital | — VAL-01 pending |
| Keep Python stack | Significant existing code — extend rather than rewrite | ✓ All new modules integrate cleanly |
| Fix bugs before features | Merge conflict blocked execution — resolved in Phase 1 | ✓ Bot starts cleanly |
| ExitManager singleton | Centralized state management for all open positions | ✓ Used throughout execute_trade() and hourly_portfolio_scan() |
| VIX scaling over halting | Trade smaller in hostile regimes rather than go silent | ✓ 100%/50%/25% tiers implemented |
| VADER threshold -0.05 | Lenient threshold — only clearly negative headlines suppress; neutral allowed | ✓ Wired; suppression events logged |
| vix_at_decision in TradeLog | Every entry records VIX so post-session analysis can correlate VIX to outcomes | ✓ Added to TradeLog schema in Phase 5 |

---
*Last updated: 2026-02-28 after v1.0 milestone*
