# BetterBot

## What This Is

An automated stock trading bot built in Python that connects to Interactive Brokers for live/paper order execution. It uses an ML ensemble (XGBoost + LightGBM) to generate entry signals from technical indicators, but currently lacks a reliable exit strategy — causing it to hold positions indefinitely or double down. The goal is a bot that enters AND exits trades intelligently, incorporating sentiment awareness and market regime detection, to produce consistent daily profit in paper trading.

## Core Value

Every trade entered must have a defined exit — no position held indefinitely. Exits matter more than entries.

## Requirements

### Validated

<!-- Already exists in codebase -->

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

### Active

- [ ] Complete exit strategy subroutine: stop loss, trailing stop, take profit targets per trade
- [ ] Fix critical blocking bugs: resolve merge conflict in main.py, remove hardcoded credentials
- [ ] News and sentiment signal integration using free sources (Yahoo Finance RSS, Reddit/WSB)
- [ ] Active market regime detection: suppress trading in unfavorable conditions (high VIX, unclear trend)
- [ ] Trade context logging: record why each trade was entered AND why it was exited
- [ ] Paper trading validation loop: consistent profitable session over 2+ weeks in paper mode

### Out of Scope

- Live trading with real money — validate in paper first
- Paid data APIs (Benzinga, Polygon.io) — free sources only for now
- Scalping or HFT — latency and broker fees make this impractical
- Options, futures, or crypto — stocks only
- Mobile/web dashboard — command-line and logs sufficient for now

## Context

- Brownfield Python codebase in `Scripts/` — significant code already written
- Codebase has real structural issues: git merge conflict in `main.py` blocks execution, hardcoded AWS RDS password and hostname in `config.py`
- Advanced RL (reinforcement learning) module exists but is optional — activated via A/B test allocation
- Paper trading account on IBKR is the target environment
- Bot has run live before with mixed results — primary failure mode was holding positions with no exit logic
- Exit logic is either absent or non-functional — confirmed by user
- PostgreSQL on AWS RDS (credentials need to move to env vars)

## Constraints

- **Tech Stack**: Python — no rewriting from scratch, extend existing code
- **Budget**: Free data sources only — no paid news/sentiment APIs
- **Safety**: Paper trading only until consistent profitability demonstrated
- **Broker**: Interactive Brokers (TWS or IB Gateway must be running)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix exit strategy first | Identified as the #1 failure mode — bot holds forever without it | — Pending |
| Free sentiment sources | Reddit/WSB + Yahoo Finance RSS before paying for Benzinga/Polygon | — Pending |
| Paper trading only | Validate performance before risking real capital | — Pending |
| Keep Python stack | Significant existing code — extend rather than rewrite | — Pending |
| Fix bugs before features | Merge conflict blocks execution — must resolve before adding intelligence | — Pending |

---
*Last updated: 2026-02-27 after initialization*
