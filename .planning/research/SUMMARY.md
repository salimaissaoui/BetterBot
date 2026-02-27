# Project Research Summary

**Project:** BetterBot — Algorithmic Stock Trading Bot (Augmentation Milestone)
**Domain:** Python / Interactive Brokers event-driven trading bot (retail, single-account)
**Researched:** 2026-02-27
**Confidence:** MEDIUM-HIGH overall

## Executive Summary

BetterBot is a Python algorithmic trading bot connected to Interactive Brokers (IBKR) that uses an ML ensemble (XGBoost + LightGBM) for entry signals and a RandomForest for position sizing. The bot's core infrastructure — data fetching, feature engineering, ML training, order execution, and PostgreSQL persistence — is already in place. This augmentation milestone addresses the three confirmed failure modes: positions that hold indefinitely with no exit logic, a market regime detector that is stubbed out and never wired to a trading gate, and a sentiment pipeline that uses a paid/rate-limited NewsAPI instead of free sources. The primary risk is capital loss from infinite holding — every other enhancement is secondary to fixing exits.

The recommended approach is to build in four strictly ordered phases driven by the codebase's actual gaps: (1) fix blocking infrastructure bugs (git merge conflict, hardcoded credentials) before touching any feature; (2) implement the ExitManager — hard stop, trailing stop, take-profit, and daily circuit breaker — which can be built with zero new dependencies and produces immediate, observable risk reduction; (3) activate the already-coded MarketRegimeDetector by wiring its `fit_regime_detector()` call at startup and making regime a position-size scaler rather than a binary on/off gate; (4) add the sentiment pipeline using feedparser + VADER as a lightweight confirming filter. Sentiment is last because it adds external HTTP dependencies and look-ahead bias risk, and is the weakest signal of the three.

The dominant risks in this augmentation are: (a) exit state is lost on restart if position data is not persisted to the database — orphaned positions will never be exited; (b) the existing `config.py` contains hardcoded AWS RDS credentials that must be rotated and removed before any further development; (c) the regime detector, if implemented as a hard on/off gate rather than a position-size multiplier, will cause the bot to miss market recoveries after every volatility spike; and (d) the existing `ib-insync` library is archived and must be migrated to the drop-in replacement `ib_async` before IBKR API version bumps break compatibility.

---

## Key Findings

### Recommended Stack

The existing stack (Python, XGBoost, LightGBM, scikit-learn, pandas, SQLAlchemy/PostgreSQL, `ta`, ib_insync) requires only targeted additions. No framework-level changes are needed. See `STACK.md` for full version details.

**Core technology additions:**

- `feedparser 6.0.12` — Parse Yahoo Finance RSS per ticker; zero API key, handles malformed XML; replaces current NewsAPI dependency
- `praw 7.8.1` — Reddit API (r/wallstreetbets) for sentiment confirmation; free at script tier (100 req/min)
- `vaderSentiment 3.3.2` — Fast lexicon-based sentiment scoring tuned for social media and financial slang; no GPU; replaces the keyword matching in `news.py`
- `transformers 5.2.0` + `torch` — Optional FinBERT for end-of-day batch scoring of full article text; too slow (0.5-2s CPU) for real-time per-bar use
- `hmmlearn 0.3.3` — GaussianHMM for supplementary regime detection layer; scikit-learn-compatible API; augments (does not replace) the existing KMeans detector
- `ib_async 2.1.0` — Drop-in replacement for archived `ib_insync`; import rename only (`ib_insync` -> `ib_async`); requires Python 3.10+
- `yfinance 1.2.0` — Already in stack; use `yf.download('^VIX')` for VIX-based risk gating (no API key required)

**What NOT to add:** `stocknews` (167 downloads/week, abandoned), `TextBlob` (not tuned for finance, VADER dominates), `pykalman` (wrong tool — Kalman filters are regression estimators, not regime classifiers), real-time per-bar FinBERT (blocks the event loop).

### Expected Features

The bot's primary failure mode — confirmed by code analysis — is holding positions indefinitely. The feature set is unambiguous: exits must come first, followed by regime gating, followed by sentiment filtering. See `FEATURES.md` for full prioritization matrix.

**Must have (table stakes — bot bleeds capital without these):**
- Hard stop loss per position (ATR-based; `ta` library already computes ATR)
- Take profit target per position (minimum 2:1 risk-reward ratio)
- Trade context logging: entry reason + exit reason as structured DB records
- Daily loss circuit breaker (halt new entries if daily P&L below -2% of account NAV)

**Should have (add after 1 week of stable paper trading):**
- Trailing stop (activate after 1x ATR profit; trail at 1.5x ATR below running high; implement in Python, not as IBKR native trailing stop order)
- Regime-gated entry suppression (wire existing KMeans detector to entry gate + VIX threshold)
- Hold-time stop (exit stalled positions after 5 trading days)
- Session close flat for intraday signals (force exit before 3:55 PM ET)

**Defer to v2+ (after 2 weeks of consistent paper profitability):**
- News sentiment filter (Yahoo Finance RSS + VADER as confirming entry gate)
- Partial profit-taking at multiple levels
- Max portfolio drawdown halt (high-water mark tracking)
- Per-stock ATR multiplier tuning by volatility profile

**Anti-features — do not build:**
- WSB Reddit sentiment as a primary entry signal (peer-reviewed research shows near-zero alpha; use only as contra-indicator for crowded retail positioning)
- Reinforcement learning for exit decisions (requires labeled exit data that does not exist yet)
- ML-predicted stop distances (will overfit to whichever stops happened not to trigger in training data)

### Architecture Approach

The flat `Scripts/` layout is preserved. This milestone adds exactly two new files (`exit_manager.py`, `sentiment.py`) and modifies four existing ones (`main.py`, `config.py`, `trade.py`, `advanced_features.py`). The key structural insight from codebase analysis is that `execute_trade()` is stateless per-call — it cannot track entry prices inline. Exit logic requires a persistent position registry (ExitManager) that survives across the 30-second bar loop. All new components follow the existing singleton pattern. See `ARCHITECTURE.md` for full component diagrams and data flows.

**Major components:**

1. `Scripts/exit_manager.py` (NEW) — ExitManager class; owns per-position EntryRecord (entry price, ATR at entry, stop level, trailing high, target price, quantity); checks exits before evaluating new entry signals on every bar; registers entry on successful BUY; clears on close
2. `Scripts/sentiment.py` (NEW) — SentimentPipeline class; fetches Yahoo Finance RSS + Reddit WSB via PRAW; scores with VADER; 15-minute TTL cache per symbol prevents rate-limit issues; blocks entries where sentiment opposes trade direction by more than 0.3 threshold
3. `Scripts/advanced_features.py` — `MarketRegimeDetector` (EXISTING, needs wiring); `fit_regime_detector()` must be called in `initialize_bot()` and its fitted KMeans/Scaler/PCA objects persisted to `regime_model.joblib`; regime output should map to a position-size multiplier (not a hard gate) to avoid missing recoveries
4. `Scripts/trade.py` — `on_bar()` gate chain: ML prediction -> sentiment gate -> execute_trade() -> exit check -> regime gate -> entry decision
5. `Scripts/model_performance.py` — Extend `log_trade_performance()` to accept `exit_reason` field ('HARD_STOP', 'TRAILING_STOP', 'TAKE_PROFIT', 'SIGNAL', 'CIRCUIT_BREAKER')

### Critical Pitfalls

1. **Hardcoded credentials in `config.py`** — The AWS RDS password and hostname appear as `os.getenv()` default values and are committed to git history. Rotate the password immediately; run `git log -S "Stocks123" --all` to confirm history exposure; remove defaults so startup fails loudly if env vars are missing. This is a blocking security issue that must be resolved before any other work.

2. **Exit state lost on restart** — The in-memory `active_positions` dict is not persisted. After any crash or restart, the bot does not know it holds open positions and never applies exit logic to them — or worse, doubles the exposure by re-entering. Prevention: query IBKR `ib.positions()` on startup and rebuild exit registry from the database before processing any new bars.

3. **Doubling down as an implicit exit strategy** — Without hard exit logic, the ML model can re-signal BUY on a falling position because indicators still pass threshold. Each additional signal allocates another tranche, creating an accidental Martingale. Prevention: check `active_positions` before any BUY; skip if already long the same symbol; enforce a hard 5-10% per-symbol portfolio cap.

4. **Regime detection used as a binary on/off gate** — All standard regime detection methods (VIX, KMeans, HMM) are lagging indicators. Declaring "high volatility regime" and halting all trading means the bot misses the early days of recovery, which are typically the highest-return days. Prevention: map VIX/regime to a position-size multiplier (VIX > 25: 50% size; VIX > 35: 25% size) rather than suppressing trading entirely.

5. **Look-ahead bias in sentiment signals** — Yahoo Finance RSS articles are frequently delayed from source; the publish timestamp does not reflect when the market priced the information. Prevention: enforce a mandatory 30-minute lag on any sentiment item before it influences trades; log the ingest timestamp separately from the publication timestamp; validate that sentiment precedes price moves in backtesting via strict timestamp inequality.

---

## Implications for Roadmap

Based on combined research findings, the suggested phase structure follows a strict dependency chain. Exits are blocked by infrastructure bugs; regime detection is blocked by exits (regime suppresses entries, not exits — so exits must exist before regime gating is meaningful); sentiment is blocked by regime detection (sentiment as a standalone filter is too noisy; it only has value as a secondary gate layered on top of regime).

### Phase 1: Infrastructure Unblock

**Rationale:** The git merge conflict in `Scripts/main.py` (lines 11-17) prevents the bot from running. The hardcoded credentials in `config.py` are a critical security exposure that must be resolved before any further commits. Nothing else can be validated until the bot can reach `trading_loop()` without crashing.

**Delivers:** A bot that starts, connects to IBKR, and reaches the trading loop without errors.

**Addresses pitfall:** Hardcoded credentials (Pitfall 4), git merge conflict blocking startup

**Avoids:** Starting feature development on a broken foundation; committing new code that inherits the credential exposure

**Research flag:** No additional research needed. The bugs are directly observed in the codebase.

### Phase 2: Exit Management

**Rationale:** Per PROJECT.md and confirmed by every research source, exits matter more than entries. The current bot has no exit logic — positions hold until the ML model reverses, which may never happen for a declining stock. This is the highest-priority feature and has zero external dependencies.

**Delivers:** Positions that close automatically via hard stop, trailing stop, or take-profit; trade context logging for every entry and exit; daily loss circuit breaker halting new entries on bad days

**Addresses features:** Hard stop loss (P1), take profit target (P1), trade context logging (P1), daily loss circuit breaker (P1), trailing stop (P1)

**Uses stack:** ATR from existing `ta` library; no new pip dependencies; existing ib_insync `placeOrder()` for close orders

**Implements architecture:** ExitManager singleton (`Scripts/exit_manager.py`); extend `log_trade_performance()` with `exit_reason` field; extend `active_positions` dict to store entry metadata

**Avoids pitfalls:** Exit state lost on restart (must persist EntryRecord to DB at entry time); doubling down (check active_positions before any BUY); inline stop logic in stateless execute_trade() (use ExitManager registry instead)

**Research flag:** Standard patterns — well-documented. No additional research needed. ATR trailing stop (Chandelier Exit formula) is the industry standard. IBKR `bracketOrder()` is documented.

### Phase 3: Regime Detection Activation

**Rationale:** The KMeans MarketRegimeDetector is already coded in `advanced_features.py` — it just is not wired. `fit_regime_detector()` is never called so `regime_model` is always None and every regime returns 'unknown'. This is a wiring bug, not a feature build. Phase 2 must come first so regime gating operates on positions that can actually be exited — there is no point suppressing entries if losses cannot be contained.

**Delivers:** A regime detector that is fitted at startup, persisted to disk, and produces real regime labels that modify position sizing on every bar

**Addresses features:** Regime-gated entry suppression (P2); VIX-based risk gating

**Uses stack:** `yfinance 1.2.0` for `^VIX` fetch (cached every 15 min); existing `hmmlearn` optional layer for forward-looking regime signal; `joblib` (already in stack) for model persistence

**Implements architecture:** Call `fit_regime_detector()` in `initialize_bot()` after DB population; persist fitted KMeans/Scaler/PCA to `regime_model.joblib`; map regime + VIX to position-size multiplier (not hard gate) in `execute_trade()`

**Avoids pitfalls:** Regime as binary gate (use position-size scaler instead); regime model refitted on every restart (persist to joblib); regime suppressing exits (ExitManager exits are unconditional; regime gate applies only to entry branch)

**Research flag:** The VIX threshold values (20/30/40) are empirical conventions, not universal constants — validate against this bot's specific return distribution during paper trading. Consider deeper research if the position-size multiplier approach needs calibration.

### Phase 4: Sentiment Pipeline

**Rationale:** Sentiment is the weakest signal of the three and has the most external complexity (HTTP fetches, rate limits, look-ahead bias risk, new credentials). It comes last because: (a) regime detection must exist for sentiment to be layered as a secondary gate; (b) PRAW credentials are a new external dependency that complicates the development environment; (c) the research is clear that sentiment as a standalone primary signal has near-zero alpha (peer-reviewed). As a confirming gate layered on top of ML + regime, it filters out entries during negative news events.

**Delivers:** A sentiment pipeline that fetches Yahoo Finance RSS + Reddit WSB, scores with VADER, caches per symbol for 15 minutes, and blocks entries where sentiment strongly opposes trade direction

**Addresses features:** News sentiment filter (P3)

**Uses stack:** `feedparser 6.0.12`, `praw 7.8.1`, `vaderSentiment 3.3.2`; replaces `Scripts/news.py` which uses the paid/rate-limited NewsAPI

**Implements architecture:** `Scripts/sentiment.py` with SentimentPipeline singleton; gate in `on_bar()` after ML prediction and before `execute_trade()`; weight: RSS 60%, Reddit 40%; block threshold: -0.3 for longs, +0.3 for shorts

**Avoids pitfalls:** Look-ahead bias (mandatory 30-minute publication lag; log ingest timestamp separately); sentiment as primary signal (VADER score is a confidence modifier on top of ML probability, never an entry initiator); rate limits (15-minute TTL cache prevents per-bar HTTP calls)

**Research flag:** Needs additional research on timestamp lag enforcement in backtesting and on PRAW Reddit credential setup (OAuth flow). The sentiment signal effectiveness on 5-minute bar signals is LOW confidence — validate that the filter reduces false entries without materially reducing true entries before treating it as a hard gate.

### Phase Ordering Rationale

- Phase 1 before all others: the bot cannot run at all with the merge conflict; credentials must be rotated before any new code is committed
- Phase 2 before Phases 3 and 4: exits are the most critical risk reduction with zero dependencies; regime and sentiment gating only matter if positions can actually be exited
- Phase 3 before Phase 4: regime detection is simpler (already coded, just needs wiring) and produces a harder, more reliable gate than sentiment; sentiment as a soft filter adds value only when layered on top of a working regime gate
- Sentiment is deferred to v2+ as a P3 feature per FEATURES.md; the roadmap can make it Phase 4 if timeline permits, or defer to a separate milestone

### Research Flags

Phases needing deeper research during planning:
- **Phase 3 (Regime Activation):** VIX threshold calibration needs validation against this bot's historical return distribution; the exact multipliers (50% at VIX > 25, 25% at VIX > 35) are starting points from research conventions, not bot-specific tuning
- **Phase 4 (Sentiment Pipeline):** Reddit PRAW OAuth credential setup and rate limit management needs a dedicated implementation guide; look-ahead bias enforcement in backtesting needs explicit timestamp audit logging designed before implementation begins

Phases with standard patterns (skip dedicated research phase):
- **Phase 1 (Unblock):** Direct bug fixes; no research needed
- **Phase 2 (Exit Management):** ATR trailing stop is the industry standard with decades of empirical backing; Chandelier Exit formula and IBKR bracket order patterns are well-documented

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All library versions verified via PyPI; ib_async drop-in compatibility confirmed via GitHub; yfinance VIX approach confirmed working |
| Features | MEDIUM | Exit strategy patterns HIGH confidence (industry standard ATR-based stops); sentiment signal effectiveness LOW-MEDIUM (contradictory research on WSB alpha; VADER vs FinBERT tradeoffs context-dependent) |
| Architecture | HIGH | Derived from direct codebase analysis of all Scripts/ modules, not documentation inference; integration points verified against actual code |
| Pitfalls | MEDIUM | Credential issue HIGH (directly observed in codebase); exit state loss pattern HIGH (directly observed); regime lag patterns MEDIUM (multiple corroborating sources but bot-specific thresholds unvalidated) |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **VIX threshold calibration:** The 20/25/30/35/40 VIX thresholds used in regime gating are conventional values cited across multiple sources but are not validated against this specific bot's signal distribution. During Phase 3 implementation, run the bot on 2020 COVID crash and 2022 rate-hike drawdown test data to confirm the multipliers do not suppress too aggressively.

- **Sentiment effectiveness on 5-minute bars:** The research consensus is that sentiment signals have 30-60 minute information lag. Running sentiment as a confirming gate on a 5-minute bar signal may reduce entry frequency without adding meaningful accuracy. Validate this with 1-2 weeks of paper trading with sentiment gate active vs. inactive.

- **FinBERT CPU latency:** STACK.md notes 0.5-2s per inference on CPU for FinBERT, but this is from training data, not direct measurement on this system. If FinBERT is included in Phase 4, measure actual latency in the bot's environment before committing to it as an online signal.

- **IBKR paper account partial order limits:** FEATURES.md flags that partial profit-taking (v2+) requires confirming that paper account supports partial fills. This needs verification before designing the multi-level take-profit logic in any future milestone.

- **Position reconciliation on restart:** ARCHITECTURE.md and PITFALLS.md both identify this as critical. The specific IBKR API call (`ib.positions()`) and the reconciliation loop design are not fully specified. During Phase 2 planning, research the exact ib_async positions reconciliation pattern.

---

## Sources

### Primary (HIGH confidence)

- Direct codebase analysis: `Scripts/trade.py`, `Scripts/main.py`, `Scripts/advanced_features.py`, `Scripts/config.py`, `Scripts/news.py`, `Scripts/model_performance.py` — architecture and pitfall findings
- feedparser PyPI — https://pypi.org/project/feedparser/ — version 6.0.12 verified
- praw PyPI — https://pypi.org/project/praw/ — version 7.8.1 verified
- vaderSentiment PyPI — https://pypi.org/project/vaderSentiment/ — version 3.3.2 verified
- ib_async PyPI + GitHub — https://pypi.org/project/ib_async/ / https://github.com/ib-api-reloaded/ib_async — drop-in replacement confirmed
- ib_insync GitHub — https://github.com/erdewit/ib_insync — archived March 2024 confirmed
- hmmlearn PyPI — https://pypi.org/project/hmmlearn/ — version 0.3.3 verified
- statsmodels PyPI — https://pypi.org/project/statsmodels/ — version 0.14.6 verified
- yfinance PyPI — https://pypi.org/project/yfinance/ — version 1.2.0, Feb 2026 verified
- transformers PyPI — https://pypi.org/project/transformers/ — version 5.2.0 verified

### Secondary (MEDIUM confidence)

- Social media attention and retail investor behavior (WallStreetBets) — ScienceDirect 2024 (peer-reviewed) — WSB long portfolios near-zero alpha finding
- Democratisation of retail trading: Reddit WSB vs investment bank analysts — Taylor & Francis 2024 (peer-reviewed) — contra-indicator behavior of WSB attention
- What 567,000 Backtests Taught Me About Algo Trading Exits — kjtradingsystems.com — simplest exits outperform complex combinations
- Market Regime Detection using Hidden Markov Models in QSTrader — quantstart.com — regime lag and dormancy problems
- ATR Chandelier Exit Strategy — quantifiedstrategies.com — Chandelier Exit formula (period=22, multiplier=3.0 defaults)
- IBKR bracket order via ib_insync — copyprogramming.com — `bracketOrder()` implementation pattern
- 3commas AI Trading Bot Risk Management Guide 2025 — stop-loss and circuit breaker patterns
- hmmlearn for regime trading — quantstart.com — GaussianHMM 2-state approach confirmed across multiple sources
- yfinance news issue #1956 — ticker relevance problem confirmed

### Tertiary (LOW confidence — needs validation)

- FinBERT CPU latency (0.5–2s per inference) — from training data, not measured; validate in environment before committing
- VIX threshold values (20/25/30/35/40) — empirical conventions from multiple sources, not this bot's validated calibration
- Sentiment signal effectiveness on 5-minute bars — low-confidence inference from 30-60 minute lag research; needs paper trading validation

---

*Research completed: 2026-02-27*
*Ready for roadmap: yes*
