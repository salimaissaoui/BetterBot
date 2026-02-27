# Pitfalls Research

**Domain:** Algorithmic stock trading bot — exit strategies, sentiment signals, market regime detection
**Researched:** 2026-02-27
**Confidence:** MEDIUM (WebSearch verified against multiple sources; IBKR-specific items LOW where unverified)

---

## Critical Pitfalls

### Pitfall 1: Overfitting Exit Parameters to Historical Data

**What goes wrong:**
Exit parameters — stop-loss percentages, take-profit multiples, trailing-stop activation levels — are tuned until the backtest equity curve looks clean. The strategy produces exceptional paper returns but collapses in live trading because the exits are fitted to past price noise, not a real edge.

**Why it happens:**
Exits have more tunable parameters than entries (stop distance, trail step, take-profit ratio, time limit, breakeven activation). Each parameter added multiplies the search space. With enough combinations, any dataset yields a profitable-looking result by chance. Developers see the improved backtest and stop searching.

**How to avoid:**
- Limit exit logic to 3 or fewer tunable parameters. Research across 567,000 backtests found that the simplest possible exit — Stop and Reverse — outperformed 14 more complex methods including parabolic stops, chandelier stops, and combination exits.
- Apply walk-forward validation: train exits on data window A, test on held-out window B. Only ship if window B also profitable.
- Sanity-test overfitting: shift the exit by 1–2 bars. If profitability collapses entirely, the signal is curve-fitted noise.
- A Sharpe Ratio above 3.0 in backtest is a red flag, not a success signal. Realistic values are 1.5–2.0.

**Warning signs:**
- Stop-loss and take-profit values are precise decimals tuned to tenths of a percent.
- Backtest shows >30% annual returns on stock data.
- Strategy fails immediately on data from a different calendar year.
- Combination exits (breakeven + trailing + target) were added iteratively as each prior version "improved" results.

**Phase to address:** Exit strategy implementation phase — before any parameter search, define fixed exit types (ATR-based stop, percentage trailing) with parameters set once from first-principles reasoning, not grid-search optimization.

---

### Pitfall 2: Look-Ahead Bias in Sentiment Signals

**What goes wrong:**
Sentiment data (Yahoo Finance RSS headlines, Reddit posts) is timestamped at publication, but the article may have been written after the market already moved. When backtesting, the system uses sentiment that was technically available but practically arrived after the tradeable window closed. The backtest shows alpha; the live system earns nothing.

**Why it happens:**
RSS feeds and Reddit APIs return items sorted by creation time, but news aggregators often republish or surface articles hours after original publication. A story published at 9:45 AM EST may reflect information the market priced at 9:31 AM open. The developer aligns by timestamp and does not account for the pre-market or pre-open information window.

**How to avoid:**
- Add a mandatory lag of one full bar to any sentiment signal before allowing it to influence trades. If the bot runs on 5-minute bars, sentiment must be at least 5 minutes old before acting.
- Record the sentiment timestamp separately from the bar timestamp in the database. Never use sentiment within 15 minutes of its publication time for intrabar decisions.
- For Yahoo Finance RSS: articles are frequently delayed from source; treat them as minimum 30 minutes stale.
- For Reddit/WSB: posts react to price movement as often as they cause it. Validate that sentiment precedes price rather than following it before including in live logic.
- In backtesting: enforce strict timestamp inequality — sentiment timestamp must be strictly less than bar open time.

**Warning signs:**
- Backtest sentiment signals show much higher Sharpe than live-equivalent period.
- Sentiment signal is profitable when aligned to bar close but not bar open.
- Removing the sentiment layer from live trading does not change P&L, but removing it from backtest significantly degrades results.

**Phase to address:** Sentiment integration phase — implement timestamp enforcement as the first gate before any signal quality evaluation. Log every sentiment item with its publication time, ingest time, and the first bar it is eligible to influence.

---

### Pitfall 3: Regime Detection Lag Causes Losses After the Fact

**What goes wrong:**
The regime detector (VIX threshold, moving-average crossover, HMM) identifies a "bad" market regime and suppresses trading — but the identification happens after the adverse move has already occurred. The system is sidelined during recoveries or re-enters in new bad regimes. Net effect: the bot misses the damage partially, misses the recovery entirely, and underperforms a simple hold strategy.

**Why it happens:**
All standard regime detection methods are lagging indicators by construction. Moving average crossovers require N bars to confirm. VIX spikes are observed after vol expands. HMM state inference runs on distributions of past returns. By the time any method declares "high volatility regime," the gap-down that caused it has already settled. Research using Hidden Markov Models on S&P 500 data found strategy did not trade at all from early 2008 to mid-2009 — catching the drawdown but missing the recovery.

**How to avoid:**
- Do not use regime detection as a binary on/off switch for trading. Use it to scale position size and widen stops. Suppressing all signals during high VIX means missing the early days of a recovery — which are typically the highest-return days of any bull cycle.
- Use VIX as a position-size scaler: if VIX > 25, reduce position size by 50%; if VIX > 35, reduce by 75%. Do not halt entirely.
- Validate regime detection on out-of-sample data before deployment. A model trained on 2020–2024 data may not correctly classify a 2025 regime shift if distributions changed.
- Schedule periodic retraining of any ML-based regime detector. Market microstructure changes (new instruments, regulatory shifts) make old training data stale.
- Require at least N consecutive bars in the new regime before switching state — reduces whipsawing on brief volatility spikes.

**Warning signs:**
- Regime detector flips state more than 3 times per week on live data (whipsawing).
- After a VIX spike, bot is idle for weeks while market recovers.
- Regime label changes the day after a major move in the opposite direction.
- Out-of-sample regime classification accuracy is below 60% on held-out data.

**Phase to address:** Regime detection implementation phase — build the regime component as a position-sizing modifier, not a trading gate. Test on 2020 COVID crash, 2022 rate-hike drawdown, and 2023–2024 bull data before accepting the implementation.

---

### Pitfall 4: Hardcoded Credentials Committed to Version Control

**What goes wrong:**
Database passwords, API keys, or broker credentials stored as default fallback values in config files are committed to the repository. If the repo is ever made public, forked, or accessed by a compromised machine, the credentials are immediately exploitable. Even in private repos, any team member or CI/CD system that clones the repo gets the credentials automatically.

**Why it happens:**
Developers add a default value for convenience during development ("I'll fix it before production"). The default value gets committed in the first commit, persists through history, and survives even after the code is "fixed" — because the credential still exists in git history.

**How to avoid:**
- Remove all default values from `os.getenv()` calls immediately. `DB_PASSWORD = os.getenv("DB_PASSWORD")` — no second argument.
- Fail loudly on startup if required environment variables are missing: `if not DB_PASSWORD: raise EnvironmentError("DB_PASSWORD not set")`.
- Add `.env` to `.gitignore` before creating the file. Never commit `.env`.
- Run `git log -S "Stocks123" --all` after removing the credential — it will appear in git history. The credential should be rotated immediately regardless of whether the repo was exposed.
- Use `git filter-repo` or BFG Repo Cleaner to scrub credentials from history if rotation is not immediately possible.
- The AWS RDS hostname (`database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com`) must also be removed from defaults. It exposes AWS account structure and is sufficient for targeted attacks when combined with weak credentials.

**Warning signs:**
- `os.getenv("DB_PASSWORD", "actual_password")` pattern anywhere in codebase.
- `config.py` is tracked by git.
- `.env` file does not exist in `.gitignore`.
- Startup does not fail or warn when environment variables are missing.

**Phase to address:** Bug-fix phase — this must be resolved before any other development. It is a blocking security issue. Rotate the database password immediately regardless of current repo visibility.

---

### Pitfall 5: Exit Logic Absent From Live Positions After Restart

**What goes wrong:**
The bot is restarted (crash, deployment, server reboot) with open positions from the previous session. The in-memory `active_positions` dict is empty because it was not persisted. The bot does not know it holds positions and never applies exit logic to them. Positions are held indefinitely — or worse, the bot re-enters the same symbol and doubles the exposure.

**Why it happens:**
Exit logic is designed around the assumption that the bot has been running continuously and tracking entries in memory. Crash recovery is not considered during implementation. The `active_positions` dictionary in `Scripts/trade.py` is the source of truth for exit decisions but is never synced with the broker on startup.

**How to avoid:**
- On startup, query IBKR for all open positions using `ib.positions()` or equivalent and populate `active_positions` before any trading logic runs.
- Persist position entry data (entry price, entry time, stop level, trailing stop high-water mark) to the PostgreSQL database on every entry. Recover this state on startup.
- After recovery, immediately apply exit checks to all recovered positions before processing new bars.
- Add a reconciliation step that compares in-memory state to broker state every N minutes.

**Warning signs:**
- Bot does not log "recovered N open positions" on startup.
- After restart, no exit orders are placed for positions visible in IBKR TWS.
- `active_positions` is never written to the database.

**Phase to address:** Exit strategy implementation phase — exit state persistence is a prerequisite for reliable exit logic, not an optional improvement.

---

### Pitfall 6: Doubling Down as an Implicit Exit Strategy

**What goes wrong:**
Without explicit exit logic, the ML model may re-signal BUY on a falling position because the indicators still meet the threshold. The position sizing logic then allocates another tranche. The result is an accidental Martingale: the bot doubles down on losing trades. Exponential position growth means a 7th consecutive signal yields 128x original size. One sustained downturn wipes the account.

**Why it happens:**
The buy signal is generated from technical indicators without awareness of existing position in the same symbol. The position-size model does not know the account is already losing on a position. The bot interprets "still looks like a buy" as "buy more."

**How to avoid:**
- Before placing any buy order, check `active_positions` for an existing long position in the same symbol. If one exists, skip the entry signal entirely — do not add to a losing trade.
- Define a maximum per-symbol allocation (e.g., 10% of portfolio). Enforce this as a hard cap regardless of signal confidence.
- The no-exit fallback must be "hold current position and do nothing," not "enter again."

**Warning signs:**
- Same symbol appears in `active_positions` twice with different entry prices.
- Per-symbol exposure grows above 15% of portfolio value.
- Bot enters a symbol on consecutive bars while price is declining.

**Phase to address:** Exit strategy implementation phase — this guard must be the first line of `execute_trade()` before any position sizing or order placement logic.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems specific to this codebase.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoded `BUY_THRESHOLD = 0.51` scattered across files | Works for current model | Cannot tune without touching 4 files; diverge silently | Never — extract to config class |
| Fixed `$100/share` fallback in position sizing | Avoids a market data call | Wrong share count for any stock not priced at $100; capital misallocation | Never |
| Global `active_positions` dict (no DB backing) | Simple to implement | Lost on restart; positions orphaned; no audit trail | MVP only if startup reconciliation added in next phase |
| `try/except: pass` on database write errors | Keeps trading loop running | Lost bars corrupt training data silently | Never in production; use retry queue |
| A/B test allocation checked on every prediction | Easy to implement | 30% overhead on prediction path | Only until model is validated; then retire test |

---

## Integration Gotchas

Common mistakes when connecting to IBKR and free sentiment sources.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| IBKR TWS/Gateway | Using two separate `IB()` connection instances for market data and trade execution | Single shared connection manager; pass one `ib` object to all modules |
| IBKR API | No rate limiting on market data requests; hardcoded `ib.sleep(1)` | Exponential backoff; track request frequency; stay below 50 msg/sec limit |
| IBKR API | Relying on TWS instead of IB Gateway for unattended operation | IB Gateway stays connected without GUI; TWS requires manual login intervention |
| Yahoo Finance RSS | Treating article publish time as signal availability time | Add mandatory 30-minute lag; record ingest timestamp separately |
| Reddit/WSB via API | Assuming posts drive price; using post count as signal strength | Academic research finds WSB long portfolios produce near-zero alpha; use as confirmation signal only, never primary |
| PostgreSQL on AWS RDS | Querying last 200 bars on every 5-minute bar for every symbol | Cache recent bars in memory; query database only on startup and periodic refresh |

---

## Performance Traps

Patterns that work for 1–3 symbols but fail as the watchlist grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full indicator recompute on every bar | CPU spikes; prediction latency > 5 seconds | Incremental indicator update — only compute new bar's contribution | >5 symbols on 5-minute bars |
| Database query per bar per symbol | Connection pool exhaustion; timeouts | In-memory rolling buffer; query only on startup and periodic refresh | >10 symbols with 20-connection pool |
| Synchronous hourly portfolio scan blocking trading loop | Missed bar callbacks during scan | Offload to thread pool; timeout after 55 minutes | Scan exceeds bar interval at 10+ symbols |
| Model prediction inside bar callback | Latency compounds — prediction blocks next bar | Precompute predictions in background thread; use cached result in bar callback | Model inference > 500ms on any symbol |

---

## Security Mistakes

Domain-specific security issues for a live-connected trading bot.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Hardcoded DB password as `os.getenv()` default | Credential exposed in git history; anyone with repo access can connect to AWS RDS | Remove default; fail on startup if env var missing; rotate password immediately |
| AWS RDS hostname as default config value | Infrastructure fingerprint exposed; combined with weak credentials enables targeted attack | Remove from code entirely; require `DB_HOST` env var |
| No order validation before placement | Duplicate orders; position limit breaches; account margin call | Check available funds, existing position, and concurrent order status before every order |
| No circuit breaker on consecutive losses | Bot trades through a broken model or data feed, compounding losses | Halt trading after N consecutive losing trades (e.g., 5) or daily loss exceeds X% of account |
| No input validation on IBKR bar data | Malformed OHLCV data corrupts training set; model degrades silently | Validate: price > 0, high >= low >= 0, volume >= 0, timestamp is current market day |
| API keys stored in process memory without rotation | If process is compromised, keys persist indefinitely | Reload env vars on scheduled interval; alert on unexpected API key usage |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces that cost real money.

- [ ] **Exit logic implemented:** Often missing exit state persistence — verify positions survive a bot restart and exit orders are still applied.
- [ ] **Stop-loss active:** Often configured but not wired to a live order — verify a stop order actually appears in IBKR TWS after entry, not just in internal state.
- [ ] **Sentiment integrated:** Often look-ahead bias not caught — verify sentiment timestamps are strictly older than the bar they influence in backtest.
- [ ] **Regime detection active:** Often regime label is computed but not actually connected to position sizing or entry suppression — verify a log entry showing "regime=high_vol; size_multiplier=0.5" on a high-VIX day.
- [ ] **Credentials secured:** Often `os.getenv()` default value removed from current code but credential still present in git history — run `git log -S "Stocks123" --all` to confirm history is clean.
- [ ] **Position reconciliation on startup:** Often assumed but not implemented — verify log shows "recovered N positions from broker" before first bar processing.
- [ ] **Circuit breaker active:** Often mentioned in comments but not enforced — verify bot actually halts (not just logs a warning) after N consecutive losses.
- [ ] **Graceful shutdown:** Often SIGTERM closes the process but leaves open positions — verify bot closes or at minimum logs all open positions on shutdown.

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Overfitted exits discovered after live deployment | HIGH | Immediately disable the tuned exit; revert to simple ATR-based stop; accept that backtest results are invalid; re-validate on new out-of-sample window |
| Hardcoded credentials exposed | HIGH | Rotate DB password and AWS RDS security group rules immediately; audit git history with `git log -S`; enable AWS CloudTrail to check for unauthorized access |
| Orphaned positions after restart | MEDIUM | Query IBKR positions API manually; add stops to all open positions via TWS directly; implement reconciliation before restarting bot |
| Regime detector whipsawing | MEDIUM | Increase confirmation bars (N=3 minimum); temporarily disable regime switching; revert to always-on trading with reduced position sizes |
| Doubling-down on losing position | HIGH | Close the duplicate position immediately via TWS; implement position deduplication check before next restart; audit all open positions for excess concentration |
| Look-ahead bias found post-deployment | MEDIUM | Add mandatory lag to sentiment signal; re-run backtest with corrected timestamps; expect significant reduction in expected returns |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Hardcoded credentials in config | Bug-fix phase (first) | `git grep "Stocks123"` returns no results; startup fails without `DB_PASSWORD` env var set |
| Git merge conflict blocking startup | Bug-fix phase (first) | `python -c "import Scripts.main"` succeeds without errors |
| Orphaned positions after restart | Exit strategy phase | Bot restart log shows "recovered N positions"; stop orders appear in TWS for all recovered positions |
| Doubling down on open positions | Exit strategy phase | `execute_trade()` logs "skipping entry — existing position in {symbol}" when position already held |
| Overfitted exit parameters | Exit strategy phase | Walk-forward validation on 2 separate data windows; exit parameters not grid-searched |
| Look-ahead bias in sentiment | Sentiment integration phase | Timestamp audit log shows all sentiment items are >= 1 bar old before influencing trades |
| WSB noise treated as primary signal | Sentiment integration phase | Sentiment used as confidence modifier (±5–10% on prediction probability), not as standalone entry signal |
| Regime detection lag causing missed recoveries | Regime detection phase | VIX regime maps to position-size multiplier, not binary on/off; bot still places (smaller) trades in high-VIX regime |
| Regime detector whipsawing | Regime detection phase | Regime label does not change more than 3 times per week on 2020 COVID-crash test data |
| No circuit breaker on consecutive losses | Risk management phase | Bot halts and logs alert after 5 consecutive losing trades within a single session |

---

## Sources

- [What 567,000 Backtests Taught Me About Algo Trading Exits](https://kjtradingsystems.com/algo-trading-exits.html) — MEDIUM confidence; primary research on exit strategy complexity vs. simplicity
- [Common Mistakes in Backtesting Trading Strategies](https://www.arihantplus.com/blogs/algo-trading/top-backtesting-mistakes-every-algo-trader-should-avoid) — MEDIUM confidence; overfitting patterns
- [Stop-Loss, Take-Profit, Triple-Barrier & Time-Exit: Advanced Strategies for Backtesting](https://medium.com/@jpolec_72972/stop-loss-take-profit-triple-barrier-time-exit-advanced-strategies-for-backtesting-8b51836ec5a2) — MEDIUM confidence; exit parameterization risks
- [Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis](https://www.researchgate.net/publication/374930572_Assessing_Look-Ahead_Bias_in_Stock_Return_Predictions_Generated_By_GPT_Sentiment_Generated_By_GPT_Sentiment_Analysis) — MEDIUM confidence; look-ahead bias mechanism in sentiment systems
- [Market Regime Detection using Hidden Markov Models in QSTrader](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/) — MEDIUM confidence; lag and dormancy problems in regime detection
- [Rage Against the Regimes: The Illusion of Market-Specific Strategies](https://www.quantconnect.com/forum/discussion/14818/rage-against-the-regimes-the-illusion-of-market-specific-strategies/) — MEDIUM confidence; overfitting regime strategies to historical data
- [Volatility Regime Shifting: How to Detect Market Shifts Early](https://www.dozendiamonds.com/volatility-regime-shifting/) — LOW confidence; signal management during regime transitions
- [The Dangers of Martingale Position Sizing in Trading](https://enlightenedstocktrading.com/martingale-position-sizing/) — MEDIUM confidence; doubling-down risk quantification
- [Hardcoded API Keys: The Rookie Mistake That Costs Millions](https://medium.com/@instatunnel/hardcoded-api-keys-the-rookie-mistake-that-costs-millions-fa6da9dcc494) — MEDIUM confidence; credential exposure mechanisms
- [GitHub Secret Leaks: The 13 Million API Credentials Sitting in Public Repos](https://medium.com/@instatunnel/github-secret-leaks-the-13-million-api-credentials-sitting-in-public-repos-1a3babfb68b1) — MEDIUM confidence; git history credential persistence
- [Social Media Sentiment for Trading: A Comprehensive Guide for 2025](https://www.shadecoder.com/topics/social-media-sentiment-for-trading-a-comprehensive-guide-for-2025) — LOW confidence; WSB signal quality assessment
- [WallStreetBets trading strategies - should you follow them?](https://alphaarchitect.com/wallstreetbets/) — MEDIUM confidence; academic evidence that WSB long portfolios produce near-zero alpha
- [AI Market Regime Detection: Smarter Real-Time Trading](https://syntiumalgo.com/ai-market-regime-detection/) — LOW confidence; position-scaling approach to regime response
- Known issues in `Scripts/config.py`, `Scripts/trade.py`, `Scripts/main.py` — HIGH confidence; directly observed in codebase analysis

---
*Pitfalls research for: BetterBot — algorithmic stock trading bot*
*Researched: 2026-02-27*
