# Feature Research

**Domain:** Algorithmic Stock Trading Bot (Retail, Python, IBKR)
**Researched:** 2026-02-27
**Confidence:** MEDIUM — exit strategy patterns HIGH; sentiment signal effectiveness LOW-MEDIUM due to contradictory research

---

## Context: What Already Exists

The existing bot has entry signals (ML ensemble), technical indicators, position sizing, IBKR execution, and PostgreSQL persistence. The research below focuses exclusively on what is **missing**: exits, sentiment, regime detection, and trade context logging.

The bot's stated primary failure mode is holding positions indefinitely with no exit. Per PROJECT.md: "Exits matter more than entries."

---

## Feature Landscape

### Table Stakes (Bot Bleeds Money Without These)

Features whose absence directly causes capital loss or makes the bot untrustworthy to run.

| Feature | Why It's Table Stakes | Complexity | Notes |
|---------|----------------------|------------|-------|
| **Hard stop loss per position** | Without it, a losing position holds forever and can wipe the account. The #1 confirmed failure mode for this bot. | LOW | ATR-based stop (e.g. 2× ATR below entry) is preferred over fixed %. ATR already computed in `indicators.py`. Attach stop price to each position record at entry time. |
| **Take profit target per position** | Without a take profit, winners also hold indefinitely and give back gains. Ensures positive expectancy over many trades. | LOW | Minimum 2:1 reward-to-risk ratio. Set at 2–4× ATR above entry. Can be a simple limit order placed at IBKR on entry. |
| **Trailing stop (activates after profit threshold)** | Protects unrealized gains once the trade moves in your favor. Without it, a 10% winner can become a 2% loser with no exit. | MEDIUM | Activate trailing after 1× ATR profit. Trail at 1.5–2× ATR below running high. Compute on each bar check in hourly portfolio scan. Do NOT use IBKR native trailing stop orders — implement in code so logic is visible and auditable. |
| **Daily loss circuit breaker** | Without a daily halt, a bad day compounds. A malfunctioning bot can drain an account in a single session. | LOW | Halt all new entries if daily P&L falls below -2% of account value. Resume next trading day. Requires reading account state from IBKR at start of each trading loop. |
| **Max portfolio drawdown halt** | Same as daily limit but cumulative. Prevents death by a thousand cuts over multiple days. | LOW | Hard stop at -7% portfolio drawdown from high-water mark. Store high-water mark in DB. |
| **Per-position risk limit** | Without sizing limits, a single position can dominate the portfolio. Partial position sizing already exists but needs a hard cap. | LOW | Cap any single position at 5% of portfolio NAV regardless of ML sizing output. |
| **Trade context logging (entry reason)** | Without recording WHY a trade entered, you cannot debug, improve, or validate the bot. Blind execution is unacceptable for any serious bot. | LOW | Log: signal probability, confidence score, model used (basic/advanced), indicators that triggered, regime label, timestamp. Structured JSON row in DB or append to CSV per trade. |
| **Trade context logging (exit reason)** | Without recording WHY a trade exited, you cannot distinguish lucky exits from correct exits. | LOW | Log: exit type (stop/trail/take-profit/signal-reversal/circuit-breaker), P&L, holding duration, exit bar indicators. Same DB table as entry log. |

---

### Differentiators (Competitive Edge for a Retail Trader)

Features that are not universally present in retail bots and create measurable edge when implemented well.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **ATR-normalized stops (per-symbol volatility)** | A fixed 2% stop works fine for low-vol stocks but gets blown through on high-beta names. ATR-scaling per symbol keeps risk consistent across the portfolio. | LOW | ATR is already computed. Multiply by configurable factor (start at 2.0). Re-evaluate stop width at each hourly scan if holding period extends. |
| **Partial profit-taking at multiple levels** | Locking in 50% of position at 1:1 R:R, holding rest to runner, materially improves risk-adjusted returns over single-exit strategies. | MEDIUM | Close half at 1× risk distance, trail the remainder. Requires fractional order splitting via IBKR. Confirm IBKR supports partial fills on market/limit orders for paper account. |
| **Regime-gated entry suppression** | Suppressing new entries when VIX > 30 or when the KMeans regime detector labels current conditions "volatile/bear" prevents entries into high-noise environments where the ML signal degrades. | MEDIUM | MarketRegimeDetector scaffold already exists in `advanced_features.py`. Need to wire its output into `trade.py` entry gate. Add VIX fetch via IBKR (IBKR provides VIX as a data feed). Regime check runs once per hourly scan, not per bar. |
| **News sentiment as a confirming signal (not primary signal)** | Adding a sentiment gate — require neutral/positive sentiment to allow an entry — filters out entries during negative news events the ML model cannot see. | MEDIUM | Use as a filter, not a driver. Sentiment confirms entry; it never initiates one. Sources: Yahoo Finance RSS (via `stocknews` PyPI package or direct RSS parse), PRAW for r/wallstreetbets mention count. Use VADER for headline scoring (rule-based, no GPU needed). FinBERT is higher quality but requires a model download. |
| **Hold-time stop (time-based exit)** | If a position hasn't reached stop or target after N bars, exit anyway. Prevents capital being tied up in stalled trades. | LOW | Simple: log entry bar timestamp. If holding > 5 trading days without target/stop hit, exit at next open. Add to hourly portfolio scan. |
| **Session close flat (no overnight risk on day signals)** | If the entry signal came from a short-term (5-min bar) model, exit before 3:55 PM ET on the same day. Eliminates gap risk. | LOW | Check signal timeframe at entry. For 5-min bar signals, tag as "intraday." Hourly scan checks tag and time. For daily-bar signals, allow overnight holds. |

---

### Anti-Features (Do NOT Build These)

Features that seem valuable but create noise, complexity, or false confidence for a solo retail trader.

| Anti-Feature | Why Requested | Why Problematic | What to Do Instead |
|--------------|---------------|-----------------|-------------------|
| **WSB/Reddit sentiment as a primary entry signal** | WSB famously moved GameStop; feels like free alpha. | Research (2022-2025 data) shows WSB sentiment predicts negative returns when attention is highest — retail "buying when it's trending" is the signal of a top. Signal-to-noise is very low outside meme-stock events. Building an entry system on it will overfit the 2021 meme period. | Use Reddit only as a confirming suppression filter: if a stock has unusually high WSB attention volume (not sentiment), skip the trade because crowded retail positioning is a contra-indicator. |
| **Sentiment as a real-time tick-by-tick signal** | More data = better signal. | Yahoo Finance RSS updates every 15-60 min. PRAW API rate limits to 1 request/sec. Neither gives tick-level sentiment. Pretending it's real-time introduces look-ahead bias. | Compute sentiment once per hourly scan. Tag each symbol with a sentiment label for the session. Use it to gate entries for that hour only. |
| **Paid news APIs (Benzinga, Polygon.io news)** | Better data quality and coverage. | Costs $50-300+/month. PROJECT.md explicitly rules out paid APIs. ROI unproven at paper trading stage. | Free Yahoo Finance RSS + VADER gets you 80% of the value. Revisit paid APIs only after paper trading proves profitability. |
| **Reinforcement learning for exit decisions** | The RL module already exists; seems natural to use it for exits. | RL exit policies are extremely sensitive to reward function design. Getting the reward signal right requires months of careful design and validation. The RL module is already optional/experimental. Adding RL exits before basic stop/trail/target exits exist is building on a shaky foundation. | Build deterministic rule-based exits first (stop/trail/target). Only layer RL exit logic after rule-based exits are validated over 2+ weeks of paper trading. |
| **Fully dynamic ML-predicted stop distances** | "Let the model decide the stop" sounds sophisticated. | An ML model predicting optimal stop distance requires labeled training data of "correct" stop distances — which you don't have. It will overfit to whatever stop distances happened to not trigger in the training set. | Use ATR × multiplier for stop distance. This is grounded in market microstructure theory and has decades of empirical backing. Simple is more robust. |
| **News scraping with Selenium/BeautifulSoup from Yahoo Finance web pages** | Yahoo deprecated its API; scraping the page gives raw data. | Yahoo Finance actively detects and blocks scrapers. Selenium adds a headless browser dependency. Fragile — breaks whenever Yahoo changes HTML. | Use the `stocknews` PyPI package (wraps Yahoo Finance RSS) or direct RSS feed parsing via `feedparser`. RSS feeds are stable interfaces Yahoo maintains intentionally. |
| **Per-stock sentiment model fine-tuning** | FinBERT fine-tuned on AAPL news would be more accurate. | Requires labeled sentiment data per ticker, GPU training time, and constant retraining as company news profiles shift. Overkill for a paper-trading bot. | Use off-the-shelf VADER (no GPU, works at runtime) or pre-trained FinBERT from Hugging Face (inference only). Both are sufficient for a confirming filter. |
| **HMM (Hidden Markov Model) regime detection** | More statistically principled than KMeans. | Significantly more implementation complexity than KMeans. The bot already has a KMeans MarketRegimeDetector scaffold — HMM would require replacing it with unproven benefit at this stage. | Wire the existing KMeans detector into the trading gate. Validate that it actually helps before evaluating alternatives. |
| **Multi-timeframe signal aggregation** | Combining 1-min, 5-min, daily signals gives fuller picture. | Managing multiple data streams, aligning timestamps, and avoiding look-ahead bias across timeframes is a major engineering problem. The existing 5-min bar architecture is already partially broken (merge conflict in main.py). | Fix the existing single-timeframe pipeline first. Only add timeframes after the base loop is solid and validated. |

---

## Feature Dependencies

```
[Hard Stop Loss]
    └──requires──> [Position Entry Tracking] (must record entry price + ATR at open)
    └──requires──> [Trade Context Logging - Entry]

[Trailing Stop]
    └──requires──> [Hard Stop Loss] (trailing stop replaces hard stop after profit threshold)
    └──requires──> [Hourly Portfolio Scan] (already exists - trail computed per bar check)

[Take Profit]
    └──requires──> [Position Entry Tracking] (must record entry price + target at open)

[Trade Context Logging - Exit]
    └──requires──> [Trade Context Logging - Entry] (exit log references entry log row)

[Daily Loss Circuit Breaker]
    └──requires──> [Daily P&L Tracking] (read from IBKR account summary at loop start)

[Max Drawdown Halt]
    └──requires──> [High-Water Mark Persistence] (store in DB)

[Regime-Gated Entry Suppression]
    └──requires──> [MarketRegimeDetector wired to trade.py] (scaffold exists, not wired)
    └──requires──> [VIX data feed from IBKR] (new data fetch needed)

[News Sentiment Filter]
    └──requires──> [Sentiment Scoring Pipeline] (VADER or FinBERT inference)
    └──requires──> [RSS / PRAW data fetch] (new module needed)
    └──enhances──> [Regime-Gated Entry Suppression] (sentiment + regime = stronger gate)

[Partial Profit-Taking]
    └──requires──> [Hard Stop Loss] (need basic exits before complex exits)
    └──requires──> [Take Profit] (partial take = fraction of TP order)
    └──requires──> [IBKR partial order capability verified] (check paper account limits)

[RL Exit Logic] (deferred)
    └──requires──> [Hard Stop Loss validated] (baseline must exist first)
    └──requires──> [Trade Context Logging] (need labeled exit data for RL reward design)
```

### Dependency Notes

- **Hard stop loss must come first.** Everything else — trailing stops, partial profits, regime gates — is useless if a position can still hold forever through a disaster.
- **Trade context logging is a prerequisite for improving the bot.** Without it, you are flying blind during paper trading validation.
- **Sentiment filter requires regime gate to be useful.** Standalone sentiment is too noisy. Sentiment confirming a regime signal is a meaningful combined filter.
- **Partial profit-taking conflicts with daily loss circuit breaker** if implemented naively — closing half a position that is a winner does not help when the portfolio is in daily drawdown from other positions. Ensure circuit breaker logic checks total portfolio P&L, not per-trade P&L.

---

## MVP Definition

### Launch With (This Milestone — Must Have Before Meaningful Paper Trading)

- [ ] **Hard stop loss per position** — attach ATR-based stop at entry, check on every bar
- [ ] **Take profit target per position** — set 2:1 R:R target at entry
- [ ] **Trade context logging (entry + exit reason)** — structured DB record per trade open/close
- [ ] **Daily loss circuit breaker** — halt new entries if daily P&L < -2%

### Add After Validation (v1.x — Once Bot Is Running Cleanly for 1 Week)

- [ ] **Trailing stop** — activate after 1× ATR profit, trail at 1.5× ATR below high
- [ ] **Regime-gated entry suppression** — wire KMeans detector + VIX threshold to entry gate
- [ ] **Hold-time stop** — exit stalled positions after 5 trading days
- [ ] **Session close flat for intraday signals** — tag entry type, force exit before 3:55 PM

### Future Consideration (v2+ — After 2 Weeks Consistent Paper Profitability)

- [ ] **News sentiment filter** — Yahoo Finance RSS + VADER as confirming entry gate
- [ ] **Partial profit-taking at multiple levels** — close 50% at 1:1 R:R, trail remainder
- [ ] **Max portfolio drawdown halt** — high-water mark tracking + halt at -7%
- [ ] **Per-stock ATR multiplier tuning** — different multipliers for high-vol vs low-vol names

---

## Feature Prioritization Matrix

| Feature | Trader Value | Implementation Cost | Priority |
|---------|-------------|---------------------|----------|
| Hard stop loss | HIGH | LOW | P1 |
| Take profit target | HIGH | LOW | P1 |
| Trade context logging | HIGH | LOW | P1 |
| Daily loss circuit breaker | HIGH | LOW | P1 |
| Trailing stop | HIGH | MEDIUM | P1 |
| Regime-gated entry suppression | MEDIUM | MEDIUM | P2 |
| Hold-time stop | MEDIUM | LOW | P2 |
| Session close flat | MEDIUM | LOW | P2 |
| News sentiment filter | LOW-MEDIUM | MEDIUM | P3 |
| Partial profit-taking | MEDIUM | MEDIUM | P3 |
| Max drawdown halt | MEDIUM | LOW | P2 |

**Priority key:**
- P1: Must have for paper trading validation to be meaningful
- P2: Should have — adds robustness once exits are working
- P3: Nice to have — add after profitability is demonstrated

---

## Competitor Feature Analysis

This is a solo retail bot, not a commercial product. The relevant "competitors" are open-source retail algo frameworks.

| Feature | Freqtrade (crypto) | QuantConnect (multi-asset) | This Bot's Approach |
|---------|-------------------|---------------------------|---------------------|
| Stop loss | Built-in, ATR-based optional | Built-in, configurable | Build into hourly portfolio scan — attach stop price to Portfolio DB record |
| Trailing stop | Built-in, activates at threshold | Built-in | Compute in Python on each bar check — do not delegate to IBKR trailing stop order (opaque) |
| Take profit | Built-in, multiple levels | Built-in | Single TP first, add multiple levels in v1.x |
| Regime detection | Plugin-based, not native | Built-in (FRED, VIX data) | KMeans scaffold exists — wire it; add VIX via IBKR data |
| Sentiment | Optional, mostly crypto | Available as alpha data feed | Yahoo Finance RSS + VADER — confirming filter only |
| Trade logging | Extensive JSON logs | Full tearsheet | Structured DB rows — extend existing `TradingPerformanceMetrics` table |
| Circuit breaker | Max open trades limit | Custom rule-based | Daily loss % halt — check IBKR account summary at loop start |

---

## Sources

- [AI Trading Bot Risk Management: Complete 2025 Guide — 3commas](https://3commas.io/blog/ai-trading-bot-risk-management-guide-2025) — MEDIUM confidence
- [Advanced Stop-Loss Logic for AI Bots in 2025 — 3commas](https://3commas.io/blog/optimizing-your-trades-advanced-stop-loss-and-take) — MEDIUM confidence
- [Stop Loss Strategies for Algorithmic Trading — TradersPost](https://blog.traderspost.io/article/stop-loss-strategies-algorithmic-trading) — MEDIUM confidence
- [Take Profit in Algorithmic Trading — FasterCapital](https://fastercapital.com/content/Take-Profit-in-Algorithmic-Trading--Enhancing-Trading-Bots.html) — LOW confidence (content aggregator)
- [Sentiment Analysis in Algo Trading — robots4forex](https://robots4forex.com/algorithmic-trading/sentiment-analysis-in-algorithmic-trading-using-news-and-social-media-for-trading-signals/) — MEDIUM confidence
- [Reddit Sentiment Analysis Strategy — Alpaca Markets](https://alpaca.markets/learn/reddit-sentiment-analysis-trading-strategy) — MEDIUM confidence (practitioner source)
- [Social media attention and retail investor behavior (WallStreetBets) — ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S1057521924006537) — HIGH confidence (peer-reviewed)
- [Democratisation of retail trading: Reddit WSB vs investment bank analysts — Taylor & Francis 2024](https://www.tandfonline.com/doi/full/10.1080/2573234X.2024.2354191) — HIGH confidence (peer-reviewed)
- [AI Market Regime Detection — Syntium Algo](https://syntiumalgo.com/ai-market-regime-detection/) — LOW confidence
- [3 Effective Ways to Detect Market Regimes — Medium/Coding Nexus, Nov 2025](https://medium.com/coding-nexus/3-effective-ways-to-detect-market-regimes-ec361712fbee) — LOW confidence
- [Step-by-Step Python Guide for Regime-Specific Trading — QuantInsti](https://blog.quantinsti.com/regime-adaptive-trading-python/) — MEDIUM confidence
- [Reducing Drawdown: 7 Risk-Management Techniques — Tradetron](https://tradetron.tech/blog/reducing-drawdown-7-risk-management-techniques-for-algo-traders) — MEDIUM confidence
- [Risk Management in Algorithmic Trading: Beyond Stop-Loss — AlgoBulls](https://algobulls.com/blog/algo-trading/risk-management) — MEDIUM confidence
- [Why Overfitting Is a Risk to Your Algo Trading — uTrade Algos](https://www.utradealgos.com/blog/why-overfitting-is-a-risk-to-your-algo-trading-success-and-how-to-avoid-it) — MEDIUM confidence
- [Top Algo Trading Mistakes — EFX Algo 2025](https://efxalgo.com/2025/12/18/top-algo-trading-mistakes-and-how-to-avoid-them/) — LOW confidence
- [stocknews PyPI package](https://pypi.org/project/stocknews/) — MEDIUM confidence (official package page)

---

*Feature research for: Algorithmic Stock Trading Bot (exits, sentiment, risk management)*
*Researched: 2026-02-27*
