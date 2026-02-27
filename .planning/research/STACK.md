# Stack Research

**Domain:** Python algorithmic stock trading bot (augmentation milestone)
**Researched:** 2026-02-27
**Confidence:** MEDIUM-HIGH (versions verified via PyPI; patterns verified via multiple sources)

---

## Context: What Already Exists

The existing stack is documented in `.planning/codebase/STACK.md`. This research covers only the **additions** needed for the active milestone:

1. Sentiment analysis (free news sources + NLP scoring)
2. Exit strategy logic (stop loss, trailing stop, take profit)
3. Market regime detection (suppress trading in unfavorable conditions)
4. Risk gating (VIX-based filter)

The existing `news.py` uses NewsAPI (paid, rate-limited) with keyword matching — both problems to solve. The `advanced_features.py` has a `MarketRegimeDetector` using KMeans clustering, but the `_add_regime_features` method is a stub returning all zeros — the detection is not wired into any trading gate. The exit logic in `trade.py` closes positions only when the ML prediction flips — no hard stops, no trailing stops, no take profit.

---

## Recommended Stack Additions

### Sentiment Analysis: News Ingestion

| Library | Version | Purpose | Why Recommended |
|---------|---------|---------|-----------------|
| `feedparser` | 6.0.12 (Sep 2025) | Parse Yahoo Finance RSS feeds per ticker | Zero API key required; Yahoo Finance RSS `https://finance.yahoo.com/rss/headline?s={TICKER}` is free and reliable; existing `requests` + `BeautifulSoup4` in stack can scrape but feedparser handles malformed XML gracefully |
| `praw` | 7.8.1 (Oct 2024) | Reddit API wrapper for r/wallstreetbets, r/investing | Official Reddit API; free at script-application tier (100 req/min); requires a Reddit app registration (5-min setup, no cost); far more reliable than scraping |
| `yfinance` | 1.2.0 (Feb 2026) | Pull news headlines alongside price data | Already in codebase (`advanced_features.py` imports it with try/except); `Ticker.news` returns recent articles; do NOT use as primary news source — the `get_news()` ticker relevance issue means articles drift off-topic. Use as secondary/backup. |

**Confidence:** HIGH for feedparser (verified via PyPI). MEDIUM for PRAW (widely used pattern, documented community implementations). MEDIUM for yfinance news (known ticker-relevance bug in recent versions, confirmed via GitHub issue #1956).

**What NOT to use:**
- `stocknews` — only 167 weekly downloads, classified inactive by Snyk, version 0.9.11 last updated 2023. Avoid.
- `NewsAPI` (current `news.py` approach) — free tier limited to 1-month lookback and 100 req/day; rate limit (429) already handled in the code but the cap is too restrictive for live trading.
- `FinNews` (scaratozzolo) — low maintenance, inconsistent RSS source availability.

---

### Sentiment Analysis: NLP Scoring

| Library | Version | Purpose | Why Recommended |
|---------|---------|---------|-----------------|
| `vaderSentiment` | 3.3.2 (stable) | Fast lexicon-based sentiment scoring for headlines and Reddit titles | No GPU required; processes thousands of headlines in milliseconds; tuned for social media slang, handles financial abbreviations; `compound` score is a single -1 to +1 float that slots directly into the feature vector; existing `analyze_sentiment()` in `news.py` uses keyword matching — VADER is strictly better and still zero-dependency |
| `transformers` | 5.2.0 (Feb 2026) | FinBERT inference for deeper financial sentiment | Use `ProsusAI/finbert` or `yiyanghkust/finbert-tone` from HuggingFace Hub; fine-tuned on financial communications; outputs positive/negative/neutral probabilities; significantly higher accuracy than VADER on earnings language |
| `torch` | latest compatible with transformers 5.x | Required PyTorch backend for transformers | CPU inference is viable for a few headlines per cycle — no GPU needed at paper trading scale |

**When to use VADER vs FinBERT:**
- **VADER** — Use as the primary signal for RSS headlines and Reddit post titles. Fast enough to run on every bar. Confidence: HIGH.
- **FinBERT** — Use for end-of-day batch scoring of full article text or earnings-related headlines. Too slow (0.5–2s per inference on CPU) for real-time per-bar use. Confidence: MEDIUM (verified HuggingFace Hub availability; CPU latency is from training data — LOW confidence on exact latency).

**What NOT to use:**
- `TextBlob` — General-purpose sentiment, not tuned for finance. VADER consistently outperforms it on financial text and is already in the NLP community standard for this domain.
- Raw keyword matching (current `news.py`) — No negation handling, no intensity weighting. VADER replaces this with no added complexity.
- Paid APIs (Benzinga, Polygon.io sentiment endpoints) — Explicitly out of scope per PROJECT.md.

---

### Exit Strategy: Implementation Pattern

No new library is required for the core exit logic — it is pure Python state management using the existing stack. The pattern to implement:

| Component | Implementation Approach | Library Dependency |
|-----------|------------------------|-------------------|
| ATR-based hard stop loss | Calculate ATR from existing `ta` library or from pandas rolling window; store `entry_price - (atr_multiplier * atr)` per position in `active_positions` dict | `ta 0.11.0` (already installed) |
| Trailing stop (Chandelier Exit) | Track `highest_high_since_entry` per position; stop = `highest_high - (atr_multiplier * atr)`; update on every bar | pandas (already installed) |
| Take profit | Store `entry_price + (risk_reward_ratio * initial_stop_distance)` per position; close on touch | None |
| Time-based exit | Store `entry_time` per position; close if `now - entry_time > max_hold_hours` | datetime (stdlib) |
| Bracket orders via IBKR | `ib.bracketOrder()` in ib_insync/ib_async creates parent + stop-loss + take-profit as linked orders | ib_insync (existing) or ib_async |

**Confidence:** HIGH for ATR trailing stop pattern (confirmed across multiple sources). MEDIUM for IBKR bracket order implementation (verified ib_insync `bracketOrder()` method exists; confirmed child orders need individual `placeOrder()` calls with `ib.sleep(1)` between them).

**The Chandelier Exit formula (standard):**
```
stop_long  = highest_high(period=22) - atr(period=22) * multiplier(default=3.0)
stop_short = lowest_low(period=22)   + atr(period=22) * multiplier(default=3.0)
```
This is the industry standard trailing stop for trend-following algos. Period=22 and multiplier=3.0 are the widely-cited defaults; ATR period 14 with multiplier 2.0 is more common for shorter hold times.

**Critical implementation note:** The current `active_positions` dict in `trade.py` stores only `'long'` or `'short'` strings. It must be extended to a dict of dicts storing `entry_price`, `entry_time`, `stop_price`, `take_profit_price`, `highest_price_since_entry`.

---

### ib_insync Replacement: ib_async

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| `ib_async` | 2.1.0 (Dec 2025) | Drop-in replacement for ib_insync | ib_insync was archived on March 14, 2024 after its creator's passing. `ib_async` (github.com/ib-api-reloaded/ib_async) is the community-maintained successor. Migration is import-name-only: `from ib_insync import *` becomes `from ib_async import *`. No API changes required. Production/Stable status. Python 3.10+ required. |

**Confidence:** HIGH (verified PyPI version, GitHub status of both repos, community discussion confirming drop-in compatibility).

**Urgency:** This is a maintenance risk, not an immediate blocker. The existing ib_insync 3.2.0 will continue to function — it is not receiving security patches or IBKR API updates. Migrate before TWS API version bumps break compatibility.

**What NOT to do:** Do not continue installing `ib-insync` from PyPI for new environments. Pin to `ib_async>=2.1.0` in requirements.txt going forward.

---

### Market Regime Detection

| Library | Version | Purpose | Why Recommended |
|---------|---------|---------|-----------------|
| `hmmlearn` | 0.3.3 (Oct 2024) | Hidden Markov Model regime detection | Industry-standard for financial regime detection; `GaussianHMM` with 2–4 hidden states on daily returns is the most common practical approach; scikit-learn-compatible API; fits naturally alongside the existing KMeans regime detector in `advanced_features.py`; limited-maintenance status (no new releases in 12 months) but stable and sufficient |
| `statsmodels` | 0.14.6 (Dec 2025) | Markov Regime Switching regression | `statsmodels.tsa.regime_switching.MarkovRegression` provides a statistically grounded regime-switching model with transition probabilities; more interpretable than KMeans; actively maintained; already a transitive dependency risk — check if in existing environment |
| `yfinance` | 1.2.0 | Fetch `^VIX` for volatility gating | `yf.download('^VIX', period='5d')` returns VIX levels free with no API key; VIX > 30 = suppress new longs; VIX > 40 = suppress all new positions |

**Confidence:** HIGH for hmmlearn (PyPI verified, multiple 2024-2025 sources confirm it as the standard). HIGH for statsmodels (PyPI verified, official docs confirm MarkovRegression in stable 0.14.6). HIGH for yfinance VIX approach (Kaggle tutorial confirms `^VIX` ticker works, approach documented on multiple sources).

**What NOT to use:**
- `pykalman` — Despite active development (version 0.11.2, Jan 2026), Kalman filters are better suited for pairs trading / dynamic hedge ratio estimation, not regime classification. Adds complexity without better classification accuracy than HMM for this use case.
- The existing KMeans approach alone — KMeans has no notion of state transition probability, assigns regimes arbitrarily across fits, and is non-deterministic. It should be replaced by or augmented with HMM or Markov switching.

**Recommended regime detection architecture:**

```
Primary gate: VIX level from yfinance
  VIX < 20 → normal trading
  20 ≤ VIX < 30 → reduce position sizing 50%
  VIX ≥ 30 → suppress all new entries

Secondary signal: GaussianHMM (2 states: low-vol / high-vol)
  State 0 (low volatility) → allow trading
  State 1 (high volatility) → tighten thresholds

Tertiary: existing KMeans MarketRegimeDetector (bullish/bearish/sideways/volatile)
  Use as feature input to the ML model, not as a hard gate
```

The VIX gate is the fastest to implement and has the highest signal-to-noise ratio. HMM adds a forward-looking regime filter. The existing KMeans detector is already computing features — wire its output as ML input features rather than using it as a hard trading gate.

---

### Risk Gating Pattern

No new library needed. The pattern:

```python
# At top of execute_trade():
vix_data = yf.download('^VIX', period='2d', progress=False)
current_vix = vix_data['Close'].iloc[-1]

if current_vix >= 40:
    logging.warning(f"VIX={current_vix:.1f} — suppressing all new entries (crisis regime)")
    return

if current_vix >= 30:
    logging.info(f"VIX={current_vix:.1f} — elevated volatility, reducing position sizing")
    # halve notional / position size
```

Cache the VIX fetch (e.g. in a module-level variable with a timestamp) so it is not fetched on every bar — once per 15 minutes is sufficient.

**Confidence:** MEDIUM (pattern derived from multiple sources; exact VIX thresholds are empirical conventions, not universal constants — validate against this bot's specific behavior).

---

## Installation

```bash
# Sentiment: News ingestion
pip install feedparser==6.0.12
pip install praw==7.8.1

# Sentiment: NLP scoring
pip install vaderSentiment==3.3.2
pip install "transformers[torch]==5.2.0"  # Optional: only if FinBERT batch scoring desired

# Regime detection
pip install hmmlearn==0.3.3
pip install statsmodels==0.14.6  # May already be present as transitive dep

# IBKR replacement (non-breaking migration)
pip install ib_async==2.1.0
# Then in requirements.txt: remove ib-insync, add ib_async>=2.1.0

# yfinance upgrade (already in stack, confirm version)
pip install --upgrade yfinance  # Target: 1.2.0
```

---

## Alternatives Considered

| Category | Recommended | Alternative | When to Use Alternative |
|----------|-------------|-------------|------------------------|
| Sentiment NLP | VADER + FinBERT | TextBlob | Never for financial text — VADER dominates on financial/social media corpora |
| Free news source | feedparser (Yahoo RSS) | direct `requests` to RSS URL | If feedparser dependency is undesirable — functionally equivalent but less XML-error-tolerant |
| Reddit sentiment | PRAW | pushshift.io / Reddit scraping | Never — pushshift is dead; scraping violates ToS |
| Regime detection | hmmlearn GaussianHMM | KMeans (existing) | Keep KMeans as ML *feature input*; do not use as the hard trading gate |
| Regime detection | statsmodels MarkovRegression | pykalman Kalman filter | pykalman if implementing pairs trading or dynamic beta; not needed for this use case |
| IBKR client | ib_async | raw `ibapi` from IBKR | ibapi is verbose and callback-based; ib_async is the right abstraction for this codebase |
| Exit execution | Python state machine + IBKR order | Pure IBKR bracket order | Hybrid: place bracket order at entry for hard stop, manage trailing stop in Python since IBKR trailing stop orders have edge cases with paper accounts |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `ib-insync` (original) | Archived March 2024 — no IBKR API updates or security fixes | `ib_async 2.1.0` |
| `stocknews` | Effectively abandoned (167 downloads/week, no 2024+ releases) | `feedparser` + Yahoo Finance RSS directly |
| `NewsAPI` (current) | Free tier: 100 req/day, 1-month lookback — insufficient for backtesting; paid for historical | Yahoo Finance RSS (feedparser) + PRAW |
| `pykalman` (for regime) | Right tool for wrong job — Kalman filters are regression estimators, not classifiers | `hmmlearn` GaussianHMM for regime classification |
| `TextBlob` | General-purpose; does not handle financial lexicon (e.g. "beat" = positive, "miss" = negative); VADER outperforms on every financial NLP benchmark | `vaderSentiment` |
| Real-time FinBERT per bar | CPU inference 0.5–2s per call; will block the event loop on every 5-second bar | Run FinBERT in batch end-of-day; use VADER for real-time per-bar |
| Full re-architecture of `MarketRegimeDetector` | The KMeans code works; rewrites risk regressions | Augment: add VIX gate + HMM layer; keep KMeans as feature source |

---

## Version Compatibility

| Package | Version in Stack | New Version | Compatibility Notes |
|---------|-----------------|-------------|---------------------|
| `ib-insync` 3.2.0 | existing | `ib_async` 2.1.0 | Import rename only: `ib_insync` → `ib_async`; requires Python 3.10+ (check venv) |
| `scikit-learn` 1.2.0 | existing | n/a | hmmlearn 0.3.3 requires scikit-learn >= 0.16 — compatible |
| `pandas` 2.2.3 | existing | n/a | statsmodels 0.14.6 requires pandas >= 1.4 — compatible |
| `transformers` 5.2.0 | new | n/a | Requires PyTorch 2.4+ — if adding FinBERT, pin torch accordingly |
| `yfinance` (unknown) | existing import | 1.2.0 | Check installed version; upgrade if below 0.2.40 for `^VIX` reliability |

---

## Sources

- feedparser PyPI — https://pypi.org/project/feedparser/ — version 6.0.12 verified
- PRAW PyPI — https://pypi.org/project/praw/ — version 7.8.1 verified
- vaderSentiment PyPI — https://pypi.org/project/vaderSentiment/ — version 3.3.2 verified
- transformers PyPI — https://pypi.org/project/transformers/ — version 5.2.0 verified
- hmmlearn PyPI — https://pypi.org/project/hmmlearn/ — version 0.3.3, Oct 2024 verified
- statsmodels PyPI — https://pypi.org/project/statsmodels/ — version 0.14.6, Dec 2025 verified
- pykalman PyPI — https://pypi.org/project/pykalman/ — version 0.11.2, Jan 2026 verified (not recommended for this use case)
- ib_async PyPI — https://pypi.org/project/ib_async/ — version 2.1.0, Dec 2025 verified
- ib_async GitHub — https://github.com/ib-api-reloaded/ib_async — confirmed drop-in replacement
- ib_insync GitHub — https://github.com/erdewit/ib_insync — confirmed archived March 2024
- yfinance PyPI — https://pypi.org/project/yfinance/ — version 1.2.0, Feb 2026 verified
- yfinance news issue — https://github.com/ranaroussi/yfinance/issues/1956 — ticker relevance problem confirmed
- FinBERT HuggingFace — https://huggingface.co/ProsusAI/finbert — available, no version pinning needed (loaded from Hub)
- statsmodels MarkovRegression docs — https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html — MEDIUM confidence
- hmmlearn for trading — https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/ — MEDIUM confidence (WebSearch, multiple corroborating sources)
- ATR chandelier exit — https://www.quantifiedstrategies.com/chandelier-exit-strategy/ — MEDIUM confidence (WebSearch)
- IBKR bracket order ib_insync — https://copyprogramming.com/howto/how-to-replicate-bracket-orders-functionality-using-interactive-brokers-ib-api-ib-insync-for-python — MEDIUM confidence (WebSearch)

---

*Stack research for: BetterBot — augmentation milestone (sentiment, exits, regime detection)*
*Researched: 2026-02-27*
