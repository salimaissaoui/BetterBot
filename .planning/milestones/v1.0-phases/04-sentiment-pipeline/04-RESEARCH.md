# Phase 4: Sentiment Pipeline - Research

**Researched:** 2026-02-27
**Domain:** Financial Sentiment Analysis / RSS Pipeline
**Confidence:** HIGH

## Summary

Phase 4 introduces a sentiment gate to the entry logic. The system will fetch recent headlines for a ticker from Yahoo Finance via RSS, score them using VADER, and suppress entries if the aggregate sentiment is negative (composite score < -0.05). To maintain efficiency and avoid rate limiting, a 15-minute TTL cache will be implemented.

**Primary recommendation:** Use `feedparser` with a custom `User-Agent` to fetch Yahoo Finance RSS, and `vaderSentiment` for lightweight scoring. Mirror the TTL cache pattern used for VIX in `Scripts/utils.py`.

## User Constraints

### Locked Decisions
- **Source:** Yahoo Finance RSS (free, no API key).
- **Library:** `feedparser` for fetching, `vaderSentiment` for scoring.
- **Threshold:** Suppress entry if VADER composite score < -0.05.
- **Cache:** 15-minute TTL per ticker.

### Claude's Discretion
- **RSS URL:** Multiple legacy URLs exist; research recommends the `feeds.finance.yahoo.com` endpoint with a browser-like User-Agent.
- **Aggregation:** Averaging the compound scores of the last 10 headlines is recommended for stability.
- **Implementation Location:** `Scripts/sentiment.py` should be created to encapsulate this logic, then wired into `Scripts/trade.py`.

### Deferred Ideas (OUT OF SCOPE)
- **FinBERT:** Batch scoring of full article text (deferred to v2).
- **Social Media:** Reddit/WSB monitoring (deferred to v2).
- **Article Body Parsing:** Only headlines are required for Phase 4.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SENT-01 | Fetch Yahoo Finance RSS headlines per ticker using feedparser | Verified URL and User-Agent requirements for `feedparser` |
| SENT-02 | Suppress entries if VADER score < -0.05 | Documented `vaderSentiment` usage and threshold logic |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| feedparser | 6.0.11 | Parse RSS feeds | Universal Python standard for RSS; handles malformed XML well. |
| vaderSentiment | 3.3.2 | Sentiment analysis | Optimized for social media and short text like headlines; no training needed. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|--------------|
| yfinance | (Already in project) | Fallback news source | Use if RSS endpoint is completely blocked; already has `ticker.news`. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| feedparser | requests + bs4 | More manual work to parse RSS structure. |
| vaderSentiment | TextBlob | VADER is generally superior for financial "intensity" (e.g., "SURGE", "CRASH"). |

**Installation:**
```bash
pip install feedparser vaderSentiment
```

## Architecture Patterns

### Recommended Implementation
1. **`Scripts/sentiment_pipeline.py`**: A new service class to handle fetching, scoring, and caching.
2. **TTL Cache**: Store `{ticker: {'score': float, 'timestamp': datetime}}` globally in the pipeline module.

### Anti-Patterns to Avoid
- **No User-Agent:** Yahoo blocks the default `urllib` or `feedparser` agent with 403 Forbidden.
- **Blocking I/O in Loop:** `feedparser.parse` is synchronous. Ensure `ib.sleep(0.1)` or similar is handled, though a single RSS fetch is usually fast enough for a 30s bar loop.
- **Over-fetching:** Requesting RSS on every 30s bar for every ticker will likely lead to an IP ban. The 15-minute TTL is critical.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sentiment Lexicon | Custom keyword list | vaderSentiment | VADER handles negations ("not good"), intensifiers ("extremely good"), and punctuation. |
| RSS Parsing | Regex/XML parser | feedparser | RSS has many versions (0.9, 1.0, 2.0, Atom); feedparser handles all transparently. |

## Common Pitfalls

### Pitfall 1: 403 Forbidden / 404 Not Found
**What goes wrong:** Yahoo Finance RSS endpoints often return errors if a Browser User-Agent is not provided.
**How to avoid:** Set `agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'` in `feedparser.parse()`.

### Pitfall 2: Empty Feed
**What goes wrong:** Tickers with low volume or no news return an empty feed, which can lead to division by zero if not handled.
**How to avoid:** Return a neutral score (0.0) if `len(feed.entries) == 0`.

## Code Examples

### RSS Fetching & VADER Scoring
```python
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_ticker_sentiment(ticker: str) -> float:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    # CRITICAL: User-Agent is mandatory
    feed = feedparser.parse(url, agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
    
    if not feed.entries:
        return 0.0 # Neutral fallback
    
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    
    for entry in feed.entries[:10]: # Check last 10 headlines
        result = analyzer.polarity_scores(entry.title)
        scores.append(result['compound'])
        
    return sum(scores) / len(scores) if scores else 0.0
```

### 15-Minute TTL Cache Pattern
```python
from datetime import datetime, timedelta

_sentiment_cache = {}

def get_cached_sentiment(ticker: str) -> float:
    now = datetime.now()
    if ticker in _sentiment_cache:
        cached_val = _sentiment_cache[ticker]
        if now - cached_val['timestamp'] < timedelta(minutes=15):
            return cached_val['score']
    
    # Refresh cache
    score = get_ticker_sentiment(ticker)
    _sentiment_cache[ticker] = {
        'score': score,
        'timestamp': now
    }
    return score
```

## Entry Filter Location

**File:** `Scripts/trade.py`
**Function:** `execute_trade(pred_results, model)`

The filter should be placed after the circuit breaker check but before the trade execution logic (around line 219).

```python
# --- Scripts/trade.py snippet ---

for symbol, prob in pred_results:
    # ... (Exit Checks) ...

    # ----------------------------------------------------------------
    # CIRCUIT BREAKER CHECK
    # ----------------------------------------------------------------
    if exit_manager.is_circuit_breaker_active():
        continue

    # ----------------------------------------------------------------
    # SENTIMENT GATE (Phase 4)
    # ----------------------------------------------------------------
    sentiment_score = get_cached_sentiment(symbol)
    if sentiment_score < -0.05:
        logging.info(f"[{symbol}] Entry suppressed by sentiment: {sentiment_score:.3f}")
        continue
    
    # ... (Buy/Short Logic) ...
```

## Validation Architecture

> Skip this section entirely if workflow.nyquist_validation is false in .planning/config.json

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | none (use default) |
| Quick run command | `pytest verify_phase4_sentiment.py` |
| Full suite command | `pytest` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SENT-01 | Feedparser returns list of headlines | unit | `pytest tests/test_sentiment.py::test_rss_fetch` | ❌ Wave 0 |
| SENT-02 | Entry suppressed if score < -0.05 | integration | `pytest tests/test_sentiment.py::test_sentiment_gate` | ❌ Wave 0 |

## Sources

### Primary (HIGH confidence)
- Official `feedparser` documentation regarding User-Agent requirements.
- `vaderSentiment` GitHub README for scoring scale and thresholds.
- Project file `Scripts/utils.py` for existing TTL cache pattern.

### Secondary (MEDIUM confidence)
- Google Search results for "Yahoo Finance RSS 2025 status" (warns of intermittent availability).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Libraries are mature and well-documented.
- Architecture: HIGH - Follows existing patterns in the project.
- Pitfalls: HIGH - User-Agent and empty feed handling are known issues.

**Research date:** 2026-02-27
**Valid until:** 2026-03-27 (30 days)
