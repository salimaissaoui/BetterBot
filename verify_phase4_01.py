from dotenv import load_dotenv
load_dotenv()

import sys
import time
from datetime import datetime, timedelta
from Scripts.news import get_ticker_sentiment, _sentiment_cache

def test_sentiment_pipeline():
    print("=== Phase 4-01: Sentiment Pipeline Test ===")
    
    ticker = "AAPL"
    
    # 1. Fetch first time
    print(f"Fetching sentiment for {ticker}...")
    score1 = get_ticker_sentiment(ticker)
    print(f"Score 1: {score1:.3f}")
    
    assert ticker in _sentiment_cache, "Ticker not in cache after fetch"
    entry1 = _sentiment_cache[ticker]
    ts1 = entry1['timestamp']
    
    # 2. Fetch again immediately (should be from cache)
    print(f"Fetching {ticker} again (should be cached)...")
    score2 = get_ticker_sentiment(ticker)
    assert score1 == score2, "Scores differ between immediate fetches"
    
    entry2 = _sentiment_cache[ticker]
    assert ts1 == entry2['timestamp'], "Cache timestamp changed on immediate fetch"
    print("PASS: Immediate fetch correctly cached")
    
    # 3. Test TTL (mocking timestamp)
    print("Testing TTL expiration...")
    _sentiment_cache[ticker]['timestamp'] = datetime.now() - timedelta(minutes=16)
    score3 = get_ticker_sentiment(ticker)
    
    entry3 = _sentiment_cache[ticker]
    assert entry3['timestamp'] > ts1, "Cache did not refresh after TTL expired"
    print(f"PASS: Cache refreshed after 16 mins (Score: {score3:.3f})")
    
    # 4. Test invalid ticker
    print("Testing invalid ticker...")
    bad_ticker = "NONEXISTENT_TICKER_12345"
    score_bad = get_ticker_sentiment(bad_ticker)
    assert score_bad == 0.0, f"Bad ticker should return 0.0, got {score_bad}"
    print("PASS: Invalid ticker returned 0.0")
    
    print("=== All Phase 4-01 tests PASSED ===")

if __name__ == "__main__":
    test_sentiment_pipeline()
