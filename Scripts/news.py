import logging
import os
import requests
import feedparser
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Global cache for ticker sentiment [SENT-01]
_sentiment_cache = {}

class SentimentAnalyzer:
    """Handles fetching and scoring news sentiment for tickers."""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

    def get_ticker_sentiment(self, ticker: str) -> float:
        """
        Fetch Yahoo Finance RSS headlines and return average VADER compound score.
        Uses 15-minute TTL cache to avoid rate limits.
        """
        global _sentiment_cache
        now = datetime.now()
        
        # 1. Check Cache
        if ticker in _sentiment_cache:
            entry = _sentiment_cache[ticker]
            if now - entry['timestamp'] < timedelta(minutes=15):
                return entry['score']

        # 2. Fetch RSS
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        try:
            # feedparser doesn't support headers directly in some versions, but we can pass agent
            feed = feedparser.parse(url, agent=self.user_agent)
            
            if not feed.entries:
                logging.warning(f"[Sentiment] No headlines found for {ticker}")
                return 0.0

            # 3. Score Headlines
            scores = []
            for entry in feed.entries[:10]:  # Use last 10 headlines
                headline = entry.title
                sentiment = self.analyzer.polarity_scores(headline)
                scores.append(sentiment['compound'])
            
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # 4. Update Cache
            _sentiment_cache[ticker] = {
                'score': avg_score,
                'timestamp': now
            }
            logging.info(f"[Sentiment] {ticker} scored {avg_score:.3f} from {len(scores)} headlines")
            return avg_score

        except Exception as e:
            logging.error(f"[Sentiment] Error fetching {ticker}: {e}")
            return 0.0

# Singleton instance
_analyzer = SentimentAnalyzer()

def get_ticker_sentiment(ticker: str) -> float:
    """Wrapper for the SentimentAnalyzer singleton."""
    return _analyzer.get_ticker_sentiment(ticker)

def fetch_articles_for_symbol(symbol, days_back=365, api_key=None):
    """
    Fetches news articles for the last 'days_back' days from NewsAPI.
    Increase days_back to get more news data.
    """
    # Provide your NewsAPI key here or from env
    NEWSAPI_KEY = api_key or os.getenv("NEWSAPI_KEY")
    if not NEWSAPI_KEY:
        logging.warning("No NewsAPI key configured. Set NEWSAPI_KEY environment variable.")
        return []
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    from_date_str = start_date.strftime("%Y-%m-%d")
    to_date_str = end_date.strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f"{symbol} stock",
        'from': from_date_str,
        'to': to_date_str,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 20,
        'apiKey': NEWSAPI_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 429:
            logging.info(f"Rate limit reached for {symbol}. Skipping article fetching.")
            return []
        
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        article_texts = []
        for article in articles:
            text = article.get('description') or article.get('title')
            if text:
                article_texts.append(text)
        return article_texts
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching articles for {symbol}: {e}")
        return []


def analyze_sentiment(articles):
    """Very basic sentiment analysis based on keyword occurrence."""
    if not articles:
        return 0.0
    positive_keywords = ["profit", "growth", "beat", "surge", "upgrade", "positive"]
    negative_keywords = ["loss", "drop", "downgrade", "negative", "miss", "fraud"]
    pos_count = 0
    neg_count = 0
    for article in articles:
        text = article.lower()
        if any(word in text for word in positive_keywords):
            pos_count += 1
        if any(word in text for word in negative_keywords):
            neg_count += 1
    total = len(articles)
    sentiment_score = (pos_count - neg_count) / total
    return sentiment_score


def get_news_sentiment(symbol, lookback_days=7, api_key=None):
    """Fetches and analyzes sentiment from news articles for 'symbol'."""
    articles = fetch_articles_for_symbol(symbol, days_back=lookback_days, api_key=api_key)
    sentiment = analyze_sentiment(articles)
    return sentiment
