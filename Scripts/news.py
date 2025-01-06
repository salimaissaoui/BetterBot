import logging
import requests
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def fetch_articles_for_symbol(symbol, days_back=365, api_key=None):
    """
    Fetches news articles for the last 'days_back' days from NewsAPI.
    Increase days_back to get more news data.
    """
    # Provide your NewsAPI key here or from env
    NEWSAPI_KEY = api_key or "YOUR_NEWSAPI_KEY"
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
