import os
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from alpaca_trade_api import REST, Stream
from alpaca_trade_api.rest import TimeFrame
from xgboost import XGBClassifier
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, TIMESTAMP, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from contextlib import contextmanager
import requests
from bs4 import BeautifulSoup
import re

# ============================================================
# Setup Logging
# ============================================================
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ============================================================
# Configuration
# ============================================================
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "PK023FOQLSDB6VR3UDRV")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "2CalVxgSuOnObttDwgBRcpQAbBMTXR4XkBa7qXdR")
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Stocks123")
DB_HOST = os.getenv("DB_HOST", "database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# We'll keep the WATCHLIST for streaming subscriptions only:
WATCHLIST = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "UNH", "V",
    "DIS", "PYPL", "ADBE", "NFLX", "INTC", "CMCSA", "PFE", "KO", "PEP", "CSCO",
    "T", "BA", "MRK", "ABT", "CRM", "XOM", "WMT", "COST", "VZ", "IBM",
    "ORCL", "CVX", "MDT", "NKE", "MCD", "LLY", "HON", "DHR", "TMO", "QCOM",
    "NEE", "LIN", "PM", "UNP", "LOW", "BMY", "AMGN", "TXN", "INTU", "C",
    "GE", "AXP", "SBUX", "MMM", "CAT", "NOW", "BLK",
    "GS", "RTX", "ISRG", "DE", "CHTR", "SYK", "AMT", "TJX", "CVS", "BKNG",
    "FIS", "USB", "PLD", "CB", "SPGI", "BDX", "ADI", "VRTX", "LMT",
    "CCI", "APD", "GILD", "TGT", "CL", "EW", "ICE", "GM",
    "ZTS", "PNC", "MS", "MO", "DUK", "CME", "ADP", "APTV", "SHW", "USB",
    "MDLZ", "COF", "MCO", "SYF", "CBRE"
]
WATCHLIST = list(set(WATCHLIST))

NOTIONAL = 1000
MODEL_LOOKBACK = 5
FEATURE_LAGS = 2
SHORT_MA = 5
LONG_MA = 20
RSI_PERIOD = 14
RETRAIN_FREQUENCY = 5

bar_count_since_last_train = 0
active_positions = {}  # {symbol: True/False or symbol: qty}
last_bar_timestamp = {}
historical_data_dict = {}
model = None  # Global model variable

# ============================================================
# Initialize Database Connection
# ============================================================
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
try:
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0, echo=False)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    Session = scoped_session(SessionFactory)
    logging.info("Connected to AWS RDS PostgreSQL database successfully.")
except Exception as e:
    logging.error(f"Failed to connect to the database: {e}")
    raise

Base = declarative_base()

class StockData(Base):
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    
    __table_args__ = (UniqueConstraint('symbol', 'timestamp', name='_symbol_timestamp_uc'),)

class CongressTradeData(Base):
    __tablename__ = 'congress_trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    timestamp = Column(TIMESTAMP, nullable=False)
    transaction_type = Column(String(10))
    representative = Column(String(100))
    amount = Column(Float)
    __table_args__ = (UniqueConstraint('symbol', 'timestamp', 'representative', name='_symbol_ts_rep_uc'),)

Base.metadata.create_all(engine)
logging.info("Database tables created or verified successfully.")

try:
    api = REST(key_id=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, base_url=APCA_API_BASE_URL)
    logging.info("Alpaca API initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Alpaca API: {e}")
    raise

@contextmanager
def get_session():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Session rollback due to error: {e}")
        raise
    finally:
        session.close()

def fetch_historical_data(symbol, limit=1000):
    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, limit=limit).df
        bars = bars.tz_localize(None)
        bars = bars.reset_index()
        
        if 'time' in bars.columns:
            bars.rename(columns={'time': 'timestamp'}, inplace=True)
        elif 'timestamp' not in bars.columns:
            logging.error(f"'timestamp' column not found for {symbol}. Available columns: {bars.columns.tolist()}")
            return pd.DataFrame()
        
        return bars
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()

def insert_stock_data(symbol, bar):
    try:
        if isinstance(bar.timestamp, (int, float)):
            timestamp_converted = pd.to_datetime(bar.timestamp, unit='ns')
        elif isinstance(bar.timestamp, str):
            timestamp_converted = pd.to_datetime(bar.timestamp)
        else:
            timestamp_converted = bar.timestamp

        record = StockData(
            symbol=symbol,
            timestamp=timestamp_converted,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume
        )

        with get_session() as session:
            session.add(record)

        logging.info(f"Inserted data for {symbol} at {timestamp_converted}.")
    except IntegrityError:
        logging.warning(f"Duplicate entry for {symbol} at {timestamp_converted}. Skipping insert.")
    except Exception as e:
        logging.error(f"Error inserting data for {symbol} at {timestamp_converted}: {e}")

def insert_historical_data(symbol, data):
    try:
        records = [
            StockData(
                symbol=symbol,
                timestamp=pd.to_datetime(row['timestamp'], unit='ns'),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            for index, row in data.iterrows()
        ]

        with get_session() as session:
            session.bulk_save_objects(records)
        logging.info(f"Inserted {len(records)} historical records for {symbol} into the database.")
    except IntegrityError:
        logging.warning(f"Duplicate entries found while inserting historical data for {symbol}. Skipping duplicates.")
    except Exception as e:
        logging.error(f"Error inserting historical data for {symbol}: {e}")

def compute_technical_indicators(data):
    if data.empty:
        logging.warning("Received empty DataFrame for technical indicators.")
        return pd.DataFrame()
    if 'close' not in data.columns:
        logging.error(f"'close' column missing. Available columns: {data.columns.tolist()}")
        return pd.DataFrame()
    
    data = data.copy()
    data['return'] = data['close'].pct_change()
    
    sma_short = SMAIndicator(close=data['close'], window=SHORT_MA)
    sma_long = SMAIndicator(close=data['close'], window=LONG_MA)
    data['ma_short'] = sma_short.sma_indicator()
    data['ma_long'] = sma_long.sma_indicator()
    data['ma_ratio'] = data['ma_short'] / data['ma_long']
    
    rsi = RSIIndicator(close=data['close'], window=RSI_PERIOD)
    data['rsi'] = rsi.rsi()
    
    for i in range(1, FEATURE_LAGS + 1):
        data[f'return_lag_{i}'] = data['return'].shift(i)
    
    data['future_return'] = data['close'].shift(-1) / data['close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)
    
    data = data.dropna()
    return data

def prepare_features(data):
    features = [
        'ma_short', 'ma_long', 'ma_ratio',
        'rsi', 'return'
    ] + [f'return_lag_{i}' for i in range(1, FEATURE_LAGS + 1)]
    if any(f not in data.columns for f in features):
        return pd.DataFrame(), pd.Series(dtype=float)
    X = data[features]
    y = data['target']
    return X, y

def train_model(X, y):
    try:
        start_time = datetime.now()
        
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        accuracy = (y_pred == y).mean()
        logging.info(f"Model training took {(datetime.now() - start_time).total_seconds():.2f} seconds.")
        logging.info(f"Model Accuracy: {accuracy:.2f}")
        
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

def load_existing_model():
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    try:
        model.load_model("model.json")
        logging.info("Loaded existing model from model.json.")
        return model
    except Exception as e:
        logging.warning(f"No existing model found or failed to load: {e}")
        return None

def retrain_model():
    with get_session() as session:
        # Get all distinct symbols from the DB
        symbols = [row[0] for row in session.query(StockData.symbol).distinct().all()]

    if not symbols:
        logging.warning("No symbols found in DB to train the model.")
        return None

    combined = []
    for sym in symbols:
        try:
            with get_session() as session:
                records = session.query(StockData).filter(StockData.symbol == sym).order_by(StockData.timestamp.asc()).all()

            if not records:
                logging.warning(f"No data found in DB for {sym}. Skipping.")
                continue

            data = pd.DataFrame([{
                'open': record.open,
                'high': record.high,
                'low': record.low,
                'close': record.close,
                'volume': record.volume,
                'timestamp': record.timestamp
            } for record in records])
            data = data.sort_values('timestamp')

            data = compute_technical_indicators(data)
            if data.empty:
                logging.warning(f"Insufficient data for {sym} after computing indicators. Skipping.")
                continue
            data['symbol'] = sym
            combined.append(data)
        except Exception as e:
            logging.error(f"Error retrieving or processing data for {sym}: {e}")
            continue

    if not combined:
        logging.warning("No data available for training the model.")
        return None

    combined_df = pd.concat(combined)
    combined_df = combined_df.dropna()

    X, y = prepare_features(combined_df)
    if X.empty:
        logging.warning("Feature set is empty. Cannot train the model.")
        return None

    new_model = train_model(X, y)
    if new_model is not None:
        # Save the model after successful training
        new_model.save_model("model.json")
        logging.info("Model saved after retraining.")
    return new_model

def get_current_position(symbol):
    try:
        position = api.get_position(symbol)
        qty = float(position.qty)
        logging.info(f"Current position for {symbol}: {qty} shares.")
        return qty
    except Exception as e:
        logging.info(f"No position for {symbol}: {e}")
        return 0.0

def close_position(symbol):
    try:
        position = api.get_position(symbol)
        qty = float(position.qty)
        side = 'sell' if qty > 0 else 'buy'
        api.submit_order(
            symbol=symbol,
            qty=abs(qty),
            side=side,
            type='market',
            time_in_force='day'
        )
        logging.info(f"Closed position for {symbol}: {side} {abs(qty)} shares.")
    except Exception as e:
        logging.error(f"Error closing position for {symbol}: {e}")

def submit_notional_order(symbol, side, notional):
    logging.info(f"Submitting {side.upper()} order for {symbol}, notional: {notional}")
    try:
        api.submit_order(
            symbol=symbol,
            notional=notional,
            side=side,
            type='market',
            time_in_force='day'
        )
        logging.info(f"Order submitted: {side.upper()} {symbol} with notional {notional}")
    except Exception as e:
        logging.error(f"Error submitting order for {symbol}: {e}")

def execute_trade(pred_results):
    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.45

    for symbol, prob in pred_results:
        logging.info(f"Symbol: {symbol}, Probability: {prob}")

        if prob > BUY_THRESHOLD:
            if symbol not in active_positions:
                submit_notional_order(symbol, 'buy', NOTIONAL)
                active_positions[symbol] = True
        elif prob < SELL_THRESHOLD:
            if symbol in active_positions:
                close_position(symbol)
                del active_positions[symbol]
        # Otherwise, hold current positions.

def initialize_data():
    # Fetch initial historical data for WATCHLIST (since we need at least some symbols to start streaming)
    for sym in WATCHLIST:
        hist = fetch_historical_data(sym, limit=1000)
        if hist.empty:
            logging.warning(f"Warning: No historical data for {sym}. Skipping.")
            continue
        hist.columns = [c.lower() for c in hist.columns]
        insert_historical_data(sym, hist)

def initialize_bot():
    logging.info("Initializing data...")
    initialize_data()
    logging.info("Data initialization complete.")
    
    logging.info("Attempting to load existing model...")
    global model
    model = load_existing_model()
    
    if model is None:
        logging.info("No existing model loaded. Retraining initial model...")
        model = retrain_model()
        if model is not None:
            logging.info("Initial model trained and saved successfully.")
        else:
            logging.warning("Initial model training failed or insufficient data.")
    else:
        logging.info("Using loaded model from previous run.")

# ============================================================
# Congress Data from News Integration
# ============================================================
NEWS_SOURCES = [
    "https://www.politico.com/search?q=congress+stock+trading",
    "https://www.cnn.com/search?size=20&q=congress%20stock%20trading",
    "https://www.cnbc.com/search/?query=congress%20trading",
    "https://www.foxnews.com/search-results/search?q=congress%20stock%20trades",
    "https://www.bloomberg.com/search?query=congress%20stock%20trading",
    "https://www.reuters.com/search/news?blob=congress+stock+trading",
    "https://thehill.com/search?q=congress%20stock%20trading",
    "https://www.wsj.com/search?query=congress+stock+trading",
]

def fetch_congress_trades_from_news(lookback_days=7):
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    records = []

    for url in NEWS_SOURCES:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " 
                              "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
                          "image/webp,image/apng,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            article_candidates = soup.find_all(['article', 'div', 'li'], 
                                               text=re.compile(r'(?i)congress.*stock|lawmakers.*trade'))
            if not article_candidates:
                article_candidates = soup.find_all('a', text=re.compile(r'(?i)congress.*stock|lawmakers.*trade'))
            
            for candidate in article_candidates:
                text = candidate.get_text(separator=' ', strip=True)
                if 'congress' in text.lower() and ('stock' in text.lower() or 'trade' in text.lower()):
                    date = datetime.now()
                    # Extract possible tickers (very naive)
                    tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
                    # Determine buy/sell
                    transaction_type = None
                    if 'bought' in text.lower():
                        transaction_type = 'buy'
                    elif 'sold' in text.lower():
                        transaction_type = 'sell'

                    if transaction_type and tickers:
                        representative = "Unknown Member"
                        amount = 1000.0  # placeholder
                        for symbol in tickers:
                            record = {
                                'symbol': symbol,
                                'timestamp': date,
                                'transaction_type': transaction_type,
                                'representative': representative,
                                'amount': amount
                            }
                            records.append(record)
        except Exception as e:
            logging.error(f"Error fetching or parsing {url}: {e}")
            continue

    if records:
        df = pd.DataFrame(records)
        df.drop_duplicates(subset=['symbol', 'timestamp', 'representative'], inplace=True)
        return df
    else:
        return pd.DataFrame()

def insert_congress_trades_from_news(df):
    if df.empty:
        logging.info("No congress trade data found from news.")
        return
    
    records = []
    for _, row in df.iterrows():
        record = CongressTradeData(
            symbol=row['symbol'],
            timestamp=row['timestamp'],
            transaction_type=row['transaction_type'],
            representative=row['representative'],
            amount=row['amount']
        )
        records.append(record)
    
    if not records:
        return
    
    with get_session() as session:
        try:
            session.bulk_save_objects(records)
        except IntegrityError:
            logging.warning("Duplicate entries while inserting congress data from news.")
    logging.info(f"Inserted {len(records)} congress trades from news into the database.")

def update_congress_data_from_news(days_back=7):
    df = fetch_congress_trades_from_news(lookback_days=days_back)
    insert_congress_trades_from_news(df)

def get_recent_congress_signal(symbol, lookback_days=7):
    with get_session() as session:
        cutoff = datetime.now() - timedelta(days=lookback_days)
        trades = session.query(CongressTradeData).filter(
            CongressTradeData.symbol == symbol,
            CongressTradeData.timestamp >= cutoff
        ).all()

    if not trades:
        return 0.0

    score = 0
    for t in trades:
        if t.transaction_type == 'buy':
            score += 1
        elif t.transaction_type == 'sell':
            score -= 1

    sentiment = score / len(trades)
    return sentiment

def scheduled_retrain():
    global model
    # Update congress data first
    update_congress_data_from_news(days_back=1)

    logging.info("Scheduled retraining started.")
    new_model = retrain_model()
    if new_model is not None:
        model = new_model
        logging.info("Scheduled retraining completed successfully.")
    else:
        logging.warning("Scheduled retraining failed or insufficient data.")

def setup_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_retrain, 'interval', minutes=5)
    scheduler.start()
    logging.info("Scheduler started for periodic retraining.")
    atexit.register(lambda: scheduler.shutdown())

def on_bar(bar):
    global model, bar_count_since_last_train, active_positions, last_bar_timestamp

    sym = bar.symbol
    ts = bar.timestamp

    if sym in last_bar_timestamp and last_bar_timestamp[sym] == ts:
        return
    last_bar_timestamp[sym] = ts

    insert_stock_data(sym, bar)

    bar_count_since_last_train += 1
    if bar_count_since_last_train >= RETRAIN_FREQUENCY:
        new_model = retrain_model()
        if new_model is not None:
            model = new_model
            logging.info("Model retrained successfully.")
        bar_count_since_last_train = 0

    if model is None:
        logging.warning("No trained model available for predictions.")
        return

    # Get all distinct symbols from DB for predictions
    with get_session() as session:
        symbols = [row[0] for row in session.query(StockData.symbol).distinct().all()]

    if not symbols:
        logging.warning("No symbols in DB for predictions.")
        return

    pred_results = []
    for s in symbols:
        try:
            with get_session() as session:
                records = session.query(StockData).filter(StockData.symbol == s).order_by(StockData.timestamp.desc()).all()
            
            if not records:
                continue
            data = pd.DataFrame([{
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume,
                'timestamp': r.timestamp
            } for r in records])
            data = data.sort_values('timestamp')
            data = compute_technical_indicators(data)
            if data.empty:
                continue
            X_all, y_all = prepare_features(data)
            if X_all.empty:
                continue
            X_pred = X_all.iloc[[-1]]
            pred_prob = model.predict_proba(X_pred)[0][1]

            # Incorporate Congress sentiment
            congress_sentiment = get_recent_congress_signal(s, lookback_days=7)
            adjusted_prob = pred_prob + 0.1 * congress_sentiment

            pred_results.append((s, adjusted_prob))
        except Exception as e:
            logging.error(f"Error processing symbol {s}: {e}")
            continue

    if not pred_results:
        logging.warning("No predictions made.")
        return

    execute_trade(pred_results)

def start_stream():
    stream = Stream(
        APCA_API_KEY_ID,
        APCA_API_SECRET_KEY,
        base_url="https://stream.data.alpaca.markets",
        data_feed='iex'
    )
    
    for symbol in WATCHLIST:
        @stream.on_bar(symbol)
        async def bar_callback(bar, symbol=symbol):
            on_bar(bar)
    
    logging.info("Starting data stream...")
    print("Starting data stream...")
    try:
        stream.run()
    except KeyboardInterrupt:
        logging.info("Stream stopped by user.")
        print("Stream stopped.")
    except Exception as e:
        logging.error(f"An error occurred during streaming: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    initialize_bot()
    setup_scheduler()
    start_stream()
