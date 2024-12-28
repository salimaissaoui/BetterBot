# ======================== IMPORTS ===========================
import os
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta, time  # Ensure time class is imported
import pytz  # Import pytz for timezone handling
from alpaca_trade_api import REST, Stream
from alpaca_trade_api.rest import TimeFrame
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, TIMESTAMP, UniqueConstraint
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.exc import IntegrityError
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from contextlib import contextmanager
import requests
import re
import traceback

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin, BaseEstimator

# Remove or avoid importing the time module:
# import time  # Remove this if present


# ======================== LOGGING CONFIG ===========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ======================== CONFIG ===========================
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID", "PK023FOQLSDB6VR3UDRV")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "2CalVxgSuOnObttDwgBRcpQAbBMTXR4XkBa7qXdR")
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Stocks123")
DB_HOST = os.getenv("DB_HOST", "database-2.ctwgq2kqgrl6.us-east-2.rds.amazonaws.com")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

NOTIONAL = 1000
MODEL_LOOKBACK = 5
FEATURE_LAGS = 2
SHORT_MA = 5
LONG_MA = 1
RSI_PERIOD = 1
RETRAIN_FREQUENCY = 5  # in minutes

bar_count_since_last_train = 0
active_positions = {}
last_bar_timestamp = {}
historical_data_dict = {}
model = None  # Ensemble model (VotingClassifier)

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
try:
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0, echo=False)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    Session = scoped_session(SessionFactory)
    logging.info("Connected to PostgreSQL database successfully.")
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

def fetch_symbols_from_api():
    assets = api.list_assets(status='active')
    major_exchanges = ['NASDAQ', 'NYSE', 'ARCA']
    tradable_stocks = [asset.symbol for asset in assets if asset.tradable and asset.exchange in major_exchanges]
    # Limit to fewer symbols to reduce data issues
    return tradable_stocks[:50]

def fetch_historical_data(symbol, limit=1000):
    try:
        bars = api.get_bars(symbol, TimeFrame.Minute, limit=limit).df
        bars = bars.tz_localize(None)
        bars = bars.reset_index()
        
        if 'time' in bars.columns:
            bars.rename(columns={'time': 'timestamp'}, inplace=True)
        elif 'timestamp' not in bars.columns:
            logging.warning(f"'timestamp' column not found for {symbol}. Returning empty.")
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
        logging.info(f"Duplicate entry for {symbol} at {timestamp_converted}. Skipping insert.")
    except Exception as e:
        logging.error(f"Error inserting data for {symbol}: {e}")

def insert_historical_data(symbol, data):
    if data.empty:
        logging.info(f"No historical data to insert for {symbol}.")
        return
    try:
        records = [
            StockData(
                symbol=symbol,
                timestamp=pd.to_datetime(row['timestamp']),
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
        logging.info(f"Duplicate entries for {symbol}. Skipping duplicates.")
    except Exception as e:
        logging.error(f"Error inserting historical data for {symbol}: {e}")

def compute_technical_indicators(data):
    if data.empty or 'close' not in data.columns:
        logging.info("No data or 'close' column missing when computing indicators.")
        return pd.DataFrame()

    data = data.copy()
    data['return'] = data['close'].pct_change()

    # Trend Indicators
    if len(data) < max(SHORT_MA, LONG_MA):
        logging.info("Not enough data for SMA computation. Returning empty.")
        return pd.DataFrame()

    sma_short = SMAIndicator(close=data['close'], window=SHORT_MA)
    sma_long = SMAIndicator(close=data['close'], window=LONG_MA)
    data['ma_short'] = sma_short.sma_indicator()
    data['ma_long'] = sma_long.sma_indicator()
    data['ma_ratio'] = data['ma_short'] / data['ma_long']

    # Momentum Indicators
    if len(data) < RSI_PERIOD:
        logging.info("Not enough data for RSI computation. Returning empty.")
        return pd.DataFrame()
    rsi = RSIIndicator(close=data['close'], window=RSI_PERIOD)
    data['rsi'] = rsi.rsi()

    macd = MACD(close=data['close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()

    # Volatility Indicators (Bollinger Bands)
    if len(data) < 20:
        logging.info("Not enough data for Bollinger Bands (need at least 20). Returning empty.")
        return pd.DataFrame()
    bb = BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bb_h'] = bb.bollinger_hband()
    data['bb_l'] = bb.bollinger_lband()
    data['bb_m'] = bb.bollinger_mavg()
    data['bb_w'] = bb.bollinger_wband()
    data['bb_p'] = bb.bollinger_pband()

    atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14)
    data['atr'] = atr.average_true_range()

    # Volume-Based Indicator
    data['obv'] = OnBalanceVolumeIndicator(close=data['close'], volume=data['volume']).on_balance_volume()

    # Candlestick pattern
    data['doji'] = np.where((abs(data['close'] - data['open']) < 0.001 * data['open']), 1, 0)

    # Remove anomalies in returns
    returns = data['return'].dropna()
    if len(returns) == 0:
        logging.info("No valid returns. Returning empty.")
        return pd.DataFrame()

    q1, q3 = returns.quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (data['return'] > q1 - 3*iqr) & (data['return'] < q3 + 3*iqr)
    data = data[mask]

    for i in range(1, FEATURE_LAGS + 1):
        data[f'return_lag_{i}'] = data['return'].shift(i)
    
    data['future_return'] = data['close'].shift(-1) / data['close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)

    data = data.dropna()

    # Check length after dropna
    if data.empty:
        logging.info("Data empty after dropna. Returning empty.")
        return pd.DataFrame()

    # Additional safety check to avoid indexing issues
    if len(data) <= FEATURE_LAGS:
        logging.info("Not enough data after processing to safely compute features. Returning empty.")
        return pd.DataFrame()

    return data

def prepare_features(data):
    if data.empty:
        logging.info("Empty data in prepare_features, returning.")
        return pd.DataFrame(), pd.Series(dtype=float)

    features = [
        'ma_short', 'ma_long', 'ma_ratio',
        'rsi', 'return',
        'macd', 'macd_signal', 'macd_diff',
        'bb_h', 'bb_l', 'bb_m', 'bb_w', 'bb_p',
        'atr', 'doji', 'obv'
    ] + [f'return_lag_{i}' for i in range(1, FEATURE_LAGS + 1)]

    available = data.columns
    features = [f for f in features if f in available]

    if not features:
        logging.info("No features available after filtering.")
        return pd.DataFrame(), pd.Series(dtype=float)

    X = data[features]
    if X.empty:
        logging.info("No data in X after selecting features.")
        return pd.DataFrame(), pd.Series(dtype=float)

    if 'target' not in data.columns:
        logging.info("No target column in data. Returning empty target.")
        return X, pd.Series(dtype=float)

    y = data['target']
    return X, y

# ======================== WRAPPER CLASSES ===========================
class XGBClassifierWrapper(XGBClassifier):
    def __sklearn_tags__(self):
        return {
            "non_deterministic": True,
            "allow_nan": True,
            "X_types": ["2darray"],
            "requires_y": True,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "no_validation": False,
            "stateless": False,
            "poor_score": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "pairwise": False,
        }

class LGBMClassifierWrapper(LGBMClassifier):
    def __sklearn_tags__(self):
        return {
            "non_deterministic": True,
            "allow_nan": True,
            "X_types": ["2darray"],
            "requires_y": True,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "no_validation": False,
            "stateless": False,
            "poor_score": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "pairwise": False,
        }

# ======================== MODEL TRAINING ===========================
def train_model(X, y):
    if X.empty or y.empty:
        logging.info("No data available for training the model. Returning.")
        return None

    try:
        start_time = datetime.now()
        scoring = make_scorer(accuracy_score)

        # Initialize individual classifiers with wrappers
        xgb = XGBClassifierWrapper(eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
        lgb = LGBMClassifierWrapper(n_jobs=-1)

        # Initialize VotingClassifier with soft voting
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb),
                ('lgb', lgb)
            ],
            voting='soft'
        )

        # Define parameter distributions for hyperparameter tuning
        param_distributions = {
            'xgb__n_estimators': [100, 200],
            'xgb__max_depth': [3, 5],
            'xgb__subsample': [0.6, 0.8],
            'xgb__colsample_bytree': [0.6, 0.8],
            'xgb__learning_rate': [0.01, 0.05],
            'lgb__n_estimators': [100, 200],
            'lgb__max_depth': [3, 5, -1],
            'lgb__subsample': [0.6, 0.8],
            'lgb__colsample_bytree': [0.6, 0.8],
            'lgb__learning_rate': [0.01, 0.05],
        }

        # Define cross-validation strategy
        cv = KFold(n_splits=2, shuffle=True, random_state=42)

        logging.info("Starting hyperparameter tuning for Voting Ensemble...")
        search = RandomizedSearchCV(
            estimator=ensemble,
            param_distributions=param_distributions,
            n_iter=20,  # Increase iterations for better tuning
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=2,
            error_score='raise'  # Raise errors during fitting for debugging
        )
        search.fit(X, y)
        logging.info(f"Best parameters: {search.best_params_}, Score: {search.best_score_:.2f}")

        best_ensemble = search.best_estimator_

        # Save the ensemble model
        joblib.dump(best_ensemble, "ensemble_model.joblib")
        logging.info("Ensemble model saved successfully.")

        # Evaluate
        ensemble_pred = best_ensemble.predict_proba(X)[:, 1]
        ensemble_acc = ((ensemble_pred > 0.5).astype(int) == y).mean()

        logging.info(f"Model training took {(datetime.now() - start_time).total_seconds():.2f} seconds.")
        logging.info(f"Final Ensemble Accuracy: {ensemble_acc:.2f}")

        return best_ensemble

    except Exception as e:
        logging.error(f"Error training model: {e}")
        logging.error(traceback.format_exc())
        return None

def load_existing_model():
    try:
        ensemble_model = joblib.load("ensemble_model.joblib")
        logging.info("Loaded existing ensemble model from ensemble_model.joblib.")
        return ensemble_model
    except FileNotFoundError:
        logging.info("No existing ensemble model found (ensemble_model.joblib not found).")
        return None
    except Exception as e:
        logging.warning(f"Failed to load existing ensemble model: {e}")
        logging.error(traceback.format_exc())
        return None

def retrain_model():
    with get_session() as session:
        symbols = [row[0] for row in session.query(StockData.symbol).distinct().all()]

    if not symbols:
        logging.info("No symbols found in DB to train the model.")
        return None

    combined = []
    for sym in symbols:
        with get_session() as session:
            records = session.query(StockData).filter(StockData.symbol == sym).order_by(StockData.timestamp.asc()).all()

        if not records or len(records) < 30:  # Need enough bars
            logging.info(f"Insufficient data for {sym}. Skipping.")
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
            logging.info(f"Insufficient processed data for {sym}. Skipping.")
            continue
        data['symbol'] = sym
        combined.append(data)

    if not combined:
        logging.info("No data available for training the model after processing.")
        return None

    combined_df = pd.concat(combined)
    combined_df = combined_df.dropna()

    X, y = prepare_features(combined_df)
    if X.empty or y.empty:
        logging.info("No features or target available for training.")
        return None

    logging.info(f"Training data shape: X={X.shape}, y={y.shape}")   

    new_model = train_model(X, y)
    if new_model is not None:
        logging.info("Models saved after retraining.")
    return new_model

# ======================== POSITION MANAGEMENT ===========================
def get_current_position(symbol):
    try:
        position = api.get_position(symbol)
        qty = float(position.qty)
        logging.info(f"Current position for {symbol}: {qty} shares.")
        return qty
    except Exception:
        # No position
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

# ======================== NEWS SENTIMENT ANALYSIS ===========================
def fetch_articles_for_symbol(symbol, days_back=7):
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "b3c33a11475c48cfbb9e3ac26a84a0e8")  # Replace with your actual NewsAPI key or retrieve from env
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

def get_news_sentiment(symbol, lookback_days=7):
    articles = fetch_articles_for_symbol(symbol, days_back=lookback_days)
    sentiment = analyze_sentiment(articles)
    return sentiment

# ======================== MARKET HOURS CONFIG ===========================
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
TIMEZONE = pytz.timezone('US/Eastern')

def is_market_open():
    now = datetime.now(TIMEZONE)
    today = now.date()
    current_time = now.time()
    
    # Check if today is a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if current time is within market hours
    if MARKET_OPEN <= current_time <= MARKET_CLOSE:
        return True
    return False

# ======================== SCHEDULER ===========================
def scheduled_retrain():
    global model
    if not is_market_open():
        logging.info("Market is closed. Scheduled retraining started.")
        new_model = retrain_model()
        if new_model is not None:
            model = new_model
            logging.info("Scheduled retraining completed successfully.")
        else:
            logging.info("Scheduled retraining had no new model.")
    else:
        logging.info("Market is open. Skipping scheduled retraining.")

def setup_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_retrain, 'interval', minutes=RETRAIN_FREQUENCY)
    scheduler.start()
    logging.info("Scheduler started for periodic retraining.")
    atexit.register(lambda: scheduler.shutdown())

# ======================== BOT INITIALIZATION ===========================
def fetch_and_load_symbols():
    symbols = fetch_symbols_from_api()
    for sym in symbols:
        hist = fetch_historical_data(sym, limit=1000)
        if hist.empty:
            logging.info(f"No historical data for {sym}. Skipping.")
            continue
        hist.columns = [c.lower() for c in hist.columns]
        insert_historical_data(sym, hist)
    return symbols

def initialize_bot():
    logging.info("Fetching symbols from API...")
    api_symbols = fetch_symbols_from_api()
    if not api_symbols:
        logging.error("No symbols returned from API. Cannot proceed.")
        return

    logging.info("Inserting historical data for fetched symbols...")
    for sym in api_symbols:
        hist = fetch_historical_data(sym, limit=1000)
        if hist.empty:
            logging.info(f"No historical data for {sym}. Skipping.")
            continue
        hist.columns = [c.lower() for c in hist.columns]
        insert_historical_data(sym, hist)

    logging.info("Attempting to load existing models...")
    global model
    model = load_existing_model()
    
    if model is None:
        logging.info("No existing models loaded. Retraining initial models...")
        model = retrain_model()
        if model is not None:
            logging.info("Initial models trained and saved successfully.")
        else:
            logging.info("Initial model training failed or insufficient data.")
    else:
        logging.info("Using loaded models from previous run.")

# ======================== STREAMING HANDLER ===========================
latest_indicators_dict = {}
previous_obv = {}

def on_bar(bar):
    global model, bar_count_since_last_train, active_positions, last_bar_timestamp, previous_obv

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
        logging.info("No trained model available for predictions.")
        return

    with get_session() as session:
        records = session.query(StockData).filter(StockData.symbol == sym).order_by(StockData.timestamp.desc()).all()

    if not records or len(records) < 30:
        return

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
        return
    X_all, y_all = prepare_features(data)
    if X_all.empty:
        return

    # Store latest indicators
    last_row = data.iloc[-1]
    macd_diff = last_row.get('macd_diff', 0.0)
    current_obv = last_row.get('obv', 0.0)
    prev_obv_val = previous_obv.get(sym, current_obv)
    previous_obv[sym] = current_obv

    latest_indicators_dict[sym] = {
        'macd_diff': macd_diff,
        'obv': current_obv,
        'obv_prev': prev_obv_val
    }

def get_latest_indicators(symbol):
    return latest_indicators_dict.get(symbol, {})


# ======================== TRADE EXECUTION ===========================
def execute_trade(pred_results):
    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.45

    for symbol, prob in pred_results:
        logging.info(f"Symbol: {symbol}, Probability: {prob}")

        indicators = get_latest_indicators(symbol)
        macd_diff = indicators.get('macd_diff', 0.0)
        obv = indicators.get('obv', 0.0)
        obv_prev = indicators.get('obv_prev', obv)
        obv_increasing = (obv > obv_prev)

        want_to_buy = (prob > BUY_THRESHOLD) and (macd_diff > 0) and obv_increasing
        want_to_sell = (prob < SELL_THRESHOLD)

        if want_to_buy:
            if symbol not in active_positions:
                logging.info(f"Buying {symbol}: prob={prob:.2f}, macd_diff={macd_diff:.4f}, obv_increasing={obv_increasing}")
                submit_notional_order(symbol, 'buy', NOTIONAL)
                active_positions[symbol] = True
            else:
                logging.info(f"Already holding {symbol}, no need to buy again.")
        elif want_to_sell:
            if symbol in active_positions:
                logging.info(f"Selling {symbol}: prob={prob:.2f}")
                close_position(symbol)
                del active_positions[symbol]
        else:
            logging.info(f"{symbol}: Prob={prob:.2f}, conditions not met for buy/sell, doing nothing.")

# ======================== STREAMING SETUP ===========================
def start_stream(symbols):
    stream = Stream(
        APCA_API_KEY_ID,
        APCA_API_SECRET_KEY,
        base_url="https://stream.data.alpaca.markets",
        data_feed='iex'
    )
    
    for symbol in symbols:
        @stream.on_bar(symbol)
        async def bar_callback(bar, symbol=symbol):
            on_bar(bar)
            # Check if market is open
            if is_market_open():
                # Execute trading logic
                if model:
                    try:
                        # Aggregate recent data for prediction
                        with get_session() as session:
                            records = session.query(StockData).filter(StockData.symbol == symbol).order_by(StockData.timestamp.asc()).all()
    
                        if not records or len(records) < 30:
                            return
    
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
                            return
                        X, y = prepare_features(data)
                        if X.empty:
                            return
    
                        ensemble_pred = model.predict_proba(X)[:, 1]
                        latest_prob = ensemble_pred[-1]
                        execute_trade([(symbol, latest_prob)])
                    except Exception as e:
                        logging.error(f"Error during prediction/trade execution for {symbol}: {e}")
                        logging.error(traceback.format_exc())
            else:
                # Outside trading hours: Train the model
                logging.info(f"Market is closed. Initiating training for {symbol}.")
                retrain_model()
    
    logging.info("Starting data stream...")
    try:
        stream.run()
    except KeyboardInterrupt:
        logging.info("Stream stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred during streaming: {e}")
        logging.error(traceback.format_exc())

# ======================== MAIN EXECUTION ===========================
if __name__ == "__main__":
    initialize_bot()
    setup_scheduler()      
    symbols = fetch_symbols_from_api()
    start_stream(symbols)
