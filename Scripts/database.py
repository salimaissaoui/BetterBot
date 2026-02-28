import logging
from contextlib import contextmanager
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, BigInteger,
    TIMESTAMP, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.exc import IntegrityError
from datetime import datetime
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Import credentials/config
from .config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

try:
    engine = create_engine(DATABASE_URL, pool_size=20, max_overflow=0, echo=False)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)
    Session = scoped_session(SessionFactory)
    logging.info("Connected to PostgreSQL database successfully.")
except Exception as e:
    logging.error(f"Failed to connect to the database: {e}")
    raise

# -----------------------
# ORM Base and Models
# -----------------------

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

    __table_args__ = (
        UniqueConstraint('symbol', 'timestamp', name='_symbol_timestamp_uc'),
    )

class Portfolio(Base):
    __tablename__ = 'portfolio'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False)
    shares = Column(Float, nullable=False, default=0.0)
    cost_basis = Column(Float, nullable=False, default=0.0)

class PositionRegistry(Base):
    """One row per open position. Upserted on register_entry(), deleted on clear_position(). EXIT-04."""
    __tablename__ = 'position_registry'

    id           = Column(Integer, primary_key=True, autoincrement=True)
    symbol       = Column(String(10), nullable=False, unique=True)
    direction    = Column(String(5), nullable=False)        # 'long' or 'short'
    entry_price  = Column(Float, nullable=False)
    entry_time   = Column(TIMESTAMP, nullable=False)
    atr_at_entry = Column(Float, nullable=False)
    quantity     = Column(Integer, nullable=False)
    stop_price   = Column(Float, nullable=False)
    target_price = Column(Float, nullable=False)
    trailing_high = Column(Float, nullable=False)
    trailing_stop = Column(Float, nullable=True)            # NULL until trailing stop activates


class TradeLog(Base):
    """Structured record per trade decision (entry and exit). OBS-01."""
    __tablename__ = 'trade_log'

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    symbol              = Column(String(10), nullable=False)
    action              = Column(String(20), nullable=False)   # BUY, SHORT, HARD_STOP_LOSS, TRAILING_STOP, TAKE_PROFIT, SELL, COVER
    direction           = Column(String(5), nullable=True)     # 'long' or 'short'
    quantity            = Column(Integer, nullable=True)
    entry_price         = Column(Float, nullable=True)
    exit_price          = Column(Float, nullable=True)
    stop_price          = Column(Float, nullable=True)
    target_price        = Column(Float, nullable=True)
    entry_reason        = Column(String(200), nullable=True)   # e.g. 'ml_prob=0.720'
    exit_reason         = Column(String(50), nullable=True)    # 'HARD_STOP_LOSS' | 'TRAILING_STOP' | 'TAKE_PROFIT' | 'SIGNAL' | None
    regime_at_decision  = Column(String(20), nullable=True)    # 'bullish' | 'bearish' | 'volatile' | 'neutral'
    sentiment_score     = Column(Float, nullable=True)         # None in Phase 2; Phase 4 fills this
    vix_at_decision     = Column(Float, nullable=True)         # Added in Phase 5 [VAL-01]
    prediction_confidence = Column(Float, nullable=True)
    decision_time       = Column(TIMESTAMP, nullable=False, default=datetime.utcnow)

Base.metadata.create_all(engine)
logging.info("Database tables created or verified successfully.")

# -----------------------
# Session Management
# -----------------------

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

# -----------------------
# Insert Single Bar
# -----------------------

def insert_stock_data(symbol, bar):
    from pandas import to_datetime

    timestamp_converted = None  # Initialize before try block
    try:
        # Defensive symbol conversion
        if isinstance(symbol, bool):
            logging.error(f"Symbol value is boolean ({symbol}) — skipping insert.")
            return
        symbol = str(symbol).upper().strip()

        # Resolve timestamp
        ts = getattr(bar, "timestamp", getattr(bar, "time", None))
        if ts is None:
            raise ValueError("Bar object has no recognizable timestamp.")

        if isinstance(ts, (int, float)):
            timestamp_converted = to_datetime(ts, unit='ns')
        elif isinstance(ts, str):
            timestamp_converted = to_datetime(ts)
        else:
            timestamp_converted = ts

        # Get open (some bars use open_, like RealTimeBar)
        open_value = getattr(bar, 'open', getattr(bar, 'open_', None))

        # Defensive insert
        record = StockData(
            symbol=symbol,
            timestamp=timestamp_converted,
            open=open_value,
            high=getattr(bar, 'high', None),
            low=getattr(bar, 'low', None),
            close=getattr(bar, 'close', None),
            volume=getattr(bar, 'volume', None)
        )

        with get_session() as session:
            session.add(record)

        logging.info(f"Inserted data for {symbol} at {timestamp_converted}.")

    except IntegrityError:
        logging.info(f"Duplicate entry for {symbol} at {timestamp_converted}. Skipping insert.")
    except Exception as e:
        logging.error(f"Error inserting data for {symbol}: {e}")

# -----------------------
# Insert Bulk Historical Data
# -----------------------

def insert_historical_data(symbol, data_df):
    import pandas as pd

    try:
        if isinstance(symbol, bool):
            logging.error(f"Symbol value is boolean ({symbol}) — skipping historical insert.")
            return
        symbol = str(symbol).upper().strip()

        if data_df.empty:
            logging.info(f"No historical data to insert for {symbol}.")
            return

        records = []
        for _, row in data_df.iterrows():
            try:
                timestamp_converted = pd.to_datetime(row['timestamp'])
                records.append({
                    'symbol': symbol,
                    'timestamp': timestamp_converted,
                    'open': row.get('open'),
                    'high': row.get('high'),
                    'low': row.get('low'),
                    'close': row.get('close'),
                    'volume': row.get('volume')
                })
            except Exception as e:
                logging.warning(f"Skipping bad row: {e}")

        with get_session() as session:
            stmt = pg_insert(StockData).values(records)
            stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'timestamp'])
            session.execute(stmt)

        logging.info(f"Upserted {len(records)} historical records for {symbol} into the database.")

    except Exception as e:
        logging.error(f"Error inserting historical data for {symbol}: {e}")
