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
