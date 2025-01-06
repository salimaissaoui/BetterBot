import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, Integer, String, Float, BigInteger, TIMESTAMP, UniqueConstraint
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.exc import IntegrityError
from datetime import datetime

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

Base.metadata.create_all(engine)
logging.info("Database tables created or verified successfully.")


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


def insert_stock_data(symbol, bar):
    """Inserts a single bar/row into the database."""
    from pandas import to_datetime  # local import to avoid circular dependencies
    try:
        timestamp_converted = None
        if isinstance(bar.timestamp, (int, float)):
            timestamp_converted = to_datetime(bar.timestamp, unit='ns')
        elif isinstance(bar.timestamp, str):
            timestamp_converted = to_datetime(bar.timestamp)
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


def insert_historical_data(symbol, data_df):
    """Bulk inserts historical DataFrame into the database."""
    import pandas as pd

    if data_df.empty:
        logging.info(f"No historical data to insert for {symbol}.")
        return
    try:
        records = []
        for _, row in data_df.iterrows():
            timestamp_converted = pd.to_datetime(row['timestamp'])
            op = row.get('open', None)
            hi = row.get('high', None)
            lo = row.get('low', None)
            cl = row.get('close', None)
            vol = row.get('volume', None)

            r = StockData(
                symbol=symbol,
                timestamp=timestamp_converted,
                open=op,
                high=hi,
                low=lo,
                close=cl,
                volume=vol
            )
            records.append(r)

        with get_session() as session:
            session.bulk_save_objects(records)
        logging.info(f"Inserted {len(records)} historical records for {symbol} into the database.")
    except IntegrityError:
        logging.info(f"Duplicate entries for {symbol}. Skipping duplicates.")
    except Exception as e:
        logging.error(f"Error inserting historical data for {symbol}: {e}")
