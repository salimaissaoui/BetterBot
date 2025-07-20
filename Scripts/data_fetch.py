import logging
import pandas as pd
from datetime import datetime
from ib_insync import IB, Stock
from .config import START_DATE
from .database import insert_historical_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def sanitize_symbol(symbol):
    try:
        if isinstance(symbol, bool):
            return None  # filter out booleans like True/False
        return str(symbol).strip().upper()
    except Exception:
        return None


def fetch_sp500_symbols_from_csv(csv_path="sp500.csv") -> list:
    try:
        df = pd.read_csv(csv_path)
        if 'Symbol' not in df.columns:
            logging.error(f"CSV is missing 'Symbol' column: {csv_path}")
            return []
        df['Symbol'] = df['Symbol'].apply(sanitize_symbol)
        symbols = df['Symbol'].unique().tolist()
        logging.info(f"Loaded {len(symbols)} valid symbols from {csv_path}")
        return symbols
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_path}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error reading {csv_path}: {e}")
        return []


def qualify_symbol(ib: IB, symbol: str):
    if symbol == 'BRK.B':
        contract = Stock('BRK B', 'SMART', 'USD')
    else:
        contract = Stock(symbol, 'SMART', 'USD')
    try:
        qualified = ib.qualifyContracts(contract)
        return qualified[0] if qualified else None
    except Exception as e:
        logging.warning(f"Could not qualify {symbol}: {e}")
        return None


def fetch_historical_data(ib: IB, symbol: str, start_date: str = START_DATE, end_date: str = None, max_retries: int = 3) -> pd.DataFrame:
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            contract = qualify_symbol(ib, symbol)
            if not contract:
                raise ValueError(f"Contract qualification failed for {symbol}")

            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_date or '',
                durationStr='1 Y',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame([{
                'timestamp': bar.date,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars])

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df['timestamp'] >= start_dt]

            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            logging.info(f"Fetched {len(df)} daily bars for {symbol}.")
            return df
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol} (Attempt {attempt}): {e}")
    logging.error(f"Failed to fetch complete historical data for {symbol} after {max_retries} attempts.")
    return pd.DataFrame()


def fetch_and_load_symbols(ib: IB, csv_path="sp500.csv", limit=None) -> list:
    all_symbols = fetch_sp500_symbols_from_csv(csv_path)
    chosen_symbols = all_symbols[:limit]
    if not chosen_symbols:
        logging.warning("No symbols to process.")
        return []

    logging.info("Fetching historical data for selected symbols...")
    processed_symbols = []

    for symbol in chosen_symbols:
        if not symbol:
            logging.warning("Skipped an empty or invalid symbol.")
            continue

        df = fetch_historical_data(ib, symbol)
        if df.empty:
            logging.info(f"No historical data for {symbol}. Skipping.")
            continue

        insert_historical_data(symbol, df)
        logging.info(f"Completed fetch and insert for {symbol}")
        processed_symbols.append(symbol)

    logging.info("Data fetching and loading completed.")
    return processed_symbols
