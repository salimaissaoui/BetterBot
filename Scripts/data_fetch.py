import logging
import pandas as pd
from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame
from datetime import datetime

from .config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
from .database import insert_historical_data

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Initialize Alpaca REST client
api = REST(
    key_id=APCA_API_KEY_ID,
    secret_key=APCA_API_SECRET_KEY,
    base_url=APCA_API_BASE_URL
)

def fetch_sp500_symbols_from_api(limit=50) -> list:
    """
    Fetch S&P 500 symbols directly from Alpaca API.
    Filters tradable assets available on IEX feed only.
    """
    try:
        assets = api.list_assets(status='active')
        major_exchanges = ['NASDAQ', 'NYSE', 'ARCA']  # Common exchanges for S&P 500
        tradable_symbols = [
            asset.symbol for asset in assets
            if asset.tradable and asset.shortable and asset.exchange in major_exchanges
        ]
        logging.info(f"Fetched {len(tradable_symbols)} tradable symbols from Alpaca.")
        return tradable_symbols[:limit]
    except Exception as e:
        logging.error(f"Error fetching S&P 500 symbols from API: {e}")
        return []

def fetch_historical_data(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    limit: int = 1000,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch historical data for a given symbol from Alpaca.
    Returns a DataFrame with columns: [timestamp, open, high, low, close, volume].
    Retries fetching if any required columns are missing.
    """
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    attempt = 0

    while attempt < max_retries:
        try:
            logging.debug(f"Fetching historical data for {symbol} with start_date={start_date}, end_date={end_date}, limit={limit} (Attempt {attempt + 1}/{max_retries})")

            # Fetch data based on provided dates or limit
            if start_date and end_date:
                bars = api.get_bars(symbol, TimeFrame.Day, start=start_date, end=end_date, feed='iex').df
                logging.debug(f"Fetched data for date range {start_date} to {end_date}")
            else:
                bars = api.get_bars(symbol, TimeFrame.Day, limit=limit, feed='iex').df
                logging.debug(f"Fetched the last {limit} records")

            # If no data is returned, log a warning
            if bars.empty:
                logging.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()

            # Processing the data
            bars = bars.tz_localize(None)
            bars.reset_index(inplace=True)
            bars.rename(columns={'time': 'timestamp'}, inplace=True)

            # Converting and sorting timestamps
            bars['timestamp'] = pd.to_datetime(bars['timestamp'])
            bars.sort_values(by='timestamp', inplace=True)
            bars.reset_index(drop=True, inplace=True)

            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in bars.columns]
            if missing_columns:
                logging.warning(f"Missing columns in fetched data: {missing_columns}. Retrying...")
                attempt += 1
                continue  # Retry fetching the data

            # All columns are present, return the processed DataFrame
            logging.info(f"Fetched historical data for {symbol}: shape={bars.shape}")
            return bars

        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol} (Attempt {attempt + 1}/{max_retries}): {e}")
            attempt += 1

    # If retries are exhausted, return an empty DataFrame
    logging.error(f"Failed to fetch complete historical data for {symbol} after {max_retries} attempts.")
    return pd.DataFrame()



def fetch_and_load_symbols():
    """
    Fetch S&P 500 symbols, download historical data, and insert into the database.
    """
    sp500_symbols = fetch_sp500_symbols_from_api()
    if not sp500_symbols:
        logging.error("No symbols fetched. Cannot proceed.")
        return []

    logging.info("Fetching historical data for symbols...")
    for sym in sp500_symbols:
        start_date = "2015-01-01"
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        hist = fetch_historical_data(sym, start_date=start_date, end_date=end_date)
        if hist.empty:
            logging.info(f"No historical data for {sym}. Skipping.")
            continue
        insert_historical_data(sym, hist)

    logging.info("Data fetching and loading completed.")
    return sp500_symbols
