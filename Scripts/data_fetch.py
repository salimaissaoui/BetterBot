import logging
import pandas as pd
import yfinance as yf
from alpaca_trade_api import REST
from alpaca_trade_api.rest import TimeFrame

from .config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, APCA_API_BASE_URL
from .database import insert_historical_data, engine
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

api = REST(
    key_id=APCA_API_KEY_ID,
    secret_key=APCA_API_SECRET_KEY,
    base_url=APCA_API_BASE_URL
)


def fetch_yahoo_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily data from Yahoo Finance for the given symbol and date range."""
    try:
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, interval="1d")
        if df.empty:
            logging.info(f"No data returned by Yahoo for {symbol}.")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        # Drop Adj Close if present
        df.drop(columns=[c for c in df.columns if "Adj" in c], inplace=True, errors='ignore')

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        logging.error(f"Error fetching Yahoo data for {symbol}: {e}")
        return pd.DataFrame()


def fetch_historical_data(symbol: str,
                          start_date: str = None,
                          end_date: str = None,
                          limit: int = 1000) -> pd.DataFrame:
    """
    Attempt to fetch minute-level data from Alpaca first.
    If Alpaca fails or returns empty, fallback to daily data from Yahoo Finance.
    """
    alpaca_df = pd.DataFrame()
    yahoo_df = pd.DataFrame()

    # 1) Try Alpaca
    try:
        if start_date and end_date:
            bars = api.get_bars(symbol, TimeFrame.Minute, start=start_date, end=end_date).df
        else:
            bars = api.get_bars(symbol, TimeFrame.Minute, limit=limit).df

        bars = bars.tz_localize(None)
        bars.reset_index(inplace=True)
        if 'time' in bars.columns:
            bars.rename(columns={'time': 'timestamp'}, inplace=True)

        alpaca_df = bars.copy()
        alpaca_df['timestamp'] = pd.to_datetime(alpaca_df['timestamp'])
        alpaca_df.sort_values(by='timestamp', inplace=True)
        alpaca_df.reset_index(drop=True, inplace=True)

        # Convert column names to lower
        alpaca_df.columns = [col.lower() for col in alpaca_df.columns]

        logging.info(f"Alpaca fetch successful for {symbol}, shape={alpaca_df.shape}")

    except Exception as e:
        logging.error(f"Alpaca error for {symbol}: {e}")
        alpaca_df = pd.DataFrame()

    # 2) Fallback to Yahoo if Alpaca fails or we want more coverage
    if (alpaca_df.empty or start_date) and (start_date and end_date):
        yahoo_df = fetch_yahoo_data(symbol, start_date, end_date)
        if not yahoo_df.empty:
            logging.info(f"Yahoo fetch successful for {symbol}, shape={yahoo_df.shape}")
        else:
            logging.info(f"No Yahoo data returned for {symbol} within {start_date} - {end_date}.")

    if alpaca_df.empty and yahoo_df.empty:
        return pd.DataFrame()

    if alpaca_df.empty:
        return yahoo_df
    elif yahoo_df.empty:
        return alpaca_df
    else:
        combined = pd.concat([alpaca_df, yahoo_df], axis=0).drop_duplicates(subset='timestamp')
        combined.sort_values(by='timestamp', inplace=True)
        combined.reset_index(drop=True, inplace=True)
        logging.info(f"Merged data for {symbol}: final shape={combined.shape}")
        return combined


def fetch_symbols_from_api(limit=50):
    """Fetch up to `limit` tradable symbols from Alpaca (major exchanges)."""
    try:
        assets = api.list_assets(status='active')
        major_exchanges = ['NASDAQ', 'NYSE', 'ARCA']
        tradable_stocks = [
            asset.symbol for asset in assets 
            if asset.tradable and asset.exchange in major_exchanges
        ]
        return tradable_stocks[:limit]
    except Exception as e:
        logging.error(f"Failed to fetch symbols from Alpaca: {e}")
        return []


def fetch_and_load_symbols():
    """
    Fetch a list of symbols and download extended historical data,
    then insert into the database.
    """
    symbols = fetch_symbols_from_api()
    for sym in symbols:
        start_date = "2022-01-01"
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        hist = fetch_historical_data(sym, start_date=start_date, end_date=end_date)
        if hist.empty:
            logging.info(f"No historical data for {sym}. Skipping.")
            continue
        insert_historical_data(sym, hist)
    return symbols
