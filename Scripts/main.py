import logging
import atexit
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from alpaca_trade_api import Stream

from .config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, RETRAIN_FREQUENCY
from .database import engine
from .data_fetch import fetch_symbols_from_api, fetch_historical_data, insert_historical_data
from .modeling import load_existing_model, retrain_model
from .trade import on_bar
from .utils import is_market_open

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

model = None  # Global model object

def initialize_bot():
    """Fetch symbols, load data, and prepare the model."""
    logging.info("Fetching symbols from API...")
    api_symbols = fetch_symbols_from_api()
    if not api_symbols:
        logging.error("No symbols returned from API. Cannot proceed.")
        return

    logging.info("Inserting historical data for fetched symbols...")
    for sym in api_symbols:
        start_date = "2022-01-01"
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
        hist = fetch_historical_data(sym, start_date=start_date, end_date=end_date)
        if hist.empty:
            logging.info(f"No historical data for {sym}. Skipping.")
            continue
        insert_historical_data(sym, hist)

    logging.info("Attempting to load existing model...")
    global model
    model = load_existing_model()

    if model is None:
        logging.info("No existing model loaded. Retraining an initial model...")
        model = retrain_model()
        if model is not None:
            logging.info("Initial model trained and saved successfully.")
        else:
            logging.info("Initial model training failed or insufficient data.")
    else:
        logging.info("Using loaded model from previous run.")


def scheduled_retrain():
    """Retrains the model periodically outside market hours."""
    global model
    if not is_market_open():
        logging.info("Market is closed. Scheduled retraining started...")
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
    # Retrain model every X minutes (defined in config.RETRAIN_FREQUENCY)
    scheduler.add_job(scheduled_retrain, 'interval', minutes=RETRAIN_FREQUENCY)
    scheduler.start()
    logging.info("Scheduler started for periodic retraining.")
    atexit.register(lambda: scheduler.shutdown())


def start_stream(symbols):
    """
    Starts the Alpaca data stream for the given symbols. 
    Attaches 'on_bar' callback to each symbol's bar updates.
    """
    global model

    # Use 'iex' data_feed unless your plan has 'sip' 
    stream = Stream(
        APCA_API_KEY_ID,
        APCA_API_SECRET_KEY,
        base_url="https://stream.data.alpaca.markets",
        data_feed='iex'
    )

    for symbol in symbols:
        @stream.on_bar(symbol)
        async def bar_callback(bar, symbol=symbol):
            global model
            model = on_bar(bar, model)

    logging.info("Starting data stream...")
    try:
        stream.run()
    except KeyboardInterrupt:
        logging.info("Stream stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred during streaming: {e}")


def main():
    initialize_bot()
    setup_scheduler()
    symbols = fetch_symbols_from_api()
    if symbols:
        start_stream(symbols)


if __name__ == "__main__":
    main()
