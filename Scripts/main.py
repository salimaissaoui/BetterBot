import logging
import atexit
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from alpaca_trade_api import Stream

from .config import APCA_API_KEY_ID, APCA_API_SECRET_KEY, RETRAIN_FREQUENCY
from .database import engine
from Scripts.data_fetch import fetch_and_load_symbols, fetch_historical_data, insert_historical_data
from .modeling import load_existing_model, retrain_model
from .trade import on_bar
from .utils import is_market_open

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

model = None  # Global model object


def initialize_bot():
    """Fetch symbols, load data, and prepare the model."""
    logging.info("Fetching and loading symbols from API...")
    symbols = fetch_and_load_symbols()  # Fetch and load S&P 500 symbols and data

    if not symbols:
        logging.error("No symbols fetched. Cannot proceed.")
        return

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

    return symbols


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
    logging.info("Initializing bot...")
    symbols = initialize_bot()  # Initialize bot and get symbols

    if not symbols:
        logging.error("Initialization failed. Exiting.")
        return

    setup_scheduler()  # Setup periodic retraining scheduler
    logging.info("Starting data stream for fetched symbols...")
    start_stream(symbols)  # Start streaming data for the fetched symbols


if __name__ == "__main__":
    main()
