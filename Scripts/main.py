import asyncio
import logging
import atexit
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from ib_insync import IB, util, Stock
from zoneinfo import ZoneInfo

from .config import IB_HOST, IB_PORT, RETRAIN_FREQUENCY, IB_CLIENT_ID
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
ib = IB()     # Global IB instance
eastern = ZoneInfo("America/New_York")
polling_symbols = []  # Symbols for historical data polling


def initialize_bot():
    """Fetch symbols, load data, and prepare the model."""
    logging.info("Fetching and loading symbols from API...")
    symbols = fetch_and_load_symbols(ib)  # Fetch and load S&P 500 symbols and data

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
    """Retrains the model periodically."""
    global model
    from .config import ALLOW_AFTER_HOURS_TRADING
    
    # If after-hours trading is enabled, retrain regardless of market hours
    if ALLOW_AFTER_HOURS_TRADING or not is_market_open():
        if ALLOW_AFTER_HOURS_TRADING:
            logging.info("After-hours trading enabled. Scheduled retraining started...")
        else:
            logging.info("Market is closed. Scheduled retraining started...")
            
        new_model = retrain_model()
        if new_model is not None:
            model = new_model
            logging.info("Scheduled retraining completed successfully.")
        else:
            logging.info("Scheduled retraining had no new model.")
    else:
        logging.info("Market is open and after-hours trading disabled. Skipping scheduled retraining.")


def setup_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_retrain, 'interval', minutes=RETRAIN_FREQUENCY)
    scheduler.start()
    logging.info("Scheduler started for background model training.")
    atexit.register(lambda: scheduler.shutdown())


def trading_loop():
    """
    Main trading loop: fetch data -> run AI predictions -> place orders -> repeat
    """
    global model, polling_symbols
    
    if not polling_symbols:
        logging.warning("No symbols to trade. Skipping trading cycle.")
        return
    
    logging.info(f"Starting trading cycle for {len(polling_symbols)} symbols...")
    
    for symbol in polling_symbols[:5]:  # Process 5 symbols per cycle
        try:
            logging.info(f"Processing {symbol}...")
            
            # Step 1: Fetch recent delayed data
            contract = Stock(symbol, 'SMART', 'USD')
            
            try:
                # Request historical data with delayed feed
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='2 D',  # Last 2 days of data
                    barSizeSetting='5 mins',
                    whatToShow='TRADES',
                    useRTH=False,  # Include extended hours
                    formatDate=1
                )
                
                if not bars or len(bars) == 0:
                    logging.warning(f"No delayed data received for {symbol}")
                    continue
                
                # Get the latest bar
                latest_bar = bars[-1]
                setattr(latest_bar, "symbol", symbol)
                
                logging.debug(f"Got delayed data for {symbol}: {latest_bar.close}")
                
            except Exception as data_error:
                logging.warning(f"Data request error for {symbol}: {data_error}")
                continue
            
            # Step 2: Process the latest bar with AI
            if model is not None:
                model = on_bar(latest_bar, model)
            else:
                logging.warning("No model available for predictions")
                
        except Exception as e:
            logging.error(f"Error in trading cycle for {symbol}: {e}")
    
    # Rotate symbols for next cycle
    if len(polling_symbols) > 5:
        polling_symbols = polling_symbols[5:] + polling_symbols[:5]
    
    logging.info("Trading cycle completed.")


def main():
    global model, polling_symbols
    
    logging.info("Starting AI Trading Bot...")
    logging.info("Connecting to IBKR...")
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    
    # Request delayed market data (no subscription needed)
    ib.reqMarketDataType(3)  # 3 = delayed data
    logging.info("Enabled delayed market data")

    # Initialize: fetch symbols and load/train model
    symbols = initialize_bot()
    if not symbols:
        logging.error("No symbols found. Exiting.")
        return

    # Set up symbols for trading
    polling_symbols = []
    for symbol_obj in symbols:
        if isinstance(symbol_obj, str):
            symbol = symbol_obj
        elif isinstance(symbol_obj, tuple) and len(symbol_obj) == 1:
            symbol = symbol_obj[0]
        else:
            continue
        polling_symbols.append(symbol)
    
    logging.info(f"Loaded {len(polling_symbols)} symbols for trading")

    # Start background model training
    setup_scheduler()
    
    logging.info("Starting main trading loop...")
    logging.info("Press Ctrl+C to stop the bot")
    
    try:
        # Main trading loop
        while True:
            # Execute one trading cycle
            trading_loop()
            
            # Wait before next cycle (30 seconds)
            ib.sleep(30)
            
    except KeyboardInterrupt:
        logging.info("Shutting down from keyboard interrupt...")
    finally:
        ib.disconnect()
        logging.info("Disconnected from IBKR. Goodbye.")


if __name__ == "__main__":
    main()