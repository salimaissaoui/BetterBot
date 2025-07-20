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
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

model = None  # Global model object
ib = IB()     # Global IB instance
eastern = ZoneInfo("America/New_York")



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
    scheduler.add_job(scheduled_retrain, 'interval', minutes=RETRAIN_FREQUENCY)
    scheduler.start()
    logging.info("Scheduler started for periodic retraining.")
    atexit.register(lambda: scheduler.shutdown())

def start_stream(symbols):
    global model, ib

    logging.info("Connecting to IBKR...")
    ib.connect(IB_HOST, IB_PORT, clientId=1)

    for symbol_obj in symbols:
        # 🌟 Sanitize the symbol extraction properly
        if isinstance(symbol_obj, str):
            symbol = symbol_obj
        elif isinstance(symbol_obj, tuple) and len(symbol_obj) == 1:
            symbol = symbol_obj[0]  # ('AAPL',) → 'AAPL'
        elif hasattr(symbol_obj, "symbol"):
            symbol = symbol_obj.symbol
        else:
            logging.warning(f"Unexpected symbol format: {symbol_obj}")
            continue

        contract = Stock(symbol, 'SMART', 'USD')
        try:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                logging.warning(f"[{symbol}] Contract could not be qualified. Skipping.")
                continue
        except Exception as e:
            logging.error(f"[{symbol}] Contract qualification failed: {e}")
            continue

        def bar_handler(bar_list, symbol=symbol):
            global model
            if not bar_list:
                return
            latest_bar = bar_list[-1]
            setattr(latest_bar, "symbol", symbol)
            model = on_bar(latest_bar, model)

        bars = ib.reqRealTimeBars(
    contract=contract,
    barSize=5,
    whatToShow='TRADES',
    useRTH=True,
    realTimeBarsOptions=[]
)

        bars.updateEvent += bar_handler
        logging.info(f"[{symbol}] Subscribed to real-time bars.")

    logging.info("Real-time streaming setup completed.")








from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

def start_streaming_realtime_bars(ib, symbols):
    logging.info("Switching to delayed-frozen market data (no subscriptions needed)…")
    ib.reqMarketDataType(4)  # 4 = delayed-frozen feed

    logging.info("Subscribing to (delayed-frozen) real-time bar streams for each symbol...")
    for symbol_obj in symbols:
        # 💡 Clean and extract the symbol safely
        if isinstance(symbol_obj, str):
            symbol = symbol_obj
        elif isinstance(symbol_obj, tuple) and len(symbol_obj) == 1:
            symbol = symbol_obj[0]
        elif hasattr(symbol_obj, 'symbol'):
            symbol = symbol_obj.symbol
        else:
            logging.warning(f"Unrecognized symbol format: {symbol_obj} — skipping.")
            continue

        contract = Stock(symbol, 'SMART', 'USD')

        try:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                logging.warning(f"[{symbol}] Contract could not be qualified. Skipping.")
                continue

            # ✅ Define the bar handler inside the loop and capture symbol correctly
            def bar_handler(bars, sym=symbol):
                global model
                if not bars:
                    return
                latest_bar = bars[-1]
                setattr(latest_bar, "symbol", sym)
                logging.debug(f"[{sym}] Bar at {latest_bar.time} (delayed)")
                model = on_bar(latest_bar, model)

            # request bars exactly as before—now they'll be delayed-frozen
            bars = ib.reqRealTimeBars(
                contract=contract,
                barSize=5,
                whatToShow='TRADES',
                useRTH=True,
                realTimeBarsOptions=[]
            )
            bars.updateEvent += bar_handler

            logging.info(f"[{symbol}] Subscribed to delayed-frozen real-time bars.")
        except Exception as e:
            logging.error(f"[{symbol}] Failed to subscribe: {e}", exc_info=True)

    logging.info("Delayed-frozen streaming setup completed.")





def main():
    logging.info("Connecting to IBKR...")
    ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)
    ib.reqMarketDataType(4)

    symbols = initialize_bot()
    if not symbols:
        logging.error("No symbols found. Exiting.")
        return

    setup_scheduler()
    start_streaming_realtime_bars(ib, symbols)  # 🔄 LIVE streaming now

    logging.info("Starting IB event loop...")
    try:
        ib.run()
    except KeyboardInterrupt:
        logging.info("Shutting down from keyboard interrupt...")
    finally:
        ib.disconnect()
        logging.info("Disconnected from IBKR. Goodbye.")




if __name__ == "__main__":
    main()
