import logging
from ib_insync import *
from datetime import datetime
from .config import (
    IB_HOST,
    IB_PORT,
    IB_CLIENT_ID_TRADE,
    NOTIONAL,
    STOP_LOSS_PCT
)
from .database import get_session, StockData
from .indicators import compute_technical_indicators, prepare_features
from .modeling import retrain_model
from .utils import is_market_open

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

ib = IB()
ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID_TRADE)

active_positions = {}
latest_indicators_dict = {}
previous_obv = {}
bar_count_since_last_train = 0

def get_current_position(symbol):
    positions = ib.positions()
    for pos in positions:
        if pos.contract.symbol == symbol:
            qty = pos.position
            logging.info(f"Current position for {symbol}: {qty} shares.")
            return qty
    return 0.0

def close_position(symbol):
    qty = get_current_position(symbol)
    if qty == 0:
        return
    side = 'SELL' if qty > 0 else 'BUY'
    order = MarketOrder(side, abs(qty))
    contract = Stock(symbol, 'SMART', 'USD')
    ib.placeOrder(contract, order)
    logging.info(f"Closed position for {symbol}: {side} {abs(qty)} shares.")

def short_position(symbol, shares):
    logging.info(f"Submitting SHORT SELL order for {symbol} with {shares} shares.")
    contract = Stock(symbol, 'SMART', 'USD')
    order = MarketOrder('SELL', shares)
    ib.placeOrder(contract, order)
    logging.info(f"Successfully submitted SHORT SELL order for {symbol} with {shares} shares.")

def submit_notional_order_with_stop_loss(symbol, side, notional, stop_loss_pct):
    try:
        market_data = ib.reqMktData(Stock(symbol, 'SMART', 'USD'), '', False, False)
        ib.sleep(1)
        current_price = float(market_data.last)
        if current_price <= 0:
            raise ValueError("Invalid current price.")

        qty = int(notional // current_price)
        if qty <= 0:
            logging.warning(f"Not enough notional to trade {symbol}.")
            return

        stop_side = 'SELL' if side.upper() == 'BUY' else 'BUY'
        stop_price = round(current_price * (1.0 - stop_loss_pct), 2) if side.upper() == 'BUY' else round(current_price * (1.0 + stop_loss_pct), 2)

        contract = Stock(symbol, 'SMART', 'USD')
        main_order = MarketOrder(side.upper(), qty)
        stop_order = StopOrder(stop_side, qty, stop_price)

        ib.placeOrder(contract, main_order)
        ib.sleep(1)
        ib.placeOrder(contract, stop_order)

        logging.info(f"{side.upper()} order with STOP LOSS: {symbol}, qty={qty}, stop_price={stop_price}")
    except Exception as e:
        logging.error(f"Error submitting order for {symbol}: {e}")

     




def execute_trade(pred_results, model):
    global active_positions
    BUY_THRESHOLD = 0.52
    SELL_THRESHOLD = 0.48
    SHORT_THRESHOLD = 0.48
    COVER_THRESHOLD = 0.52

    for symbol, prob in pred_results:
        logging.info(f"Symbol: {symbol}, Probability: {prob:.2f}")

        indicators = latest_indicators_dict.get(symbol, {})
        macd_diff = indicators.get('macd_diff', 0.0)
        obv = indicators.get('obv', 0.0)
        obv_prev = indicators.get('obv_prev', obv)
        obv_increasing = (obv > obv_prev)

        want_to_buy = (prob > BUY_THRESHOLD) and obv_increasing
        want_to_sell = (prob < SELL_THRESHOLD) and (symbol in active_positions and active_positions[symbol] == 'long')
        want_to_short = (prob < SHORT_THRESHOLD) and not obv_increasing
        want_to_cover = (prob > COVER_THRESHOLD) and (symbol in active_positions and active_positions[symbol] == 'short')

        if want_to_buy:
            if symbol not in active_positions:
                logging.info(f"Buying {symbol}: prob={prob:.2f}, macd_diff={macd_diff:.4f}")
                submit_notional_order_with_stop_loss(symbol, 'buy', NOTIONAL, STOP_LOSS_PCT)
                active_positions[symbol] = 'long'
            else:
                logging.info(f"Already holding {symbol}, no need to buy again.")

        elif want_to_sell:
            logging.info(f"Selling {symbol}: prob={prob:.2f}")
            close_position(symbol)
            del active_positions[symbol]

        elif want_to_short:
            if symbol not in active_positions:
                market_data = ib.reqMktData(Stock(symbol, 'SMART', 'USD'), '', False, False)
                ib.sleep(1)
                price = float(market_data.last)
                shares_to_short = int(NOTIONAL // price)

                if shares_to_short > 0:
                    logging.info(f"Shorting {symbol}: prob={prob:.2f}, macd_diff={macd_diff:.4f}, shares={shares_to_short}")
                    short_position(symbol, shares_to_short)
                    active_positions[symbol] = 'short'
                else:
                    logging.warning(f"Insufficient notional to short {symbol} at {price:.2f}")
            else:
                logging.info(f"Already holding {symbol} (long or short). Skipping short request.")

        elif want_to_cover:
            logging.info(f"Covering short for {symbol}: prob={prob:.2f}")
            close_position(symbol)
            del active_positions[symbol]

        else:
            logging.info(f"{symbol}: prob={prob:.2f}, no trade signal triggered.")