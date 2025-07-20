import logging
import pandas as pd
import traceback
from ib_insync import *
from datetime import datetime
from .config import (
    IB_HOST,
    IB_PORT,
    IB_CLIENT_ID_TRADE,
    NOTIONAL,
    STOP_LOSS_PCT,
    ALLOW_AFTER_HOURS_TRADING
)
from .position_sizing import ml_position_sizer
from .database import get_session, StockData
from .indicators import compute_technical_indicators, prepare_features
from .modeling import retrain_model
from .utils import is_market_open

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

ib = IB()

active_positions = {}
latest_indicators_dict = {}
previous_obv = {}
bar_count_since_last_train = 0

def ensure_ib_connected():
    """Ensure IB is connected before trading operations."""
    if not ib.isConnected():
        logging.info("Connecting to IBKR for trading operations...")
        ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID_TRADE)

def get_current_position(symbol):
    ensure_ib_connected()
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
    
    # Allow after-hours trading if enabled
    if ALLOW_AFTER_HOURS_TRADING:
        order.outsideRth = True
        
    contract = Stock(symbol, 'SMART', 'USD')
    ib.placeOrder(contract, order)
    logging.info(f"Closed position for {symbol}: {side} {abs(qty)} shares.")

def short_position(symbol, shares):
    ensure_ib_connected()
    logging.info(f"Submitting SHORT SELL order for {symbol} with {shares} shares.")
    contract = Stock(symbol, 'SMART', 'USD')
    order = MarketOrder('SELL', shares)
    
    # Allow after-hours trading if enabled
    if ALLOW_AFTER_HOURS_TRADING:
        order.outsideRth = True
        
    ib.placeOrder(contract, order)
    logging.info(f"Successfully submitted SHORT SELL order for {symbol} with {shares} shares.")

def submit_ml_sized_order(symbol, side, prediction_confidence, current_price=None):
    """Submit an order with ML-calculated position size"""
    ensure_ib_connected()
    try:
        # Use ML position sizing to determine optimal quantity
        qty = ml_position_sizer.calculate_position_size(
            ib, symbol, prediction_confidence, current_price
        )
        
        logging.info(f"ML Position Sizing: {symbol} - {qty} shares "
                    f"(confidence: {prediction_confidence:.3f})")

        contract = Stock(symbol, 'SMART', 'USD')
        main_order = MarketOrder(side.upper(), qty)
        
        # Allow after-hours trading if enabled
        if ALLOW_AFTER_HOURS_TRADING:
            main_order.outsideRth = True

        ib.placeOrder(contract, main_order)
        logging.info(f"ML-sized {side.upper()} order placed: {symbol}, qty={qty}")
        
        return qty
        
    except Exception as e:
        logging.error(f"Error submitting ML-sized order for {symbol}: {e}")
        return 0

def submit_notional_order_with_stop_loss(symbol, side, notional, stop_loss_pct):
    """Legacy function - kept for compatibility"""
    ensure_ib_connected()
    try:
        # Use a simple approach without market data - place adaptive order
        # Estimate shares using notional amount (assuming ~$100 per share average)
        estimated_price = 100.0  # Simple fallback price
        qty = max(1, int(notional // estimated_price))  # At least 1 share
        
        logging.info(f"Placing {side.upper()} order for {symbol}: {qty} shares (estimated)")

        contract = Stock(symbol, 'SMART', 'USD')
        main_order = MarketOrder(side.upper(), qty)
        
        # Allow after-hours trading if enabled
        if ALLOW_AFTER_HOURS_TRADING:
            main_order.outsideRth = True

        ib.placeOrder(contract, main_order)
        logging.info(f"{side.upper()} order placed: {symbol}, qty={qty}")
        
    except Exception as e:
        logging.error(f"Error submitting order for {symbol}: {e}")

     




def execute_trade(pred_results, model):
    global active_positions
    
    # Check if trading is allowed when market is closed
    if not is_market_open() and not ALLOW_AFTER_HOURS_TRADING:
        logging.info("Market is closed and after-hours trading is disabled. Skipping trade execution.")
        return
    elif not is_market_open() and ALLOW_AFTER_HOURS_TRADING:
        logging.info("Market is closed but after-hours trading is enabled. Proceeding with trades.")
    
    BUY_THRESHOLD = 0.51
    SELL_THRESHOLD = 0.49
    SHORT_THRESHOLD = 0.49
    COVER_THRESHOLD = 0.51

    for symbol, prob in pred_results:
        logging.info(f"Symbol: {symbol}, Probability: {prob:.2f}")

        indicators = latest_indicators_dict.get(symbol, {})
        macd_diff = indicators.get('macd_diff', 0.0)
        obv = indicators.get('obv', 0.0)
        obv_prev = indicators.get('obv_prev', obv)
        obv_increasing = (obv > obv_prev)

        # Simplified trading logic - remove OBV requirement for more trades
        want_to_buy = (prob > BUY_THRESHOLD)
        want_to_sell = (prob < SELL_THRESHOLD) and (symbol in active_positions and active_positions[symbol] == 'long')
        want_to_short = (prob < SHORT_THRESHOLD)
        want_to_cover = (prob > COVER_THRESHOLD) and (symbol in active_positions and active_positions[symbol] == 'short')
        
        logging.debug(f"[{symbol}] Conditions - Buy: {want_to_buy}, Sell: {want_to_sell}, Short: {want_to_short}, Cover: {want_to_cover}")
        logging.debug(f"[{symbol}] Prob: {prob:.3f}, OBV: {obv:.0f}, OBV_prev: {obv_prev:.0f}, OBV_increasing: {obv_increasing}")

        if want_to_buy:
            if symbol not in active_positions:
                logging.info(f"ML-Enhanced BUY signal for {symbol}: prob={prob:.2f}, macd_diff={macd_diff:.4f}")
                # Use ML position sizing for buy orders
                qty = submit_ml_sized_order(symbol, 'buy', prob)
                if qty > 0:
                    active_positions[symbol] = 'long'
            else:
                logging.info(f"Already holding {symbol}, no need to buy again.")

        elif want_to_sell:
            logging.info(f"Selling {symbol}: prob={prob:.2f}")
            close_position(symbol)
            del active_positions[symbol]

        elif want_to_short:
            if symbol not in active_positions:
                logging.info(f"ML-Enhanced SHORT signal for {symbol}: prob={prob:.2f}, macd_diff={macd_diff:.4f}")
                # Use ML position sizing for short orders
                qty = submit_ml_sized_order(symbol, 'sell', 1.0 - prob)  # Invert confidence for shorting
                if qty > 0:
                    active_positions[symbol] = 'short'
            else:
                logging.info(f"Already holding {symbol} (long or short). Skipping short request.")

        elif want_to_cover:
            logging.info(f"Covering short for {symbol}: prob={prob:.2f}")
            close_position(symbol)
            del active_positions[symbol]

        else:
            logging.debug(f"{symbol}: prob={prob:.2f}, no trade signal triggered.")


def on_bar(bar, model):
    """
    Called whenever a new real-time bar arrives.
    Processes the bar data, computes indicators, makes predictions, and executes trades.
    """
    global latest_indicators_dict, previous_obv, bar_count_since_last_train
    
    if model is None:
        logging.warning("No model available. Skipping bar processing.")
        return model
    
    symbol = getattr(bar, 'symbol', 'UNKNOWN')
    
    try:
        # Store bar data in database (using UPSERT to handle duplicates)
        try:
            with get_session() as session:
                # Handle both real-time bars and historical bars
                timestamp = getattr(bar, 'time', None) or getattr(bar, 'date', None)
                
                # Check if record already exists
                existing_record = session.query(StockData).filter(
                    StockData.symbol == symbol,
                    StockData.timestamp == timestamp
                ).first()
                
                if existing_record:
                    # Update existing record (handle NaN values)
                    existing_record.open = float(bar.open) if not pd.isna(bar.open) else existing_record.open
                    existing_record.high = float(bar.high) if not pd.isna(bar.high) else existing_record.high
                    existing_record.low = float(bar.low) if not pd.isna(bar.low) else existing_record.low
                    existing_record.close = float(bar.close) if not pd.isna(bar.close) else existing_record.close
                    existing_record.volume = int(float(bar.volume)) if not pd.isna(bar.volume) and bar.volume is not None else 0
                    logging.debug(f"Updated existing record for {symbol} at {timestamp}")
                else:
                    # Insert new record (handle NaN values)
                    stock_data = StockData(
                        symbol=symbol,
                        open=float(bar.open) if not pd.isna(bar.open) else 0.0,
                        high=float(bar.high) if not pd.isna(bar.high) else 0.0,
                        low=float(bar.low) if not pd.isna(bar.low) else 0.0, 
                        close=float(bar.close) if not pd.isna(bar.close) else 0.0,
                        volume=int(float(bar.volume)) if not pd.isna(bar.volume) and bar.volume is not None else 0,
                        timestamp=timestamp
                    )
                    session.add(stock_data)
                    logging.debug(f"Inserted new record for {symbol} at {timestamp}")
                
                session.commit()
        except Exception as db_error:
            logging.warning(f"Database error for {symbol}: {db_error}")
            # Continue with processing even if database insert fails
            pass
            
        # Get recent data for indicator calculation
        try:
            with get_session() as session:
                recent_data = session.query(StockData).filter(
                    StockData.symbol == symbol
                ).order_by(StockData.timestamp.desc()).limit(50).all()
                
                if len(recent_data) < 20:  # Need enough data for indicators
                    logging.info(f"Not enough data for {symbol} indicators. Skipping.")
                    return model
                    
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'open': d.open,
                    'high': d.high, 
                    'low': d.low,
                    'close': d.close,
                    'volume': d.volume,
                    'timestamp': d.timestamp
                } for d in reversed(recent_data)])
                
                # Compute indicators
                indicators = compute_technical_indicators(df)
                if indicators is None or indicators.empty:
                    logging.warning(f"Failed to compute indicators for {symbol}")
                    return model
                    
                # Store latest indicators
                latest_row = indicators.iloc[-1]
                latest_indicators_dict[symbol] = latest_row.to_dict()
                
                # Prepare features for prediction
                features, _ = prepare_features(indicators)  # Unpack tuple (X, y)
                if features is None or features.empty:
                    logging.warning(f"Failed to prepare features for {symbol}")
                    return model
                    
                # Make prediction
                latest_features = features.iloc[-1:].values
                prediction_prob = model.predict_proba(latest_features)[0][1]  # Probability of class 1
                
                logging.info(f"[{symbol}] Prediction made: {prediction_prob:.3f}")
                
                # Execute trade based on prediction
                execute_trade([(symbol, prediction_prob)], model)
                
                bar_count_since_last_train += 1
        except Exception as processing_error:
            logging.warning(f"Error processing indicators/predictions for {symbol}: {processing_error}")
            return model
            
    except Exception as e:
        logging.error(f"Error processing bar for {symbol}: {e}")
        logging.error(traceback.format_exc())
    
    return model


# Test function removed - using live trading loop