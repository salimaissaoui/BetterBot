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
from .advanced_modeling import get_advanced_model, run_advanced_training
from .advanced_features import create_advanced_features, detect_market_regime
from .model_performance import (
    log_trade_performance, log_portfolio_performance, log_model_prediction,
    run_adaptive_learning_cycle, setup_ab_testing, get_model_for_prediction
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

ib = IB()

active_positions = {}
latest_indicators_dict = {}
previous_obv = {}
bar_count_since_last_train = 0
advanced_model = None
use_advanced_features = True

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
    
    # Adaptive thresholds based on recent performance
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
        
        # Detect market regime for additional context
        try:
            with get_session() as session:
                recent_data = session.query(StockData).filter(
                    StockData.symbol == symbol
                ).order_by(StockData.timestamp.desc()).limit(100).all()
                
                if len(recent_data) > 50:
                    df = pd.DataFrame([{
                        'open': d.open, 'high': d.high, 'low': d.low,
                        'close': d.close, 'volume': d.volume, 'timestamp': d.timestamp
                    } for d in reversed(recent_data)])
                    
                    market_regime = detect_market_regime(df)
                    logging.debug(f"[{symbol}] Market regime: {market_regime}")
                    
                    # Adjust thresholds based on regime
                    if market_regime == 'volatile':
                        BUY_THRESHOLD = 0.55  # Be more conservative in volatile markets
                        SHORT_THRESHOLD = 0.45
                    elif market_regime == 'bullish':
                        BUY_THRESHOLD = 0.50  # Be more aggressive in bull markets
                    elif market_regime == 'bearish':
                        SHORT_THRESHOLD = 0.50  # Be more aggressive shorting in bear markets
        except Exception as e:
            logging.warning(f"Error detecting market regime for {symbol}: {e}")

        # Enhanced trading logic with regime awareness
        want_to_buy = (prob > BUY_THRESHOLD)
        want_to_sell = (prob < SELL_THRESHOLD) and (symbol in active_positions and active_positions[symbol] == 'long')
        want_to_short = (prob < SHORT_THRESHOLD)
        want_to_cover = (prob > COVER_THRESHOLD) and (symbol in active_positions and active_positions[symbol] == 'short')
        
        logging.debug(f"[{symbol}] Conditions - Buy: {want_to_buy}, Sell: {want_to_sell}, Short: {want_to_short}, Cover: {want_to_cover}")
        logging.debug(f"[{symbol}] Prob: {prob:.3f}, Thresholds - Buy: {BUY_THRESHOLD}, Short: {SHORT_THRESHOLD}")

        trade_executed = False
        entry_price = None
        
        if want_to_buy:
            if symbol not in active_positions:
                logging.info(f"ML-Enhanced BUY signal for {symbol}: prob={prob:.2f}, macd_diff={macd_diff:.4f}")
                
                # Get current price for tracking
                try:
                    contract = Stock(symbol, 'SMART', 'USD')
                    ticker = ib.reqMktData(contract, '', False, False)
                    ib.sleep(1)  # Wait for price update
                    entry_price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
                    ib.cancelMktData(contract)
                except:
                    entry_price = None
                
                # Use ML position sizing for buy orders
                qty = submit_ml_sized_order(symbol, 'buy', prob, entry_price)
                if qty > 0:
                    active_positions[symbol] = 'long'
                    trade_executed = True
                    
                    # Log trade performance
                    log_trade_performance({
                        'symbol': symbol,
                        'action': 'BUY',
                        'quantity': qty,
                        'entry_price': entry_price,
                        'entry_time': datetime.now(),
                        'prediction_confidence': prob,
                        'regime': globals().get('current_regime', 'unknown')
                    })
            else:
                logging.info(f"Already holding {symbol}, no need to buy again.")

        elif want_to_sell:
            logging.info(f"Selling {symbol}: prob={prob:.2f}")
            
            # Get exit price and calculate PnL
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                ticker = ib.reqMktData(contract, '', False, False)
                ib.sleep(1)
                exit_price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
                ib.cancelMktData(contract)
            except:
                exit_price = None
                
            close_position(symbol)
            
            # Log completed trade
            if symbol in active_positions:
                log_trade_performance({
                    'symbol': symbol,
                    'action': 'SELL',
                    'exit_price': exit_price,
                    'exit_time': datetime.now(),
                    'position_type': 'long'
                })
                
            del active_positions[symbol]
            trade_executed = True

        elif want_to_short:
            if symbol not in active_positions:
                logging.info(f"ML-Enhanced SHORT signal for {symbol}: prob={prob:.2f}, macd_diff={macd_diff:.4f}")
                
                # Get current price for tracking
                try:
                    contract = Stock(symbol, 'SMART', 'USD')
                    ticker = ib.reqMktData(contract, '', False, False)
                    ib.sleep(1)
                    entry_price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
                    ib.cancelMktData(contract)
                except:
                    entry_price = None
                
                # Use ML position sizing for short orders
                qty = submit_ml_sized_order(symbol, 'sell', 1.0 - prob, entry_price)  # Invert confidence for shorting
                if qty > 0:
                    active_positions[symbol] = 'short'
                    trade_executed = True
                    
                    # Log trade performance
                    log_trade_performance({
                        'symbol': symbol,
                        'action': 'SHORT',
                        'quantity': qty,
                        'entry_price': entry_price,
                        'entry_time': datetime.now(),
                        'prediction_confidence': 1.0 - prob,
                        'regime': globals().get('current_regime', 'unknown')
                    })
            else:
                logging.info(f"Already holding {symbol} (long or short). Skipping short request.")

        elif want_to_cover:
            logging.info(f"Covering short for {symbol}: prob={prob:.2f}")
            
            # Get exit price
            try:
                contract = Stock(symbol, 'SMART', 'USD')
                ticker = ib.reqMktData(contract, '', False, False)
                ib.sleep(1)
                exit_price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
                ib.cancelMktData(contract)
            except:
                exit_price = None
                
            close_position(symbol)
            
            # Log completed trade
            if symbol in active_positions:
                log_trade_performance({
                    'symbol': symbol,
                    'action': 'COVER',
                    'exit_price': exit_price,
                    'exit_time': datetime.now(),
                    'position_type': 'short'
                })
                
            del active_positions[symbol]
            trade_executed = True

        else:
            logging.debug(f"{symbol}: prob={prob:.2f}, no trade signal triggered.")
        
        # Log prediction for model validation
        if not trade_executed:
            log_model_prediction(symbol, prob, confidence=abs(prob - 0.5) * 2)


def on_bar(bar, model):
    """
    Called whenever a new real-time bar arrives.
    Processes the bar data, computes indicators, makes predictions, and executes trades.
    Enhanced with advanced ML features and adaptive learning.
    """
    global latest_indicators_dict, previous_obv, bar_count_since_last_train, advanced_model
    
    # Use advanced model if available, fallback to basic model
    active_model = advanced_model if advanced_model and advanced_model.is_trained else model
    
    if active_model is None:
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
            
        # Get recent data for feature calculation
        try:
            with get_session() as session:
                recent_data = session.query(StockData).filter(
                    StockData.symbol == symbol
                ).order_by(StockData.timestamp.desc()).limit(200).all()  # More data for advanced features
                
                if len(recent_data) < 50:  # Need more data for advanced features
                    logging.info(f"Not enough data for {symbol} advanced features. Skipping.")
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
                
                # Use advanced features if enabled and available
                if use_advanced_features and advanced_model and advanced_model.is_trained:
                    try:
                        # Create advanced features
                        enhanced_df = create_advanced_features(df, symbol)
                        if enhanced_df.empty:
                            # Fallback to basic indicators
                            enhanced_df = compute_technical_indicators(df)
                        
                        # Use advanced model prediction
                        if not enhanced_df.empty and len(enhanced_df) > 0:
                            # Select relevant features for prediction
                            feature_cols = [col for col in enhanced_df.columns 
                                          if col not in ['timestamp', 'target', 'future_return', 'symbol']]
                            
                            if feature_cols:
                                latest_features = enhanced_df[feature_cols].iloc[-1:].fillna(0)
                                
                                # Get prediction from A/B testing system or advanced model
                                variant_name, selected_model = get_model_for_prediction()
                                if selected_model:
                                    prediction_prob = selected_model.predict_proba(latest_features)[0][1]
                                    logging.debug(f"[{symbol}] Using model variant: {variant_name}")
                                else:
                                    prediction_prob = advanced_model.predict_proba(latest_features)[0][1]
                                
                                confidence = abs(prediction_prob - 0.5) * 2  # Convert to 0-1 confidence
                                
                                logging.info(f"[{symbol}] Advanced prediction: {prediction_prob:.3f} (confidence: {confidence:.3f})")
                                
                                # Store latest indicators for compatibility
                                latest_indicators_dict[symbol] = enhanced_df.iloc[-1].to_dict()
                                
                                # Execute trade based on prediction
                                execute_trade([(symbol, prediction_prob)], active_model)
                                
                                # Log prediction for performance tracking
                                log_model_prediction(symbol, prediction_prob, confidence=confidence)
                                
                                bar_count_since_last_train += 1
                                
                                # Check for adaptive learning triggers every 50 bars
                                if bar_count_since_last_train % 50 == 0:
                                    try:
                                        adaptation_result = run_adaptive_learning_cycle(advanced_model)
                                        if adaptation_result['status'] != 'no_adaptation_needed':
                                            logging.info(f"Adaptive learning cycle: {adaptation_result['status']}")
                                    except Exception as adapt_error:
                                        logging.warning(f"Error in adaptive learning: {adapt_error}")
                                
                                return model
                                
                    except Exception as advanced_error:
                        logging.warning(f"Error with advanced features for {symbol}: {advanced_error}")
                        # Fall back to basic processing
                
                # Fallback to basic indicators and model
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
                    
                # Make prediction with basic model
                latest_features = features.iloc[-1:].values
                prediction_prob = model.predict_proba(latest_features)[0][1]  # Probability of class 1
                
                logging.info(f"[{symbol}] Basic prediction: {prediction_prob:.3f}")
                
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


def initialize_advanced_model():
    """Initialize the advanced model system"""
    global advanced_model
    
    try:
        logging.info("Initializing advanced model system...")
        
        # Load or train advanced model
        advanced_model = get_advanced_model()
        
        # Try to load existing model
        if not advanced_model.load_model("advanced_ensemble_model.pkl"):
            logging.info("No existing advanced model found. Training new model...")
            # Train new model in background
            success = run_advanced_training()
            if success:
                logging.info("Advanced model training completed successfully")
            else:
                logging.warning("Advanced model training failed. Using basic model fallback.")
        else:
            logging.info("Advanced model loaded successfully")
        
        return True
        
    except Exception as e:
        logging.error(f"Error initializing advanced model: {e}")
        return False


# Test function removed - using live trading loop