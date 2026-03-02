from dotenv import load_dotenv
load_dotenv()  # Load .env into os.environ before config.py is imported

import asyncio
import logging
import atexit
import pandas as pd
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
from ib_insync import IB, util, Stock
from zoneinfo import ZoneInfo

from .config import IB_HOST, IB_PORT, RETRAIN_FREQUENCY, IB_CLIENT_ID
from .database import engine, get_session, StockData
from .data_fetch import fetch_and_load_symbols, fetch_historical_data, insert_historical_data
from .modeling import load_existing_model, retrain_model
from .trade import on_bar, initialize_advanced_model
from .utils import is_market_open
from .model_performance import log_portfolio_performance, setup_ab_testing

logging.basicConfig(
    level=logging.INFO,
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
    
    # Initialize advanced model system
    logging.info("Initializing advanced model system...")
    advanced_success = initialize_advanced_model()
    
    # Initialize Market Regime Detector [RISK-01]
    from .advanced_features import advanced_features
    regime_detector = advanced_features.regime_detector
    if not regime_detector.load_model():
        logging.info("Fitting new Market Regime Detector using S&P 500 data...")
        try:
            import yfinance as yf
            # Fetch 5 years of S&P 500 data for robust clustering
            gspc = yf.download("^GSPC", period="5y", interval="1d")
            if not gspc.empty:
                # Rename columns to lowercase to match calculate_regime_features expectation
                gspc.columns = [c.lower() for c in gspc.columns]
                regime_detector.fit_regime_detector({"^GSPC": gspc})
                logging.info("Market Regime Detector fitted and saved successfully.")
            else:
                logging.error("Failed to fetch S&P 500 data for regime detector.")
        except Exception as e:
            logging.error(f"Error fitting regime detector: {e}")

    if advanced_success:
        logging.info("Advanced model system initialized successfully")
        
        # Setup A/B testing if we have both basic and advanced models
        if model is not None:
            from .trade import advanced_model as _adv_model
            models = {
                'basic_model': model,
                'advanced_model': _adv_model,  # Actual model object from trade.py
            }
            allocations = {
                'basic_model': 30.0,  # 30% allocation to basic model
                'advanced_model': 70.0  # 70% allocation to advanced model
            }
            setup_ab_testing(models, allocations)
            logging.info("A/B testing setup completed")
    else:
        logging.warning("Advanced model initialization failed. Using basic model only.")

    # Reconcile position exit registry against live IBKR positions [EXIT-04]
    from .exit_manager import exit_manager
    from .trade import active_positions as ap
    logging.info("Reconciling position exit registry with IBKR...")
    account_summary = ib.accountSummary()
    account_nav = next(
        (float(x.value) for x in account_summary if x.tag == 'NetLiquidation'), 0.0
    )
    live_positions = ib.positions()
    exit_manager.reconcile_from_ibkr(live_positions, account_nav)
    # Sync active_positions dict in trade.py to match reconciled registry
    for sym, rec in exit_manager.positions.items():
        ap[sym] = rec.direction
    logging.info(f"Position reconciliation complete: {len(exit_manager.positions)} positions active")

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


def hourly_portfolio_scan():
    """
    Hourly portfolio scanning with performance analysis and trading decisions.
    Scans current portfolio, runs predictions, and decides whether to buy more, sell, or hold.
    """
    global model, advanced_model
    from .model_performance import log_portfolio_performance
    from .trade import get_current_position, close_position, active_positions as ap
    from .exit_manager import exit_manager
    from .database import TradeLog, get_session

    logging.info("Starting hourly portfolio scan...")
    
    try:
        # Get current portfolio state
        account_summary = ib.accountSummary()
        positions = ib.positions()
        
        portfolio_value = 0
        available_funds = 0
        
        for item in account_summary:
            if item.tag == 'NetLiquidation':
                portfolio_value = float(item.value)
            elif item.tag == 'AvailableFunds':
                available_funds = float(item.value)
        
        # Analyze each position
        position_analysis = []
        for pos in positions:
            if pos.position != 0:  # Only analyze active positions
                symbol = pos.contract.symbol
                current_qty = pos.position
                avg_cost = pos.avgCost
                market_value = pos.marketValue
                unrealized_pnl = pos.unrealizedPNL
                
                # Get current market price
                try:
                    contract = Stock(symbol, 'SMART', 'USD')
                    ticker = ib.reqMktData(contract, '', False, False)
                    ib.sleep(2)  # Wait for price update
                    current_price = ticker.last if ticker.last and ticker.last > 0 else ticker.close
                    ib.cancelMktData(contract)
                    
                    if current_price and current_price > 0:
                        # ExitManager exit check — fires before prediction-based decisions
                        _exit_signal = exit_manager.check_exits(symbol, current_price)
                        if _exit_signal:
                            logging.info(f"[{symbol}] Hourly scan exit: {_exit_signal} at {current_price:.2f}")
                            close_position(symbol)
                            exit_manager.clear_position(symbol)
                            if symbol in ap:
                                del ap[symbol]
                            # Log exit to TradeLog (OBS-01)
                            try:
                                with get_session() as _sess:
                                    _sess.add(TradeLog(
                                        symbol=symbol,
                                        action=_exit_signal,
                                        exit_price=current_price,
                                        exit_reason=_exit_signal,
                                        regime_at_decision=None,
                                        decision_time=datetime.now(),
                                    ))
                                    _sess.commit()
                            except Exception as _log_err:
                                logging.warning(f"[{symbol}] Failed to log hourly exit: {_log_err}")
                            # Skip prediction and action logic for this position
                            position_analysis.append({
                                'symbol': symbol, 'quantity': current_qty,
                                'avg_cost': avg_cost, 'current_price': current_price,
                                'market_value': market_value, 'unrealized_pnl': unrealized_pnl,
                                'position_return': (current_price - avg_cost) / avg_cost,
                                'prediction_prob': 0.5, 'confidence': 0.0,
                                'recommended_action': _exit_signal,
                                'action_reason': f'ExitManager: {_exit_signal}',
                            })
                            continue  # Skip rest of position processing for this symbol

                        # Calculate position performance
                        position_return = (current_price - avg_cost) / avg_cost
                        
                        # Run prediction for current holding
                        active_model = advanced_model if advanced_model and advanced_model.is_trained else model
                        prediction_prob = 0.5  # Default neutral
                        confidence = 0.0
                        
                        if active_model:
                            try:
                                # Get recent data for prediction
                                with get_session() as session:
                                    recent_data = session.query(StockData).filter(
                                        StockData.symbol == symbol
                                    ).order_by(StockData.timestamp.desc()).limit(100).all()
                                    
                                    if len(recent_data) >= 50:
                                        from .indicators import compute_technical_indicators, prepare_features
                                        
                                        df = pd.DataFrame([{
                                            'open': d.open, 'high': d.high, 'low': d.low,
                                            'close': d.close, 'volume': d.volume, 'timestamp': d.timestamp
                                        } for d in reversed(recent_data)])
                                        
                                        indicators = compute_technical_indicators(df)
                                        if not indicators.empty:
                                            # Detect market regime for this position [RISK-02]
                                            from .advanced_features import detect_market_regime
                                            market_regime = detect_market_regime(df)
                                            
                                            if hasattr(active_model, 'predict_proba') and hasattr(active_model, 'is_trained'):
                                                # Advanced model
                                                from .advanced_features import create_advanced_features
                                                enhanced_df = create_advanced_features(df, symbol)
                                                if not enhanced_df.empty:
                                                    feature_cols = [col for col in enhanced_df.columns 
                                                                  if col not in ['timestamp', 'target', 'future_return', 'symbol']]
                                                    if feature_cols:
                                                        latest_features = enhanced_df[feature_cols].iloc[-1:].fillna(0)
                                                        prediction_prob = active_model.predict_proba(latest_features)[0][1]
                                                        confidence = abs(prediction_prob - 0.5) * 2
                                            else:
                                                # Basic model
                                                features, _ = prepare_features(indicators)
                                                if not features.empty:
                                                    latest_features = features.iloc[-1:].values
                                                    prediction_prob = active_model.predict_proba(latest_features)[0][1]
                                                    confidence = abs(prediction_prob - 0.5) * 2
                                                    
                            except Exception as pred_error:
                                logging.warning(f"Error making prediction for {symbol}: {pred_error}")
                        
                        # Determine action based on prediction and current performance
                        action = "HOLD"  # Default action
                        action_reason = "Default hold"
                        
                        # Decision logic
                        if prediction_prob > 0.65 and confidence > 0.3:
                            if position_return > -0.05:  # Not deeply underwater
                                action = "BUY_MORE"
                                action_reason = f"Strong buy signal (prob: {prediction_prob:.3f}, conf: {confidence:.3f})"
                        elif prediction_prob < 0.35 and confidence > 0.3:
                            action = "SELL"
                            action_reason = f"Strong sell signal (prob: {prediction_prob:.3f}, conf: {confidence:.3f})"
                        
                        position_info = {
                            'symbol': symbol,
                            'quantity': current_qty,
                            'avg_cost': avg_cost,
                            'current_price': current_price,
                            'market_value': market_value,
                            'unrealized_pnl': unrealized_pnl,
                            'position_return': position_return,
                            'prediction_prob': prediction_prob,
                            'confidence': confidence,
                            'recommended_action': action,
                            'action_reason': action_reason
                        }
                        
                        position_analysis.append(position_info)
                        
                        logging.info(f"[{symbol}] Position Analysis:")
                        logging.info(f"  Qty: {current_qty}, Avg Cost: ${avg_cost:.2f}, Current: ${current_price:.2f}")
                        logging.info(f"  Return: {position_return:.2%}, P&L: ${unrealized_pnl:.2f}")
                        logging.info(f"  Prediction: {prediction_prob:.3f} (conf: {confidence:.3f})")
                        logging.info(f"  Recommendation: {action} - {action_reason}")
                        
                except Exception as price_error:
                    logging.warning(f"Error getting price for {symbol}: {price_error}")
        
        # Execute recommended actions
        from .trade import submit_ml_sized_order, close_position
        
        for pos_info in position_analysis:
            symbol = pos_info['symbol']
            action = pos_info['recommended_action']
            
            # Circuit breaker suppresses new entries from hourly scan too
            if action in ('BUY_MORE',) and exit_manager.is_circuit_breaker_active():
                logging.info(f"[{symbol}] Circuit breaker active — suppressing hourly scan BUY_MORE")
                continue

            # Sentiment gate for BUY_MORE [SENT-02]
            if action == 'BUY_MORE':
                from .news import get_ticker_sentiment
                _sent = get_ticker_sentiment(symbol)
                if _sent < -0.05:
                    logging.info(f"[{symbol}] BUY_MORE suppressed by sentiment: {_sent:.3f}")
                    continue

            try:
                if action == "BUY_MORE":
                    # Buy more shares with 10% of available funds
                    if available_funds > 1000:  # Minimum $1000 to trade
                        # Extract market_regime if available from pos_info
                        regime = pos_info.get('market_regime', 'unknown')
                        qty = submit_ml_sized_order(
                            symbol, 'buy', 
                            pos_info['prediction_prob'], 
                            pos_info['current_price'],
                            market_regime=regime
                        )
                        if qty > 0:
                            logging.info(f"Executed BUY_MORE for {symbol}: +{qty} shares")
                            available_funds -= qty * pos_info['current_price']
                
                elif action == "SELL":
                    close_position(symbol)
                    exit_manager.clear_position(symbol)
                    if symbol in ap:
                        del ap[symbol]
                    try:
                        with get_session() as _sess:
                            _sess.add(TradeLog(
                                symbol=symbol,
                                action='SELL',
                                exit_price=pos_info.get('current_price'),
                                exit_reason='SIGNAL',
                                decision_time=datetime.now(),
                            ))
                            _sess.commit()
                    except Exception:
                        pass
                    logging.info(f"Executed SELL for {symbol}: closed position")
                    
            except Exception as trade_error:
                logging.error(f"Error executing {action} for {symbol}: {trade_error}")
        
        # Look for new opportunities among watchlist symbols
        new_opportunities = []
        watchlist_symbols = polling_symbols[:10]  # Check top 10 symbols
        
        for symbol in watchlist_symbols:
            if symbol not in [pos['symbol'] for pos in position_analysis]:  # Not currently held
                try:
                    # Get prediction for potential new position
                    active_model = advanced_model if advanced_model and advanced_model.is_trained else model
                    
                    if active_model:
                        with get_session() as session:
                            recent_data = session.query(StockData).filter(
                                StockData.symbol == symbol
                            ).order_by(StockData.timestamp.desc()).limit(100).all()
                            
                            if len(recent_data) >= 50:
                                from .indicators import compute_technical_indicators, prepare_features
                                
                                df = pd.DataFrame([{
                                    'open': d.open, 'high': d.high, 'low': d.low,
                                    'close': d.close, 'volume': d.volume, 'timestamp': d.timestamp
                                } for d in reversed(recent_data)])
                                
                                indicators = compute_technical_indicators(df)
                                if not indicators.empty:
                                    prediction_prob = 0.5
                                    confidence = 0.0
                                    
                                    try:
                                        if hasattr(active_model, 'predict_proba') and hasattr(active_model, 'is_trained'):
                                            # Advanced model
                                            from .advanced_features import create_advanced_features
                                            enhanced_df = create_advanced_features(df, symbol)
                                            if not enhanced_df.empty:
                                                feature_cols = [col for col in enhanced_df.columns 
                                                              if col not in ['timestamp', 'target', 'future_return', 'symbol']]
                                                if feature_cols:
                                                    latest_features = enhanced_df[feature_cols].iloc[-1:].fillna(0)
                                                    prediction_prob = active_model.predict_proba(latest_features)[0][1]
                                                    confidence = abs(prediction_prob - 0.5) * 2
                                        else:
                                            # Basic model
                                            features, _ = prepare_features(indicators)
                                            if not features.empty:
                                                latest_features = features.iloc[-1:].values
                                                prediction_prob = active_model.predict_proba(latest_features)[0][1]
                                                confidence = abs(prediction_prob - 0.5) * 2
                                    
                                        # Consider new position if strong signal
                                        if prediction_prob > 0.70 and confidence > 0.4:
                                            # Sentiment gate for new opportunities [SENT-02]
                                            from .news import get_ticker_sentiment
                                            _sent = get_ticker_sentiment(symbol)
                                            if _sent < -0.05:
                                                logging.info(f"[Opportunity: {symbol}] Suppressed by sentiment: {_sent:.3f}")
                                            else:
                                                current_price = df.iloc[-1]['close']
                                                opportunity = {
                                                    'symbol': symbol,
                                                    'prediction_prob': prediction_prob,
                                                    'confidence': confidence,
                                                    'current_price': current_price,
                                                    'signal_strength': 'STRONG_BUY'
                                                }
                                                new_opportunities.append(opportunity)
                                            
                                    except Exception as pred_error:
                                        logging.warning(f"Error predicting for new opportunity {symbol}: {pred_error}")
                                        
                except Exception as opp_error:
                    logging.warning(f"Error analyzing opportunity {symbol}: {opp_error}")
        
        # Execute new opportunities (limit to top 2 to manage risk)
        for opportunity in sorted(new_opportunities, key=lambda x: x['prediction_prob'], reverse=True)[:2]:
            if available_funds > 2000:  # Minimum $2000 for new positions
                symbol = opportunity['symbol']
                try:
                    qty = submit_ml_sized_order(
                        symbol, 'buy', 
                        opportunity['prediction_prob'], 
                        opportunity['current_price']
                    )
                    if qty > 0:
                        logging.info(f"New position opened: {symbol} +{qty} shares (prob: {opportunity['prediction_prob']:.3f})")
                        available_funds -= qty * opportunity['current_price']
                        
                except Exception as new_trade_error:
                    logging.error(f"Error opening new position {symbol}: {new_trade_error}")
        
        # Log comprehensive portfolio performance
        portfolio_snapshot = {
            'timestamp': datetime.now(),
            'total_value': portfolio_value,
            'available_funds': available_funds,
            'position_count': len(positions),
            'active_positions': len(position_analysis),
            'positions_analyzed': position_analysis,
            'new_opportunities_found': len(new_opportunities),
            'scan_type': 'hourly_comprehensive'
        }
        
        log_portfolio_performance(portfolio_snapshot)
        
        logging.info(f"Hourly scan completed: {len(position_analysis)} positions analyzed, {len(new_opportunities)} new opportunities")
        
    except Exception as e:
        logging.error(f"Error in hourly portfolio scan: {e}")
        import traceback
        logging.error(traceback.format_exc())

def generate_portfolio_report():
    """
    Generate comprehensive portfolio performance report with analytics.
    """
    from .model_performance import performance_tracker
    
    try:
        logging.info("Generating portfolio performance report...")
        
        # Get current portfolio state
        account_summary = ib.accountSummary()
        positions = ib.positions()
        
        portfolio_value = 0
        available_funds = 0
        total_pnl = 0
        
        for item in account_summary:
            if item.tag == 'NetLiquidation':
                portfolio_value = float(item.value)
            elif item.tag == 'AvailableFunds':
                available_funds = float(item.value)
            elif item.tag == 'UnrealizedPnL':
                total_pnl = float(item.value)
        
        # Calculate performance metrics
        trading_metrics = performance_tracker.calculate_trading_metrics(lookback_days=30)
        prediction_metrics = performance_tracker.calculate_prediction_accuracy(lookback_days=30)
        
        # Position analysis
        position_summary = []
        total_position_value = 0
        
        for pos in positions:
            if pos.position != 0:
                position_value = abs(pos.position * pos.avgCost)
                total_position_value += position_value
                
                position_summary.append({
                    'symbol': pos.contract.symbol,
                    'quantity': pos.position,
                    'avg_cost': pos.avgCost,
                    'market_value': pos.marketValue,
                    'unrealized_pnl': pos.unrealizedPNL,
                    'position_return': (pos.marketValue - position_value) / position_value if position_value > 0 else 0
                })
        
        # Generate report
        report = f"""
=== PORTFOLIO PERFORMANCE REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PORTFOLIO OVERVIEW:
- Total Value: ${portfolio_value:,.2f}
- Available Funds: ${available_funds:,.2f}
- Cash %: {(available_funds / portfolio_value * 100) if portfolio_value > 0 else 0:.1f}%
- Total Positions: {len(position_summary)}
- Total Unrealized P&L: ${total_pnl:,.2f}

TRADING PERFORMANCE (30 days):
"""
        
        if trading_metrics:
            report += f"""- Total Trades: {trading_metrics.get('total_trades', 0)}
- Win Rate: {trading_metrics.get('win_rate', 0):.1%}
- Avg P&L per Trade: ${trading_metrics.get('avg_pnl_per_trade', 0):,.2f}
- Total P&L: ${trading_metrics.get('total_pnl', 0):,.2f}
- Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.3f}
- Max Drawdown: {trading_metrics.get('max_drawdown', 0):.1%}
- Profit Factor: {trading_metrics.get('profit_factor', 0):.2f}
"""
        else:
            report += "- No trading data available\n"
        
        report += "\nMODEL ACCURACY (30 days):\n"
        
        if prediction_metrics:
            report += f"""- Prediction Accuracy: {prediction_metrics.get('accuracy', 0):.1%}
- Direction Accuracy: {prediction_metrics.get('direction_accuracy', 0):.1%}
- ROC AUC Score: {prediction_metrics.get('roc_auc', 0):.3f}
- Correlation: {prediction_metrics.get('correlation', 0):.3f}
- High Confidence Accuracy: {prediction_metrics.get('high_confidence_accuracy', 0):.1%}
"""
        else:
            report += "- No prediction data available\n"
        
        report += "\nTOP POSITIONS:\n"
        
        # Sort positions by absolute market value
        sorted_positions = sorted(position_summary, key=lambda x: abs(x['market_value']), reverse=True)
        
        for i, pos in enumerate(sorted_positions[:10], 1):
            return_pct = pos['position_return'] * 100
            report += f"{i:2d}. {pos['symbol']:6s} | Qty: {pos['quantity']:8.0f} | "
            report += f"Value: ${pos['market_value']:10,.2f} | "
            report += f"P&L: ${pos['unrealized_pnl']:8,.2f} ({return_pct:+5.1f}%)\n"
        
        # Risk analysis
        report += "\nRISK ANALYSIS:\n"
        
        if position_summary:
            # Concentration risk
            largest_position = max(position_summary, key=lambda x: abs(x['market_value']))
            concentration = abs(largest_position['market_value']) / portfolio_value if portfolio_value > 0 else 0
            
            # Sector exposure (simplified)
            long_exposure = sum(pos['market_value'] for pos in position_summary if pos['quantity'] > 0)
            short_exposure = sum(abs(pos['market_value']) for pos in position_summary if pos['quantity'] < 0)
            net_exposure = (long_exposure - short_exposure) / portfolio_value if portfolio_value > 0 else 0
            
            report += f"- Largest Position: {largest_position['symbol']} ({concentration:.1%} of portfolio)\n"
            report += f"- Long Exposure: ${long_exposure:,.2f}\n"
            report += f"- Short Exposure: ${short_exposure:,.2f}\n"
            report += f"- Net Exposure: {net_exposure:+.1%}\n"
            
            # Diversification
            report += f"- Portfolio Diversification: {len(position_summary)} positions\n"
            
            if concentration > 0.3:
                report += "⚠️  WARNING: High concentration risk (>30% in single position)\n"
            
            if abs(net_exposure) > 0.8:
                report += "⚠️  WARNING: High directional exposure (>80%)\n"
        
        report += "\nRECOMMENDATIONS:\n"
        
        # Generate recommendations based on metrics
        recommendations = []
        
        if trading_metrics.get('win_rate', 0) < 0.4:
            recommendations.append("- Consider tightening entry criteria (win rate below 40%)")
        
        if trading_metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("- Review risk management (low Sharpe ratio)")
        
        if trading_metrics.get('max_drawdown', 0) > 0.2:
            recommendations.append("- Implement stronger stop-loss rules (high drawdown)")
        
        if prediction_metrics.get('accuracy', 0) < 0.55:
            recommendations.append("- Retrain models (prediction accuracy below 55%)")
        
        if concentration > 0.25:
            recommendations.append("- Reduce concentration risk (diversify positions)")
        
        if not recommendations:
            recommendations.append("- Portfolio performance within acceptable ranges")
        
        for rec in recommendations:
            report += rec + "\n"
        
        report += "\n" + "="*50 + "\n"
        
        logging.info("Portfolio report generated successfully")
        logging.info(report)
        
        return report
        
    except Exception as e:
        logging.error(f"Error generating portfolio report: {e}")
        return None

def setup_scheduler():
    scheduler = BackgroundScheduler()
    
    # Original model retraining job
    scheduler.add_job(scheduled_retrain, 'interval', minutes=RETRAIN_FREQUENCY)
    
    # New hourly portfolio scanning job
    scheduler.add_job(hourly_portfolio_scan, 'interval', hours=1)
    
    # Daily portfolio report generation (at 6 PM ET)
    scheduler.add_job(generate_portfolio_report, 'cron', hour=18, timezone=eastern)
    
    scheduler.start()
    logging.info("Scheduler started with model training, hourly portfolio scanning, and daily reporting.")
    atexit.register(lambda: scheduler.shutdown())


def trading_loop():
    """
    Main trading loop: fetch data -> run AI predictions -> place orders -> repeat
    Enhanced with portfolio tracking and performance monitoring
    """
    global model, polling_symbols
    
    if not polling_symbols:
        logging.warning("No symbols to trade. Skipping trading cycle.")
        return
    
    logging.info(f"Starting trading cycle for {len(polling_symbols)} symbols...")
    
    # Track portfolio performance
    try:
        account_summary = ib.accountSummary()
        portfolio_value = 0
        available_funds = 0
        
        for item in account_summary:
            if item.tag == 'NetLiquidation':
                portfolio_value = float(item.value)
            elif item.tag == 'AvailableFunds':
                available_funds = float(item.value)
        
        # Get current positions
        positions = ib.positions()
        position_count = len(positions)
        total_position_value = sum(abs(pos.position * pos.avgCost) for pos in positions)
        
        # Log portfolio snapshot
        log_portfolio_performance({
            'total_value': portfolio_value,
            'available_funds': available_funds,
            'position_count': position_count,
            'position_value': total_position_value,
            'cash_percentage': available_funds / portfolio_value if portfolio_value > 0 else 1.0
        })
        
        logging.info(f"Portfolio: ${portfolio_value:,.2f} total, ${available_funds:,.2f} available, {position_count} positions")
        
    except Exception as portfolio_error:
        logging.warning(f"Error tracking portfolio: {portfolio_error}")
    
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