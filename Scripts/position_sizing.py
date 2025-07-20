import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
import traceback

from .database import get_session, StockData

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class MLPositionSizer:
    """
    Machine Learning based position sizing system that considers:
    - Current capital/portfolio value
    - Market volatility
    - Symbol-specific risk metrics
    - Historical performance
    - Market conditions
    - Prediction confidence
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'account_value', 'available_capital', 'prediction_confidence',
            'symbol_volatility', 'symbol_volume_avg', 'symbol_price',
            'portfolio_concentration', 'market_sentiment', 'position_correlation',
            'risk_budget', 'time_of_day', 'day_of_week'
        ]
        
    def get_account_info(self, ib):
        """Get current account value and available capital"""
        try:
            account_summary = ib.accountSummary()
            
            account_value = 0
            available_funds = 0
            
            for item in account_summary:
                if item.tag == 'NetLiquidation':
                    account_value = float(item.value)
                elif item.tag == 'AvailableFunds':
                    available_funds = float(item.value)
                    
            return account_value, available_funds
            
        except Exception as e:
            logging.warning(f"Error getting account info: {e}")
            # Fallback values
            return 100000.0, 50000.0
    
    def calculate_symbol_metrics(self, symbol):
        """Calculate symbol-specific risk metrics from historical data"""
        try:
            with get_session() as session:
                # Get last 30 days of data
                cutoff_date = datetime.now() - timedelta(days=30)
                
                recent_data = session.query(StockData).filter(
                    StockData.symbol == symbol,
                    StockData.timestamp >= cutoff_date
                ).order_by(StockData.timestamp.asc()).all()
                
                if len(recent_data) < 10:
                    # Default values for insufficient data
                    return {
                        'volatility': 0.02,  # 2% default volatility
                        'avg_volume': 100000,
                        'avg_price': 100.0,
                        'price_trend': 0.0
                    }
                
                # Convert to DataFrame for analysis
                df = pd.DataFrame([{
                    'close': r.close,
                    'volume': r.volume,
                    'timestamp': r.timestamp
                } for r in recent_data])
                
                # Calculate metrics
                df['return'] = df['close'].pct_change()
                volatility = df['return'].std() * np.sqrt(252)  # Annualized volatility
                avg_volume = df['volume'].mean()
                avg_price = df['close'].mean()
                price_trend = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                
                return {
                    'volatility': max(0.01, volatility),  # Minimum 1% volatility
                    'avg_volume': max(1000, avg_volume),
                    'avg_price': max(1.0, avg_price),
                    'price_trend': price_trend
                }
                
        except Exception as e:
            logging.warning(f"Error calculating symbol metrics for {symbol}: {e}")
            return {
                'volatility': 0.02,
                'avg_volume': 100000,
                'avg_price': 100.0,
                'price_trend': 0.0
            }
    
    def get_portfolio_metrics(self, ib, symbol):
        """Calculate current portfolio concentration and correlation"""
        try:
            positions = ib.positions()
            
            if not positions:
                return 0.0, 0.0  # No concentration, no correlation
            
            # Calculate portfolio concentration
            total_value = sum(abs(pos.position * pos.avgCost) for pos in positions)
            symbol_positions = [pos for pos in positions if pos.contract.symbol == symbol]
            
            if symbol_positions:
                symbol_value = sum(abs(pos.position * pos.avgCost) for pos in symbol_positions)
                concentration = symbol_value / total_value if total_value > 0 else 0.0
            else:
                concentration = 0.0
            
            # Simple correlation proxy (could be enhanced)
            correlation_risk = min(1.0, len(positions) / 10.0)  # More positions = more correlation risk
            
            return concentration, correlation_risk
            
        except Exception as e:
            logging.warning(f"Error calculating portfolio metrics: {e}")
            return 0.0, 0.0
    
    def prepare_features(self, ib, symbol, prediction_confidence, current_price=None):
        """Prepare feature vector for position sizing prediction"""
        try:
            # Get account information
            account_value, available_capital = self.get_account_info(ib)
            
            # Get symbol metrics
            symbol_metrics = self.calculate_symbol_metrics(symbol)
            
            # Get portfolio metrics
            concentration, correlation = self.get_portfolio_metrics(ib, symbol)
            
            # Calculate risk budget (Kelly criterion inspired)
            risk_budget = min(0.25, available_capital / account_value) if account_value > 0 else 0.1
            
            # Market timing features
            now = datetime.now()
            time_of_day = now.hour + now.minute / 60.0  # Hour of day as float
            day_of_week = now.weekday()  # 0=Monday, 6=Sunday
            
            # Market sentiment proxy (could be enhanced with sentiment data)
            market_sentiment = 0.5  # Neutral default
            
            # Use current price or fallback to historical average
            symbol_price = current_price if current_price else symbol_metrics['avg_price']
            
            features = {
                'account_value': account_value,
                'available_capital': available_capital,
                'prediction_confidence': prediction_confidence,
                'symbol_volatility': symbol_metrics['volatility'],
                'symbol_volume_avg': symbol_metrics['avg_volume'],
                'symbol_price': symbol_price,
                'portfolio_concentration': concentration,
                'market_sentiment': market_sentiment,
                'position_correlation': correlation,
                'risk_budget': risk_budget,
                'time_of_day': time_of_day,
                'day_of_week': day_of_week
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error preparing features: {e}")
            return None
    
    def train_model(self):
        """Train the position sizing model using historical data and rules"""
        try:
            logging.info("Training ML position sizing model...")
            
            # Generate synthetic training data based on financial principles
            n_samples = 1000
            
            # Generate realistic feature combinations
            np.random.seed(42)
            
            account_values = np.random.lognormal(11, 0.5, n_samples)  # ~$100k average
            available_capitals = account_values * np.random.uniform(0.1, 0.8, n_samples)
            prediction_confidences = np.random.uniform(0.4, 0.9, n_samples)
            volatilities = np.random.gamma(2, 0.01, n_samples)  # Realistic volatility distribution
            volumes = np.random.lognormal(11, 1, n_samples)  # Volume distribution
            prices = np.random.lognormal(4, 0.8, n_samples)  # Price distribution
            concentrations = np.random.beta(1, 3, n_samples)  # Low concentration preferred
            market_sentiments = np.random.normal(0.5, 0.15, n_samples)
            correlations = np.random.uniform(0, 1, n_samples)
            risk_budgets = np.random.uniform(0.05, 0.25, n_samples)
            times_of_day = np.random.uniform(9.5, 16, n_samples)  # Market hours
            days_of_week = np.random.randint(0, 5, n_samples)  # Weekdays only
            
            # Create feature matrix
            X = np.column_stack([
                account_values, available_capitals, prediction_confidences,
                volatilities, volumes, prices, concentrations,
                market_sentiments, correlations, risk_budgets,
                times_of_day, days_of_week
            ])
            
            # Generate targets using financial rules (Kelly criterion inspired)
            y = []
            for i in range(n_samples):
                # Base position size on Kelly-like formula
                confidence = prediction_confidences[i]
                volatility = volatilities[i]
                available = available_capitals[i]
                price = prices[i]
                concentration = concentrations[i]
                risk_budget = risk_budgets[i]
                
                # Kelly fraction adjusted for confidence and risk
                kelly_fraction = (confidence - 0.5) / volatility if volatility > 0 else 0
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
                
                # Adjust for portfolio concentration
                concentration_adjustment = 1.0 - concentration
                
                # Calculate dollar amount to invest
                dollar_amount = available * kelly_fraction * concentration_adjustment * risk_budget
                
                # Convert to share quantity
                shares = max(1, int(dollar_amount / price)) if dollar_amount > price else 0
                
                # Cap at reasonable maximum (1000 shares)
                shares = min(1000, shares)
                
                y.append(shares)
            
            y = np.array(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_scaled, y)
            
            # Save model and scaler
            joblib.dump(self.model, "position_sizing_model.joblib")
            joblib.dump(self.scaler, "position_sizing_scaler.joblib")
            
            self.is_trained = True
            
            # Log feature importance
            importance = self.model.feature_importances_
            for i, feature in enumerate(self.feature_columns):
                logging.info(f"Feature importance - {feature}: {importance[i]:.3f}")
            
            logging.info("Position sizing model trained successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error training position sizing model: {e}")
            logging.error(traceback.format_exc())
            return False
    
    def load_model(self):
        """Load existing trained model"""
        try:
            self.model = joblib.load("position_sizing_model.joblib")
            self.scaler = joblib.load("position_sizing_scaler.joblib")
            self.is_trained = True
            logging.info("Position sizing model loaded successfully")
            return True
        except:
            logging.info("No existing position sizing model found")
            return False
    
    def calculate_position_size(self, ib, symbol, prediction_confidence, current_price=None):
        """Calculate optimal position size using ML model"""
        try:
            # Ensure model is trained
            if not self.is_trained:
                if not self.load_model():
                    if not self.train_model():
                        # Fallback to simple calculation
                        return self._fallback_position_size(ib, symbol, prediction_confidence)
            
            # Prepare features
            features = self.prepare_features(ib, symbol, prediction_confidence, current_price)
            if features is None:
                return self._fallback_position_size(ib, symbol, prediction_confidence)
            
            # Convert to array in correct order
            feature_array = np.array([[features[col] for col in self.feature_columns]])
            
            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Predict position size
            predicted_shares = self.model.predict(feature_array_scaled)[0]
            
            # Apply safety constraints
            predicted_shares = max(1, min(1000, int(predicted_shares)))
            
            logging.info(f"ML position sizing for {symbol}: {predicted_shares} shares "
                        f"(confidence: {prediction_confidence:.3f})")
            
            return predicted_shares
            
        except Exception as e:
            logging.error(f"Error in ML position sizing: {e}")
            return self._fallback_position_size(ib, symbol, prediction_confidence)
    
    def _fallback_position_size(self, ib, symbol, prediction_confidence):
        """Simple fallback position sizing when ML model fails"""
        try:
            account_value, available_capital = self.get_account_info(ib)
            
            # Simple rule: use 1-5% of available capital based on confidence
            confidence_factor = (prediction_confidence - 0.5) * 2  # Scale to 0-1
            confidence_factor = max(0.1, min(1.0, confidence_factor))
            
            allocation_pct = 0.01 + (0.04 * confidence_factor)  # 1-5%
            dollar_amount = available_capital * allocation_pct
            
            # Assume $100 average price for share calculation
            shares = max(1, int(dollar_amount / 100))
            shares = min(100, shares)  # Cap at 100 shares for safety
            
            logging.info(f"Fallback position sizing for {symbol}: {shares} shares")
            return shares
            
        except Exception as e:
            logging.error(f"Error in fallback position sizing: {e}")
            return 1  # Minimum 1 share

# Global instance
ml_position_sizer = MLPositionSizer()