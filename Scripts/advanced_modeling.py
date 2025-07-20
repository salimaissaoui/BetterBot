import logging
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Deep learning features disabled.")

# Reinforcement Learning
try:
    import gymnasium as gym
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    logging.warning("Reinforcement learning libraries not available. RL features disabled.")

from .database import get_session, StockData
from .indicators import compute_technical_indicators, prepare_features
from .config import TARGET_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class AdvancedTradingEnvironment:
    """
    Custom trading environment for reinforcement learning that simulates 
    real market conditions and provides reward feedback for learning.
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.trades_made = 0
        self.successful_trades = 0
        self.max_drawdown = 0
        self.peak_net_worth = self.initial_balance
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current state observation"""
        if self.current_step >= len(self.data):
            return np.zeros(30)  # Return zeros if out of bounds
            
        current_data = self.data.iloc[self.current_step]
        
        # Market features (20 technical indicators)
        market_features = [
            current_data.get('close', 0),
            current_data.get('rsi', 50),
            current_data.get('macd', 0),
            current_data.get('bb_p', 0.5),
            current_data.get('atr', 0),
            current_data.get('stoch', 50),
            current_data.get('obv', 0),
            current_data.get('ma_ratio', 1),
            current_data.get('return', 0),
            current_data.get('volume', 0) / 1000000,  # Normalize volume
            current_data.get('macd_diff', 0),
            current_data.get('bb_w', 0),
            current_data.get('rolling_ret_std', 0),
            current_data.get('rolling_vol_avg', 0) / 1000000,
            current_data.get('ma_short', 0),
            current_data.get('ma_long', 0),
            current_data.get('bb_h', 0),
            current_data.get('bb_l', 0),
            current_data.get('obv_ma', 0),
            current_data.get('stoch_signal', 50)
        ]
        
        # Portfolio features (10 features)
        portfolio_features = [
            self.balance / self.initial_balance,  # Normalized balance
            self.shares_held,
            self.net_worth / self.initial_balance,  # Normalized net worth
            (self.net_worth - self.previous_net_worth) / self.initial_balance,  # Return
            self.trades_made / max(1, self.current_step),  # Trade frequency
            self.successful_trades / max(1, self.trades_made),  # Win rate
            self.max_drawdown / self.initial_balance,  # Normalized drawdown
            min(1.0, self.current_step / self.max_steps),  # Progress through episode
            1 if self.shares_held > 0 else 0,  # Long position indicator
            1 if self.shares_held < 0 else 0   # Short position indicator
        ]
        
        return np.array(market_features + portfolio_features, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return new state, reward, done, info"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, {}
            
        current_price = self.data.iloc[self.current_step]['close']
        self.previous_net_worth = self.net_worth
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        reward = 0
        info = {'action': action, 'price': current_price}
        
        if action == 1:  # Buy
            if self.balance > current_price:
                shares_to_buy = int(self.balance * 0.1 / current_price)  # Use 10% of balance
                cost = shares_to_buy * current_price
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares_held += shares_to_buy
                    self.trades_made += 1
                    info['shares_bought'] = shares_to_buy
                    
        elif action == 2:  # Sell
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price
                self.balance += proceeds
                self.shares_held = 0
                self.trades_made += 1
                info['shares_sold'] = self.shares_held
        
        # Calculate portfolio value
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Calculate reward
        portfolio_return = (self.net_worth - self.previous_net_worth) / self.previous_net_worth
        reward = portfolio_return * 100  # Scale reward
        
        # Penalize excessive trading
        if self.trades_made > 0:
            trade_penalty = -0.001 * (self.trades_made / max(1, self.current_step))
            reward += trade_penalty
            
        # Track drawdown
        if self.net_worth > self.peak_net_worth:
            self.peak_net_worth = self.net_worth
        current_drawdown = (self.peak_net_worth - self.net_worth) / self.peak_net_worth
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
            
        # Penalize large drawdowns
        if current_drawdown > 0.1:  # 10% drawdown
            reward -= current_drawdown * 10
            
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, done, info


class DeepLearningModel:
    """Advanced neural network for trading predictions"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64]):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep learning features")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_size, hidden_sizes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        self.scaler = StandardScaler()
        
    def _build_model(self, input_size: int, hidden_sizes: List[int]):
        """Build neural network architecture"""
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers).to(self.device)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 64):
        """Train the neural network"""
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            if epoch % 20 == 0:
                logging.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()
            
        # Return probabilities for both classes
        return np.column_stack([1 - probabilities, probabilities])


class AdvancedEnsembleModel:
    """
    Advanced ensemble model with multiple algorithms, feature selection,
    and adaptive learning capabilities.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_history = {}
        self.model_weights = {}
        self.is_trained = False
        
        # Initialize models
        self._init_models()
        
    def _init_models(self):
        """Initialize all models in the ensemble"""
        
        # Traditional ML models
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
            self.model_weights[model_name] = 1.0 / len(self.models)
            self.performance_history[model_name] = []
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2):
        """Train all models with cross-validation and performance tracking"""
        logging.info("Training advanced ensemble model...")
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, model in self.models.items():
            try:
                logging.info(f"Training {model_name}...")
                
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_val_scaled = self.scalers[model_name].transform(X_val)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=tscv, scoring='roc_auc', n_jobs=-1
                )
                
                # Train on full training set
                model.fit(X_train_scaled, y_train)
                
                # Validate
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                val_auc = roc_auc_score(y_val, y_pred_proba)
                
                # Store performance
                performance = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'val_auc': val_auc,
                    'timestamp': datetime.now()
                }
                self.performance_history[model_name].append(performance)
                
                # Update model weight based on performance
                self.model_weights[model_name] = val_auc
                
                logging.info(f"{model_name} - CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, Val AUC: {val_auc:.4f}")
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = model.feature_importances_
                    
            except Exception as e:
                logging.error(f"Error training {model_name}: {e}")
                self.model_weights[model_name] = 0.0
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model_name in self.model_weights:
                self.model_weights[model_name] /= total_weight
        
        self.is_trained = True
        logging.info("Ensemble training completed")
        
        # Add deep learning model if available
        if TORCH_AVAILABLE and len(X_train) > 1000:
            try:
                logging.info("Training deep learning model...")
                self.models['deep'] = DeepLearningModel(X_train.shape[1])
                self.models['deep'].train(X_train.values, y_train.values)
                self.model_weights['deep'] = 0.2  # Give it some weight
                
                # Renormalize weights
                total_weight = sum(self.model_weights.values())
                for model_name in self.model_weights:
                    self.model_weights[model_name] /= total_weight
                    
            except Exception as e:
                logging.error(f"Error training deep learning model: {e}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions with weighted voting"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
            
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            if self.model_weights[model_name] > 0:
                try:
                    if model_name == 'deep':
                        pred_proba = model.predict_proba(X.values)
                    else:
                        X_scaled = self.scalers[model_name].transform(X)
                        pred_proba = model.predict_proba(X_scaled)
                    
                    predictions.append(pred_proba[:, 1])
                    weights.append(self.model_weights[model_name])
                    
                except Exception as e:
                    logging.warning(f"Error in prediction from {model_name}: {e}")
        
        if not predictions:
            # Fallback to equal prediction
            return np.full((len(X), 2), [0.5, 0.5])
        
        # Weighted ensemble prediction
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        # Return as probability matrix
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance across all models"""
        if not self.feature_importance:
            return {}
            
        # Average importance across models
        avg_importance = {}
        for model_name, importance in self.feature_importance.items():
            weight = self.model_weights.get(model_name, 0)
            if weight > 0:
                for i, imp in enumerate(importance):
                    if i not in avg_importance:
                        avg_importance[i] = 0
                    avg_importance[i] += imp * weight
        
        return avg_importance
    
    def save_model(self, filepath: str):
        """Save the entire ensemble model"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'model_weights': self.model_weights,
            'performance_history': self.performance_history,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logging.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the entire ensemble model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.model_weights = model_data['model_weights']
            self.performance_history = model_data['performance_history']
            self.feature_importance = model_data['feature_importance']
            self.is_trained = model_data['is_trained']
            
            logging.info(f"Ensemble model loaded from {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False


class SimulationBasedTraining:
    """
    Simulation-based training system that runs backtests and learns from results
    """
    
    def __init__(self, model: AdvancedEnsembleModel):
        self.model = model
        self.simulation_results = []
        self.learning_rate = 0.01
        
    def run_simulation(self, symbol: str, start_date: str, end_date: str, 
                      initial_capital: float = 100000) -> Dict[str, Any]:
        """Run a trading simulation and return performance metrics"""
        
        try:
            # Get historical data for simulation
            with get_session() as session:
                data = session.query(StockData).filter(
                    StockData.symbol == symbol,
                    StockData.timestamp >= start_date,
                    StockData.timestamp <= end_date
                ).order_by(StockData.timestamp.asc()).all()
            
            if len(data) < 100:
                logging.warning(f"Insufficient data for {symbol} simulation")
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume,
                'timestamp': d.timestamp
            } for d in data])
            
            # Compute indicators
            indicators_df = compute_technical_indicators(df)
            if indicators_df.empty:
                return {}
            
            # Prepare features
            features, targets = prepare_features(indicators_df)
            if features.empty:
                return {}
            
            # Run simulation
            capital = initial_capital
            shares = 0
            trades = []
            equity_curve = []
            
            for i in range(len(features)):
                current_price = df.iloc[i]['close']
                
                # Make prediction
                if self.model.is_trained and i > 0:
                    pred_proba = self.model.predict_proba(features.iloc[i:i+1])
                    signal_strength = pred_proba[0][1]  # Probability of positive return
                else:
                    signal_strength = 0.5  # Neutral if no model
                
                # Trading logic
                action = None
                if signal_strength > 0.6 and shares == 0:  # Buy signal
                    shares_to_buy = int(capital * 0.1 / current_price)  # Use 10% of capital
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        capital -= cost
                        shares += shares_to_buy
                        action = 'BUY'
                        trades.append({
                            'timestamp': df.iloc[i]['timestamp'],
                            'action': action,
                            'price': current_price,
                            'shares': shares_to_buy,
                            'signal_strength': signal_strength
                        })
                        
                elif signal_strength < 0.4 and shares > 0:  # Sell signal
                    proceeds = shares * current_price
                    capital += proceeds
                    shares = 0
                    action = 'SELL'
                    trades.append({
                        'timestamp': df.iloc[i]['timestamp'],
                        'action': action,
                        'price': current_price,
                        'shares': shares,
                        'signal_strength': signal_strength
                    })
                
                # Track equity
                portfolio_value = capital + (shares * current_price)
                equity_curve.append({
                    'timestamp': df.iloc[i]['timestamp'],
                    'portfolio_value': portfolio_value,
                    'capital': capital,
                    'shares': shares,
                    'price': current_price
                })
            
            # Calculate performance metrics
            final_value = capital + (shares * df.iloc[-1]['close'])
            total_return = (final_value - initial_capital) / initial_capital
            
            # Calculate Sharpe ratio
            if len(equity_curve) > 1:
                returns = []
                for i in range(1, len(equity_curve)):
                    prev_val = equity_curve[i-1]['portfolio_value']
                    curr_val = equity_curve[i]['portfolio_value']
                    returns.append((curr_val - prev_val) / prev_val)
                
                if returns:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown
            peak = initial_capital
            max_drawdown = 0
            for point in equity_curve:
                if point['portfolio_value'] > peak:
                    peak = point['portfolio_value']
                drawdown = (peak - point['portfolio_value']) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            simulation_result = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(trades),
                'trades': trades,
                'equity_curve': equity_curve,
                'timestamp': datetime.now()
            }
            
            return simulation_result
            
        except Exception as e:
            logging.error(f"Error in simulation for {symbol}: {e}")
            return {}
    
    def learn_from_simulations(self, symbols: List[str], lookback_days: int = 90):
        """Run simulations and learn from the results"""
        
        logging.info(f"Running simulations for {len(symbols)} symbols...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        simulation_results = []
        
        for symbol in symbols[:10]:  # Limit to 10 symbols for speed
            result = self.run_simulation(
                symbol, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if result:
                simulation_results.append(result)
                logging.info(f"{symbol}: Return: {result['total_return']:.2%}, "
                           f"Sharpe: {result['sharpe_ratio']:.2f}, "
                           f"Trades: {result['num_trades']}")
        
        # Analyze results and update model
        if simulation_results:
            avg_return = np.mean([r['total_return'] for r in simulation_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in simulation_results])
            
            logging.info(f"Simulation Summary - Avg Return: {avg_return:.2%}, "
                        f"Avg Sharpe: {avg_sharpe:.2f}")
            
            # Store results for learning
            self.simulation_results.extend(simulation_results)
            
            # Trigger model retraining if performance is below threshold
            if avg_return < 0.05 or avg_sharpe < 0.5:  # 5% return or 0.5 Sharpe threshold
                logging.info("Performance below threshold. Triggering model retraining...")
                return True  # Signal that retraining is needed
        
        return False


class ReinforcementLearningAgent:
    """
    Reinforcement Learning agent for trading decisions
    """
    
    def __init__(self):
        if not RL_AVAILABLE:
            logging.warning("RL libraries not available. RL agent disabled.")
            self.enabled = False
            return
            
        self.enabled = True
        self.model = None
        self.env = None
        
    def create_environment(self, data: pd.DataFrame):
        """Create trading environment from data"""
        if not self.enabled:
            return None
            
        self.env = AdvancedTradingEnvironment(data)
        return self.env
    
    def train_agent(self, data: pd.DataFrame, total_timesteps: int = 100000):
        """Train the RL agent"""
        if not self.enabled:
            logging.warning("RL agent not enabled")
            return False
            
        try:
            # Create environment
            env = self.create_environment(data)
            if env is None:
                return False
            
            # Create PPO agent
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
            
            logging.info(f"Training RL agent for {total_timesteps} timesteps...")
            self.model.learn(total_timesteps=total_timesteps)
            
            # Save model
            self.model.save("rl_trading_agent")
            logging.info("RL agent trained and saved successfully")
            
            return True
            
        except Exception as e:
            logging.error(f"Error training RL agent: {e}")
            return False
    
    def predict_action(self, observation: np.ndarray) -> int:
        """Predict action given observation"""
        if not self.enabled or self.model is None:
            return 0  # Default to hold
            
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return action
        except Exception as e:
            logging.error(f"Error in RL prediction: {e}")
            return 0
    
    def load_agent(self, filepath: str = "rl_trading_agent"):
        """Load trained agent"""
        if not self.enabled:
            return False
            
        try:
            self.model = PPO.load(filepath)
            logging.info("RL agent loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading RL agent: {e}")
            return False


# Global instances
advanced_model = AdvancedEnsembleModel()
simulation_trainer = SimulationBasedTraining(advanced_model)
rl_agent = ReinforcementLearningAgent()

def get_advanced_model():
    """Get the global advanced model instance"""
    return advanced_model

def run_advanced_training():
    """Run complete advanced training pipeline"""
    try:
        logging.info("Starting advanced training pipeline...")
        
        # Get training data from database
        with get_session() as session:
            # Get recent data for multiple symbols
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Sample symbols
            
            all_data = []
            for symbol in symbols:
                data = session.query(StockData).filter(
                    StockData.symbol == symbol
                ).order_by(StockData.timestamp.desc()).limit(1000).all()
                
                if len(data) > 100:
                    df = pd.DataFrame([{
                        'open': d.open,
                        'high': d.high,
                        'low': d.low,
                        'close': d.close,
                        'volume': d.volume,
                        'timestamp': d.timestamp,
                        'symbol': symbol
                    } for d in reversed(data)])
                    
                    # Compute indicators
                    indicators_df = compute_technical_indicators(df)
                    if not indicators_df.empty:
                        all_data.append(indicators_df)
            
            if not all_data:
                logging.error("No data available for training")
                return False
            
            # Combine all data
            combined_data = pd.concat(all_data, ignore_index=True)
            
            # Prepare features
            features, targets = prepare_features(combined_data)
            if features.empty or targets.empty:
                logging.error("No features prepared for training")
                return False
            
            # Remove any remaining NaN values
            valid_indices = ~(features.isna().any(axis=1) | targets.isna())
            features = features[valid_indices]
            targets = targets[valid_indices]
            
            if len(features) < 100:
                logging.error("Insufficient clean data for training")
                return False
            
            logging.info(f"Training with {len(features)} samples")
            
            # Train ensemble model
            advanced_model.train(features, targets)
            
            # Save model
            advanced_model.save_model("advanced_ensemble_model.pkl")
            
            # Run simulations and learn
            need_retrain = simulation_trainer.learn_from_simulations(symbols)
            
            # Train RL agent if enough data
            if len(combined_data) > 500 and RL_AVAILABLE:
                rl_agent.train_agent(combined_data, total_timesteps=50000)
            
            logging.info("Advanced training pipeline completed successfully")
            return True
            
    except Exception as e:
        logging.error(f"Error in advanced training: {e}")
        logging.error(traceback.format_exc())
        return False