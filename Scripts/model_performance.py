import logging
import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, mean_squared_error, mean_absolute_error
)

from .database import get_session, StockData
from .config import TARGET_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class TradingPerformanceMetrics:
    """
    Comprehensive trading performance tracking and analysis
    """
    
    def __init__(self):
        self.trade_history = []
        self.portfolio_history = []
        self.model_predictions = []
        self.performance_cache = {}
        
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a completed trade"""
        trade_data['timestamp'] = datetime.now()
        self.trade_history.append(trade_data)
        
    def log_portfolio_snapshot(self, portfolio_data: Dict[str, Any]):
        """Log portfolio state"""
        portfolio_data['timestamp'] = datetime.now()
        self.portfolio_history.append(portfolio_data)
        
    def log_prediction(self, symbol: str, prediction: float, actual_return: float = None, 
                      confidence: float = None, features: Dict = None):
        """Log model prediction for later evaluation"""
        prediction_data = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'prediction': prediction,
            'actual_return': actual_return,
            'confidence': confidence,
            'features': features
        }
        self.model_predictions.append(prediction_data)
        
    def calculate_trading_metrics(self, lookback_days: int = 30) -> Dict[str, float]:
        """Calculate comprehensive trading performance metrics"""
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_trades = [t for t in self.trade_history if t['timestamp'] >= cutoff_date]
        recent_portfolio = [p for p in self.portfolio_history if p['timestamp'] >= cutoff_date]
        
        if not recent_trades or not recent_portfolio:
            return {}
        
        metrics = {}
        
        # Basic trade statistics
        metrics['total_trades'] = len(recent_trades)
        metrics['winning_trades'] = len([t for t in recent_trades if t.get('pnl', 0) > 0])
        metrics['losing_trades'] = len([t for t in recent_trades if t.get('pnl', 0) < 0])
        metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
        
        # PnL analysis
        pnls = [t.get('pnl', 0) for t in recent_trades]
        if pnls:
            metrics['total_pnl'] = sum(pnls)
            metrics['avg_pnl_per_trade'] = np.mean(pnls)
            metrics['median_pnl_per_trade'] = np.median(pnls)
            metrics['std_pnl'] = np.std(pnls)
            
            winning_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p < 0]
            
            if winning_pnls:
                metrics['avg_winning_trade'] = np.mean(winning_pnls)
                metrics['largest_win'] = max(winning_pnls)
            else:
                metrics['avg_winning_trade'] = 0
                metrics['largest_win'] = 0
                
            if losing_pnls:
                metrics['avg_losing_trade'] = np.mean(losing_pnls)
                metrics['largest_loss'] = min(losing_pnls)
            else:
                metrics['avg_losing_trade'] = 0
                metrics['largest_loss'] = 0
            
            # Profit factor
            total_wins = sum(winning_pnls) if winning_pnls else 0
            total_losses = abs(sum(losing_pnls)) if losing_pnls else 0
            metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Portfolio performance
        if len(recent_portfolio) > 1:
            initial_value = recent_portfolio[0]['total_value']
            final_value = recent_portfolio[-1]['total_value']
            
            metrics['total_return'] = (final_value - initial_value) / initial_value
            
            # Calculate daily returns for Sharpe ratio
            daily_values = {}
            for snapshot in recent_portfolio:
                date = snapshot['timestamp'].date()
                daily_values[date] = snapshot['total_value']
            
            daily_returns = []
            sorted_dates = sorted(daily_values.keys())
            for i in range(1, len(sorted_dates)):
                prev_value = daily_values[sorted_dates[i-1]]
                curr_value = daily_values[sorted_dates[i]]
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)
            
            if daily_returns:
                metrics['daily_return_mean'] = np.mean(daily_returns)
                metrics['daily_return_std'] = np.std(daily_returns)
                metrics['sharpe_ratio'] = (metrics['daily_return_mean'] / metrics['daily_return_std'] * 
                                         np.sqrt(252)) if metrics['daily_return_std'] > 0 else 0
                
                # Maximum drawdown
                peak_value = initial_value
                max_drawdown = 0
                for snapshot in recent_portfolio:
                    current_value = snapshot['total_value']
                    if current_value > peak_value:
                        peak_value = current_value
                    drawdown = (peak_value - current_value) / peak_value
                    max_drawdown = max(max_drawdown, drawdown)
                
                metrics['max_drawdown'] = max_drawdown
        
        # Trading frequency and efficiency
        trading_days = (datetime.now() - cutoff_date).days
        metrics['trades_per_day'] = metrics['total_trades'] / trading_days if trading_days > 0 else 0
        
        # Hold time analysis
        hold_times = []
        for trade in recent_trades:
            if 'entry_time' in trade and 'exit_time' in trade:
                hold_time = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # hours
                hold_times.append(hold_time)
        
        if hold_times:
            metrics['avg_hold_time_hours'] = np.mean(hold_times)
            metrics['median_hold_time_hours'] = np.median(hold_times)
        
        return metrics
    
    def calculate_prediction_accuracy(self, lookback_days: int = 30) -> Dict[str, float]:
        """Calculate model prediction accuracy metrics"""
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_predictions = [p for p in self.model_predictions 
                            if p['timestamp'] >= cutoff_date and p['actual_return'] is not None]
        
        if not recent_predictions:
            return {}
        
        predictions = [p['prediction'] for p in recent_predictions]
        actuals = [p['actual_return'] for p in recent_predictions]
        
        # Convert to binary classification (positive return vs negative)
        binary_predictions = [1 if p > 0.5 else 0 for p in predictions]
        binary_actuals = [1 if a > TARGET_THRESHOLD else 0 for a in actuals]
        
        metrics = {}
        
        # Classification metrics
        if len(set(binary_actuals)) > 1:  # Need both classes for these metrics
            metrics['accuracy'] = accuracy_score(binary_actuals, binary_predictions)
            metrics['precision'] = precision_score(binary_actuals, binary_predictions, zero_division=0)
            metrics['recall'] = recall_score(binary_actuals, binary_predictions, zero_division=0)
            metrics['f1_score'] = f1_score(binary_actuals, binary_predictions, zero_division=0)
            
            try:
                metrics['roc_auc'] = roc_auc_score(binary_actuals, predictions)
            except ValueError:
                metrics['roc_auc'] = 0.5
        
        # Regression metrics
        metrics['mse'] = mean_squared_error(actuals, predictions)
        metrics['mae'] = mean_absolute_error(actuals, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Correlation
        metrics['correlation'] = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
        
        # Direction accuracy
        pred_directions = [1 if p > 0 else -1 for p in predictions]
        actual_directions = [1 if a > 0 else -1 for a in actuals]
        metrics['direction_accuracy'] = accuracy_score(actual_directions, pred_directions)
        
        # Confidence calibration
        if any(p.get('confidence') for p in recent_predictions):
            high_conf_predictions = [p for p in recent_predictions if p.get('confidence', 0) > 0.8]
            if high_conf_predictions:
                high_conf_accuracy = accuracy_score(
                    [1 if p['actual_return'] > TARGET_THRESHOLD else 0 for p in high_conf_predictions],
                    [1 if p['prediction'] > 0.5 else 0 for p in high_conf_predictions]
                )
                metrics['high_confidence_accuracy'] = high_conf_accuracy
        
        return metrics


class AdaptiveLearningSystem:
    """
    System that continuously adapts model parameters based on performance
    """
    
    def __init__(self):
        self.performance_tracker = TradingPerformanceMetrics()
        self.adaptation_history = []
        self.performance_thresholds = {
            'min_accuracy': 0.55,
            'min_sharpe': 0.5,
            'max_drawdown': 0.15,
            'min_win_rate': 0.45
        }
        self.adaptation_triggers = {
            'performance_check_interval': 24,  # hours
            'min_trades_for_evaluation': 10,
            'consecutive_losses_trigger': 5,
            'drawdown_trigger': 0.10
        }
        
    def check_adaptation_triggers(self) -> List[str]:
        """Check if any adaptation triggers have been activated"""
        
        triggers = []
        
        # Performance-based triggers
        trading_metrics = self.performance_tracker.calculate_trading_metrics(lookback_days=7)
        prediction_metrics = self.performance_tracker.calculate_prediction_accuracy(lookback_days=7)
        
        if trading_metrics:
            # Win rate trigger
            if trading_metrics.get('win_rate', 0) < self.performance_thresholds['min_win_rate']:
                triggers.append('low_win_rate')
            
            # Drawdown trigger
            if trading_metrics.get('max_drawdown', 0) > self.performance_thresholds['max_drawdown']:
                triggers.append('high_drawdown')
            
            # Sharpe ratio trigger
            if trading_metrics.get('sharpe_ratio', 0) < self.performance_thresholds['min_sharpe']:
                triggers.append('low_sharpe')
            
            # Consecutive losses
            recent_trades = self.performance_tracker.trade_history[-10:]
            consecutive_losses = 0
            for trade in reversed(recent_trades):
                if trade.get('pnl', 0) <= 0:
                    consecutive_losses += 1
                else:
                    break
            
            if consecutive_losses >= self.adaptation_triggers['consecutive_losses_trigger']:
                triggers.append('consecutive_losses')
        
        if prediction_metrics:
            # Accuracy trigger
            if prediction_metrics.get('accuracy', 0) < self.performance_thresholds['min_accuracy']:
                triggers.append('low_accuracy')
        
        # Time-based trigger
        last_adaptation = self.adaptation_history[-1] if self.adaptation_history else None
        if (last_adaptation is None or 
            (datetime.now() - last_adaptation['timestamp']).total_seconds() > 
            self.adaptation_triggers['performance_check_interval'] * 3600):
            triggers.append('scheduled_check')
        
        return triggers
    
    def suggest_adaptations(self, triggers: List[str]) -> Dict[str, Any]:
        """Suggest model adaptations based on triggers"""
        
        suggestions = {
            'retrain_model': False,
            'adjust_thresholds': False,
            'feature_selection': False,
            'parameter_tuning': False,
            'ensemble_reweighting': False,
            'details': {}
        }
        
        trading_metrics = self.performance_tracker.calculate_trading_metrics()
        prediction_metrics = self.performance_tracker.calculate_prediction_accuracy()
        
        for trigger in triggers:
            if trigger == 'low_accuracy':
                suggestions['retrain_model'] = True
                suggestions['feature_selection'] = True
                suggestions['details']['accuracy_issue'] = (
                    f"Prediction accuracy: {prediction_metrics.get('accuracy', 0):.3f}"
                )
                
            elif trigger == 'low_win_rate':
                suggestions['adjust_thresholds'] = True
                suggestions['parameter_tuning'] = True
                suggestions['details']['win_rate_issue'] = (
                    f"Win rate: {trading_metrics.get('win_rate', 0):.3f}"
                )
                
            elif trigger == 'high_drawdown':
                suggestions['adjust_thresholds'] = True
                suggestions['ensemble_reweighting'] = True
                suggestions['details']['risk_issue'] = (
                    f"Max drawdown: {trading_metrics.get('max_drawdown', 0):.3f}"
                )
                
            elif trigger == 'consecutive_losses':
                suggestions['retrain_model'] = True
                suggestions['parameter_tuning'] = True
                suggestions['details']['streak_issue'] = "Consecutive losses detected"
                
            elif trigger == 'low_sharpe':
                suggestions['ensemble_reweighting'] = True
                suggestions['parameter_tuning'] = True
                suggestions['details']['sharpe_issue'] = (
                    f"Sharpe ratio: {trading_metrics.get('sharpe_ratio', 0):.3f}"
                )
        
        return suggestions
    
    def apply_adaptations(self, suggestions: Dict[str, Any], model) -> bool:
        """Apply suggested adaptations to the model"""
        
        adaptations_applied = []
        
        try:
            if suggestions.get('adjust_thresholds'):
                # Adjust trading thresholds based on recent performance
                trading_metrics = self.performance_tracker.calculate_trading_metrics()
                
                if trading_metrics.get('win_rate', 0) < 0.5:
                    # Increase threshold for entering trades (be more selective)
                    adaptations_applied.append('increased_entry_threshold')
                
                if trading_metrics.get('max_drawdown', 0) > 0.1:
                    # Decrease position sizes
                    adaptations_applied.append('reduced_position_size')
            
            if suggestions.get('ensemble_reweighting'):
                # Reweight ensemble models based on recent performance
                if hasattr(model, 'model_weights') and hasattr(model, 'performance_history'):
                    # Update weights based on recent performance
                    for model_name in model.model_weights:
                        recent_performance = model.performance_history.get(model_name, [])
                        if recent_performance:
                            latest_performance = recent_performance[-1]
                            weight_adjustment = latest_performance.get('val_auc', 0.5) - 0.5
                            model.model_weights[model_name] *= (1 + weight_adjustment * 0.1)
                    
                    # Renormalize weights
                    total_weight = sum(model.model_weights.values())
                    if total_weight > 0:
                        for model_name in model.model_weights:
                            model.model_weights[model_name] /= total_weight
                    
                    adaptations_applied.append('reweighted_ensemble')
            
            if suggestions.get('parameter_tuning'):
                # Adjust model hyperparameters
                adaptations_applied.append('parameter_adjustment')
            
            # Log adaptation
            adaptation_record = {
                'timestamp': datetime.now(),
                'triggers': list(suggestions.get('details', {}).keys()),
                'adaptations_applied': adaptations_applied,
                'suggestions': suggestions
            }
            self.adaptation_history.append(adaptation_record)
            
            logging.info(f"Applied adaptations: {adaptations_applied}")
            return True
            
        except Exception as e:
            logging.error(f"Error applying adaptations: {e}")
            return False
    
    def run_adaptive_cycle(self, model) -> Dict[str, Any]:
        """Run complete adaptive learning cycle"""
        
        # Check triggers
        triggers = self.check_adaptation_triggers()
        
        if not triggers:
            return {'status': 'no_adaptation_needed', 'triggers': []}
        
        logging.info(f"Adaptation triggers detected: {triggers}")
        
        # Get suggestions
        suggestions = self.suggest_adaptations(triggers)
        
        # Apply adaptations
        success = self.apply_adaptations(suggestions, model)
        
        return {
            'status': 'adaptation_applied' if success else 'adaptation_failed',
            'triggers': triggers,
            'suggestions': suggestions,
            'success': success
        }


class ModelValidationSystem:
    """
    Real-time model validation and A/B testing framework
    """
    
    def __init__(self):
        self.model_variants = {}
        self.validation_results = {}
        self.ab_test_config = {
            'test_duration_days': 7,
            'min_samples_per_variant': 50,
            'significance_level': 0.05
        }
        
    def register_model_variant(self, variant_name: str, model, allocation_percent: float):
        """Register a model variant for A/B testing"""
        
        self.model_variants[variant_name] = {
            'model': model,
            'allocation': allocation_percent / 100.0,
            'predictions': [],
            'performance': {}
        }
        
        logging.info(f"Registered model variant '{variant_name}' with {allocation_percent}% allocation")
    
    def select_model_for_prediction(self) -> Tuple[str, Any]:
        """Select which model variant to use for prediction"""
        
        if not self.model_variants:
            return None, None
        
        # Simple random allocation based on configured percentages
        rand_val = np.random.random()
        cumulative_allocation = 0
        
        for variant_name, variant_data in self.model_variants.items():
            cumulative_allocation += variant_data['allocation']
            if rand_val <= cumulative_allocation:
                return variant_name, variant_data['model']
        
        # Fallback to first model
        first_variant = next(iter(self.model_variants.items()))
        return first_variant[0], first_variant[1]['model']
    
    def log_prediction_result(self, variant_name: str, prediction: float, 
                            actual_outcome: float, symbol: str):
        """Log prediction result for a specific model variant"""
        
        if variant_name in self.model_variants:
            self.model_variants[variant_name]['predictions'].append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'actual': actual_outcome,
                'symbol': symbol
            })
    
    def evaluate_model_variants(self) -> Dict[str, Dict[str, float]]:
        """Evaluate performance of all model variants"""
        
        results = {}
        
        for variant_name, variant_data in self.model_variants.items():
            predictions = variant_data['predictions']
            
            if len(predictions) < self.ab_test_config['min_samples_per_variant']:
                results[variant_name] = {'status': 'insufficient_data', 'sample_size': len(predictions)}
                continue
            
            # Calculate metrics
            pred_values = [p['prediction'] for p in predictions]
            actual_values = [p['actual'] for p in predictions]
            
            # Binary classification metrics
            binary_preds = [1 if p > 0.5 else 0 for p in pred_values]
            binary_actuals = [1 if a > TARGET_THRESHOLD else 0 for a in actual_values]
            
            metrics = {
                'sample_size': len(predictions),
                'accuracy': accuracy_score(binary_actuals, binary_preds),
                'precision': precision_score(binary_actuals, binary_preds, zero_division=0),
                'recall': recall_score(binary_actuals, binary_preds, zero_division=0),
                'f1_score': f1_score(binary_actuals, binary_preds, zero_division=0),
                'correlation': np.corrcoef(pred_values, actual_values)[0, 1] if len(pred_values) > 1 else 0,
                'mse': mean_squared_error(actual_values, pred_values),
                'mae': mean_absolute_error(actual_values, pred_values)
            }
            
            results[variant_name] = metrics
            
            # Update variant performance
            self.model_variants[variant_name]['performance'] = metrics
        
        return results
    
    def statistical_significance_test(self, variant_a: str, variant_b: str, metric: str = 'accuracy') -> Dict[str, Any]:
        """Perform statistical significance test between two variants"""
        
        from scipy import stats
        
        if variant_a not in self.model_variants or variant_b not in self.model_variants:
            return {'error': 'One or both variants not found'}
        
        perf_a = self.model_variants[variant_a]['performance']
        perf_b = self.model_variants[variant_b]['performance']
        
        if metric not in perf_a or metric not in perf_b:
            return {'error': f'Metric {metric} not available for comparison'}
        
        # Get raw predictions for statistical test
        preds_a = self.model_variants[variant_a]['predictions']
        preds_b = self.model_variants[variant_b]['predictions']
        
        if metric == 'accuracy':
            # For accuracy, we need to compare success rates
            binary_a = [1 if p['prediction'] > 0.5 and p['actual'] > TARGET_THRESHOLD else 0 for p in preds_a]
            binary_b = [1 if p['prediction'] > 0.5 and p['actual'] > TARGET_THRESHOLD else 0 for p in preds_b]
            
            # Two-proportion z-test
            count_a, n_a = sum(binary_a), len(binary_a)
            count_b, n_b = sum(binary_b), len(binary_b)
            
            if n_a == 0 or n_b == 0:
                return {'error': 'Insufficient data for statistical test'}
            
            p_a, p_b = count_a / n_a, count_b / n_b
            p_pooled = (count_a + count_b) / (n_a + n_b)
            
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
            z_score = (p_a - p_b) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
        else:
            # For other metrics, use t-test on the raw differences
            values_a = [abs(p['prediction'] - p['actual']) for p in preds_a]  # Prediction errors
            values_b = [abs(p['prediction'] - p['actual']) for p in preds_b]
            
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            z_score = t_stat
        
        result = {
            'variant_a': variant_a,
            'variant_b': variant_b,
            'metric': metric,
            'p_value': p_value,
            'z_score': z_score,
            'significant': p_value < self.ab_test_config['significance_level'],
            'winner': variant_a if perf_a[metric] > perf_b[metric] else variant_b,
            'performance_a': perf_a[metric],
            'performance_b': perf_b[metric]
        }
        
        return result
    
    def auto_promote_best_model(self) -> Optional[str]:
        """Automatically promote the best performing model variant"""
        
        results = self.evaluate_model_variants()
        
        # Find best performing variant by accuracy
        best_variant = None
        best_score = -1
        
        for variant_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                if metrics['accuracy'] > best_score and metrics['sample_size'] >= self.ab_test_config['min_samples_per_variant']:
                    best_score = metrics['accuracy']
                    best_variant = variant_name
        
        if best_variant:
            # Increase allocation for best variant
            total_other_allocation = sum(
                variant_data['allocation'] 
                for name, variant_data in self.model_variants.items() 
                if name != best_variant
            )
            
            # Give best variant 60% allocation, distribute rest proportionally
            self.model_variants[best_variant]['allocation'] = 0.6
            
            for name, variant_data in self.model_variants.items():
                if name != best_variant:
                    original_allocation = variant_data['allocation']
                    new_allocation = 0.4 * (original_allocation / total_other_allocation) if total_other_allocation > 0 else 0.1
                    variant_data['allocation'] = new_allocation
            
            logging.info(f"Promoted model variant '{best_variant}' to 60% allocation (accuracy: {best_score:.3f})")
            
        return best_variant


# Global instances
performance_tracker = TradingPerformanceMetrics()
adaptive_system = AdaptiveLearningSystem()
validation_system = ModelValidationSystem()

def log_trade_performance(trade_data: Dict[str, Any]):
    """Log trade performance data"""
    performance_tracker.log_trade(trade_data)

def log_portfolio_performance(portfolio_data: Dict[str, Any]):
    """Log portfolio performance data"""
    performance_tracker.log_portfolio_snapshot(portfolio_data)

def log_model_prediction(symbol: str, prediction: float, actual_return: float = None, 
                        confidence: float = None):
    """Log model prediction for tracking"""
    performance_tracker.log_prediction(symbol, prediction, actual_return, confidence)

def run_adaptive_learning_cycle(model):
    """Run adaptive learning cycle"""
    return adaptive_system.run_adaptive_cycle(model)

def setup_ab_testing(models: Dict[str, Any], allocations: Dict[str, float]):
    """Setup A/B testing for multiple models"""
    for name, model in models.items():
        allocation = allocations.get(name, 100.0 / len(models))
        validation_system.register_model_variant(name, model, allocation)

def get_model_for_prediction():
    """Get model for prediction from A/B test"""
    return validation_system.select_model_for_prediction()

def evaluate_ab_test_results():
    """Evaluate A/B test results"""
    return validation_system.evaluate_model_variants()