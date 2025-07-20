import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Financial libraries
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from .database import get_session, StockData
from .config import TARGET_THRESHOLD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class MarketRegimeDetector:
    """
    Detects different market regimes (bull, bear, sideways, volatile)
    using unsupervised learning and statistical methods.
    """
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.regimes = {0: 'bearish', 1: 'sideways', 2: 'bullish', 3: 'volatile'}
        self.regime_model = None
        self.scaler = StandardScaler()
        
    def calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for regime detection"""
        
        if len(data) < self.lookback_window:
            logging.warning("Insufficient data for regime detection")
            return pd.DataFrame()
        
        # Calculate rolling statistics
        window = min(50, len(data) // 4)
        
        features = pd.DataFrame(index=data.index)
        
        # Price trend features
        features['return_mean'] = data['close'].pct_change().rolling(window).mean()
        features['return_std'] = data['close'].pct_change().rolling(window).std()
        features['return_skew'] = data['close'].pct_change().rolling(window).skew()
        features['return_kurt'] = data['close'].pct_change().rolling(window).kurt()
        
        # Volatility features
        features['volatility'] = data['close'].pct_change().rolling(window).std() * np.sqrt(252)
        features['volatility_ma'] = features['volatility'].rolling(window//2).mean()
        features['volatility_ratio'] = features['volatility'] / features['volatility_ma']
        
        # Trend strength
        features['trend_strength'] = abs(data['close'].rolling(window).apply(
            lambda x: stats.linregress(range(len(x)), x)[0]
        ))
        
        # Price momentum
        features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['momentum_20'] = data['close'] / data['close'].shift(20) - 1
        features['momentum_60'] = data['close'] / data['close'].shift(60) - 1
        
        # Volume features
        if 'volume' in data.columns:
            features['volume_ma'] = data['volume'].rolling(window).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma']
            features['price_volume_trend'] = ((data['close'] - data['close'].shift(1)) / 
                                            data['close'].shift(1) * data['volume']).rolling(window).sum()
        else:
            features['volume_ma'] = 0
            features['volume_ratio'] = 1
            features['price_volume_trend'] = 0
        
        # Market structure features
        features['new_highs'] = (data['high'] == data['high'].rolling(window).max()).astype(int)
        features['new_lows'] = (data['low'] == data['low'].rolling(window).min()).astype(int)
        
        # Moving average relationships
        features['ma_5'] = data['close'].rolling(5).mean()
        features['ma_20'] = data['close'].rolling(20).mean()
        features['ma_50'] = data['close'].rolling(50).mean()
        
        features['price_vs_ma5'] = data['close'] / features['ma_5'] - 1
        features['price_vs_ma20'] = data['close'] / features['ma_20'] - 1
        features['price_vs_ma50'] = data['close'] / features['ma_50'] - 1
        
        features['ma5_vs_ma20'] = features['ma_5'] / features['ma_20'] - 1
        features['ma20_vs_ma50'] = features['ma_20'] / features['ma_50'] - 1
        
        # Support/Resistance levels
        features['support_strength'] = self._calculate_support_resistance(data, 'support')
        features['resistance_strength'] = self._calculate_support_resistance(data, 'resistance')
        
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_support_resistance(self, data: pd.DataFrame, level_type: str) -> pd.Series:
        """Calculate support/resistance strength"""
        
        if level_type == 'support':
            prices = data['low']
            comparison = lambda x, y: x <= y
        else:
            prices = data['high']
            comparison = lambda x, y: x >= y
        
        window = 20
        strength = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(data)):
            window_prices = prices.iloc[i-window:i]
            current_price = prices.iloc[i]
            
            # Count how many times price has touched this level
            touches = sum(1 for p in window_prices 
                         if abs(p - current_price) / current_price < 0.02)
            
            strength.iloc[i] = touches / window
        
        return strength.fillna(0)
    
    def fit_regime_detector(self, market_data: Dict[str, pd.DataFrame]):
        """Fit regime detection model using multiple symbols"""
        
        all_features = []
        
        for symbol, data in market_data.items():
            features = self.calculate_regime_features(data)
            if not features.empty:
                # Add symbol identifier
                features['symbol_hash'] = hash(symbol) % 1000  # Simple symbol encoding
                all_features.append(features)
        
        if not all_features:
            logging.error("No valid features for regime detection")
            return False
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Remove rows with NaN
        combined_features = combined_features.dropna()
        
        if len(combined_features) < 100:
            logging.error("Insufficient data for regime clustering")
            return False
        
        # Scale features
        feature_columns = [col for col in combined_features.columns if col != 'symbol_hash']
        scaled_features = self.scaler.fit_transform(combined_features[feature_columns])
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(10, len(feature_columns)))
        pca_features = pca.fit_transform(scaled_features)
        
        # Cluster to identify regimes
        self.regime_model = KMeans(n_clusters=4, random_state=42, n_init=10)
        regime_labels = self.regime_model.fit_predict(pca_features)
        
        # Analyze clusters to assign regime names
        regime_analysis = {}
        for regime in range(4):
            mask = regime_labels == regime
            if np.sum(mask) > 0:
                avg_return = combined_features.loc[mask, 'return_mean'].mean()
                avg_volatility = combined_features.loc[mask, 'return_std'].mean()
                
                regime_analysis[regime] = {
                    'avg_return': avg_return,
                    'avg_volatility': avg_volatility,
                    'count': np.sum(mask)
                }
        
        # Sort regimes by return and volatility
        sorted_regimes = sorted(regime_analysis.items(), 
                              key=lambda x: (x[1]['avg_return'], -x[1]['avg_volatility']))
        
        # Reassign regime meanings
        regime_mapping = {}
        for i, (original_regime, _) in enumerate(sorted_regimes):
            if i == 0:
                regime_mapping[original_regime] = 'bearish'
            elif i == 1:
                regime_mapping[original_regime] = 'sideways'
            elif i == 2:
                regime_mapping[original_regime] = 'bullish'
            else:
                regime_mapping[original_regime] = 'volatile'
        
        self.regimes = regime_mapping
        
        # Store PCA model
        self.pca = pca
        self.feature_columns = feature_columns
        
        logging.info(f"Regime detector fitted with {len(combined_features)} samples")
        logging.info(f"Regime distribution: {regime_analysis}")
        
        return True
    
    def predict_regime(self, data: pd.DataFrame) -> str:
        """Predict current market regime"""
        
        if self.regime_model is None:
            return 'unknown'
        
        try:
            features = self.calculate_regime_features(data)
            if features.empty:
                return 'unknown'
            
            # Use most recent feature vector
            latest_features = features[self.feature_columns].iloc[-1:].fillna(0)
            
            # Scale and transform
            scaled_features = self.scaler.transform(latest_features)
            pca_features = self.pca.transform(scaled_features)
            
            # Predict regime
            regime_id = self.regime_model.predict(pca_features)[0]
            
            return self.regimes.get(regime_id, 'unknown')
            
        except Exception as e:
            logging.error(f"Error predicting regime: {e}")
            return 'unknown'


class AdvancedFeatureEngineering:
    """
    Advanced feature engineering with technical, fundamental, 
    sentiment, and market microstructure features.
    """
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        self.feature_cache = {}
        
    def create_advanced_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Create comprehensive feature set"""
        
        if data.empty or len(data) < 50:
            logging.warning("Insufficient data for advanced features")
            return pd.DataFrame()
        
        features = data.copy()
        
        # Basic technical indicators (if not already present)
        features = self._add_basic_technicals(features)
        
        # Advanced technical features
        features = self._add_advanced_technicals(features)
        
        # Market microstructure features
        features = self._add_microstructure_features(features)
        
        # Time-based features
        features = self._add_temporal_features(features)
        
        # Statistical features
        features = self._add_statistical_features(features)
        
        # Momentum and trend features
        features = self._add_momentum_features(features)
        
        # Volatility features
        features = self._add_volatility_features(features)
        
        # Pattern recognition features
        features = self._add_pattern_features(features)
        
        # Market regime features
        features = self._add_regime_features(features)
        
        # Cross-asset features (if symbol provided)
        if symbol:
            features = self._add_cross_asset_features(features, symbol)
        
        return features
    
    def _add_basic_technicals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators if not present"""
        
        # Only add if not already present
        if 'rsi' not in data.columns:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
        
        if 'macd' not in data.columns:
            # MACD
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema12 - ema26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_diff'] = data['macd'] - data['macd_signal']
        
        return data
    
    def _add_advanced_technicals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators"""
        
        # Williams %R
        high_14 = data['high'].rolling(14).max()
        low_14 = data['low'].rolling(14).min()
        data['williams_r'] = -100 * (high_14 - data['close']) / (high_14 - low_14)
        
        # Commodity Channel Index (CCI)
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        data['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Money Flow Index (MFI)
        if 'volume' in data.columns:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            money_ratio = positive_flow / negative_flow
            data['mfi'] = 100 - (100 / (1 + money_ratio))
        else:
            data['mfi'] = 50  # Neutral if no volume data
        
        # Accumulation/Distribution Line
        if 'volume' in data.columns:
            clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
            clv = clv.fillna(0)
            data['ad_line'] = (clv * data['volume']).cumsum()
        else:
            data['ad_line'] = 0
        
        # Parabolic SAR
        data['sar'] = self._calculate_parabolic_sar(data)
        
        # Chaikin Oscillator
        if 'volume' in data.columns:
            ad_line = data['ad_line']
            data['chaikin_osc'] = ad_line.ewm(span=3).mean() - ad_line.ewm(span=10).mean()
        else:
            data['chaikin_osc'] = 0
        
        return data
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame, step: float = 0.02, maximum: float = 0.2):
        """Calculate Parabolic SAR"""
        
        high, low, close = data['high'], data['low'], data['close']
        sar = np.zeros(len(data))
        trend = np.zeros(len(data))
        af = np.zeros(len(data))
        ep = np.zeros(len(data))
        
        # Initialize
        sar[0] = low[0]
        trend[0] = 1  # 1 for uptrend, -1 for downtrend
        af[0] = step
        ep[0] = high[0]
        
        for i in range(1, len(data)):
            if trend[i-1] == 1:  # Uptrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if low[i] <= sar[i]:
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    af[i] = step
                    ep[i] = low[i]
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(maximum, af[i-1] + step)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                if high[i] >= sar[i]:
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    af[i] = step
                    ep[i] = high[i]
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(maximum, af[i-1] + step)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return sar
    
    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features"""
        
        # Price efficiency measures
        data['price_efficiency'] = abs(data['close'].pct_change()) / abs(data['high'] - data['low']) * data['close']
        
        # Intraday range
        data['intraday_range'] = (data['high'] - data['low']) / data['close']
        
        # Gap analysis
        data['gap_up'] = np.where(data['open'] > data['close'].shift(1), 
                                (data['open'] - data['close'].shift(1)) / data['close'].shift(1), 0)
        data['gap_down'] = np.where(data['open'] < data['close'].shift(1), 
                                  (data['close'].shift(1) - data['open']) / data['close'].shift(1), 0)
        
        # Price clustering (round number effects)
        data['round_number'] = (data['close'] % 1 == 0).astype(int)
        data['half_dollar'] = (data['close'] % 0.5 == 0).astype(int)
        
        # Tick direction
        data['tick_direction'] = np.sign(data['close'].diff())
        data['tick_streak'] = self._calculate_streaks(data['tick_direction'])
        
        if 'volume' in data.columns:
            # Volume-weighted average price (VWAP)
            data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
            
            # Volume profile features
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            data['price_volume_correlation'] = data['close'].rolling(20).corr(data['volume'])
            
            # Order flow approximation
            data['buy_volume'] = np.where(data['close'] > data['open'], data['volume'], 0)
            data['sell_volume'] = np.where(data['close'] < data['open'], data['volume'], 0)
            data['order_flow_ratio'] = (data['buy_volume'].rolling(10).sum() / 
                                       (data['sell_volume'].rolling(10).sum() + 1))
        
        return data
    
    def _calculate_streaks(self, series: pd.Series) -> pd.Series:
        """Calculate consecutive streaks in a series"""
        
        streaks = pd.Series(index=series.index, dtype=int)
        current_streak = 0
        
        for i in range(len(series)):
            if i == 0:
                current_streak = 1
            elif series.iloc[i] == series.iloc[i-1] and series.iloc[i] != 0:
                current_streak += 1
            else:
                current_streak = 1
            
            streaks.iloc[i] = current_streak
        
        return streaks
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Time components
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['day_of_month'] = data['timestamp'].dt.day
            data['month'] = data['timestamp'].dt.month
            data['quarter'] = data['timestamp'].dt.quarter
            
            # Market timing
            data['market_open'] = ((data['hour'] >= 9) & (data['hour'] < 16)).astype(int)
            data['pre_market'] = (data['hour'] < 9).astype(int)
            data['after_hours'] = (data['hour'] >= 16).astype(int)
            
            # Weekly patterns
            data['monday_effect'] = (data['day_of_week'] == 0).astype(int)
            data['friday_effect'] = (data['day_of_week'] == 4).astype(int)
            
            # Monthly patterns
            data['month_end'] = (data['day_of_month'] >= 28).astype(int)
            data['month_start'] = (data['day_of_month'] <= 5).astype(int)
            
            # Seasonal patterns
            data['q1'] = (data['quarter'] == 1).astype(int)
            data['q4'] = (data['quarter'] == 4).astype(int)
        
        return data
    
    def _add_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features"""
        
        returns = data['close'].pct_change()
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            # Returns statistics
            data[f'return_mean_{window}'] = returns.rolling(window).mean()
            data[f'return_std_{window}'] = returns.rolling(window).std()
            data[f'return_skew_{window}'] = returns.rolling(window).skew()
            data[f'return_kurt_{window}'] = returns.rolling(window).kurt()
            
            # Price statistics
            data[f'price_zscore_{window}'] = ((data['close'] - data['close'].rolling(window).mean()) / 
                                            data['close'].rolling(window).std())
            
            # Percentile ranks
            data[f'price_rank_{window}'] = data['close'].rolling(window).rank(pct=True)
            
            if 'volume' in data.columns:
                data[f'volume_zscore_{window}'] = ((data['volume'] - data['volume'].rolling(window).mean()) / 
                                                  data['volume'].rolling(window).std())
        
        # Autocorrelation
        for lag in [1, 2, 5, 10]:
            data[f'return_autocorr_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag) if len(x) > lag else 0
            )
        
        return data
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and trend features"""
        
        # Price momentum across multiple timeframes
        for period in [1, 3, 5, 10, 20, 50, 100]:
            data[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        
        # Acceleration (momentum of momentum)
        data['momentum_acceleration'] = data['momentum_10'].diff()
        
        # Trend strength using linear regression slope
        for window in [10, 20, 50]:
            data[f'trend_slope_{window}'] = data['close'].rolling(window).apply(
                lambda x: stats.linregress(range(len(x)), x)[0] if len(x) == window else 0
            )
            
            data[f'trend_r2_{window}'] = data['close'].rolling(window).apply(
                lambda x: stats.linregress(range(len(x)), x)[2]**2 if len(x) == window else 0
            )
        
        # Moving average slopes
        for ma_period in [5, 10, 20, 50]:
            ma = data['close'].rolling(ma_period).mean()
            data[f'ma_slope_{ma_period}'] = ma.diff()
            data[f'ma_acceleration_{ma_period}'] = data[f'ma_slope_{ma_period}'].diff()
        
        # Relative strength vs market (placeholder - would use market index)
        data['relative_strength'] = data['momentum_20']  # Simplified
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features"""
        
        returns = data['close'].pct_change()
        
        # Realized volatility
        for window in [5, 10, 20, 50]:
            data[f'realized_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            
            # Upside/downside volatility
            upside_returns = returns.where(returns > 0, 0)
            downside_returns = returns.where(returns < 0, 0)
            
            data[f'upside_vol_{window}'] = upside_returns.rolling(window).std() * np.sqrt(252)
            data[f'downside_vol_{window}'] = downside_returns.rolling(window).std() * np.sqrt(252)
            
            # Volatility ratio
            data[f'vol_ratio_{window}'] = (data[f'upside_vol_{window}'] / 
                                         (data[f'downside_vol_{window}'] + 1e-6))
        
        # GARCH-like volatility clustering
        data['vol_clustering'] = returns.rolling(20).apply(
            lambda x: (x**2).autocorr(1) if len(x) >= 2 else 0
        )
        
        # Parkinson estimator (high-low volatility)
        data['parkinson_vol'] = (0.25 * np.log(2) * 
                               (np.log(data['high'] / data['low'])**2).rolling(20).mean())
        
        return data
    
    def _add_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick and chart pattern features"""
        
        # Candlestick patterns
        body_size = abs(data['close'] - data['open'])
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        lower_shadow = np.minimum(data['open'], data['close']) - data['low']
        total_range = data['high'] - data['low']
        
        # Basic candlestick features
        data['body_ratio'] = body_size / (total_range + 1e-6)
        data['upper_shadow_ratio'] = upper_shadow / (total_range + 1e-6)
        data['lower_shadow_ratio'] = lower_shadow / (total_range + 1e-6)
        
        # Doji pattern
        data['doji'] = (body_size / (total_range + 1e-6) < 0.1).astype(int)
        
        # Hammer/hanging man
        data['hammer'] = ((lower_shadow > 2 * body_size) & 
                         (upper_shadow < 0.1 * total_range)).astype(int)
        
        # Engulfing patterns
        data['bullish_engulfing'] = ((data['close'] > data['open']) & 
                                   (data['close'].shift(1) < data['open'].shift(1)) &
                                   (data['open'] < data['close'].shift(1)) &
                                   (data['close'] > data['open'].shift(1))).astype(int)
        
        data['bearish_engulfing'] = ((data['close'] < data['open']) & 
                                   (data['close'].shift(1) > data['open'].shift(1)) &
                                   (data['open'] > data['close'].shift(1)) &
                                   (data['close'] < data['open'].shift(1))).astype(int)
        
        # Support/resistance touches
        data['support_touch'] = self._detect_level_touches(data, 'support')
        data['resistance_touch'] = self._detect_level_touches(data, 'resistance')
        
        # Price channel position
        data['channel_position'] = self._calculate_channel_position(data)
        
        return data
    
    def _detect_level_touches(self, data: pd.DataFrame, level_type: str, window: int = 20) -> pd.Series:
        """Detect support/resistance level touches"""
        
        touches = pd.Series(index=data.index, dtype=int)
        
        for i in range(window, len(data)):
            if level_type == 'support':
                level = data['low'].iloc[i-window:i].min()
                current_low = data['low'].iloc[i]
                touches.iloc[i] = int(abs(current_low - level) / level < 0.02)
            else:  # resistance
                level = data['high'].iloc[i-window:i].max()
                current_high = data['high'].iloc[i]
                touches.iloc[i] = int(abs(current_high - level) / level < 0.02)
        
        return touches.fillna(0)
    
    def _calculate_channel_position(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate position within price channel"""
        
        position = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(data)):
            high = data['high'].iloc[i-window:i].max()
            low = data['low'].iloc[i-window:i].min()
            current = data['close'].iloc[i]
            
            if high > low:
                position.iloc[i] = (current - low) / (high - low)
            else:
                position.iloc[i] = 0.5
        
        return position.fillna(0.5)
    
    def _add_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        
        # Placeholder for regime detection
        # In practice, this would use the regime detector
        data['regime_bullish'] = 0
        data['regime_bearish'] = 0
        data['regime_sideways'] = 0
        data['regime_volatile'] = 0
        
        # Regime change detection
        data['regime_change'] = 0
        
        return data
    
    def _add_cross_asset_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add cross-asset correlation features"""
        
        # Placeholder for cross-asset features
        # Would correlate with market indices, sectors, etc.
        data['market_correlation'] = 0.5
        data['sector_correlation'] = 0.6
        data['beta'] = 1.0
        
        return data


# Global instance
advanced_features = AdvancedFeatureEngineering()

def create_advanced_features(data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Create advanced feature set for a given dataset"""
    return advanced_features.create_advanced_features(data, symbol)

def detect_market_regime(data: pd.DataFrame) -> str:
    """Detect current market regime"""
    return advanced_features.regime_detector.predict_regime(data)