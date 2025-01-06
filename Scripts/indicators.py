import logging
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

from .config import SHORT_MA, LONG_MA, RSI_PERIOD, FEATURE_LAGS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def compute_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Computes a variety of technical indicators and returns a processed DataFrame."""
    if data.empty or 'close' not in data.columns:
        logging.info("No data or 'close' column missing when computing indicators.")
        return pd.DataFrame()

    data = data.copy()
    data['return'] = data['close'].pct_change()

    # Trend Indicators
    if len(data) < max(SHORT_MA, LONG_MA):
        logging.info("Not enough data for SMA computation. Returning empty.")
        return pd.DataFrame()

    sma_short = SMAIndicator(close=data['close'], window=SHORT_MA)
    sma_long = SMAIndicator(close=data['close'], window=LONG_MA)
    data['ma_short'] = sma_short.sma_indicator()
    data['ma_long'] = sma_long.sma_indicator()
    data['ma_ratio'] = data['ma_short'] / data['ma_long']

    # Momentum Indicators
    if len(data) < RSI_PERIOD:
        logging.info("Not enough data for RSI computation. Returning empty.")
        return pd.DataFrame()
    rsi = RSIIndicator(close=data['close'], window=RSI_PERIOD)
    data['rsi'] = rsi.rsi()

    macd = MACD(close=data['close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()

    # Bollinger Bands (need at least 20 bars)
    if len(data) < 20:
        logging.info("Not enough data for Bollinger Bands (need at least 20). Returning empty.")
        return pd.DataFrame()
    bb = BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bb_h'] = bb.bollinger_hband()
    data['bb_l'] = bb.bollinger_lband()
    data['bb_m'] = bb.bollinger_mavg()
    data['bb_w'] = bb.bollinger_wband()
    data['bb_p'] = bb.bollinger_pband()

    # ATR
    atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=14)
    data['atr'] = atr.average_true_range()

    # OBV
    data['obv'] = OnBalanceVolumeIndicator(close=data['close'], volume=data['volume']).on_balance_volume()

    # Simple candlestick pattern: doji
    data['doji'] = np.where((abs(data['close'] - data['open']) < 0.001 * data['open']), 1, 0)

    # Remove anomalies in returns
    returns = data['return'].dropna()
    if len(returns) == 0:
        logging.info("No valid returns. Returning empty.")
        return pd.DataFrame()

    q1, q3 = returns.quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (data['return'] > q1 - 3*iqr) & (data['return'] < q3 + 3*iqr)
    data = data[mask]

    # Feature lags
    for i in range(1, FEATURE_LAGS + 1):
        data[f'return_lag_{i}'] = data['return'].shift(i)
    
    # Future return and target
    data['future_return'] = data['close'].shift(-1) / data['close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)

    data.dropna(inplace=True)

    # Additional safety check
    if len(data) <= FEATURE_LAGS:
        logging.info("Not enough data after processing to safely compute features. Returning empty.")
        return pd.DataFrame()

    return data


def prepare_features(data: pd.DataFrame):
    """Splits the DataFrame into features (X) and target (y)."""
    if data.empty:
        logging.info("Empty data in prepare_features.")
        return pd.DataFrame(), pd.Series(dtype=float)

    features = [
        'ma_short', 'ma_long', 'ma_ratio',
        'rsi', 'return',
        'macd', 'macd_signal', 'macd_diff',
        'bb_h', 'bb_l', 'bb_m', 'bb_w', 'bb_p',
        'atr', 'doji', 'obv'
    ] + [f'return_lag_{i}' for i in range(1, FEATURE_LAGS + 1)]

    available = data.columns
    features = [f for f in features if f in available]

    if not features:
        logging.info("No features available after filtering.")
        return pd.DataFrame(), pd.Series(dtype=float)

    X = data[features]
    if X.empty:
        logging.info("No data in X after selecting features.")
        return pd.DataFrame(), pd.Series(dtype=float)

    if 'target' not in data.columns:
        logging.info("No target column in data. Returning empty target.")
        return X, pd.Series(dtype=float)

    y = data['target']
    return X, y
