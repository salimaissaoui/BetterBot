import logging
import pandas as pd
import numpy as np

# If you want to integrate TA-Lib, uncomment and install:
# import talib

from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# Example of user-defined parameters (import from .config or define inline):
SHORT_MA = 10
LONG_MA = 50
RSI_PERIOD = 14
FEATURE_LAGS = 3
ATR_WINDOW = 14  # More standard than 1 for smoothing

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def compute_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a variety of technical indicators on the input DataFrame and returns a processed DataFrame.

    Best Practices/Improvements:
    1. Have enough warm-up data loaded (e.g., extra historical rows) so the earliest indicator values are not skewed.
    2. Reindex the data if needed to ensure it’s at a consistent frequency (daily, hourly, etc.).
    3. Don't return empty too early; instead compute indicators and only drop rows after they're all populated.
    4. Use a higher ATR window (14) if you want more "standard" ATR smoothing.
    5. (Optional) Use TA-Lib to cross-check indicator values if exact reproducibility with charting platforms is desired.
    """

    logging.debug("Starting computation of technical indicators...")
    logging.debug(f"Initial data shape: {data.shape}")
    logging.debug(f"Initial data columns: {data.columns.tolist()}")

    # -------------------------------------------------------------------------
    # 1. Basic checks & optional reindexing
    # -------------------------------------------------------------------------
    if data.empty or not {'open', 'high', 'low', 'close', 'volume'}.issubset(data.columns):
        logging.warning("Data is empty or missing essential columns (open, high, low, close, volume).")
        return pd.DataFrame()

    # Example: reindex to daily frequency if needed (commented out)
    # data = data.sort_index()
    # full_idx = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    # data = data.reindex(full_idx).fillna(method='ffill')

    # Make a copy to avoid mutating the original
    data = data.copy()

    # -------------------------------------------------------------------------
    # 2. Basic returns
    # -------------------------------------------------------------------------
    data['return'] = data['close'].pct_change()
    logging.debug("Computed daily returns.")

    # -------------------------------------------------------------------------
    # 3. SMA Indicators (Short & Long)
    # -------------------------------------------------------------------------
    sma_short = SMAIndicator(close=data['close'], window=SHORT_MA)
    sma_long = SMAIndicator(close=data['close'], window=LONG_MA)
    data['ma_short'] = sma_short.sma_indicator()
    data['ma_long'] = sma_long.sma_indicator()
    data['ma_ratio'] = data['ma_short'] / data['ma_long']
    logging.debug("Computed SMA indicators.")

    # -------------------------------------------------------------------------
    # 4. RSI
    # -------------------------------------------------------------------------
    rsi = RSIIndicator(close=data['close'], window=RSI_PERIOD)
    data['rsi'] = rsi.rsi()
    logging.debug("Computed RSI.")

    # -------------------------------------------------------------------------
    # 5. MACD
    # -------------------------------------------------------------------------
    macd = MACD(close=data['close'])
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    logging.debug("Computed MACD indicators.")

    # -------------------------------------------------------------------------
    # 6. Bollinger Bands
    # -------------------------------------------------------------------------
    # Typically needs at least 20 bars, so ensure you have enough warm-up data loaded.
    bb = BollingerBands(close=data['close'], window=20, window_dev=2)
    data['bb_h'] = bb.bollinger_hband()
    data['bb_l'] = bb.bollinger_lband()
    data['bb_m'] = bb.bollinger_mavg()
    data['bb_w'] = bb.bollinger_wband()
    data['bb_p'] = bb.bollinger_pband()
    logging.debug("Computed Bollinger Bands.")

    # -------------------------------------------------------------------------
    # 7. ATR (using a 14-day window for smoothing)
    # -------------------------------------------------------------------------
    atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=ATR_WINDOW)
    data['atr'] = atr.average_true_range()
    logging.debug("Computed ATR.")

    # -------------------------------------------------------------------------
    # 8. OBV
    # -------------------------------------------------------------------------
    obv = OnBalanceVolumeIndicator(close=data['close'], volume=data['volume'])
    data['obv'] = obv.on_balance_volume()
    logging.debug("Computed OBV.")

    # -------------------------------------------------------------------------
    # 9. Identify Doji candlesticks
    # -------------------------------------------------------------------------
    data['doji'] = np.where(
        (np.abs(data['close'] - data['open']) < 0.001 * data['open']), 
        1, 
        0
    )
    logging.debug("Computed Doji candlestick flags.")


    # -------------------------------------------------------------------------
    # 11. Compute feature lags
    # -------------------------------------------------------------------------
    for i in range(1, FEATURE_LAGS + 1):
        data[f'return_lag_{i}'] = data['return'].shift(i)
    logging.debug("Computed feature lags for returns.")

    # -------------------------------------------------------------------------
    # 12. Future return and target
    # -------------------------------------------------------------------------
    data['future_return'] = data['close'].shift(-1) / data['close'] - 1
    data['target'] = (data['future_return'] > 0).astype(int)
    logging.debug("Computed future return and target.")

    # -------------------------------------------------------------------------
    # 13. Drop rows where any indicator is NaN due to warm-up or shifting
    # -------------------------------------------------------------------------
    # If you want to keep partial data, you can skip dropping or do it selectively.
    logging.debug(f"Rows before final dropna: {len(data)}")
    data.dropna(inplace=True)
    logging.debug(f"Rows after final dropna: {len(data)}")
    logging.debug("Final processed data preview:")
    logging.debug(data.head().to_string())

    return data


def prepare_features(data: pd.DataFrame):
    """
    Splits the DataFrame into features (X) and target (y).

    Assumes 'target' is already created as a binary label in the DataFrame.
    Returns:
        X (DataFrame): Feature matrix
        y (Series): Target labels
    """
    if data.empty:
        logging.warning("Input data is empty in prepare_features. Returning empty X, y.")
        return pd.DataFrame(), pd.Series(dtype=float)

    # List of features you want for your model
    features = [
        'ma_short', 'ma_long', 'ma_ratio',
        'rsi', 'return',
        'macd', 'macd_signal', 'macd_diff',
        'bb_h', 'bb_l', 'bb_m', 'bb_w', 'bb_p',
        'atr', 'doji', 'obv'
    ] + [f'return_lag_{i}' for i in range(1, FEATURE_LAGS + 1)]

    # Keep only columns that exist
    available = data.columns
    features = [f for f in features if f in available]

    if not features:
        logging.warning("No requested features found in data. Returning empty X, y.")
        return pd.DataFrame(), pd.Series(dtype=float)

    X = data[features].copy()
    y = data['target'].copy() if 'target' in data.columns else pd.Series(dtype=float)

    # Log info
    logging.debug(f"Prepared features X shape: {X.shape}, y length: {len(y)}")
    return X, y
