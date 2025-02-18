import logging
import pandas as pd
import numpy as np

from .config import (
    SHORT_MA,
    LONG_MA,
    RSI_PERIOD,
    FEATURE_LAGS,
    ATR_WINDOW,
    DOJI_THRESHOLD,
    TARGET_THRESHOLD
)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def compute_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a variety of technical indicators manually, without external libraries.
    Returns a processed DataFrame with columns for each indicator.
    """
    logging.debug("Starting computation of technical indicators...")
    logging.debug(f"Initial data shape: {data.shape}")
    logging.debug(f"Initial data columns: {data.columns.tolist()}")

    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if data.empty or not required_cols.issubset(data.columns):
        logging.warning("Data is empty or missing essential columns.")
        return pd.DataFrame()

    # Convert to numeric, just in case
    for col in ['open', 'high', 'low', 'close', 'volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Sort by timestamp if present
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data.sort_values('timestamp', inplace=True)
        data.reset_index(drop=True, inplace=True)
    else:
        if isinstance(data.index, pd.DatetimeIndex):
            data.sort_index(inplace=True)

    data = data.copy()

    # 1. Returns
    data['return'] = data['close'].pct_change()

    # 2. SMA (Short & Long)
    data['ma_short'] = data['close'].rolling(window=SHORT_MA).mean()
    data['ma_long'] = data['close'].rolling(window=LONG_MA).mean()
    data['ma_ratio'] = data['ma_short'] / data['ma_long']

    # 3. RSI
    diff = data['close'].diff(1)
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_PERIOD).mean()
    avg_loss = loss.rolling(window=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # 4. MACD
    ema_fast = data['close'].ewm(span=12, adjust=False).mean()
    ema_slow = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_fast - ema_slow
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']

    # 5. Bollinger Bands
    bb_window = 20
    rolling_mean = data['close'].rolling(window=bb_window).mean()
    rolling_std = data['close'].rolling(window=bb_window).std()
    data['bb_m'] = rolling_mean
    data['bb_h'] = rolling_mean + (2 * rolling_std)
    data['bb_l'] = rolling_mean - (2 * rolling_std)
    data['bb_w'] = (data['bb_h'] - data['bb_l']) / data['bb_m']
    data['bb_p'] = (data['close'] - data['bb_l']) / (data['bb_h'] - data['bb_l'])

    # 6. ATR
    data['hl'] = data['high'] - data['low']
    data['hc'] = (data['high'] - data['close'].shift(1)).abs()
    data['lc'] = (data['low'] - data['close'].shift(1)).abs()
    data['tr'] = data[['hl', 'hc', 'lc']].max(axis=1)
    data['atr'] = data['tr'].rolling(window=ATR_WINDOW).mean()
    data.drop(['hl', 'hc', 'lc', 'tr'], axis=1, inplace=True)

    # 7. OBV
    price_diff_sign = np.sign(data['close'].diff())
    price_diff_sign.iloc[0] = 0
    data['obv'] = (price_diff_sign * data['volume']).fillna(0).cumsum()
    data['obv_ma'] = data['obv'].rolling(window=10, min_periods=10).mean()

    # 8. Stochastic
    stoch_window = 14
    lowest_low = data['low'].rolling(stoch_window).min()
    highest_high = data['high'].rolling(stoch_window).max()
    data['stoch'] = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    data['stoch_signal'] = data['stoch'].rolling(window=3).mean()

    # 9. Doji
    data['doji'] = np.where(
        (np.abs(data['close'] - data['open']) < DOJI_THRESHOLD * data['open']),
        1,
        0
    )

    # ---- NEW FEATURES ----
    # Rolling standard deviation of returns
    data['rolling_ret_std'] = data['return'].rolling(window=10).std()  # 10-day example

    # Rolling average volume
    data['rolling_vol_avg'] = data['volume'].rolling(window=10).mean()

    # 10. Lagged returns
    for i in range(1, FEATURE_LAGS + 1):
        data[f'return_lag_{i}'] = data['return'].shift(i)

    # 11. Future return + target with threshold
    data['future_return'] = data['close'].shift(-1) / data['close'] - 1
    # Instead of (future_return > 0), use a threshold from config
    data['target'] = (data['future_return'] >= TARGET_THRESHOLD).astype(int)

    # Optional: fill or drop NaNs
    # data.dropna(inplace=True)     # Traditional approach: drop incomplete rows
    # Or fill them forward:
    # data.fillna(method='ffill', inplace=True)

    logging.debug(f"Final processed data preview:\n{data.head(15).to_string()}")
    return data


def prepare_features(data: pd.DataFrame):
    """
    Prepares the feature matrix (X) and target vector (y) for modeling.
    """
    # Expanded to include the new features: rolling_ret_std, rolling_vol_avg
    expected_features = [
        'ma_short', 'ma_long', 'ma_ratio',
        'rsi', 'return',
        'macd', 'macd_signal', 'macd_diff',
        'bb_h', 'bb_l', 'bb_m', 'bb_w', 'bb_p',
        'atr', 'doji',
        'obv', 'obv_ma', 'stoch', 'stoch_signal',
        'rolling_ret_std', 'rolling_vol_avg'
    ]

    available_features = list(data.columns)
    print(f"Available features: {available_features}")
    print(f"Expected features: {expected_features}")

    X = data.reindex(columns=expected_features, fill_value=0)
    if 'target' in data.columns:
        y = data['target']
    else:
        y = pd.Series(dtype=float)

    print(f"Final X shape: {X.shape}, y length: {len(y)}")
    return X, y
