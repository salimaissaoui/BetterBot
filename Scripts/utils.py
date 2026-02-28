import pytz
import logging
import yfinance as yf
from datetime import datetime, time, timedelta
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import TIMEZONE_NAME, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE

# Global cache for VIX [RISK-02]
_vix_cache = {
    'value': 20.0,  # Default fallback
    'timestamp': datetime.min
}

def get_current_vix() -> float:
    """Fetch current VIX level from yfinance with 15-minute TTL cache."""
    global _vix_cache
    now = datetime.now()
    
    if now - _vix_cache['timestamp'] < timedelta(minutes=15):
        return _vix_cache['value']
    
    try:
        ticker = yf.Ticker("^VIX")
        data = ticker.history(period="1d")
        if not data.empty:
            vix = float(data['Close'].iloc[-1])
            _vix_cache['value'] = vix
            _vix_cache['timestamp'] = now
            logging.debug(f"VIX cache updated: {vix:.2f}")
            return vix
    except Exception as e:
        logging.warning(f"Failed to fetch VIX: {e}. Using cached/fallback: {_vix_cache['value']:.2f}")
    
    return _vix_cache['value']

class XGBClassifierWrapper(XGBClassifier):
    """Wrapper to ensure correct scikit-learn tags for XGBClassifier."""
    def __sklearn_tags__(self):
        return {
            "non_deterministic": True,
            "allow_nan": True,
            "X_types": ["2darray"],
            "requires_y": True,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "no_validation": False,
            "stateless": False,
            "poor_score": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "pairwise": False,
        }


class LGBMClassifierWrapper(LGBMClassifier):
    """Wrapper to ensure correct scikit-learn tags for LGBMClassifier."""
    def __sklearn_tags__(self):
        return {
            "non_deterministic": True,
            "allow_nan": True,
            "X_types": ["2darray"],
            "requires_y": True,
            "binary_only": False,
            "multilabel": False,
            "multioutput": False,
            "no_validation": False,
            "stateless": False,
            "poor_score": False,
            "requires_positive_X": False,
            "requires_positive_y": False,
            "pairwise": False,
        }


def is_market_open() -> bool:
    """Checks whether the current time is within standard US market hours."""
    tz = pytz.timezone(TIMEZONE_NAME)
    now = datetime.now(tz)
    current_time = now.time()

    # Check if today is a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False

    # Check if current time is within market hours
    market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    return market_open <= current_time <= market_close
