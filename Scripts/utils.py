import pytz
from datetime import datetime, time
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import TIMEZONE_NAME, MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE

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
