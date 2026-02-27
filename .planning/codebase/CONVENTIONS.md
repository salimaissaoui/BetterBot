# Coding Conventions

**Analysis Date:** 2026-02-27

## Naming Patterns

**Files:**
- Lowercase with underscores: `main.py`, `data_fetch.py`, `modeling.py`
- Grouped by functional domain: data fetching, modeling, trading, database, indicators
- Single responsibility pattern: each module focuses on one area

**Functions:**
- Lowercase with underscores: `fetch_historical_data()`, `compute_technical_indicators()`, `initialize_bot()`
- Helper functions prefixed with underscore: `_calculate_support_resistance()`, `_add_basic_technicals()`
- Getter functions use plain names or `get_*`: `get_session()`, `get_current_position()`, `get_account_info()`
- Setter/insert functions use `set_*` or action verbs: `insert_stock_data()`, `insert_historical_data()`

**Variables:**
- Lowercase with underscores for module-level: `active_positions`, `latest_indicators_dict`, `polling_symbols`
- Global variables marked with comment (see `Scripts/main.py` line 30): `model = None  # Global model object`
- UPPERCASE for constants: `IB_HOST`, `DB_USER`, `MODEL_LOOKBACK`, `ATR_WINDOW`, `NOTIONAL`
- Descriptive names for data collections: `symbol_metrics`, `regime_features`, `combined_df`

**Types:**
- Class names use PascalCase: `XGBClassifierWrapper`, `LGBMClassifierWrapper`, `MarketRegimeDetector`, `MLPositionSizer`
- Dataframe variables often suffixed with `_df`: `combined_df`, `data_df`, `recent_data`
- Dictionary/collection variables suffixed with appropriate type: `models`, `allocations`, `positions`

## Code Style

**Formatting:**
- No linting or formatting configuration detected
- Inconsistent spacing patterns observed (variable standards)
- Line length appears to vary (80-100+ characters)
- Indentation: 4 spaces (Python standard)

**Import Organization:**
- Standard library imports first: `import logging`, `import asyncio`, `from datetime import ...`
- Third-party packages next: `import pandas as pd`, `import numpy as np`, `from sklearn...`
- Local imports last: `from .config import ...`, `from .database import ...`
- Each category separated by blank line

**Path Aliases:**
- Relative imports used: `from .config import ...`, `from .utils import ...`
- Package structure: `Scripts/` is the main package directory

## Error Handling

**Patterns:**
- Try-except blocks used extensively for external API calls and I/O operations
- Most exceptions logged, not re-raised: `except Exception as e: logging.error(f"Error: {e}")`
- Some critical errors propagate with re-raise: `except Exception as e: ... raise` in database connection
- Graceful fallback values provided when operations fail: `except Exception: return None`
- Traceback logging for debugging: `logging.error(traceback.format_exc())`

**Examples from codebase:**
- `Scripts/database.py` (line 89-134): Defensive insert with specific IntegrityError handling
- `Scripts/modeling.py` (line 175-180): Try-except-log pattern for model loading
- `Scripts/data_fetch.py` (line 51-95): Retry mechanism with `max_retries` parameter
- `Scripts/main.py` (line 114+): Nested try-catch for complex workflows

## Logging

**Framework:** `logging` standard library

**Configuration:**
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
```
Pattern repeated across most modules (`Scripts/config.py`, `Scripts/database.py`, `Scripts/modeling.py`, etc.)

**Log Levels Used:**
- `logging.info()`: Normal operation flow, important milestones
- `logging.warning()`: Recoverable issues, missing optional config
- `logging.error()`: Error conditions but process continues
- `logging.debug()`: Detailed info (used sparingly, see `Scripts/indicators.py` line 31)

**Logging Examples:**
- `logging.info(f"Loaded {len(symbols)} valid symbols from {csv_path}")` - Informational
- `logging.error(f"No symbols fetched. Cannot proceed.")` - Critical error
- `logging.warning(f"Could not qualify {symbol}: {e}")` - Recoverable issue
- Context-rich messages with f-strings: `logging.info(f"Inserted data for {symbol} at {timestamp_converted}.")`

## Comments

**When to Comment:**
- Module docstrings present: `Scripts/__init__.py` has file-level docstring
- Function docstrings used for complex operations: `Scripts/position_sizing.py` class methods documented
- Inline comments sparse; code intended to be self-documenting
- TODO/FIXME comments not systematically used

**JSDoc/TSDoc:**
- Python project - docstrings used instead
- Triple-quoted docstrings for functions/classes: `"""Docstring here."""`
- Variable documentation via comments limited

**Examples:**
```python
# Scripts/database.py (line 62)
Base = declarative_base()

class StockData(Base):
    __tablename__ = 'stock_data'
    # Column definitions follow

# Scripts/position_sizing.py (line 23-24)
class MLPositionSizer:
    """
    Machine Learning based position sizing system that considers:
    - Current capital/portfolio value
    - Market volatility
    ...
    """
```

## Function Design

**Size:**
- Functions typically 20-100 lines
- Longer functions in `Scripts/main.py` (on_bar, scan_positions up to 200+ lines)
- Shorter utility functions in `Scripts/utils.py` (40-60 lines)
- Model training functions complex: `Scripts/modeling.py` train_model ~150 lines

**Parameters:**
- Type hints used in some functions: `def fetch_historical_data(ib: IB, symbol: str, ...) -> pd.DataFrame:`
- Optional parameters use defaults: `max_retries: int = 3`, `lookback_window: int = 252`
- Mixed approach: some functions heavily typed, others not

**Return Values:**
- Explicit returns preferred (not implicit None)
- Often return None on error: `return None`
- Dataframes returned from data processing: `return pd.DataFrame()`
- Empty collections on no data: `return []`, `return {}`

## Module Design

**Exports:**
- No explicit `__all__` used
- Barrel files not used
- Public functions (no underscore) exported implicitly

**Module Purposes:**
- `Scripts/config.py`: Configuration constants
- `Scripts/database.py`: ORM models and session management
- `Scripts/modeling.py`: Model training and loading
- `Scripts/indicators.py`: Technical indicator calculations
- `Scripts/data_fetch.py`: IBKR API data fetching
- `Scripts/trade.py`: Order execution and position management
- `Scripts/utils.py`: Market hours checking, classifier wrappers
- `Scripts/position_sizing.py`: ML-based position sizing logic
- `Scripts/advanced_features.py`: Advanced feature engineering
- `Scripts/advanced_modeling.py`: Deep learning and RL models
- `Scripts/main.py`: Bot orchestration and scheduling

## Pattern Examples

**Config Access:**
```python
# Scripts/main.py (line 8)
from .config import IB_HOST, IB_PORT, RETRAIN_FREQUENCY, IB_CLIENT_ID
```

**Context Manager Pattern:**
```python
# Scripts/database.py (line 69-76)
@contextmanager
def get_session():
    session = Session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Session rollback due to error: {e}")
        raise
    finally:
        session.close()
```

**Class with Private Methods:**
```python
# Scripts/advanced_features.py (line 247+)
class AdvancedFeatureEngine:
    def __init__(self):
        ...

    def create_advanced_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        ...

    def _add_basic_technicals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Private method"""
        ...
```

---

*Convention analysis: 2026-02-27*
