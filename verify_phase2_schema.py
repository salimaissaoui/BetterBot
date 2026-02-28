import os, sys
os.environ.setdefault('DB_PASSWORD', 'test')
os.environ.setdefault('DB_HOST', 'localhost')
import sqlalchemy
_real_create_engine = sqlalchemy.create_engine
def _patched_create_engine(url, **kw):
    kw.pop('pool_size', None); kw.pop('max_overflow', None)
    return _real_create_engine('sqlite:///:memory:', **kw)
sqlalchemy.create_engine = _patched_create_engine

import inspect
print("=== Phase 2 Schema and Wiring Tests ===")

from Scripts.database import PositionRegistry, TradeLog, Base
pr_cols = {c.name for c in PositionRegistry.__table__.columns}
required_pr = {'symbol', 'direction', 'entry_price', 'entry_time', 'atr_at_entry',
               'quantity', 'stop_price', 'target_price', 'trailing_high', 'trailing_stop'}
assert required_pr.issubset(pr_cols), f"PositionRegistry missing columns: {required_pr - pr_cols}"
print("PositionRegistry schema PASS:", sorted(pr_cols))

tl_cols = {c.name for c in TradeLog.__table__.columns}
required_tl = {'symbol', 'action', 'exit_reason', 'entry_reason', 'regime_at_decision',
               'sentiment_score', 'prediction_confidence', 'stop_price', 'target_price', 'decision_time'}
assert required_tl.issubset(tl_cols), f"TradeLog missing columns: {required_tl - tl_cols}"
print("TradeLog schema PASS (OBS-01):", sorted(tl_cols))

from sqlalchemy import UniqueConstraint
constraints = [c for c in PositionRegistry.__table__.constraints if isinstance(c, UniqueConstraint)]
unique_cols = {col.name for uc in constraints for col in uc.columns}
assert 'symbol' in unique_cols or any(
    col.unique for col in PositionRegistry.__table__.columns if col.name == 'symbol'
), "PositionRegistry.symbol not unique"
print("PositionRegistry.symbol unique constraint PASS")

from Scripts.config import (HARD_STOP_ATR_MULT, TAKE_PROFIT_ATR_MULT, TRAILING_ATR_MULT,
                             TRAILING_TRIGGER_ATR, DAILY_LOSS_LIMIT_PCT)
assert HARD_STOP_ATR_MULT == 2.0
assert TAKE_PROFIT_ATR_MULT == 4.0
assert TRAILING_ATR_MULT == 1.5
assert TRAILING_TRIGGER_ATR == 1.0
assert DAILY_LOSS_LIMIT_PCT == 0.02
print("Config constants PASS: all five correct defaults")

from Scripts.trade import execute_trade
src = inspect.getsource(execute_trade)
for token in ['check_exits', 'is_circuit_breaker_active', 'register_entry', 'clear_position',
              'exit_reason', 'entry_reason']:
    assert token in src, f"execute_trade missing: {token}"
print("execute_trade wiring PASS: all ExitManager hooks present")

from Scripts import main
scan_src = inspect.getsource(main.hourly_portfolio_scan)
assert 'exit_manager.check_exits' in scan_src, "hourly_portfolio_scan missing check_exits"
assert 'position_return < -0.15' not in scan_src, "hardcoded -15% stop still present"
assert 'position_return > 0.25' not in scan_src, "hardcoded +25% target still present"
print("hourly_portfolio_scan PASS: hardcoded thresholds removed, ExitManager wired")

init_src = inspect.getsource(main.initialize_bot)
assert 'reconcile_from_ibkr' in init_src, "initialize_bot missing reconcile_from_ibkr"
print("initialize_bot PASS: reconcile_from_ibkr wired")

print()
print("=== All schema and wiring assertions PASSED ===")
