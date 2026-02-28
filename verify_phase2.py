import os, sys
# Provide dummy DB credentials so config.py can import without error.
# database.py will attempt create_engine at import time — patch it to use SQLite in-memory.
os.environ.setdefault('DB_PASSWORD', 'test')
os.environ.setdefault('DB_HOST', 'localhost')

# Monkey-patch SQLAlchemy engine creation before database.py loads
import unittest.mock as mock

# We need to patch create_engine to use sqlite in-memory
import sqlalchemy
_real_create_engine = sqlalchemy.create_engine

def _patched_create_engine(url, **kw):
    kw.pop('pool_size', None)
    kw.pop('max_overflow', None)
    return _real_create_engine('sqlite:///:memory:', **kw)

sqlalchemy.create_engine = _patched_create_engine

from Scripts.exit_manager import ExitManager, EntryRecord
from Scripts.config import (
    HARD_STOP_ATR_MULT, TAKE_PROFIT_ATR_MULT,
    TRAILING_ATR_MULT, TRAILING_TRIGGER_ATR, DAILY_LOSS_LIMIT_PCT
)

sqlalchemy.create_engine = _real_create_engine  # restore

print("=== Phase 2 ExitManager Logic Tests ===")

em = ExitManager()
em.register_entry('TEST', 'long', 100.0, 2.0, 10)
rec = em.positions['TEST']
assert rec.stop_price == 96.0, f"EXIT-01 FAIL: stop={rec.stop_price}, expected 96.0"
print("EXIT-01 PASS: stop_price = entry - 2*ATR = 96.0")
assert rec.target_price == 108.0, f"EXIT-02 FAIL: target={rec.target_price}, expected 108.0"
print("EXIT-02 PASS: target_price = entry + 4*ATR = 108.0")
em2 = ExitManager()
em2.register_entry('SHORT_TEST', 'short', 100.0, 2.0, 10)
rec2 = em2.positions['SHORT_TEST']
assert rec2.stop_price == 104.0, f"EXIT-01 short FAIL: stop={rec2.stop_price}, expected 104.0"
print("EXIT-01 short PASS: stop_price = entry + 2*ATR = 104.0")
em3 = ExitManager()
em3.register_entry('TRAIL_TEST', 'long', 100.0, 2.0, 10)
result = em3.check_exits('TRAIL_TEST', 101.5)
assert em3.positions['TRAIL_TEST'].trailing_stop is None, "EXIT-03 FAIL: trailing activated too early"
print("EXIT-03 PASS: trailing stop not activated at 1.5 profit (threshold is 2.0)")
result = em3.check_exits('TRAIL_TEST', 102.5)
assert em3.positions['TRAIL_TEST'].trailing_stop is not None, "EXIT-03 FAIL: trailing not activated at 2.5 profit"
trail = em3.positions['TRAIL_TEST'].trailing_stop
expected_trail = 102.5 - (1.5 * 2.0)
assert trail == expected_trail, f"EXIT-03 FAIL: trail={trail}, expected {expected_trail}"
print(f"EXIT-03 PASS: trailing stop activated at {trail} (high=102.5 - 1.5*2.0=99.5)")
result = em3.check_exits('TRAIL_TEST', 99.0)
assert result == 'TRAILING_STOP', f"EXIT-03 FAIL: expected TRAILING_STOP, got {result}"
print("EXIT-03 PASS: TRAILING_STOP triggered when price falls below trailing stop")
em4 = ExitManager()
em4.register_entry('STOP_TEST', 'long', 100.0, 2.0, 10)
result = em4.check_exits('STOP_TEST', 95.9)
assert result == 'HARD_STOP_LOSS', f"EXIT-01 FAIL: expected HARD_STOP_LOSS, got {result}"
print("EXIT-01 PASS: HARD_STOP_LOSS triggered when price falls to stop level")
em5 = ExitManager()
em5.register_entry('TP_TEST', 'long', 100.0, 2.0, 10)
result = em5.check_exits('TP_TEST', 108.1)
assert result == 'TAKE_PROFIT', f"EXIT-02 FAIL: expected TAKE_PROFIT, got {result}"
print("EXIT-02 PASS: TAKE_PROFIT triggered at target price")
em6 = ExitManager()
assert not em6.is_circuit_breaker_active(), "EXIT-05 FAIL: circuit breaker active on fresh instance"
em6.record_trade_pnl(-200.0, 9000.0)
assert em6.is_circuit_breaker_active(), "EXIT-05 FAIL: circuit breaker did not activate at -2.22%"
print("EXIT-05 PASS: circuit breaker activates when daily P&L breaches -2% NAV")
em7 = ExitManager()
em7.register_entry('CB_TEST', 'long', 100.0, 2.0, 10)
em7.record_trade_pnl(-300.0, 5000.0)
assert em7.is_circuit_breaker_active()
result = em7.check_exits('CB_TEST', 95.9)
assert result == 'HARD_STOP_LOSS', f"EXIT-05 FAIL: exits suppressed by circuit breaker, got {result}"
print("EXIT-05 PASS: check_exits unaffected by circuit breaker state")
em6._circuit_breaker_date = '2000-01-01'
assert not em6.is_circuit_breaker_active(), "EXIT-05 FAIL: circuit breaker did not reset on new day"
print("EXIT-05 PASS: circuit breaker resets on new calendar day")
em8 = ExitManager()
em8.register_entry('NAN_TEST', 'long', 100.0, float('nan'), 5)
rec8 = em8.positions['NAN_TEST']
assert rec8.stop_price is not None and not (rec8.stop_price != rec8.stop_price), f"ATR NaN guard FAIL: stop={rec8.stop_price}"
print(f"ATR NaN guard PASS: fallback atr=0.5%, stop={rec8.stop_price:.4f}")
print()
print("=== All EXIT logic assertions PASSED ===")
