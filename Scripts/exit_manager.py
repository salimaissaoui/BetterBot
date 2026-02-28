import logging
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict

from .database import get_session
from .config import (
    HARD_STOP_ATR_MULT,
    TAKE_PROFIT_ATR_MULT,
    TRAILING_ATR_MULT,
    TRAILING_TRIGGER_ATR,
    DAILY_LOSS_LIMIT_PCT,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


@dataclass
class EntryRecord:
    symbol: str
    direction: str           # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    atr_at_entry: float
    quantity: int
    stop_price: float        # EXIT-01: entry_price - (HARD_STOP_ATR_MULT * atr)  [long]
    target_price: float      # EXIT-02: entry_price + (TAKE_PROFIT_ATR_MULT * atr) [long]
    trailing_high: float     # highest price seen since entry (long) / lowest (short)
    trailing_stop: Optional[float]  # None until 1x ATR profit reached (EXIT-03)


class ExitManager:
    """
    Owns the position registry and enforces all exit rules.

    Usage pattern per bar (in execute_trade):
      1. exit_check = exit_manager.check_exits(symbol, current_price)
      2. if exit_check: close, clear, log, continue
      3. if exit_manager.is_circuit_breaker_active(): skip entry
      4. on new entry: exit_manager.register_entry(...)
      5. on signal-driven exit: exit_manager.clear_position(...)
    """

    def __init__(self):
        self.positions: Dict[str, EntryRecord] = {}
        self._daily_pnl: float = 0.0
        self._circuit_breaker_active: bool = False
        self._circuit_breaker_date: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_entry(self, symbol: str, direction: str, entry_price: float,
                       atr: float, quantity: int) -> EntryRecord:
        """
        Called immediately after a BUY/SHORT order fills.
        Computes stop and target, persists to DB, adds to in-memory registry.

        EXIT-01: stop_price = entry_price - (HARD_STOP_ATR_MULT * atr)  [long]
                              entry_price + (HARD_STOP_ATR_MULT * atr)  [short]
        EXIT-02: target_price = entry_price + (TAKE_PROFIT_ATR_MULT * atr) [long]
                                entry_price - (TAKE_PROFIT_ATR_MULT * atr) [short]

        ATR NaN guard: if atr is None, NaN, or <= 0, use 0.5% of entry_price as fallback.
        """
        if atr is None or pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.005
            logging.warning(
                f"[ExitManager] ATR unavailable for {symbol}, using fallback: {atr:.4f}"
            )

        if direction == 'long':
            stop_price   = entry_price - (HARD_STOP_ATR_MULT * atr)
            target_price = entry_price + (TAKE_PROFIT_ATR_MULT * atr)
            trailing_high = entry_price
        else:  # short — mirror formulae
            stop_price   = entry_price + (HARD_STOP_ATR_MULT * atr)
            target_price = entry_price - (TAKE_PROFIT_ATR_MULT * atr)
            trailing_high = entry_price  # tracks low-water mark for shorts

        record = EntryRecord(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(),
            atr_at_entry=atr,
            quantity=quantity,
            stop_price=stop_price,
            target_price=target_price,
            trailing_high=trailing_high,
            trailing_stop=None,
        )
        self.positions[symbol] = record
        self._persist_to_db(record)
        logging.info(
            f"[ExitManager] Registered {direction} entry for {symbol}: "
            f"entry={entry_price:.2f} stop={stop_price:.2f} target={target_price:.2f} atr={atr:.4f}"
        )
        return record

    def check_exits(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Returns exit reason string ('HARD_STOP_LOSS', 'TRAILING_STOP', 'TAKE_PROFIT') or None.
        MUST be called BEFORE new entry evaluation for this symbol (exits-before-entries rule).

        Updates trailing high-water mark and activates trailing stop per EXIT-03.
        Uses atr_at_entry (frozen at entry) for all multiplier math — not current ATR.
        """
        if symbol not in self.positions:
            return None
        rec = self.positions[symbol]

        if rec.direction == 'long':
            # Update high-water mark
            if current_price > rec.trailing_high:
                rec.trailing_high = current_price
                self._update_db_trailing(symbol, rec.trailing_high, rec.trailing_stop)

            # Activate trailing stop once profit >= 1x ATR (EXIT-03)
            profit = current_price - rec.entry_price
            if rec.trailing_stop is None and profit >= (TRAILING_TRIGGER_ATR * rec.atr_at_entry):
                rec.trailing_stop = rec.trailing_high - (TRAILING_ATR_MULT * rec.atr_at_entry)
                self._update_db_trailing(symbol, rec.trailing_high, rec.trailing_stop)
                logging.info(
                    f"[ExitManager] Trailing stop activated for {symbol}: "
                    f"trail={rec.trailing_stop:.2f} (high={rec.trailing_high:.2f})"
                )

            # Ratchet trailing stop upward
            if rec.trailing_stop is not None:
                new_trail = rec.trailing_high - (TRAILING_ATR_MULT * rec.atr_at_entry)
                if new_trail > rec.trailing_stop:
                    rec.trailing_stop = new_trail
                    self._update_db_trailing(symbol, rec.trailing_high, rec.trailing_stop)

            # Check exit conditions — hard stop takes priority
            if current_price <= rec.stop_price:
                return 'HARD_STOP_LOSS'
            if rec.trailing_stop is not None and current_price <= rec.trailing_stop:
                return 'TRAILING_STOP'
            if current_price >= rec.target_price:
                return 'TAKE_PROFIT'

        else:  # short — mirror direction
            # Low-water mark for shorts
            if current_price < rec.trailing_high:
                rec.trailing_high = current_price
                self._update_db_trailing(symbol, rec.trailing_high, rec.trailing_stop)

            profit = rec.entry_price - current_price
            if rec.trailing_stop is None and profit >= (TRAILING_TRIGGER_ATR * rec.atr_at_entry):
                rec.trailing_stop = rec.trailing_high + (TRAILING_ATR_MULT * rec.atr_at_entry)
                self._update_db_trailing(symbol, rec.trailing_high, rec.trailing_stop)
                logging.info(
                    f"[ExitManager] Trailing stop activated for short {symbol}: "
                    f"trail={rec.trailing_stop:.2f}"
                )

            if rec.trailing_stop is not None:
                new_trail = rec.trailing_high + (TRAILING_ATR_MULT * rec.atr_at_entry)
                if new_trail < rec.trailing_stop:
                    rec.trailing_stop = new_trail
                    self._update_db_trailing(symbol, rec.trailing_high, rec.trailing_stop)

            if current_price >= rec.stop_price:
                return 'HARD_STOP_LOSS'
            if rec.trailing_stop is not None and current_price >= rec.trailing_stop:
                return 'TRAILING_STOP'
            if current_price <= rec.target_price:
                return 'TAKE_PROFIT'

        return None

    def clear_position(self, symbol: str) -> None:
        """Called after close_position() executes. Removes from registry and DB."""
        if symbol in self.positions:
            del self.positions[symbol]
        self._delete_from_db(symbol)
        logging.info(f"[ExitManager] Cleared position registry for {symbol}")

    def is_circuit_breaker_active(self) -> bool:
        """
        Returns True if daily P&L has breached -DAILY_LOSS_LIMIT_PCT of NAV (EXIT-05).
        Automatically resets on a new calendar day.
        """
        today = datetime.now().strftime('%Y-%m-%d')
        if self._circuit_breaker_date != today:
            self._circuit_breaker_active = False
            self._daily_pnl = 0.0
            self._circuit_breaker_date = today
        return self._circuit_breaker_active

    def record_trade_pnl(self, pnl: float, account_nav: float) -> None:
        """
        Update daily P&L accumulator and trigger circuit breaker if limit breached.
        Call this immediately after a position closes with the realized P&L and current NAV.
        """
        # Ensure the date bucket is initialised for today before accumulating P&L.
        # This prevents is_circuit_breaker_active() from resetting a breaker that was
        # triggered within the same calendar day by record_trade_pnl().
        today = datetime.now().strftime('%Y-%m-%d')
        if self._circuit_breaker_date != today:
            self._circuit_breaker_active = False
            self._daily_pnl = 0.0
            self._circuit_breaker_date = today

        self._daily_pnl += pnl
        if account_nav > 0 and (self._daily_pnl / account_nav) < -DAILY_LOSS_LIMIT_PCT:
            self._circuit_breaker_active = True
            logging.warning(
                f"[ExitManager] Circuit breaker ACTIVATED: daily_pnl={self._daily_pnl:.2f} "
                f"({self._daily_pnl / account_nav:.2%} of NAV={account_nav:.2f})"
            )

    def reconcile_from_ibkr(self, ib_positions, account_nav: float) -> None:
        """
        Startup reconciliation (EXIT-04). Call in initialize_bot() after IBKR connect.

        Steps:
        1. Load all rows from position_registry DB table into self.positions
        2. Cross-check against live ib_positions list
        3. For each IBKR position not in DB: create_stub record with conservative stops
        4. For each DB record where IBKR shows no position: delete stale record
        Returns: populates self.positions; caller should sync active_positions dict.
        """
        # Step 1: load DB registry into memory
        self._load_from_db()
        logging.info(f"[ExitManager] Loaded {len(self.positions)} positions from DB registry")

        # Build set of symbols IBKR currently holds
        ibkr_symbols = {}
        for pos in ib_positions:
            if pos.position != 0:
                sym = pos.contract.symbol
                ibkr_symbols[sym] = pos

        # Step 3: IBKR position not in DB — create conservative stub
        for sym, pos in ibkr_symbols.items():
            if sym not in self.positions:
                avg_cost = pos.avgCost if pos.avgCost > 0 else 1.0
                direction = 'long' if pos.position > 0 else 'short'
                stub_atr = avg_cost * 0.005  # 0.5% fallback ATR
                logging.warning(
                    f"[ExitManager] {sym} found in IBKR but not in DB registry — "
                    f"creating stub record with conservative stops"
                )
                self.register_entry(
                    symbol=sym,
                    direction=direction,
                    entry_price=avg_cost,
                    atr=stub_atr,
                    quantity=int(abs(pos.position)),
                )

        # Step 4: DB record with no IBKR position — remove stale record
        stale = [sym for sym in list(self.positions.keys()) if sym not in ibkr_symbols]
        for sym in stale:
            logging.warning(
                f"[ExitManager] {sym} in DB registry but not in IBKR — removing stale record"
            )
            self.clear_position(sym)

        logging.info(
            f"[ExitManager] Reconciliation complete: {len(self.positions)} positions tracked"
        )

    # ------------------------------------------------------------------
    # Private DB helpers
    # ------------------------------------------------------------------

    def _persist_to_db(self, record: EntryRecord) -> None:
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from .database import PositionRegistry
        try:
            with get_session() as session:
                stmt = pg_insert(PositionRegistry).values(
                    symbol=record.symbol,
                    direction=record.direction,
                    entry_price=record.entry_price,
                    entry_time=record.entry_time,
                    atr_at_entry=record.atr_at_entry,
                    quantity=record.quantity,
                    stop_price=record.stop_price,
                    target_price=record.target_price,
                    trailing_high=record.trailing_high,
                    trailing_stop=record.trailing_stop,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['symbol'],
                    set_={
                        'stop_price':    stmt.excluded.stop_price,
                        'target_price':  stmt.excluded.target_price,
                        'trailing_high': stmt.excluded.trailing_high,
                        'trailing_stop': stmt.excluded.trailing_stop,
                    }
                )
                session.execute(stmt)
        except Exception as e:
            logging.error(f"[ExitManager] Failed to persist registry for {record.symbol}: {e}")

    def _update_db_trailing(self, symbol: str, trailing_high: float,
                            trailing_stop: Optional[float]) -> None:
        from .database import PositionRegistry
        try:
            with get_session() as session:
                row = session.query(PositionRegistry).filter_by(symbol=symbol).first()
                if row:
                    row.trailing_high = trailing_high
                    row.trailing_stop = trailing_stop
        except Exception as e:
            logging.warning(f"[ExitManager] Failed to update trailing for {symbol}: {e}")

    def _delete_from_db(self, symbol: str) -> None:
        from .database import PositionRegistry
        try:
            with get_session() as session:
                session.query(PositionRegistry).filter_by(symbol=symbol).delete()
        except Exception as e:
            logging.warning(f"[ExitManager] Failed to delete registry row for {symbol}: {e}")

    def _load_from_db(self) -> None:
        from .database import PositionRegistry
        try:
            with get_session() as session:
                rows = session.query(PositionRegistry).all()
                for row in rows:
                    self.positions[row.symbol] = EntryRecord(
                        symbol=row.symbol,
                        direction=row.direction,
                        entry_price=row.entry_price,
                        entry_time=row.entry_time,
                        atr_at_entry=row.atr_at_entry,
                        quantity=row.quantity,
                        stop_price=row.stop_price,
                        target_price=row.target_price,
                        trailing_high=row.trailing_high,
                        trailing_stop=row.trailing_stop,
                    )
        except Exception as e:
            logging.error(f"[ExitManager] Failed to load registry from DB: {e}")


# Module-level singleton — import this instance everywhere
exit_manager = ExitManager()
