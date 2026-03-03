---
status: investigating
trigger: "Feature mismatch + market data 'Not connected' + advanced model training/prediction pipeline mismatch"
created: 2026-03-02T00:00:00Z
updated: 2026-03-02T00:00:00Z
---

## Current Focus

hypothesis: run_advanced_training() trains on basic ~21 features but create_advanced_features() generates 134 features for prediction — mismatch causes sub-model failures. reqMktData "Not connected" may be clientId conflict or wrong IB instance.
test: Read advanced_modeling.py, trade.py, features.py, main.py to verify training vs prediction feature pipelines
expecting: Confirmed: run_advanced_training() uses prepare_features() (21 cols), execute_trade() uses create_advanced_features() (134 cols)
next_action: Read all key files

## Symptoms

expected: Bot trains advanced model using 134 advanced features, gets live prices for symbols, places trades with valid prices
actual:
  1. Advanced model always returns 0.500 — sub-models fail with "feature names should match those at fit time" (unseen: ad_line, after_hours, bearish_engulfing, etc.; missing: atr, bb_h, bb_l, bb_m, etc.)
  2. reqMktData returns "Not connected" during trading cycle for all symbols — entry_price=None, all trades skipped
  3. ~115 unintended short positions placed after hours from prior buggy run (paper account only, no code fix)
errors:
  - "feature names should match those that were passed during fit. Features seen at fit time, yet now missing: [atr, bb_h, bb_l, bb_m, bb_p, bb_width, ...]. Features seen now, yet not present at fit time: [ad_line, after_hours, bearish_engulfing, beta, ...]"
  - "Error getting market data for MMM: Not connected"
reproduction: Run bot during market hours — feature mismatch on every prediction; "Not connected" on every reqMktData call
started: Feature mismatch after advanced_ensemble_model.pkl deleted and retrained. "Not connected" pre-existing.

## Eliminated

- hypothesis: "Not connected" caused by clientId conflict between IB instances
  evidence: trade.py line 32 creates its own `ib = IB()` (never connected). main.py creates a SEPARATE `ib = IB()` and connects it (clientId=1). execute_trade() calls `ib.reqMktData()` on the trade.py ib instance which is NEVER connected. This is confirmed root cause, not a clientId conflict per se.
  timestamp: 2026-03-02

- hypothesis: "Not connected" caused by market data type not set on the trading ib object
  evidence: main.py calls `ib.reqMarketDataType(3)` on main.py's ib instance, not trade.py's ib. But moot — the trade.py ib is never even connected.
  timestamp: 2026-03-02

## Evidence

- timestamp: 2026-03-02
  checked: advanced_modeling.py run_advanced_training() lines 799-874
  found: Training uses compute_technical_indicators() (line 827) + prepare_features() (line 839). prepare_features() in indicators.py returns exactly 21 features: ma_short, ma_long, ma_ratio, rsi, return, macd, macd_signal, macd_diff, bb_h, bb_l, bb_m, bb_w, bb_p, atr, doji, obv, obv_ma, stoch, stoch_signal, rolling_ret_std, rolling_vol_avg
  implication: The saved advanced_ensemble_model.pkl was trained on 21 basic features

- timestamp: 2026-03-02
  checked: trade.py on_bar() lines 608-642
  found: During prediction, on_bar() calls create_advanced_features(df, symbol) from advanced_features.py, which generates ~134 columns (basic technicals + advanced technicals + microstructure + temporal + statistical + momentum + volatility + pattern + regime + cross-asset features including: ad_line, after_hours, bearish_engulfing, beta, body_ratio, etc.)
  implication: The advanced model receives 134-column feature matrix but was trained on 21 columns — confirmed mismatch. Sub-models throw sklearn warning, all fail, fallback returns 0.5.

- timestamp: 2026-03-02
  checked: trade.py line 32, execute_trade() lines 337-343, main.py lines 27, 752-755
  found: trade.py has its own module-level `ib = IB()` (line 32) that is NEVER connected. main.py has a different `ib = IB()` (line 27) that does `ib.connect(IB_HOST, IB_PORT, clientId=IB_CLIENT_ID)` (line 752) and `ib.reqMarketDataType(3)` (line 755). execute_trade() calls `ib.reqMktData()` using trade.py's own disconnected ib instance.
  implication: Every reqMktData call in execute_trade() uses a disconnected IB instance → "Not connected" error for all symbols.

- timestamp: 2026-03-02
  checked: trade.py ensure_ib_connected() lines 41-45
  found: ensure_ib_connected() exists and would connect trade.py's ib using IB_CLIENT_ID_TRADE=2. BUT execute_trade() does NOT call ensure_ib_connected() before reqMktData calls. The function is only called in get_current_position(), close_position(), short_position(), submit_ml_sized_order(), submit_notional_order_with_stop_loss(). All reqMktData blocks in execute_trade() are bare try/except without calling ensure_ib_connected() first.
  implication: Fix is twofold: (1) call ensure_ib_connected() before reqMktData, OR (2) pass main's ib into trade functions. Better: add ensure_ib_connected() call at top of each reqMktData block in execute_trade(), or at the top of execute_trade() itself.

- timestamp: 2026-03-02
  checked: main.py line 755, trade.py reqMarketDataType usage
  found: main.py calls ib.reqMarketDataType(3) on its own ib object. trade.py's ib object (if reconnected via ensure_ib_connected) will NOT have reqMarketDataType(3) set — it will default to live data type 1, which requires market data subscription. Paper accounts may not have live subscription during market hours.
  implication: After fixing the disconnected ib issue, must also call ib.reqMarketDataType(3) on trade.py's ib after connecting, to enable delayed data.

- timestamp: 2026-03-02
  checked: bot.log tail
  found: Bot is currently in data fetch phase (loading ~500 symbols), has not yet entered trading loop. This is safe — fixes can be applied now before trading begins.
  implication: No race condition with live trading. Can safely edit trade.py and advanced_modeling.py now.

## Resolution

root_cause: |
  ISSUE 1 (Feature mismatch): run_advanced_training() in advanced_modeling.py trains on basic 21 features
  (via compute_technical_indicators + prepare_features), but on_bar() in trade.py predicts using
  create_advanced_features() which produces ~134 features. The saved model's sklearn sub-models all fail
  with "feature names should match" error, causing ensemble to fall back to 0.5.

  ISSUE 2 (Not connected): trade.py has its own module-level `ib = IB()` instance that is NEVER
  connected. main.py connects a completely separate `ib` instance. execute_trade() calls
  `ib.reqMktData()` on the disconnected trade.py instance. Additionally, ensure_ib_connected() is
  defined but never called within execute_trade()'s reqMktData blocks. After reconnecting,
  reqMarketDataType(3) must also be set on the trade ib instance.

fix:
  ISSUE 1: Modify run_advanced_training() in advanced_modeling.py to use create_advanced_features()
  instead of compute_technical_indicators()/prepare_features(). Need to import create_advanced_features
  from advanced_features. Also need to add target column computation since create_advanced_features
  doesn't produce it. Then delete the stale .pkl and retrain.

  ISSUE 2: Add ensure_ib_connected() call at the top of execute_trade() (before the loop), and add
  ib.reqMarketDataType(3) call inside ensure_ib_connected() (after connect, before returning).

verification:
files_changed:
  - Scripts/advanced_modeling.py
  - Scripts/trade.py
