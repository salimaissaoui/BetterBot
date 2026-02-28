import pandas as pd
import numpy as np
from Scripts.advanced_features import MarketRegimeDetector
import os

def generate_dummy_data(n_days=500):
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_days)
    
    # Generate 4 different regimes
    data_list = []
    
    # Regime 1: Bullish
    close = 100
    for _ in range(n_days // 4):
        close *= (1 + np.random.normal(0.001, 0.01))
        data_list.append({'close': close, 'high': close*1.01, 'low': close*0.99, 'volume': 1000000})
        
    # Regime 2: Bearish
    for _ in range(n_days // 4):
        close *= (1 + np.random.normal(-0.001, 0.015))
        data_list.append({'close': close, 'high': close*1.01, 'low': close*0.99, 'volume': 1200000})
        
    # Regime 3: Sideways
    for _ in range(n_days // 4):
        close *= (1 + np.random.normal(0, 0.005))
        data_list.append({'close': close, 'high': close*1.005, 'low': close*0.995, 'volume': 800000})
        
    # Regime 4: Volatile
    for _ in range(n_days - 3*(n_days//4)):
        close *= (1 + np.random.normal(0, 0.03))
        data_list.append({'close': close, 'high': close*1.03, 'low': close*0.97, 'volume': 2000000})
        
    df = pd.DataFrame(data_list, index=dates)
    return df

print("=== Phase 3-01: Regime Detector Persistence Test ===")

# 1. Generate data
df = generate_dummy_data()
market_data = {"TEST": df}

# 2. Fit detector
detector1 = MarketRegimeDetector()
print("Fitting detector 1...")
success = detector1.fit_regime_detector(market_data)
assert success, "Failed to fit detector 1"

# 3. Verify file exists (save_model called in fit_regime_detector)
model_path = "regime_model.joblib"
assert os.path.exists(model_path), f"Model file {model_path} not found"
print(f"PASS: {model_path} created")

# 4. Predict with detector 1
# Need at least lookback_window (252) data points
regime1 = detector1.predict_regime(df.tail(300))
print(f"Detector 1 prediction: {regime1}")
assert regime1 != 'unknown', "Detector 1 returned unknown regime"

# 5. Load into detector 2
detector2 = MarketRegimeDetector()
print("Loading into detector 2...")
load_success = detector2.load_model(model_path)
assert load_success, "Failed to load model into detector 2"

# 6. Verify identical prediction
regime2 = detector2.predict_regime(df.tail(300))
print(f"Detector 2 prediction: {regime2}")
assert regime1 == regime2, f"Predictions mismatch! {regime1} != {regime2}"
print("PASS: Predictions match")

# 7. Cleanup
# os.remove(model_path) # Keep it for bot use

print("=== All Phase 3-01 tests PASSED ===")
