import logging
import traceback
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import VotingClassifier

# XGBoost & LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import RETRAIN_FREQUENCY
from .indicators import compute_technical_indicators, prepare_features
from .database import get_session, StockData
from .utils import XGBClassifierWrapper, LGBMClassifierWrapper

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains a VotingClassifier ensemble with XGBoost and LightGBM,
    using time-series cross-validation and an expanded hyperparam search.
    """

    if X.empty or y.empty:
        logging.info("No data available for training. Returning None.")
        return None

    try:
        # Optional: Downsampling (if dataset is extremely large)
        MAX_TRAIN_SIZE = 100_000
        if len(X) > MAX_TRAIN_SIZE:
            logging.info(f"Downsampling from {len(X)} to {MAX_TRAIN_SIZE}.")
            X, _, y, _ = train_test_split(
                X, y,
                train_size=MAX_TRAIN_SIZE,
                random_state=42,
                stratify=y
            )

        scoring = make_scorer(accuracy_score)

        # 1. Define base estimators
        xgb = XGBClassifierWrapper(use_label_encoder=False, n_jobs=-1)
        lgb = LGBMClassifierWrapper(n_jobs=-1)

        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('lgb', lgb)],
            voting='soft'
        )

        # 2. Expanded hyperparam distributions
        param_distributions = {
            'xgb__n_estimators': [100, 200],
            'xgb__max_depth': [3, 5, 7],
            'xgb__subsample': [0.6, 0.8],
            'xgb__colsample_bytree': [0.6, 0.8],
            'xgb__learning_rate': [0.01, 0.05, 0.1],
            'xgb__min_child_weight': [1, 3],
            'xgb__gamma': [0, 1],
            'xgb__scale_pos_weight': [1, 2, 3],  # class imbalance parameter

            'lgb__n_estimators': [100, 200],
            'lgb__max_depth': [3, 5, 7],
            'lgb__subsample': [0.6, 0.8],
            'lgb__colsample_bytree': [0.6, 0.8],
            'lgb__learning_rate': [0.01, 0.05, 0.1],
            'lgb__min_child_weight': [1, 3],
            'lgb__scale_pos_weight': [1, 2, 3],  # class imbalance for LGB
        }

        # 3. TimeSeriesSplit ensures forward-only splits
        tscv = TimeSeriesSplit(n_splits=3)

        # 4. Early stopping sets
        # We can't pass eval_set directly to RandomizedSearchCV easily with time series,
        # so we do minimal early_stopping in the final fit instead.
        # We'll do partial early stopping in the final training after search.
        search = RandomizedSearchCV(
            estimator=ensemble,
            param_distributions=param_distributions,
            n_iter=10,               # more search iterations
            scoring=scoring,
            cv=tscv,                 # time-series CV
            n_jobs=-1,
            random_state=42,
            verbose=2,
            error_score='raise'
        )

        logging.info("Starting hyperparameter tuning (TimeSeriesSplit)...")
        search.fit(X, y)

        logging.info(f"Best parameters: {search.best_params_}, Score: {search.best_score_:.4f}")
        best_ensemble = search.best_estimator_

        # 5. Final fit on the entire dataset with early stopping
        # We do a manual partial approach:
        # We'll simply re-fit XGB & LGB inside best_ensemble with early_stopping.
        xgb_best = best_ensemble.named_estimators_['xgb']
        lgb_best = best_ensemble.named_estimators_['lgb']

        # Convert X, y to np arrays for the fit
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        # We’ll slice out a small portion of data for valid (e.g. last 10%) for early stopping:
        val_size = int(len(X) * 0.1)
        train_size = len(X) - val_size
        X_train, X_val = X_np[:train_size], X_np[train_size:]
        y_train, y_val = y_np[:train_size], y_np[train_size:]

        # Fit XGB with early stopping
        xgb_best.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=5,
            eval_metric='logloss',
            verbose=False
        )
        # Fit LGB with early stopping
        lgb_best.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=5,
            eval_metric='logloss',
            verbose=False
        )

        # Now rebuild the ensemble with these updated fitted estimators
        final_ensemble = VotingClassifier(
            estimators=[('xgb', xgb_best), ('lgb', lgb_best)],
            voting='soft'
        )
        # Must manually fit the VotingClassifier to unify it, 
        # but we do a "dummy" fit that won't break the internal fits:
        final_ensemble.fit(X, y)  # effectively just merges the fitted sub-estimators.

        # 6. Save the final model
        joblib.dump(final_ensemble, "ensemble_model.joblib")
        logging.info("Final ensemble model saved successfully.")

        # Evaluate on entire data
        ensemble_pred = final_ensemble.predict_proba(X)[:, 1]
        ensemble_acc = ((ensemble_pred > 0.5).astype(int) == y).mean()
        logging.info(f"Final Ensemble Accuracy (on full dataset): {ensemble_acc:.3f}")

        return final_ensemble

    except Exception as e:
        logging.error(f"Error training model: {e}")
        logging.error(traceback.format_exc())
        return None


def load_existing_model():
    """Loads an existing trained model from joblib file, if available."""
    try:
        ensemble_model = joblib.load("ensemble_model.joblib")
        logging.info("Loaded existing ensemble model from ensemble_model.joblib.")
        return ensemble_model
    except FileNotFoundError:
        logging.info("No existing ensemble model found (ensemble_model.joblib not found).")
        return None
    except Exception as e:
        logging.warning(f"Failed to load existing ensemble model: {e}")
        logging.error(traceback.format_exc())
        return None


def retrain_model():
    """
    Retrieves all distinct symbols from the DB, processes their historical data,
    and retrains the ensemble model using the combined data.
    """
    from .database import get_session, StockData

    # Retrieve distinct symbols and convert to a list of clean strings.
    with get_session() as session:
        symbol_tuples = session.query(StockData.symbol).distinct().all()
        symbols = [str(row[0]).upper().strip() for row in symbol_tuples if row and row[0]]

    if not symbols:
        logging.info("No symbols found in DB to train the model.")
        return None

    combined = []
    for sym in symbols:
        # Use the sanitized symbol (which is now a proper string) for queries.
        with get_session() as session:
            records = (
                session.query(StockData)
                .filter(StockData.symbol == sym)
                .order_by(StockData.timestamp.asc())
                .all()
            )

        if not records:
            logging.info(f"Insufficient data for {sym}. Skipping.")
            continue

        # Build DataFrame from the records
        data = pd.DataFrame([{
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume,
            'timestamp': r.timestamp
        } for r in records])
        data.sort_values('timestamp', inplace=True)

        # Process technical indicators
        data = compute_technical_indicators(data)
        if data.empty:
            logging.info(f"Insufficient processed data for {sym}. Skipping.")
            continue

        # Attach the clean symbol to the DataFrame
        data['symbol'] = sym
        combined.append(data)

    if not combined:
        logging.info("No data available for training the model after processing.")
        return None

    combined_df = pd.concat(combined)

    X, y = prepare_features(combined_df)
    if X.empty or y.empty:
        logging.info("No features or target available for training.")
        return None

    logging.info(f"Training data shape: X={X.shape}, y={y.shape}")
    new_model = train_model(X, y)
    return new_model

