import logging
import traceback
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import VotingClassifier

# XGBoost & LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Local imports (assuming they exist in your codebase)
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
    using a smaller hyperparameter search space, fewer iterations,
    optional downsampling, and early stopping to speed up training.
    """

    if X.empty or y.empty:
        logging.info("No data available for training the model. Returning.")
        return None

    try:
        # ----------------------------------
        # 1. (Optional) Downsample for Speed
        # ----------------------------------
        # If you have a huge dataset, you can sample a portion for hyperparam search:
        MAX_TRAIN_SIZE = 50_000  # Example threshold
        if len(X) > MAX_TRAIN_SIZE:
            logging.info(f"Downsampling from {len(X)} to {MAX_TRAIN_SIZE} rows for faster tuning.")
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X, y, 
                train_size=MAX_TRAIN_SIZE, 
                random_state=42, 
                stratify=y
            )
        else:
            X_train_sample, y_train_sample = X, y

        # ----------------------------------
        # 2. Train/Validation Split for Early Stopping
        # ----------------------------------
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_sample, y_train_sample, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_train_sample
        )

        start_time = datetime.now()
        scoring = make_scorer(accuracy_score)

        # ----------------------------------
        # 3. Define Wrappers / Classifiers
        # ----------------------------------
        xgb = XGBClassifierWrapper(
            # We'll let early_stopping_rounds come via fit_params
            use_label_encoder=False,
            n_jobs=-1
        )
        lgb = LGBMClassifierWrapper(
            n_jobs=-1
        )

        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('lgb', lgb)],
            voting='soft'
        )

        # ----------------------------------
        # 4. Reduced Hyperparameter Search
        # ----------------------------------
        # Keep only a few possible values to drastically speed up the search.
        param_distributions = {
            'xgb__n_estimators': [100],           # fixed at 100
            'xgb__max_depth': [3, 5],             # only 2 values
            'xgb__subsample': [0.6],              # fixed
            'xgb__colsample_bytree': [0.6],       # fixed
            'xgb__learning_rate': [0.01, 0.05],   # 2 values

            'lgb__n_estimators': [100], 
            'lgb__max_depth': [3, 5], 
            'lgb__subsample': [0.6], 
            'lgb__colsample_bytree': [0.6], 
            'lgb__learning_rate': [0.01, 0.05],
        }

        # ----------------------------------
        # 5. Early Stopping in fit_params
        # ----------------------------------
        # We'll pass evaluation sets & early_stopping_rounds for both XGB & LGB.
        # Note: Each library has slightly different param naming, see docs.
        fit_params = {
            # For XGBoost
            'xgb__eval_set': [(X_val, y_val)],
            'xgb__eval_metric': 'logloss',
            'xgb__early_stopping_rounds': 10,

            # For LightGBM
            'lgb__eval_set': [(X_val, y_val)],
            'lgb__eval_metric': 'logloss',
            'lgb__early_stopping_rounds': 10
        }

        # ----------------------------------
        # 6. RandomizedSearch with Fewer Iterations
        # ----------------------------------
        cv = KFold(n_splits=2, shuffle=True, random_state=42)
        logging.info("Starting hyperparameter tuning for Voting Ensemble...")
        search = RandomizedSearchCV(
            estimator=ensemble,
            param_distributions=param_distributions,
            n_iter=5,            # fewer iterations
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=2,
            error_score='raise'
        )

        search.fit(X_train, y_train, **fit_params)  # pass in early stopping params

        logging.info(f"Best parameters: {search.best_params_}, Score: {search.best_score_:.2f}")

        best_ensemble = search.best_estimator_

        # ----------------------------------
        # 7. Final Retrain on ALL Sampled Data (Optional)
        # ----------------------------------
        # Now that we have best hyperparams, we can skip cross-validation
        # and do a single fit on the entire sample, again with early stopping.
        # This step is optional if you want to strictly use the CV best_estimator.
        best_ensemble.set_params(**search.best_params_)
        best_ensemble.fit(
            X_train_sample, 
            y_train_sample,
            **fit_params
        )

        # Save the final model
        joblib.dump(best_ensemble, "ensemble_model.joblib")
        logging.info("Ensemble model saved successfully.")

        # ----------------------------------
        # Evaluate on the Entire Data
        # ----------------------------------
        ensemble_pred = best_ensemble.predict_proba(X)[:, 1]
        ensemble_acc = ((ensemble_pred > 0.5).astype(int) == y).mean()
        logging.info(f"Model training took {(datetime.now() - start_time).total_seconds():.2f} seconds.")
        logging.info(f"Final Ensemble Accuracy (on full dataset): {ensemble_acc:.2f}")

        return best_ensemble

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

    with get_session() as session:
        symbols = [row[0] for row in session.query(StockData.symbol).distinct().all()]

    if not symbols:
        logging.info("No symbols found in DB to train the model.")
        return None

    combined = []
    for sym in symbols:
        with get_session() as session:
            records = (session.query(StockData)
                               .filter(StockData.symbol == sym)
                               .order_by(StockData.timestamp.asc())
                               .all())
        if not records or len(records) < 1:
            logging.info(f"Insufficient data for {sym}. Skipping.")
            continue

        data = pd.DataFrame([{
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume,
            'timestamp': r.timestamp
        } for r in records])
        data.sort_values('timestamp', inplace=True)

        data = compute_technical_indicators(data)
        if data.empty:
            logging.info(f"Insufficient processed data for {sym}. Skipping.")
            continue
        data['symbol'] = sym
        combined.append(data)

    if not combined:
        logging.info("No data available for training the model after processing.")
        return None

    combined_df = pd.concat(combined)
    combined_df.dropna(inplace=True)

    X, y = prepare_features(combined_df)
    if X.empty or y.empty:
        logging.info("No features or target available for training.")
        return None

    logging.info(f"Training data shape: X={X.shape}, y={y.shape}")
    new_model = train_model(X, y)
    return new_model
