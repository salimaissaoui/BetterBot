import logging
import traceback
import joblib
from datetime import datetime
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .config import RETRAIN_FREQUENCY
from .indicators import compute_technical_indicators, prepare_features
from .database import get_session, StockData
from .utils import XGBClassifierWrapper, LGBMClassifierWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def train_model(X, y):
    """Trains a VotingClassifier ensemble with XGBoost and LightGBM."""
    if X.empty or y.empty:
        logging.info("No data available for training the model. Returning.")
        return None

    try:
        start_time = datetime.now()
        scoring = make_scorer(accuracy_score)

        xgb = XGBClassifierWrapper(eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
        lgb = LGBMClassifierWrapper(n_jobs=-1)

        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('lgb', lgb)],
            voting='soft'
        )

        param_distributions = {
            'xgb__n_estimators': [100, 200],
            'xgb__max_depth': [3, 5],
            'xgb__subsample': [0.6, 0.8],
            'xgb__colsample_bytree': [0.6, 0.8],
            'xgb__learning_rate': [0.01, 0.05],
            'lgb__n_estimators': [100, 200],
            'lgb__max_depth': [3, 5, -1],
            'lgb__subsample': [0.6, 0.8],
            'lgb__colsample_bytree': [0.6, 0.8],
            'lgb__learning_rate': [0.01, 0.05],
        }

        cv = KFold(n_splits=2, shuffle=True, random_state=42)

        logging.info("Starting hyperparameter tuning for Voting Ensemble...")
        search = RandomizedSearchCV(
            estimator=ensemble,
            param_distributions=param_distributions,
            n_iter=20,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=2,
            error_score='raise'
        )
        search.fit(X, y)
        logging.info(f"Best parameters: {search.best_params_}, Score: {search.best_score_:.2f}")

        best_ensemble = search.best_estimator_
        joblib.dump(best_ensemble, "ensemble_model.joblib")
        logging.info("Ensemble model saved successfully.")

        ensemble_pred = best_ensemble.predict_proba(X)[:, 1]
        ensemble_acc = ((ensemble_pred > 0.5).astype(int) == y).mean()

        logging.info(f"Model training took {(datetime.now() - start_time).total_seconds():.2f} seconds.")
        logging.info(f"Final Ensemble Accuracy: {ensemble_acc:.2f}")

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
    with get_session() as session:
        symbols = [row[0] for row in session.query(StockData.symbol).distinct().all()]

    if not symbols:
        logging.info("No symbols found in DB to train the model.")
        return None

    combined = []
    for sym in symbols:
        with get_session() as session:
            records = session.query(StockData).filter(StockData.symbol == sym).order_by(StockData.timestamp.asc()).all()

        if not records or len(records) < 30:
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
