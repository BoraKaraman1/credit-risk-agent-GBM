"""
Training Pipeline
Trains a gradient boosting model on Gold features, logs to MLflow, manages champion/challenger.
Uses sklearn HistGradientBoostingClassifier (XGBoost-equivalent, no OpenMP dependency).
"""

import json
import logging
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models"


def _model_path(directory):
    """Resolve model path with backward compat (joblib first, then pkl fallback)."""
    p = directory / "model.joblib"
    if p.exists():
        return p
    return directory / "model.pkl"


def compute_ks(y_true, y_score):
    """Kolmogorov-Smirnov statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def compute_gini(auc):
    """Gini coefficient from AUC."""
    return 2 * auc - 1


def load_gold_data():
    """Load train/val/test splits from Gold."""
    with open(GOLD_DIR / "feature_metadata.json") as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]

    train = pd.read_parquet(GOLD_DIR / "features_train.parquet")
    val = pd.read_parquet(GOLD_DIR / "features_val.parquet")
    test = pd.read_parquet(GOLD_DIR / "features_test.parquet")

    X_train, y_train = train[feature_cols], train["default"]
    X_val, y_val = val[feature_cols], val["default"]
    X_test, y_test = test[feature_cols], test["default"]

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_model(X_train, y_train, X_val, y_val, params=None):
    """Train HistGradientBoosting with early stopping on validation set."""
    if params is None:
        params = {
            "max_iter": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
            "max_bins": 255,
            "early_stopping": True,
            "validation_fraction": None,
            "n_iter_no_change": 50,
            "scoring": "roc_auc",
            "random_state": 42,
            "verbose": 1,
        }

    combined_X = pd.concat([X_train, X_val], ignore_index=True)
    combined_y = pd.concat([y_train, y_val], ignore_index=True)
    val_fraction = len(X_val) / len(combined_X)

    params["validation_fraction"] = val_fraction
    params["early_stopping"] = True

    model = HistGradientBoostingClassifier(**params)
    model.fit(combined_X, combined_y)

    logger.info(f"Best iteration: {model.n_iter_}")

    return model, params


def evaluate_model(model, X, y, split_name="test"):
    """Compute AUC, KS, Gini for a given split."""
    y_score = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, y_score)
    ks = compute_ks(y, y_score)
    gini = compute_gini(auc)

    logger.info(f"[{split_name}] AUC={auc:.4f}  KS={ks:.4f}  Gini={gini:.4f}")
    return {"auc": round(auc, 4), "ks": round(ks, 4), "gini": round(gini, 4)}


def save_model(model, feature_cols, metrics, version, dest_dir):
    """Save model and metadata JSON."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    model_path = dest_dir / "model.joblib"
    joblib.dump(model, model_path)

    meta = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "metrics": metrics,
    }
    with open(dest_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved to {dest_dir}")
    return model_path


def run(as_challenger=False):
    """Full training pipeline with MLflow logging."""
    mlflow.set_tracking_uri(str(MODELS_DIR / "mlruns"))
    mlflow.set_experiment("credit_risk_gbm")

    logger.info("Loading Gold data ...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_gold_data()
    logger.info(f"Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    with mlflow.start_run():
        logger.info("Training HistGradientBoosting ...")
        model, params = train_model(X_train, y_train, X_val, y_val)

        # Log params (convert non-string values for MLflow)
        log_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(log_params)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_iterations", model.n_iter_)

        # Evaluate on all splits
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        val_metrics = evaluate_model(model, X_val, y_val, "val")
        test_metrics = evaluate_model(model, X_test, y_test, "test")

        # Log metrics
        for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            for k, v in metrics.items():
                mlflow.log_metric(f"{split}_{k}", v)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Determine version
        champion_meta = MODELS_DIR / "champion" / "model_metadata.json"
        if champion_meta.exists():
            with open(champion_meta) as f:
                prev = json.load(f)
            prev_version = prev.get("version", "v1.0")
            major, minor = prev_version.lstrip("v").split(".")
            version = f"v{major}.{int(minor) + 1}"
        else:
            version = "v1.0"

        # Save as challenger or champion
        dest = MODELS_DIR / ("challenger" if as_challenger else "champion")

        all_metrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        save_model(model, feature_cols, all_metrics, version, dest)
        mlflow.log_param("model_version", version)
        mlflow.log_param("role", "challenger" if as_challenger else "champion")

    logger.info(f"Done. Model {version} saved as {'challenger' if as_challenger else 'champion'}.")
    return model, all_metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
