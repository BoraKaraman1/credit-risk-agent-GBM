"""
Training Pipeline
Trains a gradient boosting model on Gold features, logs to MLflow, manages champion/challenger.
Uses sklearn HistGradientBoostingClassifier (XGBoost-equivalent, no OpenMP dependency).
"""

import json
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models"


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
            "validation_fraction": None,  # we provide our own val set
            "n_iter_no_change": 50,
            "scoring": "roc_auc",
            "random_state": 42,
            "verbose": 1,
        }

    model = HistGradientBoostingClassifier(**params)

    # HistGradientBoosting uses fit with validation set for early stopping
    model.fit(X_train, y_train, sample_weight=None)

    # Re-fit with early stopping using validation data
    # HistGradientBoosting handles this internally when validation_fraction is set
    # For explicit val set, we use a different approach:
    params_with_val = {**params, "validation_fraction": None, "early_stopping": False}

    # Actually, let's use the built-in early stopping properly
    # We combine train+val and set validation_fraction
    combined_X = pd.concat([X_train, X_val], ignore_index=True)
    combined_y = pd.concat([y_train, y_val], ignore_index=True)
    val_fraction = len(X_val) / len(combined_X)

    params["validation_fraction"] = val_fraction
    params["early_stopping"] = True

    model = HistGradientBoostingClassifier(**params)
    model.fit(combined_X, combined_y)

    print(f"[TRAIN] Best iteration: {model.n_iter_}")

    return model, params


def evaluate_model(model, X, y, split_name="test"):
    """Compute AUC, KS, Gini for a given split."""
    y_score = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, y_score)
    ks = compute_ks(y, y_score)
    gini = compute_gini(auc)

    print(f"[{split_name}] AUC={auc:.4f}  KS={ks:.4f}  Gini={gini:.4f}")
    return {"auc": round(auc, 4), "ks": round(ks, 4), "gini": round(gini, 4)}


def save_model(model, feature_cols, metrics, version, dest_dir):
    """Save model pickle and metadata JSON."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    model_path = dest_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    meta = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": feature_cols,
        "n_features": len(feature_cols),
        "metrics": metrics,
    }
    with open(dest_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[MODEL] Saved to {dest_dir}")
    return model_path


def run(as_challenger=False):
    """Full training pipeline with MLflow logging."""
    mlflow.set_tracking_uri(str(MODELS_DIR / "mlruns"))
    mlflow.set_experiment("credit_risk_gbm")

    print("[TRAIN] Loading Gold data ...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_gold_data()
    print(f"[TRAIN] Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    with mlflow.start_run():
        print("[TRAIN] Training HistGradientBoosting ...")
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

    print(f"[TRAIN] Done. Model {version} saved as {'challenger' if as_challenger else 'champion'}.")
    return model, all_metrics


if __name__ == "__main__":
    run()
