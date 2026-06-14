"""
Reject Inference Pipeline
Corrects selection bias by incorporating rejected applicants via parcelling.

Steps:
1. Load champion model (trained on accepted-only)
2. Align rejected applicant features to accepted feature space
3. Score rejected applicants → assign pseudo-labels
4. Retrain on combined (accepted + pseudo-labeled rejected) with sample weights
5. Compare augmented vs champion → save as challenger if improved
"""

import json
import logging
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime, timezone

from pipeline.train import _model_path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models"

# Reject inference config
REJECT_SAMPLE_SIZE = 500_000      # cap rejected rows for memory
REJECT_DEFAULT_MULTIPLIER = 2.0   # assumed reject default rate = multiplier * accepted rate
REJECT_DEFAULT_CAP = 0.60         # max assumed reject default rate
REJECT_SAMPLE_WEIGHT = 0.3        # weight for pseudo-labeled samples (uncertain labels)


def compute_ks(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def load_data():
    """Load accepted Gold data, rejected Silver data, and champion model."""
    with open(GOLD_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]

    train = pd.read_parquet(GOLD_DIR / "features_train.parquet")
    val = pd.read_parquet(GOLD_DIR / "features_val.parquet")
    test = pd.read_parquet(GOLD_DIR / "features_test.parquet")

    rejected = pd.read_parquet(SILVER_DIR / "rejected_clean.parquet")

    champion_path = _model_path(MODELS_DIR / "champion")
    if not champion_path.exists():
        raise FileNotFoundError("No champion model found. Run train.py first.")
    champion_model = joblib.load(champion_path)

    return train, val, test, rejected, champion_model, feature_cols


def align_rejected_features(rejected, feature_cols, train):
    """Align rejected applicant features to the accepted feature space.

    Rejected applicants have fewer columns (only application-level data).
    Overlapping features are mapped directly; missing features are filled
    with training set medians (conservative imputation).
    """
    np.random.seed(42)
    sample_size = min(REJECT_SAMPLE_SIZE, len(rejected))
    rejected_sample = rejected.sample(n=sample_size, random_state=42).copy()

    aligned = pd.DataFrame(index=rejected_sample.index)

    # Direct mappings (columns that exist in both accepted and rejected)
    overlap_cols = [c for c in feature_cols if c in rejected_sample.columns]
    for col in overlap_cols:
        aligned[col] = rejected_sample[col]

    logger.info(f"Overlapping features: {overlap_cols}")

    # Fill remaining features with training medians
    train_medians = train[feature_cols].median()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = train_medians[col]

    # Ensure column order and fill any remaining NaN
    aligned = aligned[feature_cols].fillna(train_medians)

    logger.info(f"Aligned {len(aligned):,} rejected applicants to {len(feature_cols)} features")
    return aligned


def assign_pseudo_labels(champion_model, rejected_aligned, training_default_rate):
    """Score rejected applicants and assign pseudo-labels via parcelling.

    Assumes rejected applicants would default at a higher rate than accepted
    (they were rejected for a reason). Uses the champion model's score as
    the ordering, then assigns default=1 to the top-scoring fraction.
    """
    scores = champion_model.predict_proba(rejected_aligned)[:, 1]

    reject_default_rate = min(
        training_default_rate * REJECT_DEFAULT_MULTIPLIER,
        REJECT_DEFAULT_CAP,
    )
    threshold = np.percentile(scores, 100 * (1 - reject_default_rate))

    pseudo_labels = (scores >= threshold).astype(int)

    logger.info(f"Training default rate (accepted): {training_default_rate:.3f}")
    logger.info(f"Assumed reject default rate: {reject_default_rate:.3f}")
    logger.info(f"Score threshold: {threshold:.4f}")
    logger.info(f"Pseudo-labels — default: {pseudo_labels.sum():,}, "
                f"non-default: {(pseudo_labels == 0).sum():,}")

    return pseudo_labels, scores


def train_augmented_model(X_accepted, y_accepted, X_rejected, y_rejected,
                          X_val, y_val):
    """Train on combined accepted + pseudo-labeled rejected data."""
    X_combined = pd.concat([X_accepted, X_rejected], ignore_index=True)
    y_combined = pd.concat([y_accepted, y_rejected], ignore_index=True)

    sample_weights = np.concatenate([
        np.ones(len(X_accepted)),
        np.full(len(X_rejected), REJECT_SAMPLE_WEIGHT),
    ])

    # Same data usage as the champion: val joins the training pool and a
    # random stratified carve-out drives early stopping.
    X_all = pd.concat([X_combined, X_val], ignore_index=True)
    y_all = pd.concat([y_combined, y_val], ignore_index=True)
    weights_all = np.concatenate([sample_weights, np.ones(len(X_val))])
    val_fraction = len(X_val) / len(X_all)

    X_fit, X_es, y_fit, y_es, w_fit, _ = train_test_split(
        X_all, y_all, weights_all, test_size=val_fraction,
        random_state=42, stratify=y_all,
    )

    logger.info(f"Combined training set: {len(X_combined):,} rows "
                f"(accepted={len(X_accepted):,}, rejected={len(X_rejected):,})")

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        reg_lambda=1.0,
        max_bin=255,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_es, y_es)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    logger.info(f"Augmented model iterations: {model.best_iteration_}")

    return model


def compare_models(champion, augmented, X_test, y_test, feature_cols):
    """Compare champion vs augmented model on held-out test set."""
    scores_champ = champion.predict_proba(X_test)[:, 1]
    scores_aug = augmented.predict_proba(X_test)[:, 1]

    auc_champ = roc_auc_score(y_test, scores_champ)
    auc_aug = roc_auc_score(y_test, scores_aug)
    ks_champ = compute_ks(y_test, scores_champ)
    ks_aug = compute_ks(y_test, scores_aug)

    # PSI between score distributions
    bin_edges = np.linspace(0, 1, 11)
    counts_c, _ = np.histogram(scores_champ, bins=bin_edges)
    counts_a, _ = np.histogram(scores_aug, bins=bin_edges)
    pct_c = np.clip(counts_c / counts_c.sum(), 1e-6, None)
    pct_a = np.clip(counts_a / counts_a.sum(), 1e-6, None)
    psi = float(np.sum((pct_a - pct_c) * np.log(pct_a / pct_c)))

    results = {
        "champion": {"auc": round(auc_champ, 4), "ks": round(ks_champ, 4),
                      "gini": round(2 * auc_champ - 1, 4)},
        "augmented": {"auc": round(auc_aug, 4), "ks": round(ks_aug, 4),
                       "gini": round(2 * auc_aug - 1, 4)},
        "psi_between_models": round(psi, 4),
        "auc_delta": round(auc_aug - auc_champ, 4),
    }

    logger.info("")
    logger.info("Model Comparison on Test Set")
    logger.info("=" * 55)
    logger.info(f"{'Metric':<20} {'Champion':>15} {'Augmented':>15}")
    logger.info("-" * 55)
    logger.info(f"{'AUC':.<20} {auc_champ:>15.4f} {auc_aug:>15.4f}")
    logger.info(f"{'KS':.<20} {ks_champ:>15.4f} {ks_aug:>15.4f}")
    logger.info(f"{'Gini':.<20} {2*auc_champ-1:>15.4f} {2*auc_aug-1:>15.4f}")
    logger.info(f"PSI between models: {psi:.4f}")
    logger.info(f"AUC delta: {auc_aug - auc_champ:+.4f}")

    return results


def save_augmented_model(model, feature_cols, metrics, comparison):
    """Save augmented model as challenger with reject inference metadata."""
    dest = MODELS_DIR / "challenger"
    dest.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, dest / "model.joblib")

    # Remove any stale calibrator from a previous challenger: this path
    # does not fit one, and the JSON exporter would otherwise embed a
    # calibrator fit on a different model.
    (dest / "calibrator.joblib").unlink(missing_ok=True)

    # Determine version
    champion_meta_path = MODELS_DIR / "champion" / "model_metadata.json"
    if champion_meta_path.exists():
        with open(champion_meta_path) as f:
            prev = json.load(f)
        prev_version = prev.get("version", "v1.0")
        major, minor = prev_version.lstrip("v").split(".")
        version = f"v{major}.{int(minor) + 1}-ri"
    else:
        version = "v1.0-ri"

    meta = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "method": "reject_inference_parcelling",
        "features": feature_cols,
        "n_features": len(feature_cols),
        "reject_inference_config": {
            "reject_sample_size": REJECT_SAMPLE_SIZE,
            "default_multiplier": REJECT_DEFAULT_MULTIPLIER,
            "default_cap": REJECT_DEFAULT_CAP,
            "sample_weight": REJECT_SAMPLE_WEIGHT,
        },
        "metrics": metrics,
        "comparison_vs_champion": comparison,
    }

    with open(dest / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Augmented model saved as challenger {version} at {dest}")
    return version


def run():
    """Full reject inference pipeline."""
    mlflow.set_tracking_uri(str(MODELS_DIR / "mlruns"))
    mlflow.set_experiment("credit_risk_reject_inference")

    logger.info("Loading data ...")
    train, val, test, rejected, champion_model, feature_cols = load_data()

    X_train, y_train = train[feature_cols], train["default"]
    X_val, y_val = val[feature_cols], val["default"]
    X_test, y_test = test[feature_cols], test["default"]

    logger.info(f"Accepted train: {len(train):,} | Rejected pool: {len(rejected):,}")

    # Step 1: Align rejected features
    logger.info("Step 1: Aligning rejected features ...")
    rejected_aligned = align_rejected_features(rejected, feature_cols, train)

    # Step 2: Assign pseudo-labels
    logger.info("Step 2: Assigning pseudo-labels ...")
    training_default_rate = y_train.mean()
    pseudo_labels, rejected_scores = assign_pseudo_labels(
        champion_model, rejected_aligned, training_default_rate
    )
    rejected_aligned = rejected_aligned.copy()
    rejected_aligned["default"] = pseudo_labels

    # Step 3: Train augmented model
    logger.info("Step 3: Training augmented model ...")
    with mlflow.start_run():
        augmented_model = train_augmented_model(
            X_train, y_train,
            rejected_aligned[feature_cols], rejected_aligned["default"],
            X_val, y_val,
        )

        # Step 4: Compare vs champion
        logger.info("Step 4: Comparing models ...")
        comparison = compare_models(champion_model, augmented_model,
                                    X_test, y_test, feature_cols)

        # Evaluate augmented model metrics
        aug_metrics = {
            "train": comparison["augmented"],
            "test": comparison["augmented"],
        }

        # Log to MLflow
        mlflow.log_params({
            "method": "parcelling",
            "reject_sample_size": REJECT_SAMPLE_SIZE,
            "default_multiplier": REJECT_DEFAULT_MULTIPLIER,
            "sample_weight": REJECT_SAMPLE_WEIGHT,
        })
        for k, v in comparison["augmented"].items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("auc_delta", comparison["auc_delta"])
        mlflow.log_metric("psi_vs_champion", comparison["psi_between_models"])
        mlflow.sklearn.log_model(augmented_model, "model")

        # Step 5: Save as challenger
        version = save_augmented_model(
            augmented_model, feature_cols, aug_metrics, comparison
        )
        mlflow.log_param("model_version", version)

    # Recommendation
    auc_delta = comparison["auc_delta"]
    psi = comparison["psi_between_models"]
    logger.info("=" * 55)
    if auc_delta > 0 and psi < 0.25:
        logger.info(f"RECOMMENDATION: Promote augmented model (AUC +{auc_delta:.4f}, PSI {psi:.4f})")
    elif auc_delta > 0 and psi >= 0.25:
        logger.info(f"CAUTION: AUC improved (+{auc_delta:.4f}) but high PSI ({psi:.4f}). "
                     "Review score distribution shift before promoting.")
    else:
        logger.info(f"RECOMMENDATION: Keep champion (augmented AUC delta: {auc_delta:+.4f})")
    logger.info("=" * 55)

    logger.info("Reject inference done.")
    return augmented_model, comparison


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
