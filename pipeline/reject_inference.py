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
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
from datetime import datetime, timezone

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

    champion_path = MODELS_DIR / "champion" / "model.pkl"
    if not champion_path.exists():
        raise FileNotFoundError("No champion model found. Run train.py first.")
    with open(champion_path, "rb") as f:
        champion_model = pickle.load(f)

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

    print(f"[REJECT] Overlapping features: {overlap_cols}")

    # Fill remaining features with training medians
    train_medians = train[feature_cols].median()
    for col in feature_cols:
        if col not in aligned.columns:
            aligned[col] = train_medians[col]

    # Ensure column order and fill any remaining NaN
    aligned = aligned[feature_cols].fillna(train_medians)

    print(f"[REJECT] Aligned {len(aligned):,} rejected applicants to {len(feature_cols)} features")
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

    print(f"[REJECT] Training default rate (accepted): {training_default_rate:.3f}")
    print(f"[REJECT] Assumed reject default rate: {reject_default_rate:.3f}")
    print(f"[REJECT] Score threshold: {threshold:.4f}")
    print(f"[REJECT] Pseudo-labels — default: {pseudo_labels.sum():,}, "
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

    # Combine with val for early stopping
    X_all = pd.concat([X_combined, X_val], ignore_index=True)
    y_all = pd.concat([y_combined, y_val], ignore_index=True)
    weights_all = np.concatenate([sample_weights, np.ones(len(X_val))])
    val_fraction = len(X_val) / len(X_all)

    print(f"[REJECT] Combined training set: {len(X_combined):,} rows "
          f"(accepted={len(X_accepted):,}, rejected={len(X_rejected):,})")

    model = HistGradientBoostingClassifier(
        max_iter=1000,
        max_depth=6,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=True,
        validation_fraction=val_fraction,
        n_iter_no_change=50,
        scoring="roc_auc",
        random_state=42,
        verbose=1,
    )

    model.fit(X_all, y_all, sample_weight=weights_all)
    print(f"[REJECT] Augmented model iterations: {model.n_iter_}")

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

    print("\n[REJECT] Model Comparison on Test Set")
    print("=" * 55)
    print(f"{'Metric':<20} {'Champion':>15} {'Augmented':>15}")
    print("-" * 55)
    print(f"{'AUC':.<20} {auc_champ:>15.4f} {auc_aug:>15.4f}")
    print(f"{'KS':.<20} {ks_champ:>15.4f} {ks_aug:>15.4f}")
    print(f"{'Gini':.<20} {2*auc_champ-1:>15.4f} {2*auc_aug-1:>15.4f}")
    print(f"\nPSI between models: {psi:.4f}")
    print(f"AUC delta: {auc_aug - auc_champ:+.4f}")

    return results


def save_augmented_model(model, feature_cols, metrics, comparison):
    """Save augmented model as challenger with reject inference metadata."""
    dest = MODELS_DIR / "challenger"
    dest.mkdir(parents=True, exist_ok=True)

    with open(dest / "model.pkl", "wb") as f:
        pickle.dump(model, f)

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

    print(f"\n[REJECT] Augmented model saved as challenger {version} at {dest}")
    return version


def run():
    """Full reject inference pipeline."""
    mlflow.set_tracking_uri(str(MODELS_DIR / "mlruns"))
    mlflow.set_experiment("credit_risk_reject_inference")

    print("[REJECT] Loading data ...")
    train, val, test, rejected, champion_model, feature_cols = load_data()

    X_train, y_train = train[feature_cols], train["default"]
    X_val, y_val = val[feature_cols], val["default"]
    X_test, y_test = test[feature_cols], test["default"]

    print(f"[REJECT] Accepted train: {len(train):,} | Rejected pool: {len(rejected):,}")

    # Step 1: Align rejected features
    print("\n[REJECT] Step 1: Aligning rejected features ...")
    rejected_aligned = align_rejected_features(rejected, feature_cols, train)

    # Step 2: Assign pseudo-labels
    print("\n[REJECT] Step 2: Assigning pseudo-labels ...")
    training_default_rate = y_train.mean()
    pseudo_labels, rejected_scores = assign_pseudo_labels(
        champion_model, rejected_aligned, training_default_rate
    )
    rejected_aligned = rejected_aligned.copy()
    rejected_aligned["default"] = pseudo_labels

    # Step 3: Train augmented model
    print("\n[REJECT] Step 3: Training augmented model ...")
    with mlflow.start_run():
        augmented_model = train_augmented_model(
            X_train, y_train,
            rejected_aligned[feature_cols], rejected_aligned["default"],
            X_val, y_val,
        )

        # Step 4: Compare vs champion
        print("\n[REJECT] Step 4: Comparing models ...")
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
    print("\n" + "=" * 55)
    if auc_delta > 0 and psi < 0.25:
        print(f"RECOMMENDATION: Promote augmented model (AUC +{auc_delta:.4f}, PSI {psi:.4f})")
    elif auc_delta > 0 and psi >= 0.25:
        print(f"CAUTION: AUC improved (+{auc_delta:.4f}) but high PSI ({psi:.4f}). "
              "Review score distribution shift before promoting.")
    else:
        print(f"RECOMMENDATION: Keep champion (augmented AUC delta: {auc_delta:+.4f})")
    print("=" * 55)

    print("\n[REJECT] Done.")
    return augmented_model, comparison


if __name__ == "__main__":
    run()
