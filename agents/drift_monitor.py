"""
Drift Monitor Agent
Computes PSI (Population Stability Index) on score distributions
and CSI (Characteristic Stability Index) on individual features.
Designed to be invoked by Claude Code as a subagent.
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text

from agents.config import (
    MODELS_DIR, GOLD_DIR, SUPABASE_DB_URL,
    PSI_WARNING, PSI_CRITICAL, CSI_THRESHOLD,
)


def compute_psi(expected, actual, bins=10):
    """
    Population Stability Index.
    Compares two score distributions using equal-width bins.
    PSI < 0.10: no shift
    PSI 0.10-0.25: moderate shift
    PSI > 0.25: significant shift → retrain
    """
    bin_edges = np.linspace(0, 1, bins + 1)
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Avoid division by zero
    expected_pct = (expected_counts + 1) / (expected_counts.sum() + bins)
    actual_pct = (actual_counts + 1) / (actual_counts.sum() + bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi), expected_pct.tolist(), actual_pct.tolist()


def compute_csi(train_col, production_col, bins=10):
    """
    Characteristic Stability Index for a single feature.
    Same math as PSI but applied to a feature distribution instead of scores.
    """
    combined = np.concatenate([train_col, production_col])
    bin_edges = np.percentile(combined[~np.isnan(combined)], np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        return 0.0

    train_counts, _ = np.histogram(train_col[~np.isnan(train_col)], bins=bin_edges)
    prod_counts, _ = np.histogram(production_col[~np.isnan(production_col)], bins=bin_edges)

    n_bins = len(train_counts)
    train_pct = (train_counts + 1) / (train_counts.sum() + n_bins)
    prod_pct = (prod_counts + 1) / (prod_counts.sum() + n_bins)

    csi = np.sum((prod_pct - train_pct) * np.log(prod_pct / train_pct))
    return float(csi)


def run(production_scores=None, production_features=None):
    """
    Run drift monitoring.

    If production_scores/features are None, attempts to read from Supabase scoring_log.
    Falls back to using the test set as a proxy for production.

    Returns a report dict with PSI, CSI, and recommendations.
    """
    # Load champion model and training data
    model_path = MODELS_DIR / "champion" / "model.pkl"
    meta_path = MODELS_DIR / "champion" / "model_metadata.json"

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    feature_cols = meta["features"]

    # Training scores (reference distribution)
    train = pd.read_parquet(GOLD_DIR / "features_train.parquet")
    X_train = train[feature_cols]
    train_scores = model.predict_proba(X_train)[:, 1]

    # Production scores — try Supabase first, fall back to test set
    if production_scores is None:
        if SUPABASE_DB_URL:
            try:
                engine = create_engine(SUPABASE_DB_URL)
                with engine.connect() as conn:
                    rows = conn.execute(
                        text("SELECT score FROM scoring_log ORDER BY scored_at DESC LIMIT 50000")
                    ).fetchall()
                if rows:
                    production_scores = np.array([r[0] for r in rows], dtype=float)
                    print(f"[DRIFT] Using {len(production_scores):,} scores from scoring_log")
            except Exception as e:
                print(f"[DRIFT] Could not read scoring_log: {e}")

    if production_scores is None:
        # Fall back to test set as proxy
        test = pd.read_parquet(GOLD_DIR / "features_test.parquet")
        X_test = test[feature_cols]
        production_scores = model.predict_proba(X_test)[:, 1]
        production_features = X_test
        print("[DRIFT] Using test set as production proxy")

    # --- PSI on score distribution ---
    psi, train_pct, prod_pct = compute_psi(train_scores, production_scores)

    psi_status = "OK"
    if psi > PSI_CRITICAL:
        psi_status = "CRITICAL"
    elif psi > PSI_WARNING:
        psi_status = "WARNING"

    print(f"[DRIFT] Score PSI = {psi:.4f} ({psi_status})")

    # --- CSI on individual features ---
    csi_results = {}
    if production_features is not None:
        for col in feature_cols:
            if col in production_features.columns:
                csi = compute_csi(
                    X_train[col].values.astype(float),
                    production_features[col].values.astype(float),
                )
                csi_results[col] = round(csi, 4)

        drifted_features = {k: v for k, v in csi_results.items() if v > CSI_THRESHOLD}
        if drifted_features:
            print(f"[DRIFT] Features with CSI > {CSI_THRESHOLD}: {drifted_features}")
        else:
            print(f"[DRIFT] No individual features above CSI threshold ({CSI_THRESHOLD})")

    # --- Build report ---
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": meta["version"],
        "psi": round(psi, 4),
        "psi_status": psi_status,
        "psi_thresholds": {"warning": PSI_WARNING, "critical": PSI_CRITICAL},
        "train_distribution": train_pct,
        "production_distribution": prod_pct,
        "csi_results": csi_results,
        "drifted_features": {k: v for k, v in csi_results.items() if v > CSI_THRESHOLD},
        "recommendation": _make_recommendation(psi, psi_status, csi_results),
    }

    # Log to Supabase if available
    if SUPABASE_DB_URL:
        try:
            engine = create_engine(SUPABASE_DB_URL)
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO drift_log (metric_name, metric_value, model_version, details)
                        VALUES (:metric_name, :metric_value, :model_version, :details::jsonb)
                    """),
                    {
                        "metric_name": "psi",
                        "metric_value": psi,
                        "model_version": meta["version"],
                        "details": json.dumps({
                            "csi": csi_results,
                            "drifted_features": list(report["drifted_features"].keys()),
                        }),
                    },
                )
            print("[DRIFT] Results logged to drift_log table")
        except Exception as e:
            print(f"[DRIFT] Could not log to Supabase: {e}")

    return report


def _make_recommendation(psi, psi_status, csi_results):
    """Generate human-readable recommendation based on drift metrics."""
    if psi_status == "CRITICAL":
        return (
            f"RETRAIN RECOMMENDED. Score PSI ({psi:.4f}) exceeds critical threshold. "
            f"The population applying for loans has shifted significantly from training data."
        )
    elif psi_status == "WARNING":
        drifted = [k for k, v in csi_results.items() if v > CSI_THRESHOLD]
        if drifted:
            return (
                f"MONITOR CLOSELY. Score PSI ({psi:.4f}) shows moderate drift. "
                f"Features driving drift: {', '.join(drifted)}. "
                f"If this persists for 2+ weeks, consider retraining."
            )
        return (
            f"MONITOR CLOSELY. Score PSI ({psi:.4f}) shows moderate drift "
            f"but no individual features exceed CSI threshold."
        )
    return f"No action needed. Score PSI ({psi:.4f}) is within normal range."


if __name__ == "__main__":
    report = run()
    print(json.dumps(report, indent=2))
