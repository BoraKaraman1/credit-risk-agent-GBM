"""
Performance Monitor Agent
Tracks model AUC, KS, and Gini on cohorts with known outcomes.
Compares against training metrics to detect model degradation.
Designed to be invoked by Claude Code as a subagent.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score, roc_curve
from sqlalchemy import create_engine, text

from agents.config import (
    MODELS_DIR, GOLD_DIR, SUPABASE_DB_URL, AUC_DROP_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _model_path(directory):
    """Resolve model path with backward compat."""
    p = directory / "model.joblib"
    if p.exists():
        return p
    return directory / "model.pkl"


def compute_ks(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def run():
    """
    Run performance monitoring.

    Tries to use scoring_log entries with known outcomes from Supabase.
    Falls back to test set evaluation.

    Returns a report dict with current vs training metrics and recommendation.
    """
    # Load champion model and metadata
    meta_path = MODELS_DIR / "champion" / "model_metadata.json"
    model_path = _model_path(MODELS_DIR / "champion")

    with open(meta_path) as f:
        meta = json.load(f)
    model = joblib.load(model_path)

    training_metrics = meta["metrics"]
    feature_cols = meta["features"]

    # Try Supabase for real outcomes
    outcomes_source = "test_set_proxy"
    y_true = None
    y_score = None

    if SUPABASE_DB_URL:
        try:
            engine = create_engine(SUPABASE_DB_URL)
            with engine.connect() as conn:
                rows = conn.execute(
                    text("""
                        SELECT score, actual_default
                        FROM scoring_log
                        WHERE actual_default IS NOT NULL
                        ORDER BY outcome_observed_at DESC
                        LIMIT 100000
                    """)
                ).fetchall()

            if len(rows) >= 100:
                y_score = np.array([r[0] for r in rows], dtype=float)
                y_true = np.array([int(r[1]) for r in rows])
                outcomes_source = f"scoring_log ({len(rows):,} outcomes)"
                logger.info(f"Using {len(rows):,} outcomes from scoring_log")
        except Exception as e:
            logger.warning(f"Could not read scoring_log: {e}")

    if y_true is None:
        # Fall back to test set
        test = pd.read_parquet(GOLD_DIR / "features_test.parquet")
        X_test = test[feature_cols]
        y_true = test["default"].values
        y_score = model.predict_proba(X_test)[:, 1]
        logger.info("Using test set as production proxy")

    # Compute current metrics
    current_auc = roc_auc_score(y_true, y_score)
    current_ks = compute_ks(y_true, y_score)
    current_gini = 2 * current_auc - 1

    current_metrics = {
        "auc": round(current_auc, 4),
        "ks": round(current_ks, 4),
        "gini": round(current_gini, 4),
    }

    # Compare to training metrics
    train_auc = training_metrics.get("val", training_metrics.get("test", {})).get("auc", 0)
    auc_drop = train_auc - current_auc

    # Decile analysis
    decile_df = pd.DataFrame({"score": y_score, "default": y_true})
    decile_df["decile"] = pd.qcut(decile_df["score"], 10, labels=False, duplicates="drop")
    decile_stats = (
        decile_df.groupby("decile")
        .agg(count=("default", "size"), default_rate=("default", "mean"), avg_score=("score", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )

    # Check rank ordering (each decile should have higher default rate than previous)
    default_rates = [d["default_rate"] for d in decile_stats]
    rank_order_breaks = sum(1 for i in range(1, len(default_rates)) if default_rates[i] < default_rates[i-1])

    logger.info(f"Current AUC={current_auc:.4f}  Training AUC={train_auc:.4f}  Drop={auc_drop:.4f}")
    logger.info(f"Rank-ordering breaks: {rank_order_breaks}")

    # Build report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": meta["version"],
        "outcomes_source": outcomes_source,
        "n_observations": len(y_true),
        "current_metrics": current_metrics,
        "training_metrics": training_metrics,
        "auc_drop": round(auc_drop, 4),
        "auc_drop_threshold": AUC_DROP_THRESHOLD,
        "rank_order_breaks": rank_order_breaks,
        "decile_analysis": decile_stats,
        "recommendation": _make_recommendation(auc_drop, rank_order_breaks, current_metrics),
    }

    # Log to Supabase
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
                        "metric_name": "auc",
                        "metric_value": current_auc,
                        "model_version": meta["version"],
                        "details": json.dumps({
                            "auc_drop": round(auc_drop, 4),
                            "ks": current_metrics["ks"],
                            "rank_order_breaks": rank_order_breaks,
                        }),
                    },
                )
        except Exception as e:
            logger.warning(f"Could not log to Supabase: {e}")

    return report


def _make_recommendation(auc_drop, rank_breaks, current_metrics):
    if auc_drop > AUC_DROP_THRESHOLD:
        return (
            f"RETRAIN RECOMMENDED. AUC has dropped {auc_drop:.4f} from training "
            f"(threshold: {AUC_DROP_THRESHOLD}). Current AUC: {current_metrics['auc']}."
        )
    if rank_breaks > 2:
        return (
            f"INVESTIGATE. Rank ordering has {rank_breaks} breaks across deciles. "
            f"The model's discrimination is degrading even though AUC drop ({auc_drop:.4f}) "
            f"hasn't crossed the threshold yet."
        )
    return (
        f"Model performance stable. AUC drop: {auc_drop:.4f} "
        f"(within {AUC_DROP_THRESHOLD} threshold). "
        f"Current KS: {current_metrics['ks']}."
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    report = run()
    print(json.dumps(report, indent=2))
