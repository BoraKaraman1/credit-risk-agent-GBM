"""
Retrain Orchestrator Agent
End-to-end retraining pipeline triggered by drift or performance degradation.
Trains a challenger model, compares to champion, and presents results for human review.
Designed to be invoked by Claude Code as a subagent.
"""

import json
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from agents.config import MODELS_DIR, GOLD_DIR
from agents.drift_monitor import compute_psi


def run(reason="manual"):
    """
    Run the full retraining pipeline.

    Steps:
    1. Train a new challenger model
    2. Compare challenger vs champion on test set
    3. Compute PSI between champion and challenger score distributions
    4. Generate a comparison report for human review

    Args:
        reason: Why retraining was triggered (e.g. "psi_critical", "auc_drop", "manual")

    Returns:
        Report dict with champion vs challenger comparison.
    """
    print(f"[RETRAIN] Starting retraining. Reason: {reason}")

    # Import train module
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.train import load_gold_data, train_model, evaluate_model, save_model

    # Load data
    print("[RETRAIN] Loading Gold data ...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_gold_data()

    # Load current champion for comparison
    champion_meta_path = MODELS_DIR / "champion" / "model_metadata.json"
    champion_model_path = MODELS_DIR / "champion" / "model.pkl"

    champion_meta = None
    champion_model = None
    if champion_meta_path.exists():
        with open(champion_meta_path) as f:
            champion_meta = json.load(f)
        with open(champion_model_path, "rb") as f:
            champion_model = pickle.load(f)
        print(f"[RETRAIN] Current champion: {champion_meta['version']}")

    # Train challenger
    print("[RETRAIN] Training challenger model ...")
    from pipeline.train import train_model as _train, evaluate_model as _eval

    from sklearn.ensemble import HistGradientBoostingClassifier

    # Slightly different hyperparams for challenger (exploring the space)
    challenger_params = {
        "max_iter": 1200,
        "max_depth": 7,
        "learning_rate": 0.03,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 30,
        "l2_regularization": 2.0,
        "max_bins": 255,
        "early_stopping": True,
        "validation_fraction": len(X_val) / (len(X_train) + len(X_val)),
        "n_iter_no_change": 50,
        "scoring": "roc_auc",
        "random_state": 42,
        "verbose": 0,
    }

    combined_X = pd.concat([X_train, X_val], ignore_index=True)
    combined_y = pd.concat([y_train, y_val], ignore_index=True)

    challenger_model = HistGradientBoostingClassifier(**challenger_params)
    challenger_model.fit(combined_X, combined_y)

    print(f"[RETRAIN] Challenger trained. Iterations: {challenger_model.n_iter_}")

    # Evaluate both on test set
    challenger_scores = challenger_model.predict_proba(X_test)[:, 1]
    from sklearn.metrics import roc_auc_score, roc_curve

    def _ks(y, s):
        fpr, tpr, _ = roc_curve(y, s)
        return float(np.max(tpr - fpr))

    challenger_auc = roc_auc_score(y_test, challenger_scores)
    challenger_ks = _ks(y_test, challenger_scores)
    challenger_gini = 2 * challenger_auc - 1

    challenger_metrics = {
        "test": {
            "auc": round(challenger_auc, 4),
            "ks": round(challenger_ks, 4),
            "gini": round(challenger_gini, 4),
        }
    }

    # Champion test metrics
    champion_test_metrics = None
    champion_scores = None
    if champion_model is not None:
        champion_scores = champion_model.predict_proba(X_test)[:, 1]
        champion_auc = roc_auc_score(y_test, champion_scores)
        champion_ks = _ks(y_test, champion_scores)
        champion_test_metrics = {
            "auc": round(champion_auc, 4),
            "ks": round(champion_ks, 4),
            "gini": round(2 * champion_auc - 1, 4),
        }

    # PSI between champion and challenger score distributions
    score_psi = None
    if champion_scores is not None:
        score_psi, _, _ = compute_psi(champion_scores, challenger_scores)
        score_psi = round(score_psi, 4)

    # Determine version
    if champion_meta:
        prev_version = champion_meta.get("version", "v1.0")
        major, minor = prev_version.lstrip("v").split(".")
        version = f"v{major}.{int(minor) + 1}"
    else:
        version = "v1.0"

    # Save challenger
    save_model(challenger_model, feature_cols, challenger_metrics, version, MODELS_DIR / "challenger")

    # Build comparison report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "champion": {
            "version": champion_meta["version"] if champion_meta else None,
            "test_metrics": champion_test_metrics,
        },
        "challenger": {
            "version": version,
            "test_metrics": challenger_metrics["test"],
            "params": {k: str(v) for k, v in challenger_params.items()},
            "n_iterations": challenger_model.n_iter_,
        },
        "comparison": {
            "score_psi": score_psi,
            "auc_improvement": round(challenger_auc - (champion_test_metrics["auc"] if champion_test_metrics else 0), 4),
            "ks_improvement": round(challenger_ks - (champion_test_metrics["ks"] if champion_test_metrics else 0), 4),
        },
        "recommendation": _make_recommendation(
            challenger_metrics["test"], champion_test_metrics, score_psi
        ),
        "action_required": "Human review required before promoting challenger to champion (SR 11-7).",
        "promote_command": "python -c \"import shutil; shutil.copytree('data/models/challenger', 'data/models/champion', dirs_exist_ok=True)\"",
    }

    print(f"\n[RETRAIN] === Comparison ===")
    if champion_test_metrics:
        print(f"  Champion ({champion_meta['version']}): AUC={champion_test_metrics['auc']}  KS={champion_test_metrics['ks']}")
    print(f"  Challenger ({version}): AUC={challenger_metrics['test']['auc']}  KS={challenger_metrics['test']['ks']}")
    if score_psi is not None:
        print(f"  Score PSI between models: {score_psi}")
    print(f"  Recommendation: {report['recommendation']}")
    print(f"\n[RETRAIN] Challenger saved. Awaiting human review for promotion.")

    return report


def _make_recommendation(challenger, champion, score_psi):
    if champion is None:
        return "PROMOTE. No existing champion — challenger becomes the first production model."

    auc_diff = challenger["auc"] - champion["auc"]
    if auc_diff > 0.005:
        return (
            f"PROMOTE. Challenger AUC ({challenger['auc']}) exceeds champion ({champion['auc']}) "
            f"by {auc_diff:.4f}. Score PSI: {score_psi}."
        )
    elif auc_diff > -0.005:
        return (
            f"CONSIDER. Challenger AUC ({challenger['auc']}) is within 0.5% of champion ({champion['auc']}). "
            f"Promotion optional — may prefer stability. Score PSI: {score_psi}."
        )
    else:
        return (
            f"DO NOT PROMOTE. Challenger AUC ({challenger['auc']}) is worse than champion ({champion['auc']}) "
            f"by {abs(auc_diff):.4f}. Investigate hyperparameters or data quality."
        )


if __name__ == "__main__":
    reason = sys.argv[1] if len(sys.argv) > 1 else "manual"
    report = run(reason=reason)
    print(json.dumps(report, indent=2))
