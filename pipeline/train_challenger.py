"""
Challenger Training Entry Point
Invoked by the Go retrain orchestrator (go/cmd/retrain-orchestrator).
Trains a challenger HistGradientBoostingClassifier, saves it to
data/models/challenger, and exports model.json for the Go runtime.

Logs go to stderr; stdout carries a single JSON result for the caller.

Usage:
    python pipeline/train_challenger.py <version>
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.train import MODELS_DIR, evaluate_model, load_gold_data, save_model
from pipeline.export_model_json import export_model

logger = logging.getLogger(__name__)


def main(version):
    logger.info("Loading Gold data ...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_gold_data()

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

    logger.info("Training challenger model ...")
    model = HistGradientBoostingClassifier(**challenger_params)
    model.fit(combined_X, combined_y)
    logger.info(f"Challenger trained. Iterations: {model.n_iter_}")

    test_metrics = evaluate_model(model, X_test, y_test, "test")
    save_model(model, feature_cols, {"test": test_metrics}, version, MODELS_DIR / "challenger")
    export_model(MODELS_DIR / "challenger")

    print(json.dumps({
        "version": version,
        "n_iterations": int(model.n_iter_),
        "params": {k: str(v) for k, v in challenger_params.items()},
        "test_metrics": test_metrics,
    }))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if len(sys.argv) != 2:
        sys.exit("usage: train_challenger.py <version>")
    main(sys.argv[1])
