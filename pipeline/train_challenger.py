"""
Challenger Training Entry Point
Invoked by the Go retrain orchestrator (gbm retrain, go/monitoring).
Trains a challenger LightGBM model, saves it to
data/models/challenger, and exports model.json for the Go runtime.

Logs go to stderr; stdout carries a single JSON result for the caller.

Usage:
    python pipeline/train_challenger.py <version>
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config, fairness
from pipeline.train import early_stopping_split, evaluate_model, load_gold_data, save_model
from pipeline.calibrate import calibrate_model, save_calibration
from pipeline.export_model_json import export_model

logger = logging.getLogger(__name__)


def main(version):
    logger.info("Loading Gold data ...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_gold_data()

    # Slightly different hyperparams for challenger (exploring the space)
    challenger_params = {
        "n_estimators": 1200,
        "max_depth": 7,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 30,
        "reg_lambda": 2.0,
        "max_bin": 255,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Same carve-out as the champion path (and as calibrate.run's
    # standalone recalibration): delegated so the split can never drift.
    X_fit, X_es, y_fit, y_es = early_stopping_split(X_train, y_train, X_val, y_val)

    logger.info("Training challenger model ...")
    model = lgb.LGBMClassifier(**challenger_params)
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_es, y_es)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    logger.info(f"Challenger trained. Iterations: {model.best_iteration_}")

    # Test scores are computed once and reused by evaluation,
    # calibration, and fairness below.
    test_scores = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(model, X_test, y_test, "test", y_score=test_scores)
    save_model(model, feature_cols, {"test": test_metrics}, version, config.challenger_dir())

    # Calibrate on the early-stopping carve-out so the exported JSON
    # carries the calibrator, same as the champion path.
    calibrator, cal_report = calibrate_model(model, X_es, y_es, X_test, y_test,
                                             raw_test=test_scores)
    save_calibration(config.challenger_dir(), calibrator, cal_report)

    # Fairness for both models on the same test set: the orchestrator
    # gates promotion on the challenger relative to the champion, so it
    # needs both summaries (fairness lives only in Python).
    challenger_fairness = fairness.summarize(
        fairness.run(model=model, X_test=X_test, y_test=y_test, raw_score=test_scores))
    fairness.save_fairness(config.challenger_dir(), challenger_fairness)

    champion_fairness = None
    if config.metadata_path(config.champion_dir()).exists():
        champion_model = joblib.load(config.model_path(config.champion_dir()))
        champion_fairness = fairness.summarize(
            fairness.run(model=champion_model, X_test=X_test, y_test=y_test))

    export_model(config.challenger_dir())

    print(json.dumps({
        "version": version,
        "n_iterations": int(model.best_iteration_ or model.n_estimators_),
        "params": {k: str(v) for k, v in challenger_params.items()},
        "test_metrics": test_metrics,
        "fairness": challenger_fairness,
        "champion_fairness": champion_fairness,
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
