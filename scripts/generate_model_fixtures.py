"""
Regenerate the Go parity-test artifacts in go/shared/model/testdata/:

- parity_model.json: a small dedicated LightGBM model (trained on a
  deterministic Gold sample, isotonic-calibrated) exported through the
  production export path. Small enough to commit, so the Go<->Python
  parity tests always run — including in CI, where the champion model
  does not exist.
- fixtures.json: Python-computed predictions, TreeSHAP values,
  calibrated PDs, and scorecard scores for 20 test rows (rows 0-1 carry
  injected NaNs to exercise missing-value routing).

Rerun and commit both whenever the export format, calibration, or
scorecard logic changes:

    python scripts/generate_model_fixtures.py
"""

import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.calibrate import calibrate_model, save_calibration, pd_to_score
from pipeline.export_model_json import export_model

logger = logging.getLogger(__name__)

N_FIXTURE_ROWS = 20
N_TRAIN_SAMPLE = 50_000
N_CAL_SAMPLE = 10_000
# feature -> row index in which it is nulled, to exercise NaN routing
NAN_INJECTIONS = {"home_ownership": 0, "loan_amnt": 1, "mths_since_last_delinq": 1}

TESTDATA = (Path(__file__).resolve().parent.parent
            / "go" / "shared" / "model" / "testdata")


def main():
    with open(config.gold_dir() / "feature_metadata.json") as f:
        features = json.load(f)["feature_columns"]

    train = pd.read_parquet(config.gold_dir() / "features_train.parquet")
    val = pd.read_parquet(config.gold_dir() / "features_val.parquet")
    test = pd.read_parquet(config.gold_dir() / "features_test.parquet")

    # Deterministic samples: heads, no RNG involved.
    X_fit, y_fit = train[features].head(N_TRAIN_SAMPLE), train["default"].head(N_TRAIN_SAMPLE)
    X_cal, y_cal = val[features].head(N_CAL_SAMPLE), val["default"].head(N_CAL_SAMPLE)

    model = lgb.LGBMClassifier(
        n_estimators=40, max_depth=4, num_leaves=15, min_child_samples=50,
        learning_rate=0.1, random_state=0, n_jobs=-1, verbose=-1,
    )
    model.fit(X_fit, y_fit)

    # Export through the production path so the parity model exercises
    # the same format the champion ships in.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        joblib.dump(model, tmp / "model.joblib")
        with open(tmp / "model_metadata.json", "w") as f:
            json.dump({
                "version": "parity-test",
                "features": features,
                "n_features": len(features),
                "metrics": {},
                # Stub clean fairness block: _validation_status fails
                # closed on missing fairness data, and the committed
                # parity fixture should stay semantically APPROVED.
                "fairness": {"dir_threshold": config.DIR_THRESHOLD, "attributes": {}},
            }, f)
        calibrator, report = calibrate_model(
            model, X_cal, y_cal, X_cal, y_cal)
        save_calibration(tmp, calibrator, report)
        export_model(tmp)
        TESTDATA.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp / "model.json", TESTDATA / "parity_model.json")

    X = test[features].head(N_FIXTURE_ROWS).reset_index(drop=True).astype(float)
    for col, row in NAN_INJECTIONS.items():
        X.loc[row, col] = np.nan

    proba = model.predict_proba(X)[:, 1]

    # TreeSHAP in log-odds space, matching the Go implementation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # some versions return [class0, class1]
        shap_values = shap_values[1]
    expected_value = explainer.expected_value
    if np.ndim(expected_value) > 0:
        expected_value = expected_value[-1]

    calibrated = calibrator.predict(proba)
    scaled = [int(pd_to_score(p)) for p in calibrated]

    rows = [[None if np.isnan(v) else float(v) for v in row]
            for row in X.to_numpy()]

    fixtures = {
        "features": features,
        "rows": rows,
        "proba": [float(p) for p in proba],
        "shap_values": [[float(v) for v in row] for row in shap_values],
        "shap_expected_value": float(expected_value),
        "calibrated_pd": [float(p) for p in calibrated],
        "scaled_score": scaled,
    }

    with open(TESTDATA / "fixtures.json", "w") as f:
        json.dump(fixtures, f)
    logger.info(f"Wrote parity_model.json and {N_FIXTURE_ROWS} fixture rows to {TESTDATA}")

    # Committed Gold sample so the Go parquet-reader test also runs in CI.
    gold_testdata = TESTDATA.parent.parent / "gold" / "testdata"
    gold_testdata.mkdir(parents=True, exist_ok=True)
    test.head(200).to_parquet(gold_testdata / "sample.parquet", index=False)
    logger.info(f"Wrote 200-row Gold sample to {gold_testdata}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
