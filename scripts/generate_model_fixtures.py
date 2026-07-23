"""
Regenerate go/shared/model/testdata/fixtures.json from the current
champion. The Go model tests verify pure-Go inference, TreeSHAP,
calibration, and scorecard scaling against these Python-computed values,
so this must be rerun whenever the champion is retrained:

    python scripts/generate_model_fixtures.py

Rows 0-1 carry injected NaNs to exercise missing-value routing.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.calibrate import pd_to_score

logger = logging.getLogger(__name__)

N_ROWS = 20
# feature -> row index in which it is nulled, to exercise NaN routing
NAN_INJECTIONS = {"home_ownership": 0, "loan_amnt": 1, "mths_since_last_delinq": 1}

OUT_PATH = (Path(__file__).resolve().parent.parent
            / "go" / "shared" / "model" / "testdata" / "fixtures.json")


def main():
    champion = config.champion_dir()
    model = joblib.load(config.model_path(champion))
    calibrator = joblib.load(champion / "calibrator.joblib")
    with open(config.metadata_path(champion)) as f:
        features = json.load(f)["features"]

    test = pd.read_parquet(config.gold_dir() / "features_test.parquet")
    X = test[features].head(N_ROWS).reset_index(drop=True).astype(float)
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

    with open(OUT_PATH, "w") as f:
        json.dump(fixtures, f)
    logger.info(f"Wrote {N_ROWS} fixture rows to {OUT_PATH}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
