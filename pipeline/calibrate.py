"""
Probability Calibration & Scorecard Scaling
Raw GBM scores rank-order well but are not calibrated PDs; downstream
consumers (pricing, provisioning, IFRS 9 / CECL) assume they are.
Fits an isotonic calibrator on the early-stopping holdout (held out
from gradient fitting), reports Brier score and reliability curves on
the test split, and maps calibrated PDs to an industry-style scorecard
score via points-to-double-odds (600 = 30:1 good:bad odds, PDO 20).

Usage (standalone, calibrates an already-trained model):
    python pipeline/calibrate.py [model_dir]   # default: data/models/champion
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config

logger = logging.getLogger(__name__)

DATA_DIR = config.data_dir()
MODELS_DIR = config.models_dir()

# Scorecard anchors: a score of 600 corresponds to 30:1 good:bad odds,
# and every 20 points doubles the odds (PDO).
BASE_SCORE = 600.0
BASE_ODDS = 30.0
PDO = 20.0
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

# Isotonic calibration can output PD of exactly 0 or 1; clip before the
# odds transform so the score stays finite.
PD_CLIP = 1e-6


def pd_to_score(pd):
    """Map PD(s) to integer scorecard score(s).

    floor(x + 0.5) rather than round() so the Go runtime (math.Floor)
    produces identical integers.
    """
    pd = np.clip(np.asarray(pd, dtype=float), PD_CLIP, 1 - PD_CLIP)
    score = OFFSET + FACTOR * np.log((1 - pd) / pd)
    return np.floor(score + 0.5).astype(int)


def scorecard_params():
    """Scorecard constants in exportable form."""
    return {
        "base_score": BASE_SCORE,
        "base_odds": BASE_ODDS,
        "pdo": PDO,
        "factor": float(FACTOR),
        "offset": float(OFFSET),
    }


def fit_calibrator(raw_proba, y):
    """Fit isotonic regression mapping raw scores to calibrated PDs."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_proba, y)
    return iso


def reliability_table(y, proba, n_bins=10):
    """Quantile-binned mean predicted PD vs observed default rate."""
    y = np.asarray(y, dtype=float)
    proba = np.asarray(proba, dtype=float)
    edges = np.unique(np.quantile(proba, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return [{
            "n": int(len(y)),
            "mean_predicted": round(float(proba.mean()), 6),
            "observed_default_rate": round(float(y.mean()), 6),
        }]
    idx = np.clip(np.searchsorted(edges, proba, side="right") - 1, 0, len(edges) - 2)
    table = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        table.append({
            "n": int(mask.sum()),
            "mean_predicted": round(float(proba[mask].mean()), 6),
            "observed_default_rate": round(float(y[mask].mean()), 6),
        })
    return table


def calibrate_model(model, X_cal, y_cal, X_test, y_test):
    """Fit the calibrator on held-out data and evaluate on test."""
    raw_cal = model.predict_proba(X_cal)[:, 1]
    calibrator = fit_calibrator(raw_cal, y_cal)

    raw_test = model.predict_proba(X_test)[:, 1]
    cal_test = calibrator.predict(raw_test)

    report = {
        "method": "isotonic",
        "n_calibration_rows": int(len(X_cal)),
        "n_breakpoints": int(len(calibrator.X_thresholds_)),
        "brier_raw": round(float(brier_score_loss(y_test, raw_test)), 6),
        "brier_calibrated": round(float(brier_score_loss(y_test, cal_test)), 6),
        "reliability_raw": reliability_table(y_test, raw_test),
        "reliability_calibrated": reliability_table(y_test, cal_test),
    }
    logger.info(
        f"[test] Brier raw={report['brier_raw']:.6f}  "
        f"calibrated={report['brier_calibrated']:.6f}  "
        f"({report['n_breakpoints']} isotonic breakpoints)"
    )
    return calibrator, report


def save_calibration(dest_dir, calibrator, report):
    """Save calibrator.joblib and record the report in model_metadata.json."""
    dest_dir = Path(dest_dir)
    joblib.dump(calibrator, dest_dir / "calibrator.joblib")

    meta_path = dest_dir / "model_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["calibration"] = report
    meta["scorecard"] = scorecard_params()
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Calibrator saved to {dest_dir}")


def run(model_dir=None):
    """Calibrate an already-trained model without retraining."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline.train import _model_path, early_stopping_split, load_gold_data

    model_dir = Path(model_dir) if model_dir else MODELS_DIR / "champion"
    model = joblib.load(_model_path(model_dir))

    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_gold_data()
    _, X_es, _, y_es = early_stopping_split(X_train, y_train, X_val, y_val)

    calibrator, report = calibrate_model(model, X_es, y_es, X_test, y_test)
    save_calibration(model_dir, calibrator, report)
    return calibrator, report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    target = sys.argv[1] if len(sys.argv) > 1 else MODELS_DIR / "champion"
    run(target)
