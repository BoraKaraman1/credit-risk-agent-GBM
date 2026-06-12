"""
Model Export for Go Runtime
Dumps a trained HistGradientBoostingClassifier to a portable JSON format
(tree nodes + baseline + metadata) that go/internal/model can load.

Usage:
    python pipeline/export_model_json.py [model_dir]   # default: data/models/champion
"""

import json
import logging
import sys
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

FORMAT_VERSION = 1


def _model_path(directory):
    """Resolve model path with backward compat."""
    p = directory / "model.joblib"
    if p.exists():
        return p
    return directory / "model.pkl"


def export_tree(predictor):
    """Convert one TreePredictor's node array to columnar lists."""
    nodes = predictor.nodes
    if nodes["is_categorical"].any():
        raise ValueError("Categorical splits are not supported by the Go runtime.")
    return {
        "value": nodes["value"].tolist(),
        "count": nodes["count"].tolist(),
        "feature_idx": nodes["feature_idx"].tolist(),
        "num_threshold": nodes["num_threshold"].tolist(),
        "missing_go_to_left": nodes["missing_go_to_left"].astype(int).tolist(),
        "left": nodes["left"].tolist(),
        "right": nodes["right"].tolist(),
        "is_leaf": nodes["is_leaf"].astype(int).tolist(),
    }


def export_model(model_dir):
    model_dir = Path(model_dir)
    model = joblib.load(_model_path(model_dir))
    with open(model_dir / "model_metadata.json") as f:
        meta = json.load(f)

    predictors = model._predictors
    if any(len(per_iter) != 1 for per_iter in predictors):
        raise ValueError("Expected binary classification (one tree per iteration).")

    payload = {
        "format_version": FORMAT_VERSION,
        "model_version": meta["version"],
        "n_features": meta["n_features"],
        "features": meta["features"],
        "metrics": meta.get("metrics", {}),
        "baseline_prediction": float(model._baseline_prediction.ravel()[0]),
        "trees": [export_tree(per_iter[0]) for per_iter in predictors],
    }

    out_path = model_dir / "model.json"
    with open(out_path, "w") as f:
        json.dump(payload, f)

    size_mb = out_path.stat().st_size / 1e6
    logger.info(
        f"Exported {meta['version']}: {len(payload['trees'])} trees, "
        f"{meta['n_features']} features -> {out_path} ({size_mb:.1f} MB)"
    )
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    target = sys.argv[1] if len(sys.argv) > 1 else MODELS_DIR / "champion"
    export_model(target)
