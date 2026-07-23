"""
Model Export for Go Runtime
Dumps a trained LightGBM binary classifier to the portable JSON format
(tree nodes + baseline + metadata) that go/shared/model loads. The
nested LightGBM tree dump is flattened into columnar node arrays with
sklearn-style semantics (value <= threshold goes left, NaN follows
missing_go_to_left), so the Go runtime is model-library agnostic.

Usage:
    python pipeline/export_model_json.py [model_dir]   # default: data/models/challenger
(A promoted champion is immutable; its model.json was published by
`gbm promote`. Export the challenger, then promote.)
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config, io_utils
from pipeline.calibrate import scorecard_params
from pipeline.model_card import _validation_status

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1


def export_tree(tree_structure):
    """Flatten one LightGBM tree (nested dicts) to columnar arrays."""
    cols = {
        "value": [], "count": [], "feature_idx": [], "num_threshold": [],
        "missing_go_to_left": [], "left": [], "right": [], "is_leaf": [],
    }

    def add_node(node):
        idx = len(cols["value"])
        for c in cols.values():
            c.append(0)

        if "leaf_value" in node and "split_feature" not in node:
            cols["value"][idx] = node["leaf_value"]
            cols["count"][idx] = node.get("leaf_count", 0)
            cols["is_leaf"][idx] = 1
            return idx

        if node["decision_type"] != "<=":
            raise ValueError(f"Unsupported decision_type {node['decision_type']!r} "
                             "(categorical splits are not supported by the Go runtime).")

        threshold = float(node["threshold"])
        missing_type = node.get("missing_type", "None")
        if missing_type == "NaN":
            missing_left = bool(node["default_left"])
        elif missing_type == "None":
            # LightGBM treats NaN as 0.0 when the split saw no missing values
            missing_left = 0.0 <= threshold
        else:
            raise ValueError(f"Unsupported missing_type {missing_type!r} "
                             "(train with zero_as_missing=False).")

        cols["count"][idx] = node["internal_count"]
        cols["feature_idx"][idx] = node["split_feature"]
        cols["num_threshold"][idx] = threshold
        cols["missing_go_to_left"][idx] = int(missing_left)
        cols["left"][idx] = add_node(node["left_child"])
        cols["right"][idx] = add_node(node["right_child"])
        return idx

    add_node(tree_structure)
    return cols


def _leaf_value_for_row(tree, row):
    """Walk one exported (columnar) tree for a single row."""
    i = 0
    while not tree["is_leaf"][i]:
        v = row[tree["feature_idx"][i]]
        if np.isnan(v):
            go_left = tree["missing_go_to_left"][i]
        else:
            go_left = v <= tree["num_threshold"][i]
        i = tree["left"][i] if go_left else tree["right"][i]
    return tree["value"][i]


def export_model(model_dir):
    config.assert_mutable_model_dir(model_dir)
    model_dir = Path(model_dir)
    model = joblib.load(config.model_path(model_dir))
    with open(config.metadata_path(model_dir)) as f:
        meta = json.load(f)

    booster = model.booster_
    num_iteration = getattr(model, "best_iteration_", None) or 0
    dump = booster.dump_model(num_iteration=num_iteration)

    if not dump["objective"].startswith("binary"):
        raise ValueError(f"Expected binary objective, got {dump['objective']!r}")

    trees = [export_tree(t["tree_structure"]) for t in dump["tree_info"]]

    # LightGBM bakes any init score into the leaves; measure the residual
    # offset against raw predictions and verify it is constant across a
    # randomized check set (one all-NaN row exercises missing routing).
    rng = np.random.default_rng(0)
    X_check = rng.normal(0, 100, size=(64, meta["n_features"]))
    X_check[0, :] = np.nan
    raw = booster.predict(X_check, raw_score=True, num_iteration=num_iteration)
    sums = np.array([
        sum(_leaf_value_for_row(t, row) for t in trees) for row in X_check
    ])
    offsets = raw - sums
    baseline = float(offsets[0])
    if np.max(np.abs(offsets - baseline)) > 1e-9:
        raise AssertionError("Exported trees do not reproduce LightGBM raw scores "
                             "up to a constant baseline.")

    # Embed the model-card verdict so the serving governance gate can
    # refuse a non-APPROVED champion without re-deriving it (single source
    # of truth: model_card._validation_status over the same metadata).
    status, rationale = _validation_status(meta)

    # Carry the feature-store schema version (from the Gold feature
    # contract) so serving can reject snapshots built under an
    # incompatible feature schema.
    feature_version = None
    fm_path = config.gold_dir() / "feature_metadata.json"
    if fm_path.exists():
        with open(fm_path) as f:
            feature_version = json.load(f).get("feature_version")

    payload = {
        "format_version": FORMAT_VERSION,
        "model_version": meta["version"],
        "n_features": meta["n_features"],
        "features": meta["features"],
        "metrics": meta.get("metrics", {}),
        "baseline_prediction": baseline,
        "trees": trees,
        "validation_status": {"status": status, "rationale": rationale},
    }
    if feature_version is not None:
        payload["feature_version"] = feature_version

    # Optional isotonic calibrator (pipeline/calibrate.py): exported as
    # breakpoints so the Go runtime interpolates without sklearn.
    calibrator_path = model_dir / "calibrator.joblib"
    if calibrator_path.exists():
        iso = joblib.load(calibrator_path)
        payload["calibration"] = {
            "method": "isotonic",
            "x": iso.X_thresholds_.tolist(),
            "y": iso.y_thresholds_.tolist(),
        }
        payload["scorecard"] = scorecard_params()

    # Atomic write: serialize to a temp file, fsync, then rename into
    # place so a starting/reloading runtime never reads a partial model.
    out_path = model_dir / "model.json"
    tmp_path = model_dir / "model.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)

    size_mb = out_path.stat().st_size / 1e6
    logger.info(
        f"Exported {meta['version']}: {len(trees)} trees, "
        f"{meta['n_features']} features -> {out_path} ({size_mb:.1f} MB)"
    )
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    target = sys.argv[1] if len(sys.argv) > 1 else config.challenger_dir()
    with io_utils.registry_lock():
        export_model(target)
