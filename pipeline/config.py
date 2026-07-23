"""
Pipeline configuration: environment-aware paths and flags shared across
the pipeline modules. Mirrors go/shared/config so the Python pipeline and
the Go services agree on data/model locations in every deployment
(CREDIT_RISK_DATA_DIR / CREDIT_RISK_MODELS_DIR), instead of each module
hardcoding paths relative to its own file.
"""

import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

# Single source of truth for the thresholds shared with the Go services
# (go:embed of the same file) and the UI. Loaded eagerly so a missing or
# malformed contract fails at import, never at decision time.
CONTRACT_PATH = ROOT / "go" / "shared" / "config" / "contract.json"
with open(CONTRACT_PATH) as _f:
    _CONTRACT = json.load(_f)

APPROVE_THRESHOLD = _CONTRACT["decision"]["approve_below"]
REVIEW_THRESHOLD = _CONTRACT["decision"]["review_below"]
PSI_WARNING = _CONTRACT["monitoring"]["psi_warning"]
PSI_CRITICAL = _CONTRACT["monitoring"]["psi_critical"]
CSI_THRESHOLD = _CONTRACT["monitoring"]["csi_threshold"]
AUC_DROP_THRESHOLD = _CONTRACT["monitoring"]["auc_drop_threshold"]
DIR_THRESHOLD = _CONTRACT["fairness"]["dir_threshold"]
DIR_WORSEN_TOLERANCE = _CONTRACT["fairness"]["dir_worsen_tolerance"]


def data_dir() -> Path:
    """Root of the data lake. Override with CREDIT_RISK_DATA_DIR."""
    override = os.getenv("CREDIT_RISK_DATA_DIR")
    return Path(override) if override else ROOT / "data"


def bronze_dir() -> Path:
    return data_dir() / "bronze"


def silver_dir() -> Path:
    return data_dir() / "silver"


def gold_dir() -> Path:
    return data_dir() / "gold"


def models_dir() -> Path:
    """Model registry. Override with CREDIT_RISK_MODELS_DIR."""
    override = os.getenv("CREDIT_RISK_MODELS_DIR")
    return Path(override) if override else data_dir() / "models"


def champion_dir() -> Path:
    return models_dir() / "champion"


def challenger_dir() -> Path:
    return models_dir() / "challenger"


def model_path(directory) -> Path:
    """Resolve a saved model with backward compat (joblib first, then pkl)."""
    p = Path(directory) / "model.joblib"
    if p.exists():
        return p
    return Path(directory) / "model.pkl"


def parse_version(version: str) -> tuple[int, int]:
    """Parse "v<major>.<minor>[-tag]" into (major, minor). A "-tag"
    suffix (e.g. reject inference's "-ri") is ignored, so a tagged
    challenger can be promoted without breaking later version bumps.
    Must stay in sync with nextVersion in go/monitoring/retrain.go."""
    core = version.lstrip("v").split("-", 1)[0]
    major, minor = core.split(".")
    return int(major), int(minor)


def next_version(suffix: str = "") -> str:
    """Next model version: champion minor + 1 (v1.2 -> v1.3), or v1.0
    when no champion exists (a broken champion symlink reads as absent,
    restarting at v1.0 — promote then refuses the duplicate version).
    `suffix` tags the produced version without affecting the bump."""
    meta = metadata_path(champion_dir())
    if not meta.exists():
        return f"v1.0{suffix}"
    with open(meta) as f:
        prev = json.load(f).get("version", "v1.0")
    try:
        major, minor = parse_version(prev)
    except ValueError:
        logger.warning("unparseable champion version %r; restarting at v1.0", prev)
        return f"v1.0{suffix}"
    return f"v{major}.{minor + 1}{suffix}"


def assert_mutable_model_dir(directory) -> None:
    """Refuse writes into a promoted champion. After `gbm promote`,
    models/champion is a symlink into the immutable versions/ archive;
    new models are written to challenger/ and promoted (single door).
    Legacy real champion directories remain writable. Checked before
    any mkdir/write, since mkdir silently traverses symlinks."""
    d = Path(directory)
    if d.is_symlink():
        raise RuntimeError(
            f"{d} is a promoted champion (symlink into the immutable versions/ "
            "archive); write to the challenger and run `gbm promote` instead"
        )


def metadata_path(directory) -> Path:
    return Path(directory) / "model_metadata.json"


def strict_data_quality() -> bool:
    """When true, data-quality validation failures abort the pipeline
    instead of only logging a warning. Off by default for local
    exploration; set CREDIT_RISK_STRICT_DQ=true in CI/production so bad
    data cannot silently flow through to training and serving."""
    return os.getenv("CREDIT_RISK_STRICT_DQ", "").strip().lower() in {"1", "true", "yes"}


def enforce_data_quality(context: str, message: str) -> None:
    """Handle a data-quality failure: raise in strict mode so the
    pipeline/DAG task fails, otherwise log a warning. `context` names the
    layer (e.g. "Silver"), `message` describes what failed."""
    full = f"{context} data quality failed: {message}"
    if strict_data_quality():
        raise RuntimeError(full)
    logger.warning(full)
