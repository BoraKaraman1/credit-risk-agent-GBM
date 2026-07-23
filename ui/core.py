"""
Pure helpers for the Streamlit UI: shaping drift_log rows and API
responses into DataFrames and presentation dicts. No streamlit imports,
so everything here is unit-testable with plain pandas.
"""

import json
from pathlib import Path

import pandas as pd

# Chart colors. Marks use one categorical hue (every chart here is a
# single series); status colors annotate thresholds and always appear
# with a text label, never as the only carrier of meaning.
SERIES = "#2a78d6"
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}
TEXT_SECONDARY = "#52514e"

# Monitor thresholds from the cross-language contract — the same file
# the Go monitors embed, so chart annotations can never drift from the
# thresholds actually enforced. (Dockerfile.ui copies it alongside ui/.)
_CONTRACT_PATH = Path(__file__).resolve().parent.parent / "go" / "shared" / "config" / "contract.json"
_CONTRACT = json.loads(_CONTRACT_PATH.read_text())

PSI_WARNING = _CONTRACT["monitoring"]["psi_warning"]
PSI_CRITICAL = _CONTRACT["monitoring"]["psi_critical"]
AUC_DROP_THRESHOLD = _CONTRACT["monitoring"]["auc_drop_threshold"]

DECISION_PRESENTATION = {
    "approve": {"label": "Approve", "icon": "✅", "status": "good"},
    "review": {"label": "Manual review", "icon": "⚠️", "status": "warning"},
    "decline": {"label": "Decline", "icon": "⛔", "status": "critical"},
}


def decision_presentation(decision: str) -> dict:
    """Label/icon/status for a decision band; unknown values pass through."""
    return DECISION_PRESENTATION.get(
        decision, {"label": decision, "icon": "❔", "status": "warning"}
    )


def psi_status(psi: float) -> str:
    if psi > PSI_CRITICAL:
        return "CRITICAL"
    if psi > PSI_WARNING:
        return "WARNING"
    return "OK"


def metric_history_frame(rows) -> pd.DataFrame:
    """Shape drift_log rows (measured_at, metric_value, model_version,
    details) into a chart-ready frame; empty input yields empty frame."""
    df = pd.DataFrame(rows, columns=["measured_at", "metric_value", "model_version", "details"])
    if df.empty:
        return df
    df["measured_at"] = pd.to_datetime(df["measured_at"])
    return df.sort_values("measured_at").reset_index(drop=True)


def csi_frame(details: dict) -> pd.DataFrame:
    """Per-feature CSI from a psi row's details, highest first."""
    csi = (details or {}).get("csi") or {}
    df = pd.DataFrame(
        [{"feature": k, "csi": v} for k, v in csi.items()],
        columns=["feature", "csi"],
    )
    return df.sort_values("csi", ascending=False).reset_index(drop=True)


def decile_frame(details: dict) -> pd.DataFrame:
    """Decile analysis from an auc row's details (decile, count,
    default_rate, avg_score)."""
    deciles = (details or {}).get("decile_analysis") or []
    return pd.DataFrame(deciles, columns=["decile", "count", "default_rate", "avg_score"])


def fairness_frame(details: dict) -> pd.DataFrame:
    """Flatten a fairness summary (drift_log details) into one row per
    attribute/group with DIR, EOD, SPD, rates, and violation flags."""
    rows = []
    for attr, a in (details or {}).get("attributes", {}).items():
        violations = set(a.get("violations", []))
        for group, g in a.get("groups", {}).items():
            rows.append({
                "attribute": a.get("description", attr),
                "group": group,
                "privileged": group == a.get("privileged_group"),
                "dir": g.get("dir"),
                "eod": g.get("eod"),
                "spd": g.get("spd"),
                "approval_rate": g.get("approval_rate"),
                "default_rate": g.get("default_rate"),
                "violation": group in violations,
            })
    return pd.DataFrame(rows, columns=[
        "attribute", "group", "privileged", "dir", "eod", "spd",
        "approval_rate", "default_rate", "violation",
    ])


def adverse_actions_frame(actions) -> pd.DataFrame:
    """Adverse actions from a /score response, one row per reason."""
    rows = [{
        "code": a.get("code"),
        "reason": a.get("reason"),
        "feature": a.get("feature_name"),
        "shap": a.get("shap_value"),
        "value": a.get("feature_value"),
    } for a in (actions or [])]
    return pd.DataFrame(rows, columns=["code", "reason", "feature", "shap", "value"])


# --- Model governance helpers (champion metadata + /health shaping) ---


def metrics_frame(metadata: dict) -> pd.DataFrame:
    """Discrimination metrics per split (train/early_stopping/test) from
    champion model_metadata.json, one row per split."""
    rows = []
    for split, m in ((metadata or {}).get("metrics") or {}).items():
        rows.append({
            "split": split.replace("_", " "),
            "auc": m.get("auc"), "ks": m.get("ks"), "gini": m.get("gini"),
        })
    return pd.DataFrame(rows, columns=["split", "auc", "ks", "gini"])


def reliability_frame(calibration: dict) -> pd.DataFrame:
    """Reliability (calibration) curve points, long form with a series
    column distinguishing raw vs calibrated predictions."""
    rows = []
    for series, key in (("raw", "reliability_raw"),
                        ("calibrated", "reliability_calibrated")):
        for b in (calibration or {}).get(key) or []:
            rows.append({
                "series": series,
                "mean_predicted": b.get("mean_predicted"),
                "observed": b.get("observed_default_rate"),
                "n": b.get("n"),
            })
    return pd.DataFrame(rows, columns=["series", "mean_predicted", "observed", "n"])


def calibration_summary(metadata: dict) -> dict:
    """Headline calibration facts from champion metadata; missing fields
    come back as None so the page can degrade gracefully."""
    cal = (metadata or {}).get("calibration") or {}
    raw, fitted = cal.get("brier_raw"), cal.get("brier_calibrated")
    gain = (raw - fitted) if isinstance(raw, (int, float)) and isinstance(fitted, (int, float)) else None
    return {
        "method": cal.get("method"),
        "n_calibration_rows": cal.get("n_calibration_rows"),
        "n_breakpoints": cal.get("n_breakpoints"),
        "brier_raw": raw,
        "brier_calibrated": fitted,
        "brier_gain": gain,
    }


def validation_status_from_card(card_text: str) -> dict:
    """Extract the governance verdict from the generated model card's
    Validation Status section. The card is the human-readable source of
    the same verdict model.json carries for the serving gate."""
    text = card_text or ""
    if "REVIEW REQUIRED" in text:
        return {"status": "REVIEW REQUIRED", "kind": "critical", "icon": "⛔"}
    if "APPROVED" in text:
        return {"status": "APPROVED", "kind": "good", "icon": "✅"}
    return {"status": "UNKNOWN", "kind": "warning", "icon": "❔"}


def health_presentation(health: dict) -> dict:
    """Shape a /health response into status tiles. `status` is ok/degraded
    as reported by the API (it returns 503 when degraded)."""
    status = (health or {}).get("status", "unknown")
    kind = {"ok": "good", "degraded": "critical"}.get(status, "warning")
    return {
        "status": status,
        "kind": kind,
        "model_version": (health or {}).get("model_version", "?"),
        "calibrated": bool((health or {}).get("calibrated")),
        "database": (health or {}).get("database", "unknown"),
        "loaded_at": (health or {}).get("model_loaded_at"),
    }
