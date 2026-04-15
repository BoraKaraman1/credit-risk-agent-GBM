"""
Fairness Analysis Module
Analyzes model fairness across protected attribute proxies.
Computes Disparate Impact Ratio (DIR), Equal Opportunity Difference (EOD),
Statistical Parity Difference (SPD), and per-group performance metrics.

Uses home_ownership, verification_status, and emp_length_missing as proxy
attributes. No external fairness library required — metrics computed directly.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models"

# Decision thresholds (mirrored from API)
APPROVE_THRESHOLD = 0.15
REVIEW_THRESHOLD = 0.30

# 80% rule threshold for Disparate Impact Ratio
DIR_THRESHOLD = 0.80

# Reverse mappings for label-encoded categoricals
# (pandas cat.codes sorts alphabetically)
HOME_OWNERSHIP_MAP = {0: "MORTGAGE", 1: "OTHER", 2: "OWN", 3: "RENT"}
VERIFICATION_STATUS_MAP = {0: "Not Verified", 1: "Source Verified", 2: "Verified"}
EMP_LENGTH_MISSING_MAP = {0: "Reported", 1: "Not Reported"}

# Protected attribute configuration
PROTECTED_ATTRIBUTES = {
    "home_ownership": {
        "reverse_map": HOME_OWNERSHIP_MAP,
        "privileged_code": 0,  # MORTGAGE
        "description": "Home Ownership Status",
    },
    "verification_status": {
        "reverse_map": VERIFICATION_STATUS_MAP,
        "privileged_code": 2,  # Verified
        "description": "Income Verification Status",
    },
    "emp_length_missing": {
        "reverse_map": EMP_LENGTH_MISSING_MAP,
        "privileged_code": 0,  # Reported
        "description": "Employment Length Reporting",
    },
}


def _model_path(directory):
    """Resolve model path with backward compat."""
    p = directory / "model.joblib"
    if p.exists():
        return p
    return directory / "model.pkl"


def compute_disparate_impact(approval_rates: dict, privileged_group: str) -> dict:
    """
    Disparate Impact Ratio: approval_rate(group) / approval_rate(privileged).
    A ratio < 0.80 for any group signals potential disparate impact (80% rule).

    Returns dict of {group_name: DIR value} plus a flag for any violations.
    """
    priv_rate = approval_rates.get(privileged_group, 0)
    if priv_rate == 0:
        return {"ratios": {}, "violations": [], "privileged_rate": 0}

    ratios = {}
    violations = []
    for group, rate in approval_rates.items():
        if group == privileged_group:
            ratios[group] = 1.0
            continue
        dir_val = rate / priv_rate
        ratios[group] = round(dir_val, 4)
        if dir_val < DIR_THRESHOLD:
            violations.append(group)

    return {
        "ratios": ratios,
        "violations": violations,
        "privileged_rate": round(priv_rate, 4),
    }


def compute_equal_opportunity_diff(tpr_by_group: dict, privileged_group: str) -> dict:
    """
    Equal Opportunity Difference: TPR(group) - TPR(privileged).
    Measures whether the model equally identifies non-defaulters across groups.
    Positive = group has higher TPR, Negative = group has lower TPR.
    """
    priv_tpr = tpr_by_group.get(privileged_group, 0)
    diffs = {}
    for group, tpr in tpr_by_group.items():
        diffs[group] = round(tpr - priv_tpr, 4)
    return diffs


def compute_statistical_parity_diff(approval_rates: dict, privileged_group: str) -> dict:
    """
    Statistical Parity Difference: P(approve|group) - P(approve|privileged).
    Measures whether approval rates differ across groups.
    """
    priv_rate = approval_rates.get(privileged_group, 0)
    diffs = {}
    for group, rate in approval_rates.items():
        diffs[group] = round(rate - priv_rate, 4)
    return diffs


def compute_group_metrics(y_true: np.ndarray, y_score: np.ndarray,
                          group_labels: np.ndarray, group_names: dict) -> list[dict]:
    """
    Per-group performance metrics: AUC, default rate, approval rate, mean score.
    """
    decisions = np.where(y_score < APPROVE_THRESHOLD, "approve",
                np.where(y_score < REVIEW_THRESHOLD, "review", "decline"))

    results = []
    for code, name in sorted(group_names.items()):
        mask = group_labels == code
        if mask.sum() < 10:
            continue

        group_y = y_true[mask]
        group_scores = y_score[mask]
        group_decisions = decisions[mask]

        # AUC (only if both classes present)
        auc = None
        if len(np.unique(group_y)) > 1:
            auc = round(float(roc_auc_score(group_y, group_scores)), 4)

        results.append({
            "group": name,
            "code": int(code),
            "count": int(mask.sum()),
            "default_rate": round(float(group_y.mean()), 4),
            "approval_rate": round(float((group_decisions == "approve").mean()), 4),
            "decline_rate": round(float((group_decisions == "decline").mean()), 4),
            "mean_score": round(float(group_scores.mean()), 4),
            "auc": auc,
        })
    return results


def analyze_attribute(y_true: np.ndarray, y_score: np.ndarray,
                      attribute_values: np.ndarray, attr_config: dict,
                      attr_name: str) -> dict:
    """Full fairness analysis for a single protected attribute."""
    reverse_map = attr_config["reverse_map"]
    privileged_code = attr_config["privileged_code"]
    privileged_name = reverse_map[privileged_code]

    # Per-group metrics
    group_metrics = compute_group_metrics(y_true, y_score, attribute_values, reverse_map)

    # Extract approval rates and TPRs by group name
    approval_rates = {m["group"]: m["approval_rate"] for m in group_metrics}

    # True Positive Rate: P(approve | non-defaulter) per group
    decisions = np.where(y_score < APPROVE_THRESHOLD, 1, 0)  # 1=approve, 0=not approve
    tpr_by_group = {}
    for m in group_metrics:
        mask = attribute_values == m["code"]
        non_defaulters = y_true[mask] == 0
        if non_defaulters.sum() > 0:
            tpr_by_group[m["group"]] = float(decisions[mask][non_defaulters].mean())
        else:
            tpr_by_group[m["group"]] = 0.0

    # Fairness metrics
    dir_result = compute_disparate_impact(approval_rates, privileged_name)
    eod_result = compute_equal_opportunity_diff(tpr_by_group, privileged_name)
    spd_result = compute_statistical_parity_diff(approval_rates, privileged_name)

    return {
        "attribute": attr_name,
        "description": attr_config["description"],
        "privileged_group": privileged_name,
        "group_metrics": group_metrics,
        "disparate_impact": dir_result,
        "equal_opportunity_diff": eod_result,
        "statistical_parity_diff": spd_result,
        "has_dir_violation": len(dir_result["violations"]) > 0,
    }


def run(model=None, X_test=None, y_test=None) -> dict:
    """
    Run fairness analysis across all protected attributes.
    If model/X_test/y_test not provided, loads from disk.
    """
    if model is None or X_test is None or y_test is None:
        with open(GOLD_DIR / "feature_metadata.json") as f:
            meta = json.load(f)
        feature_cols = meta["feature_columns"]

        model_path = _model_path(MODELS_DIR / "champion")
        model = joblib.load(model_path)

        test = pd.read_parquet(GOLD_DIR / "features_test.parquet")
        X_test = test[feature_cols]
        y_test = test["default"].values
    else:
        feature_cols = list(X_test.columns) if hasattr(X_test, "columns") else None

    y_score = model.predict_proba(X_test)[:, 1]
    y_true = np.asarray(y_test)

    if isinstance(X_test, pd.DataFrame):
        X_values = X_test
    else:
        X_values = pd.DataFrame(X_test, columns=feature_cols)

    logger.info(f"Running fairness analysis on {len(y_true):,} observations")

    results = {}
    for attr_name, attr_config in PROTECTED_ATTRIBUTES.items():
        if attr_name not in X_values.columns:
            logger.warning(f"Attribute {attr_name} not found in features, skipping")
            continue

        attr_values = X_values[attr_name].values.astype(int)
        results[attr_name] = analyze_attribute(
            y_true, y_score, attr_values, attr_config, attr_name
        )

        if results[attr_name]["has_dir_violation"]:
            violations = results[attr_name]["disparate_impact"]["violations"]
            logger.warning(f"DIR violation for {attr_name}: groups {violations} below 80% threshold")

    report = {
        "n_observations": len(y_true),
        "overall_approval_rate": round(float((y_score < APPROVE_THRESHOLD).mean()), 4),
        "overall_default_rate": round(float(y_true.mean()), 4),
        "thresholds": {"approve": APPROVE_THRESHOLD, "review": REVIEW_THRESHOLD},
        "dir_threshold": DIR_THRESHOLD,
        "attributes": results,
    }

    logger.info(format_report(report))
    return report


def format_report(report: dict) -> str:
    """Format the report dict as a human-readable summary."""
    lines = [
        "",
        "=" * 70,
        "FAIRNESS ANALYSIS REPORT",
        "=" * 70,
        f"Observations: {report['n_observations']:,}",
        f"Overall approval rate: {report['overall_approval_rate']:.2%}",
        f"Overall default rate: {report['overall_default_rate']:.2%}",
        "",
    ]

    for attr_name, attr_result in report["attributes"].items():
        lines.append(f"--- {attr_result['description']} ({attr_name}) ---")
        lines.append(f"Privileged group: {attr_result['privileged_group']}")
        lines.append("")

        # Group metrics table
        lines.append(f"{'Group':<20} {'Count':>8} {'Def Rate':>10} {'Appr Rate':>10} {'Mean PD':>10} {'AUC':>8}")
        lines.append("-" * 70)
        for m in attr_result["group_metrics"]:
            auc_str = f"{m['auc']:.4f}" if m["auc"] is not None else "N/A"
            lines.append(
                f"{m['group']:<20} {m['count']:>8,} {m['default_rate']:>10.4f} "
                f"{m['approval_rate']:>10.4f} {m['mean_score']:>10.4f} {auc_str:>8}"
            )

        # DIR
        dir_result = attr_result["disparate_impact"]
        lines.append(f"\nDisparate Impact Ratios (threshold: {report['dir_threshold']}):")
        for group, ratio in dir_result["ratios"].items():
            flag = " ** VIOLATION **" if group in dir_result["violations"] else ""
            lines.append(f"  {group}: {ratio:.4f}{flag}")

        # SPD
        spd = attr_result["statistical_parity_diff"]
        lines.append(f"\nStatistical Parity Differences:")
        for group, diff in spd.items():
            lines.append(f"  {group}: {diff:+.4f}")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    report = run()
    print(json.dumps(report, indent=2, default=str))
