"""
Model Card / Validation Report Generator
Assembles a markdown model card from a trained model's metadata: data
window, discrimination metrics, calibration, scorecard, fairness, and a
validation status. This is the artifact a Model Risk Management team
reviews (SR 11-7) before promotion. Every training run writes the card
into the model's own directory (model_card.md), and `gbm promote`
carries it into the immutable version dir alongside model.json.

Usage:
    python pipeline/model_card.py [model_dir] [output_path]
    # defaults: data/models/champion -> docs/model_card.md
    # (regenerates the committed repo snapshot from the champion)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config

logger = logging.getLogger(__name__)

ROOT = config.ROOT
GOLD_DIR = config.gold_dir()
MODELS_DIR = config.models_dir()
DEFAULT_OUTPUT = ROOT / "docs" / "model_card.md"


def _pct(x):
    return f"{x * 100:.2f}%" if x is not None else "N/A"


def _validation_status(meta, champion_fairness=None):
    """Derive a one-line validation verdict from the metadata.

    Absolute rule: any DIR violation fails review. With
    `champion_fairness` (the incumbent champion's fairness summary) the
    rule is champion-relative, mirroring the Go retrain gate
    (go/monitoring/retrain.go fairnessGate): a violation blocks only if
    it is new, or worsens the champion's DIR on that group by more than
    the contract's dir_worsen_tolerance. Missing fairness data fails
    closed; absence of analysis is never approval.
    """
    fair = meta.get("fairness")
    if not fair:
        return "REVIEW REQUIRED", (
            "No fairness analysis recorded; DIR compliance cannot be verified."
        )

    thr = fair.get("dir_threshold", config.DIR_THRESHOLD)
    blocking, inherited = [], []
    for attr, a in fair.get("attributes", {}).items():
        champ_groups = {}
        if champion_fairness:
            champ_groups = (champion_fairness.get("attributes", {})
                            .get(attr, {}).get("groups", {}))
        for group in a.get("violations", []):
            champ_dir = champ_groups.get(group, {}).get("dir")
            if champ_dir is not None and champ_dir < thr:
                chal_dir = a.get("groups", {}).get(group, {}).get("dir")
                if chal_dir is not None and \
                        chal_dir < champ_dir - config.DIR_WORSEN_TOLERANCE:
                    blocking.append(f"{attr}/{group} (worsened: "
                                    f"{champ_dir:.2f} -> {chal_dir:.2f})")
                else:
                    inherited.append(f"{attr}/{group}")
            else:
                blocking.append(f"{attr}/{group}")

    if blocking:
        return "REVIEW REQUIRED", (
            "Disparate Impact Ratio below the 0.80 four-fifths rule for: "
            + ", ".join(blocking) + "."
        )
    if "calibration" not in meta:
        return "REVIEW REQUIRED", "Model is not calibrated; predicted scores are not usable as PDs."
    if inherited:
        return "APPROVED", (
            "Discrimination, calibration, and fairness checks passed "
            "(champion-relative: inherited DIR violations on "
            + ", ".join(inherited) + " not worsened)."
        )
    return "APPROVED", "Discrimination, calibration, and fairness checks passed."


def champion_fairness_for(model_dir):
    """The incumbent champion's fairness summary, when `model_dir` is
    the challenger and a champion exists; that context makes
    _validation_status champion-relative. None otherwise, so the
    absolute rule applies at bootstrap and when rendering the champion
    itself (a champion must never be scored against its own summary)."""
    if Path(model_dir).resolve() != config.challenger_dir().resolve():
        return None
    champ_meta = config.metadata_path(config.champion_dir())
    if not champ_meta.exists():
        return None
    with open(champ_meta) as f:
        return json.load(f).get("fairness")


def _metrics_table(metrics):
    lines = ["| Split | AUC | KS | Gini |", "|-------|-----|----|----|"]
    # Preferred display order; any other keys follow in insertion order.
    order = ["train", "early_stopping", "val", "test"]
    keys = [k for k in order if k in metrics] + [k for k in metrics if k not in order]
    for split in keys:
        m = metrics[split]
        label = split.replace("_", " ").title()
        lines.append(f"| {label} | {m['auc']:.4f} | {m['ks']:.4f} | {m['gini']:.4f} |")
    return "\n".join(lines)


def _data_window(feature_meta):
    splits = feature_meta.get("splits", {})
    lines = [
        "Time-aware split: " + feature_meta.get("split_method", "not recorded") + ".",
        "",
        "| Split | Rows | Default rate |",
        "|-------|------|--------------|",
    ]
    for split in ("train", "val", "test"):
        s = splits.get(split)
        if not s:
            continue
        lines.append(f"| {split.capitalize()} | {s['rows']:,} | {_pct(s.get('default_rate'))} |")
    return "\n".join(lines)


def _calibration_section(cal):
    lines = [
        f"Method: {cal.get('method', 'n/a')}, fit on the early-stopping holdout "
        f"({cal.get('n_calibration_rows', 0):,} rows), {cal.get('n_breakpoints', 0)} breakpoints.",
        "",
        f"Test Brier score: {cal.get('brier_raw')} raw, {cal.get('brier_calibrated')} calibrated.",
        "",
        "Reliability after calibration (quantile bins):",
        "",
        "| Mean predicted PD | Observed default rate | N |",
        "|-------------------|-----------------------|---|",
    ]
    for row in cal.get("reliability_calibrated", []):
        lines.append(
            f"| {row['mean_predicted']:.4f} | {row['observed_default_rate']:.4f} | {row['n']:,} |")
    return "\n".join(lines)


def _scorecard_section(sc):
    return (
        f"Points-to-double-odds scaling: base score {sc['base_score']:.0f} at "
        f"{sc['base_odds']:.0f}:1 good:bad odds, PDO {sc['pdo']:.0f}. "
        f"score = {sc['offset']:.2f} + {sc['factor']:.2f} * ln((1 - pd) / pd)."
    )


def _fairness_section(fair):
    thr = fair.get("dir_threshold", 0.80)
    lines = [
        f"Disparate Impact Ratio threshold: {thr} (four-fifths rule). "
        "Protected attributes are proxies, not collected directly.",
    ]
    for attr, a in fair.get("attributes", {}).items():
        lines.append("")
        lines.append(f"### {a.get('description', attr)} (privileged: {a.get('privileged_group')})")
        lines.append("")
        lines.append("| Group | DIR | Approval rate | Default rate | Status |")
        lines.append("|-------|-----|---------------|--------------|--------|")
        violations = set(a.get("violations", []))
        for group, g in a.get("groups", {}).items():
            status = "VIOLATION" if group in violations else "ok"
            lines.append(
                f"| {group} | {g['dir']:.4f} | {_pct(g.get('approval_rate'))} | "
                f"{_pct(g.get('default_rate'))} | {status} |")
    return "\n".join(lines)


def _hyperparameters_section(params):
    if not params:
        return "Not recorded for this model."
    lines = ["| Parameter | Value |", "|-----------|-------|"]
    for k in sorted(params):
        lines.append(f"| {k} | {params[k]} |")
    return "\n".join(lines)


def render(meta, feature_meta, champion_fairness=None):
    """Render the model card markdown from metadata dicts."""
    status, rationale = _validation_status(meta, champion_fairness)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections = [
        f"# Model Card: Credit Risk GBM {meta.get('version', '')}",
        "",
        f"Auto-generated by `pipeline/model_card.py` on {generated_at}. "
        "Written alongside the model on every training run; the reviewed "
        "card travels with the model through promotion.",
        "",
        "## Validation Status",
        "",
        f"**{status}.** {rationale}",
        "",
        "## Model",
        "",
        f"- Version: {meta.get('version', 'n/a')}",
        f"- Trained at: {meta.get('trained_at', 'n/a')}",
        "- Algorithm: LightGBM gradient-boosted trees (binary classification, log-odds output)",
        f"- Features: {meta.get('n_features', 'n/a')}",
        "",
        "## Data Window",
        "",
        _data_window(feature_meta),
        "",
        "## Discrimination Metrics",
        "",
        _metrics_table(meta.get("metrics", {})),
    ]

    if "calibration" in meta:
        sections += ["", "## Calibration", "", _calibration_section(meta["calibration"])]
    if "scorecard" in meta:
        sections += ["", "## Scorecard Scaling", "", _scorecard_section(meta["scorecard"])]
    if "fairness" in meta:
        sections += ["", "## Fairness", "", _fairness_section(meta["fairness"])]

    sections += [
        "",
        "## Hyperparameters",
        "",
        _hyperparameters_section(meta.get("hyperparameters")),
        "",
    ]
    return "\n".join(sections)


def generate(model_dir=None, output_path=None):
    """Generate the model card for a model directory. A challenger's
    verdict is computed champion-relative (champion_fairness_for), so
    the card a human reviews always matches the gate the exported
    model.json enforces."""
    model_dir = Path(model_dir) if model_dir else MODELS_DIR / "champion"
    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT

    with open(config.metadata_path(model_dir)) as f:
        meta = json.load(f)
    with open(GOLD_DIR / "feature_metadata.json") as f:
        feature_meta = json.load(f)

    card = render(meta, feature_meta, champion_fairness_for(model_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(card)

    logger.info(f"Model card written to {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    model_dir = sys.argv[1] if len(sys.argv) > 1 else None
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    generate(model_dir, output_path)
