"""
Model Risk Review Agent
An LLM analyst that reads the weekly monitoring output (drift,
performance, retrain reports, model card) and writes the review memo a
human model-risk reviewer would otherwise assemble by hand.

The agent is advisory only. Every gate in this repository stays
deterministic: PSI/AUC thresholds trigger retraining, the champion-
relative fairness gate blocks promotion, and promotion itself is a
human decision (SR 11-7). This agent prepares the review; it decides
nothing, and the monitoring loop runs unchanged without it.

Governance controls:
- The model ID is pinned via REVIEW_AGENT_MODEL (default claude-opus-4-8).
- Tools are read-only: the two read-only monitors, the model card, and
  model metadata. No database writes, no filesystem writes, no promote.
- Every run appends a full audit record (inputs, memo, token usage) to
  data/review_agent_audit.jsonl.

Usage:
    ANTHROPIC_API_KEY=... python agents/review_agent.py
    # or as the optional llm_review task in the monitoring DAG
"""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config

# The monitoring loop must not depend on the anthropic package: the
# module stays importable without it and run() fails with a clear error.
try:
    import anthropic
    from anthropic import beta_tool
except ImportError:  # pragma: no cover - exercised only without the SDK
    anthropic = None
    beta_tool = None

logger = logging.getLogger(__name__)

ROOT = config.ROOT
DEFAULT_MODEL = "claude-opus-4-8"
MEMO_PATH = ROOT / "docs" / "monitoring_review.md"
MONITOR_TIMEOUT_SECONDS = 600

SYSTEM_PROMPT = """You are a model risk analyst reviewing the weekly \
monitoring run of a consumer credit scoring model (LightGBM, Lending Club \
data, served by a Go API). Your memo is read by the human reviewer who \
makes all decisions; you decide nothing. The deterministic controls \
(PSI/AUC retrain triggers, the champion-relative fairness gate, human \
promotion approval) are the system of record. Never recommend overriding \
them.

Write a concise markdown memo with these sections:

## Summary
Two or three sentences: overall model health and whether anything needs \
the reviewer's attention this week.

## Findings
The important facts from the reports, each with its exact number and \
threshold. Distinguish "breached", "trending toward a threshold", and \
"stable". Say which data the performance monitor used (real backfilled \
outcomes or the test-set proxy).

## Risks and trends
What the numbers imply for the coming weeks if trends continue. Be \
specific about which feature or segment is moving, not just that "drift \
exists".

## Recommended follow-ups
Concrete, small actions for the human reviewer. If nothing is needed, \
say so plainly.

Rules: cite only numbers that appear in the reports or tool results; \
never invent values. If a report is missing, say it is missing rather \
than guessing. Use the tools when you need detail the provided reports \
do not contain. Keep the memo under 600 words."""


def _gbm_path() -> Path:
    import os

    bin_dir = os.getenv("CREDIT_RISK_GO_BIN")
    if bin_dir:
        return Path(bin_dir) / "gbm"
    return ROOT / "go" / "bin" / "gbm"


def run_monitor(monitor: str) -> str:
    """Run a read-only Go monitor and return its JSON report.

    Args:
        monitor: Which monitor to run, either "drift" (PSI on the score
            distribution, CSI per feature) or "performance" (AUC/KS/Gini
            against the training baseline).
    """
    if monitor not in {"drift", "performance"}:
        return f"ERROR: unknown monitor {monitor!r}; use 'drift' or 'performance'."
    gbm = _gbm_path()
    if not gbm.exists():
        return f"ERROR: gbm binary not found at {gbm} (build with: cd go && go build -o bin/gbm .)"
    try:
        result = subprocess.run(
            [str(gbm), monitor], capture_output=True, text=True,
            timeout=MONITOR_TIMEOUT_SECONDS, check=True,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return f"ERROR: {monitor} monitor timed out after {MONITOR_TIMEOUT_SECONDS}s."
    except subprocess.CalledProcessError as e:
        return f"ERROR: {monitor} monitor exited {e.returncode}: {e.stderr[:500]}"


def read_model_card() -> str:
    """Read the current champion's model card (the model_card.md that
    ships inside the champion model directory, MODEL_CARD_PATH override).

    The card carries the validation status, data window, discrimination
    metrics, calibration reliability table, scorecard parameters, and
    the full fairness breakdown.
    """
    override = os.getenv("MODEL_CARD_PATH")
    path = Path(override) if override else config.champion_dir() / "model_card.md"
    if not path.exists():
        return f"ERROR: model card not found at {path}."
    return path.read_text()


def read_model_metadata(role: str) -> str:
    """Read a model's metadata JSON (metrics, calibration, fairness).

    Args:
        role: Which model to read, either "champion" or "challenger".
    """
    if role not in {"champion", "challenger"}:
        return f"ERROR: unknown role {role!r}; use 'champion' or 'challenger'."
    path = config.models_dir() / role / "model_metadata.json"
    if not path.exists():
        return f"ERROR: no {role} metadata at {path}."
    return path.read_text()


def build_prompt(reports: Optional[dict]) -> str:
    """Assemble the user prompt from whatever reports the caller has."""
    lines = [
        "Review this week's monitoring run and write the review memo.",
        "",
    ]
    provided = {k: v for k, v in (reports or {}).items() if v is not None}
    if provided:
        lines.append("Reports from this run:")
        for name, report in provided.items():
            lines.append(f"\n### {name} report\n```json\n{json.dumps(report, indent=2)}\n```")
    else:
        lines.append(
            "No reports were passed in. Run the drift and performance "
            "monitors yourself with the run_monitor tool."
        )
    missing = [k for k, v in (reports or {}).items() if v is None]
    if missing:
        lines.append(f"\nMissing from this run (do not guess their contents): {', '.join(missing)}.")
    lines.append(
        "\nUse read_model_card or read_model_metadata if you need "
        "calibration, fairness, or baseline detail beyond the reports."
    )
    return "\n".join(lines)


def _write_memo(memo: str, model: str) -> Path:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        "# Weekly Monitoring Review\n\n"
        f"Generated by `agents/review_agent.py` ({model}) on {generated_at}. "
        "Advisory only: retrain triggers, the fairness gate, and promotion "
        "remain deterministic and human-approved.\n\n---\n\n"
    )
    MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMO_PATH.write_text(header + memo + "\n")
    return MEMO_PATH


def _append_audit(record: dict) -> Path:
    """Append one JSONL audit record; every run is reconstructable."""
    path = config.data_dir() / "review_agent_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def run(reports: Optional[dict] = None, model: Optional[str] = None) -> str:
    """Run the review agent and return the memo text.

    `reports` maps report names (e.g. "drift", "performance", "retrain")
    to already-parsed JSON reports; None values are reported as missing.
    With no reports, the agent runs the read-only monitors itself.
    """
    if anthropic is None:
        raise RuntimeError(
            "The anthropic package is not installed. "
            "Install it with: pip install -r requirements/agent.txt"
        )
    import os

    model = model or os.getenv("REVIEW_AGENT_MODEL", DEFAULT_MODEL)
    prompt = build_prompt(reports)

    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        tools=[
            beta_tool(run_monitor),
            beta_tool(read_model_card),
            beta_tool(read_model_metadata),
        ],
        messages=[{"role": "user", "content": prompt}],
    )

    usage = {"input_tokens": 0, "output_tokens": 0}
    final = None
    for message in runner:
        final = message
        usage["input_tokens"] += message.usage.input_tokens
        usage["output_tokens"] += message.usage.output_tokens

    memo = "\n".join(b.text for b in final.content if b.type == "text").strip()
    if not memo:
        raise RuntimeError(f"review agent returned no text (stop_reason={final.stop_reason})")

    memo_path = _write_memo(memo, model)
    audit_path = _append_audit({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "prompt": prompt,
        "memo": memo,
        "stop_reason": final.stop_reason,
        "usage": usage,
    })
    logger.info(f"Review memo written to {memo_path} (audit: {audit_path})")
    return memo


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    print(run())
