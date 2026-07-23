"""
Airflow DAG: Credit Risk Monitoring
Weekly monitoring: outcome backfill + drift check + performance check →
conditional retrain → LLM review memo. The backfill matures scored
outcomes first so the performance monitor reads real labels; drift runs
in parallel (it uses scores, not outcomes). A branching decision then
determines whether to trigger the retrain orchestrator. Finally the
optional review agent summarizes the run for the human reviewer; it is
advisory and never blocks or changes the loop.

The monitors are gbm subcommands (see go/) that print a JSON report to
stdout; this DAG shells out to them and routes the reports via XCom.
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

# Go binaries live in /opt/airflow/bin inside the Docker image; for a
# local Airflow run from the repo root they are in go/bin.
GO_BIN_DIR = os.environ.get("CREDIT_RISK_GO_BIN", "/opt/airflow/bin")

default_args = {
    "owner": "credit_risk",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run_go(subcommand, *args):
    """Run a gbm subcommand and return its parsed JSON report."""
    cmd = [os.path.join(GO_BIN_DIR, "gbm"), subcommand, *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.stderr:
        logger.info("%s stderr:\n%s", subcommand, result.stderr)
    return json.loads(result.stdout)


def _run_outcome_backfill(**kwargs):
    """Mature scored outcomes before the performance monitor reads them.
    Tolerant of a missing database so monitoring still proceeds."""
    try:
        report = _run_go("backfill")
        logger.info("outcome backfill: %s", report)
        return report
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning("outcome backfill skipped: %s", e)
        return None


def _run_drift_monitor(**kwargs):
    report = _run_go("drift")
    kwargs["ti"].xcom_push(key="drift_report", value=report)
    return report


def _run_prune(**kwargs):
    """Enforce scoring_log retention (Reg B window, default 750 days).
    Tolerant of a missing database, like the backfill."""
    try:
        report = _run_go("prune")
        logger.info("scoring_log prune: %s", report)
        return report
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning("scoring_log prune skipped: %s", e)
        return None


def _run_performance_monitor(**kwargs):
    report = _run_go("performance")
    kwargs["ti"].xcom_push(key="perf_report", value=report)
    return report


def _decide_retrain(**kwargs):
    """Route on the monitors' machine-readable verdicts. The retrain
    rule has exactly one implementation — the Go monitors, thresholds
    from contract.json — and this branch only reads their
    needs_retrain/retrain_reasons fields."""
    ti = kwargs["ti"]
    drift_report = ti.xcom_pull(task_ids="drift_monitor", key="drift_report")
    perf_report = ti.xcom_pull(task_ids="performance_monitor", key="perf_report")

    reasons = []
    for report in (drift_report, perf_report):
        if report and report.get("needs_retrain"):
            reasons.extend(report.get("retrain_reasons", []))

    if reasons:
        ti.xcom_push(key="retrain_reason", value=", ".join(reasons))
        return "retrain"
    return "skip_retrain"


def _run_retrain(**kwargs):
    reason = kwargs["ti"].xcom_pull(task_ids="decide_retrain", key="retrain_reason") or "monitoring_trigger"
    report = _run_go("retrain", reason)
    return report


def _run_llm_review(**kwargs):
    """Write the advisory review memo for this run (agents/review_agent.py).
    Optional by design: a missing anthropic install or API key logs a
    warning and the task succeeds, so the deterministic loop never
    depends on the LLM."""
    ti = kwargs["ti"]
    reports = {
        "drift": ti.xcom_pull(task_ids="drift_monitor", key="drift_report"),
        "performance": ti.xcom_pull(task_ids="performance_monitor", key="perf_report"),
        "retrain": ti.xcom_pull(task_ids="retrain"),
    }
    try:
        from agents.review_agent import run as run_review

        memo = run_review(reports=reports)
        logger.info("LLM review memo written to docs/monitoring_review.md")
        return memo
    except Exception as e:
        logger.warning("LLM review skipped: %s", e)
        return None


with DAG(
    dag_id="credit_risk_monitoring",
    default_args=default_args,
    description="Weekly model monitoring: drift, performance, conditional retrain",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit_risk", "monitoring"],
) as dag:

    backfill = PythonOperator(task_id="outcome_backfill", python_callable=_run_outcome_backfill)
    drift = PythonOperator(task_id="drift_monitor", python_callable=_run_drift_monitor)
    perf = PythonOperator(task_id="performance_monitor", python_callable=_run_performance_monitor)
    decide = BranchPythonOperator(task_id="decide_retrain", python_callable=_decide_retrain)
    retrain = PythonOperator(task_id="retrain", python_callable=_run_retrain)
    skip = EmptyOperator(task_id="skip_retrain")
    review = PythonOperator(
        task_id="llm_review",
        python_callable=_run_llm_review,
        # Runs after whichever branch executed (the other is skipped).
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    prune = PythonOperator(task_id="scoring_log_prune", python_callable=_run_prune)

    # Backfill matures outcomes before the performance monitor reads them;
    # drift runs in parallel (it uses scores, not outcomes). Then decide
    # whether to retrain, and close the run with the advisory LLM memo.
    # Retention pruning runs after backfill so it never races outcome
    # writes on the rows it deletes.
    backfill >> perf
    backfill >> prune
    [drift, perf] >> decide >> [retrain, skip]
    [retrain, skip] >> review
