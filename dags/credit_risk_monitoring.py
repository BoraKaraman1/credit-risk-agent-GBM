"""
Airflow DAG: Credit Risk Monitoring
Weekly monitoring: drift check + performance check → conditional retrain.
Drift and performance monitors run in parallel, then a branching decision
determines whether to trigger the retrain orchestrator.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "credit_risk",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run_drift_monitor(**kwargs):
    from agents.drift_monitor import run
    report = run()
    kwargs["ti"].xcom_push(key="drift_report", value=report)
    return report


def _run_performance_monitor(**kwargs):
    from agents.performance_monitor import run
    report = run()
    kwargs["ti"].xcom_push(key="perf_report", value=report)
    return report


def _decide_retrain(**kwargs):
    """Decide whether to trigger retraining based on drift and performance reports."""
    ti = kwargs["ti"]
    drift_report = ti.xcom_pull(task_ids="drift_monitor", key="drift_report")
    perf_report = ti.xcom_pull(task_ids="performance_monitor", key="perf_report")

    needs_retrain = False
    reason_parts = []

    if drift_report and drift_report.get("psi_status") == "CRITICAL":
        needs_retrain = True
        reason_parts.append(f"psi_critical ({drift_report['psi']:.4f})")

    if perf_report and perf_report.get("auc_drop", 0) > perf_report.get("auc_drop_threshold", 0.03):
        needs_retrain = True
        reason_parts.append(f"auc_drop ({perf_report['auc_drop']:.4f})")

    if needs_retrain:
        reason = ", ".join(reason_parts)
        ti.xcom_push(key="retrain_reason", value=reason)
        return "retrain"
    return "skip_retrain"


def _run_retrain(**kwargs):
    from agents.retrain_orchestrator import run
    reason = kwargs["ti"].xcom_pull(task_ids="decide_retrain", key="retrain_reason") or "monitoring_trigger"
    report = run(reason=reason)
    return report


with DAG(
    dag_id="credit_risk_monitoring",
    default_args=default_args,
    description="Weekly model monitoring: drift, performance, conditional retrain",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit_risk", "monitoring"],
) as dag:

    drift = PythonOperator(task_id="drift_monitor", python_callable=_run_drift_monitor)
    perf = PythonOperator(task_id="performance_monitor", python_callable=_run_performance_monitor)
    decide = BranchPythonOperator(task_id="decide_retrain", python_callable=_decide_retrain)
    retrain = PythonOperator(task_id="retrain", python_callable=_run_retrain)
    skip = EmptyOperator(task_id="skip_retrain")

    # Drift and performance run in parallel, then decide whether to retrain
    [drift, perf] >> decide >> [retrain, skip]
