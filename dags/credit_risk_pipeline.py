"""
Airflow DAG: Credit Risk Pipeline
Monthly full pipeline run: Bronze → Silver → Gold → Train → Sync to Supabase.
Optional branch: Reject Inference after training (enabled via DAG config).
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


def _run_bronze():
    from pipeline.bronze_ingest import run
    run()


def _run_silver():
    from pipeline.silver_transform import run
    run()


def _run_gold():
    from pipeline.gold_features import run
    run()


def _run_train():
    from pipeline.train import run
    run()


def _run_sync():
    from pipeline.sync_to_supabase import run
    run()


def _run_reject_inference():
    from pipeline.reject_inference import run
    run()


def _run_fairness():
    from pipeline.fairness import run
    run()


def _decide_reject_inference(**kwargs):
    """Branch: run reject inference only if explicitly enabled via DAG config."""
    conf = kwargs.get("dag_run")
    run_ri = False
    if conf and conf.conf:
        run_ri = conf.conf.get("run_reject_inference", False)
    if run_ri:
        return "reject_inference"
    return "skip_reject_inference"


with DAG(
    dag_id="credit_risk_pipeline",
    default_args=default_args,
    description="Monthly credit risk model pipeline: ingest → transform → features → train → sync",
    schedule="@monthly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["credit_risk", "ml_pipeline"],
) as dag:

    bronze = PythonOperator(task_id="bronze_ingest", python_callable=_run_bronze)
    silver = PythonOperator(task_id="silver_transform", python_callable=_run_silver)
    gold = PythonOperator(task_id="gold_features", python_callable=_run_gold)
    train = PythonOperator(task_id="train_model", python_callable=_run_train)
    sync = PythonOperator(task_id="sync_to_supabase", python_callable=_run_sync)
    fairness = PythonOperator(task_id="fairness_analysis", python_callable=_run_fairness)

    decide_ri = BranchPythonOperator(
        task_id="decide_reject_inference",
        python_callable=_decide_reject_inference,
    )
    reject_inf = PythonOperator(task_id="reject_inference", python_callable=_run_reject_inference)
    skip_ri = EmptyOperator(task_id="skip_reject_inference")

    # Linear pipeline: Bronze → Silver → Gold → Train → Sync + Fairness
    bronze >> silver >> gold >> train >> sync
    train >> fairness

    # Optional reject inference branch after train
    train >> decide_ri >> [reject_inf, skip_ri]
