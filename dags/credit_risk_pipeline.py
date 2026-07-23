"""
Airflow DAG: Credit Risk Pipeline
Monthly full pipeline run: Bronze → Silver → Gold → Train → Export → Sync.
Optional branch: Reject Inference after export (enabled via DAG config).

Training stays in Python and always produces a *challenger* with its
model card; the champion is only ever created by `gbm promote` after
human review (single promotion door, SR 11-7). The export task dumps
the challenger's model.json for the Go runtime, then gbm sync pushes
features to the feature store.
"""

import os
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

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


def _run_export_model():
    from pipeline import config
    from pipeline.export_model_json import export_model
    from pipeline.io_utils import registry_lock
    with registry_lock():
        export_model(config.challenger_dir())


def _run_sync():
    subprocess.run(
        [
            os.path.join(GO_BIN_DIR, "gbm"),
            "sync",
            "--model",
            "challenger",
        ],
        check=True,
    )


def _run_reject_inference():
    from pipeline.reject_inference import run
    run()


def _run_fairness():
    """Standalone fairness snapshot of the current champion. At
    bootstrap there is no champion yet (promotion is human-gated), so
    skip instead of failing the first pipeline run."""
    from pipeline import config
    from pipeline.fairness import run
    if not config.model_path(config.champion_dir()).exists():
        import logging
        logging.getLogger(__name__).warning(
            "no champion model yet; skipping fairness snapshot until first promote")
        return
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
    train = PythonOperator(task_id="train_challenger", python_callable=_run_train)
    export = PythonOperator(task_id="export_model_json", python_callable=_run_export_model)
    sync = PythonOperator(task_id="sync_to_supabase", python_callable=_run_sync)
    fairness = PythonOperator(task_id="fairness_analysis", python_callable=_run_fairness)

    decide_ri = BranchPythonOperator(
        task_id="decide_reject_inference",
        python_callable=_decide_reject_inference,
    )
    reject_inf = PythonOperator(task_id="reject_inference", python_callable=_run_reject_inference)
    skip_ri = EmptyOperator(task_id="skip_reject_inference")
    finalize_challenger = EmptyOperator(
        task_id="finalize_challenger",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Linear pipeline through export; sync waits until the optional reject
    # inference branch has selected the deterministic final challenger.
    bronze >> silver >> gold >> train >> export
    train >> fairness

    # Optional reject inference branch AFTER export: both write the
    # challenger slot, so sequencing makes the RI model (when enabled)
    # the deterministic final challenger of the run.
    export >> decide_ri >> [reject_inf, skip_ri]
    [reject_inf, skip_ri] >> finalize_challenger >> sync
