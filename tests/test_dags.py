"""
Airflow DAG validation.

Two layers of protection:
- AST-based: every `from X import Y` written inside a DAG task callable
  must resolve against the real module, without executing the task.
  Task imports are lazy, so a bad symbol otherwise only explodes at run
  time inside Airflow. Needs the pipeline stack but not Airflow.
- DagBag: the DAG files must parse under a real Airflow install.
  Skipped when Airflow is not installed (runs in the CI DAG job).
"""

import ast
import importlib
from pathlib import Path

import pytest

DAG_DIR = Path(__file__).resolve().parent.parent / "dags"
DAG_FILES = sorted(p for p in DAG_DIR.glob("*.py") if p.name != "__init__.py")

EXPECTED_DAG_IDS = {"credit_risk_pipeline", "credit_risk_monitoring"}


def _function_imports(path):
    """Yield (module, symbol, lineno) for every absolute
    `from X import Y` nested inside a function definition."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.ImportFrom) and inner.level == 0:
                for alias in inner.names:
                    yield inner.module, alias.name, inner.lineno


def test_dag_files_exist():
    assert DAG_FILES, f"no DAG files found in {DAG_DIR}"


@pytest.mark.parametrize("dag_file", DAG_FILES, ids=lambda p: p.name)
def test_task_callable_imports_resolve(dag_file):
    checked = 0
    for module_name, symbol, lineno in _function_imports(dag_file):
        if module_name.split(".")[0] == "airflow":
            continue
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), (
            f"{dag_file.name}:{lineno} does "
            f"`from {module_name} import {symbol}`, "
            f"but {module_name} has no attribute {symbol!r}"
        )
        checked += 1
    assert checked > 0, f"{dag_file.name}: no task imports found to check"


def _dagbag():
    pytest.importorskip("airflow", reason="airflow not installed")
    from airflow.models import DagBag
    return DagBag(dag_folder=str(DAG_DIR), include_examples=False)


def test_dagbag_imports_cleanly():
    bag = _dagbag()
    assert not bag.import_errors, f"DAG import errors: {bag.import_errors}"
    assert EXPECTED_DAG_IDS <= set(bag.dag_ids)


def test_pipeline_dag_structure():
    # bag.dags (not get_dag) so no metadata-DB access is needed
    dag = _dagbag().dags["credit_risk_pipeline"]
    task_ids = {t.task_id for t in dag.tasks}
    assert {"bronze_ingest", "silver_transform", "gold_features",
            "train_model", "export_model_json", "sync_to_supabase",
            "fairness_analysis"} <= task_ids
    export = dag.get_task("export_model_json")
    assert {t.task_id for t in export.upstream_list} == {"train_model"}


def test_monitoring_dag_structure():
    dag = _dagbag().dags["credit_risk_monitoring"]
    task_ids = {t.task_id for t in dag.tasks}
    assert {"outcome_backfill", "drift_monitor", "performance_monitor",
            "decide_retrain", "retrain"} <= task_ids
