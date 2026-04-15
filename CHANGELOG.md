# Changelog

## [1.1.0] - 2026-04-15

### Overview

Major update introducing production-readiness improvements across the entire credit risk scoring pipeline. This release adds regulatory compliance features (ECOA adverse action reasons, fairness analysis), operational tooling (Airflow orchestration, data quality validation), and resolves several code quality issues.

---

### New Features

#### SHAP Adverse Action Reasons (ECOA Compliance)
- Integrated `shap.TreeExplainer` into the scoring API to generate real-time, model-driven explanations for every credit decision.
- Decline and manual-review responses now include the **top 4 adverse action reasons** — each with a human-readable feature name, SHAP contribution value, the applicant's actual value, and a plain-language direction (e.g., *"increases risk"*).
- Added `AdverseAction` Pydantic schema and updated `ScoreResponse` to carry adverse actions.
- **Why it matters:** The Equal Credit Opportunity Act (ECOA) and Regulation B require lenders to provide specific reasons when credit is denied. SHAP values give a principled, model-consistent way to satisfy this — far more accurate than generic reason-code tables.

#### Fairness Analysis Module (`pipeline/fairness.py`)
- New module that evaluates model fairness across three proxy-protected attributes available in the feature set: **home ownership status**, **income verification status**, and **employment length reporting**.
- Computes three industry-standard fairness metrics per attribute:
  - **Disparate Impact Ratio (DIR)** — flags groups below the 80% four-fifths rule threshold.
  - **Equal Opportunity Difference** — measures gap in true positive rates between groups.
  - **Statistical Parity Difference** — measures gap in approval rates between groups.
- Produces per-group breakdowns (AUC, default rate, approval rate, mean PD) and a formatted summary report.
- **Why it matters:** Fair lending regulations (ECOA, Fair Housing Act) prohibit discrimination. Quantitative fairness audits surface disparate impact *before* deployment, giving the team evidence to present to model risk management and compliance reviewers.

#### Great Expectations Data Quality Validation (`pipeline/data_quality.py`)
- New validation module with layer-specific expectation suites:
  - **Bronze:** Row count > 0, `ingested_at` and `source_file` columns present and non-null.
  - **Silver:** Core columns non-null, FICO score 300–850, income non-negative, DTI 0–100, binary default target, default rate 5–50%.
  - **Gold:** All 33 feature columns present, `grade_numeric` 1–7, `sub_grade_numeric` 1–35, binary flags are 0/1, row count > 1000, sensible default rate.
- Validation is called automatically at the end of each pipeline step (Bronze ingest, Silver transform, Gold features) but is wrapped in `try/except ImportError` so Great Expectations remains an optional dependency.
- **Why it matters:** Silent data corruption is the most common cause of model degradation in production ML. Automated quality gates at each medallion layer catch schema drift, upstream ETL bugs, and data anomalies before they propagate into model training.

#### Apache Airflow Orchestration (`dags/`)
- **`credit_risk_pipeline`** — Monthly DAG: `Bronze → Silver → Gold → Train → Sync to Supabase`, with a parallel `Train → Fairness Analysis` branch and an optional `Reject Inference` branch (enabled via DAG config).
- **`credit_risk_monitoring`** — Weekly DAG: Drift monitor and performance monitor run in parallel, then a `BranchPythonOperator` decides whether to trigger automated retraining based on PSI status and AUC degradation thresholds. Retrain reasons are passed via XCom.
- **Why it matters:** Manual pipeline execution doesn't scale and is error-prone. Airflow provides scheduling, dependency management, retries, alerting, and a visual DAG UI — essential for operationalizing an ML pipeline in production.

---

### Code Quality Improvements

#### `print()` → `logging` (All Modules)
- Replaced all `print(f"[TAG] ...")` calls with Python's `logging` module across every pipeline step, agent, and API module.
- Each module now uses `logger = logging.getLogger(__name__)` with appropriate log levels (`info`, `warning`, `error`).
- **Why it matters:** Print statements vanish in production — they can't be filtered, routed to log aggregators (Datadog, CloudWatch, ELK), or correlated across services. Structured logging is a baseline requirement for debugging and monitoring in any deployed system.

#### `pickle` → `joblib` (Model Serialization)
- Replaced `pickle.dump`/`pickle.load` with `joblib.dump`/`joblib.load` for all model save/load operations.
- Model files now use the `.joblib` extension, with a backward-compatible `_model_path()` helper that falls back to `.pkl` for pre-migration models.
- **Why it matters:** `joblib` is significantly faster for large numpy arrays (common in sklearn models), produces smaller files through compression, and is the serialization method recommended by scikit-learn. The fallback ensures existing trained models continue to work without retraining.

#### Removed Wasted First `model.fit()` (`pipeline/train.py`)
- The original training code fit the model on the training set, then immediately discarded it and re-fit on the combined train+validation set. The first fit was dead code — it consumed compute time but its output was never used.
- **Why it matters:** Clean code is easier to audit. Redundant operations waste compute and confuse future readers (and model risk reviewers) about which fit actually produced the deployed model.

#### Replaced Deprecated `@app.on_event("startup")` (`api/scoring_service.py`)
- Migrated to FastAPI's `lifespan` async context manager pattern.
- **Why it matters:** `@app.on_event("startup")` is deprecated in FastAPI ≥ 0.95 and will be removed in a future release. The lifespan pattern is the officially supported replacement.

#### Replaced `iterrows()` with Vectorized Approach (`pipeline/sync_to_supabase.py`)
- Replaced the row-by-row `iterrows()` loop with `batch.to_dict("records")` for constructing Supabase upload payloads.
- **Why it matters:** `iterrows()` is one of the slowest ways to iterate a DataFrame — it converts each row to a Series, losing dtype information and adding overhead. The vectorized approach is faster and more idiomatic.

---

### Tests

- **17 new data quality tests** (`tests/test_data_quality.py`) — validates each expectation suite with synthetic valid and invalid DataFrames.
- **17 new fairness tests** (`tests/test_fairness.py`) — covers all metric functions and end-to-end `run()` with a synthetic model.
- **4 new API tests** (`tests/test_api.py`) — validates adverse action presence for decline/review decisions and absence for approvals.
- **Total: 99 tests passing.**

---

### Dependencies Added

| Package | Version | Purpose |
|---------|---------|---------|
| `joblib` | ≥ 1.3 | Model serialization (replaces pickle) |
| `apache-airflow` | ≥ 2.8 | Pipeline and monitoring orchestration |
| `great_expectations` | ≥ 1.0 | Data quality validation |

---

### Files Changed

| File | Change |
|------|--------|
| `requirements.txt` | Added 3 dependencies |
| `pipeline/train.py` | Logging, joblib, removed wasted fit |
| `pipeline/bronze_ingest.py` | Logging, GX validation hook |
| `pipeline/silver_transform.py` | Logging, GX validation hook |
| `pipeline/gold_features.py` | Logging, GX validation hook |
| `pipeline/reject_inference.py` | Logging, joblib |
| `pipeline/sync_to_supabase.py` | Logging, joblib, fixed iterrows |
| `agents/drift_monitor.py` | Logging, joblib |
| `agents/performance_monitor.py` | Logging, joblib |
| `agents/retrain_orchestrator.py` | Logging, joblib |
| `api/models.py` | Added AdverseAction schema |
| `api/scoring_service.py` | Logging, joblib, lifespan, SHAP |
| `tests/test_api.py` | Added adverse action tests |
| `pipeline/fairness.py` | **New** — Fairness analysis |
| `pipeline/data_quality.py` | **New** — Data quality validation |
| `tests/test_fairness.py` | **New** — Fairness tests |
| `tests/test_data_quality.py` | **New** — Data quality tests |
| `dags/__init__.py` | **New** — Package init |
| `dags/credit_risk_pipeline.py` | **New** — Monthly pipeline DAG |
| `dags/credit_risk_monitoring.py` | **New** — Weekly monitoring DAG |
