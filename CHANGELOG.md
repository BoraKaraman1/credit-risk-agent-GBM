# Changelog

## [2.1.0] - 2026-06-13

### Overview

The model moved from scikit-learn's `HistGradientBoostingClassifier` to LightGBM without touching a single line of the Go serving layer. The library-agnostic JSON export format is what made this possible: all parity tests pass unchanged. The Go test suite grew from 12 to 154 tests in the same release.

---

### Changed

#### LightGBM Training (`pipeline/train.py`, `train_challenger.py`, `reject_inference.py`)
- All three training paths now use `lgb.LGBMClassifier` with AUC-based early stopping (patience 50) on an explicit eval set.
- Data usage preserved from the previous champion: train+val combined with a stratified random carve-out for early stopping. (A first attempt trained on the train split only dropped test AUC to 0.7053; with a temporal split, the val period carries the most recent signal.)
- New champion **v1.2**: 940 trees, test AUC 0.7168 / KS 0.3164 (statistically on par with the previous champion: 0.7180 / 0.3182).
- **Why it matters:** LightGBM is what credit risk teams actually deploy; native categorical support and faster training open future feature work. The near-identical AUC confirms the previous model family was already at this dataset's ceiling.

#### Library-Agnostic Model Export (`pipeline/export_model_json.py`)
- Rewritten to flatten LightGBM's nested tree dump into the existing columnar JSON format the Go runtime executes: `value <= threshold` goes left, NaN follows `missing_go_to_left`.
- Handles LightGBM-specific semantics: truncation to `best_iteration_` when early stopping fires, `default_left` for NaN-aware splits, NaN-treated-as-zero for splits that never saw missing values (`missing_type: None`), and the baked-in baseline measured empirically.
- The exporter **refuses to export** unless the flattened trees reproduce LightGBM's raw scores to 1e-9 on a randomized check set.
- **Why it matters:** The Go serving layer required zero changes: predictions match LightGBM to 1e-9 and Go TreeSHAP matches the Python `shap` library to 1e-6 on regenerated fixtures. The serving layer is provably decoupled from the training library.

#### OpenMP Runtime Dependency
- LightGBM requires OpenMP: `Dockerfile.pipeline`, `Dockerfile.test`, and `Dockerfile.airflow` now install `libgomp1` (absent from slim/Airflow base images). On macOS, `brew install libomp`.

---

### Tests

- **Go suite expanded from 12 to 154 tests**: hand-verified metric edge cases (histogram bin semantics, quantile interpolation, ROC-AUC midrank ties, KS, decile rank-order breaks), synthetic hand-built trees with analytically exact SHAP values, model loader error paths, a parquet write→read round-trip, config env handling, and HTTP handler tests (`httptest`).
- Decision thresholds extracted into a testable `decide()` function in the scoring API.
- Python suite: 79 tests passing on LightGBM.

---

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `lightgbm` | ≥ 4.0 | Gradient boosting (replaces sklearn HistGradientBoosting) |

---

## [2.0.0] - 2026-06-12

### Overview

**Forked the serving layer to Go.** The scoring API, drift/performance monitors, retrain orchestrator, and feature-store sync now run as five static Go binaries with pure-Go GBM inference and TreeSHAP. No Python at serving time. Training, fairness analysis, data quality validation, and Airflow DAG definitions stay in Python. The superseded Python services are preserved in `backup_python/`.

---

### New Features

#### Go Serving Layer (`go/`)
- **`cmd/scoring-api`:** `net/http` port of the FastAPI service: `/score`, `/score/batch`, `/health`, `/reload`, with identical decision thresholds, ECOA adverse action reasons, and `scoring_log` audit inserts.
- **`cmd/drift-monitor`**, **`cmd/performance-monitor`:** PSI/CSI and AUC/KS/Gini/decile monitoring with the same Supabase-or-test-set fallback logic and `drift_log` writes.
- **`cmd/retrain-orchestrator`:** shells out to Python for the model fit, then evaluates champion vs challenger natively (AUC/KS/Gini + score PSI) and emits the SR 11-7 review report.
- **`cmd/supabase-sync`:** bulk feature upsert (multi-row `unnest` INSERT) plus the training score distribution used as the PSI reference.
- **`internal/model`:** loads a JSON export of the champion's trees and implements `predict_proba` (parallel batch scoring) and **path-dependent TreeSHAP** (Lundberg et al., Algorithm 2) in pure Go.
- **Why it matters:** The serving image drops from a full scientific-Python stack to one static binary on a slim base. Attack surface shrinks, cold starts get faster, and batch scoring hits 831K rows in ~7s.

#### Model JSON Export (`pipeline/export_model_json.py`)
- One-time export of the trained model's trees (nodes, thresholds, missing directions, covers, baseline) to `model.json` for the Go runtime.

#### Verified Cross-Language Parity
- Predictions match the Python stack to **1e-9**; TreeSHAP adverse-action values match the `shap` library to **1e-6**, including rows with missing values.
- Drift monitor PSI and all 33 CSI values bit-identical to the Python agents; recommendation strings equal.
- Live side-by-side API comparison: identical scores (5 decimals), decisions, adverse actions, and audit rows against the same Supabase instance.

---

### Changed

- **Airflow DAGs** now shell out to the Go binaries (`/opt/airflow/bin`) and parse their JSON reports; the pipeline DAG gained an `export_model_json` task between train and sync.
- **`Dockerfile.api`** is a Go multi-stage build (no Python in the image); **`Dockerfile.airflow`** packages the Go binaries alongside the DAGs; pipeline/test images dropped the moved directories.
- **Requirements** slimmed: `api.txt` removed (the API has no Python dependencies); `sqlalchemy`, `psycopg2-binary`, and `python-dotenv` dropped from `base.txt` (database access moved to Go/pgx); `httpx` dropped; `shap` moved to dev (notebooks).
- **`docker-compose.yml`**: `audit-tests` service removed (its test moved to backup with the code it tested).

### Moved to `backup_python/`

`api/` (FastAPI service), `agents/` (drift, performance, retrain, config, db_logging), `pipeline/sync_to_supabase.py`, `tests/test_api.py`, `tests/test_audit_logging_sql.py`, and a self-contained `requirements-api.txt`.

---

### Tests

- Go test suites cross-check against fixtures generated by the Python stack: sklearn `predict_proba`, `shap` TreeSHAP values, numpy PSI/CSI, sklearn AUC/KS, pandas decile analysis.
- Python suite: 79 tests passing (API and audit-SQL tests moved to backup with their code; PSI/CSI coverage moved to Go).

---

## [1.1.0] - 2026-04-15

### Overview

Added ECOA adverse action reasons, fairness analysis, Airflow orchestration, Great Expectations data quality validation, and a handful of code quality fixes across the pipeline.

---

### New Features

#### SHAP Adverse Action Reasons (ECOA Compliance)
- Integrated `shap.TreeExplainer` into the scoring API to generate real-time, model-driven explanations for every credit decision.
- Decline and manual-review responses now include the **top 4 adverse action reasons**: a human-readable feature name, SHAP contribution value, the applicant's actual value, and a plain-language direction (e.g., *"increases risk"*).
- Added `AdverseAction` Pydantic schema and updated `ScoreResponse` to carry adverse actions.
- **Why it matters:** ECOA and Regulation B require lenders to provide specific reasons when credit is denied. SHAP values give a principled, model-consistent way to satisfy this, and are more accurate than generic reason-code tables.

#### Fairness Analysis Module (`pipeline/fairness.py`)
- New module that evaluates model fairness across three proxy-protected attributes available in the feature set: **home ownership status**, **income verification status**, and **employment length reporting**.
- Computes three industry-standard fairness metrics per attribute:
  - **Disparate Impact Ratio (DIR):** flags groups below the 80% four-fifths rule threshold.
  - **Equal Opportunity Difference:** measures gap in true positive rates between groups.
  - **Statistical Parity Difference:** measures gap in approval rates between groups.
- Produces per-group breakdowns (AUC, default rate, approval rate, mean PD) and a formatted summary report.
- **Why it matters:** Fair lending regulations (ECOA, Fair Housing Act) prohibit discrimination. Quantitative fairness audits surface disparate impact *before* deployment, giving the team evidence to present to model risk management and compliance reviewers.

#### Great Expectations Data Quality Validation (`pipeline/data_quality.py`)
- New validation module with layer-specific expectation suites:
  - **Bronze:** Row count > 0, `ingested_at` and `source_file` columns present and non-null.
  - **Silver:** Core columns non-null, FICO score 300–850, income non-negative, DTI 0–100, binary default target, default rate 5–50%.
  - **Gold:** All 33 feature columns present, `grade_numeric` 1–7, `sub_grade_numeric` 1–35, binary flags are 0/1, row count > 1000, sensible default rate.
- Validation is called automatically at the end of each pipeline step (Bronze ingest, Silver transform, Gold features) but is wrapped in `try/except ImportError` so Great Expectations remains an optional dependency.
- **Why it matters:** Silent data corruption is the most common cause of model degradation in production. Automated quality gates at each medallion layer catch problems before they reach training.

#### Apache Airflow Orchestration (`dags/`)
- **`credit_risk_pipeline`** — Monthly DAG: `Bronze → Silver → Gold → Train → Sync to Supabase`, with a parallel `Train → Fairness Analysis` branch and an optional `Reject Inference` branch (enabled via DAG config).
- **`credit_risk_monitoring`** — Weekly DAG: Drift monitor and performance monitor run in parallel, then a `BranchPythonOperator` decides whether to trigger automated retraining based on PSI status and AUC degradation thresholds. Retrain reasons are passed via XCom.
- **Why it matters:** Running pipeline steps by hand doesn't scale. Airflow gives you scheduling, retries, alerting, and a visual DAG UI; it's the difference between a script and a system.

---

### Code Quality Improvements

#### `print()` → `logging` (All Modules)
- Replaced all `print(f"[TAG] ...")` calls with Python's `logging` module across every pipeline step, agent, and API module.
- Each module now uses `logger = logging.getLogger(__name__)` with appropriate log levels (`info`, `warning`, `error`).
- **Why it matters:** Print statements vanish in production. They can't be filtered, routed to a log aggregator, or correlated across services. The logging module fixes all of that.

#### `pickle` → `joblib` (Model Serialization)
- Replaced `pickle.dump`/`pickle.load` with `joblib.dump`/`joblib.load` for all model save/load operations.
- Model files now use the `.joblib` extension, with a backward-compatible `_model_path()` helper that falls back to `.pkl` for pre-migration models.
- **Why it matters:** `joblib` is faster than pickle for large numpy arrays and produces smaller files; it's also what scikit-learn recommends. The fallback ensures existing models still load without retraining.

#### Removed Wasted First `model.fit()` (`pipeline/train.py`)
- The original training code fit the model on the training set, then immediately discarded it and re-fit on the combined train+validation set. The first fit was dead code; it consumed compute time but its output was never used.
- **Why it matters:** Clean code is easier to audit. Redundant operations waste compute and confuse future readers (and model risk reviewers) about which fit actually produced the deployed model.

#### Replaced Deprecated `@app.on_event("startup")` (`api/scoring_service.py`)
- Migrated to FastAPI's `lifespan` async context manager pattern.
- **Why it matters:** `@app.on_event("startup")` is deprecated in FastAPI ≥ 0.95 and will be removed in a future release. The lifespan pattern is the officially supported replacement.

#### Replaced `iterrows()` with Vectorized Approach (`pipeline/sync_to_supabase.py`)
- Replaced the row-by-row `iterrows()` loop with `batch.to_dict("records")` for constructing Supabase upload payloads.
- **Why it matters:** `iterrows()` converts each row to a Series, losing dtype information and adding overhead. It's one of the slowest DataFrame iteration patterns. The vectorized approach is faster.

---

### Tests

- **17 new data quality tests** (`tests/test_data_quality.py`): validates each expectation suite with synthetic valid and invalid DataFrames.
- **17 new fairness tests** (`tests/test_fairness.py`): covers all metric functions and end-to-end `run()` with a synthetic model.
- **4 new API tests** (`tests/test_api.py`): validates adverse action presence for decline/review decisions and absence for approvals.
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
