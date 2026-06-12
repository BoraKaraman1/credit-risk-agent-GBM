# Credit Risk Scoring Pipeline

> **TL;DR:** Production-grade credit risk pipeline scoring 2.3M+ Lending Club loans — featuring a medallion data architecture, SHAP-based regulatory-compliant adverse action reasons, automated drift monitoring, and fairness analysis. Training runs in Python (LightGBM); serving and monitoring run as static Go binaries with pure-Go GBM inference and TreeSHAP. Test AUC of 0.717 on origination-only features (upper end for this dataset, where typical models achieve 0.68–0.73).

---

End-to-end credit risk modeling pipeline built on Lending Club data (2.3M+ loans). Implements a medallion architecture (Bronze/Silver/Gold), gradient boosting model training, a real-time scoring API with SHAP-based adverse action reasons, automated drift monitoring, fairness analysis, data quality validation, and Airflow orchestration — all on free-tier infrastructure.

The system is split across two runtimes: **Python** owns everything that needs the scientific stack (data prep, LightGBM training, fairness, Great Expectations, Airflow DAG definitions), while **Go** owns the serving layer (`go/`) — the scoring API, drift/performance monitors, retrain orchestration, and feature-store sync. The trained model is exported to a library-agnostic JSON tree format (`pipeline/export_model_json.py`) and the Go services run inference and TreeSHAP natively, with no Python at serving time. Go outputs are verified against the Python stack: predictions to 1e-9 vs LightGBM, SHAP values to 1e-6 vs the `shap` library.

## Architecture

```
                              APACHE AIRFLOW
                    ┌──────────────────────────────────┐
                    │ credit_risk_pipeline   (@monthly)│
                    │ credit_risk_monitoring (@weekly) │
                    └──────────────┬───────────────────┘
                                   │ orchestrates
                                   ▼
LOCAL FILESYSTEM (medallion)          SUPABASE POSTGRES          REST API (Go)
┌─────────────────────────┐          ┌─────────────────┐        ┌────────────┐
│ data/                   │          │ applicant_feats │───────>│ POST /score│
│   bronze/  (raw)        │──[GX]──> │ scoring_log     │        │ pure-Go    │
│   silver/  (clean)      │─ sync ──>│ drift_log       │        │ GBM + SHAP │
│   gold/    (features)   │  (Go)    │ training_dist   │        └────────────┘
│   models/  (champion)   │──[GX]──> └─────────────────┘              │
└─────────────────────────┘                 ▲                         │
         │                          Monitoring Agents (Go) ───────────┘
         └── Fairness Analysis      (drift, performance, retrain)
             (DIR, EOD, SPD)

[GX] = Great Expectations data quality validation at each layer
```

## Stack

| Component | Technology |
|-----------|-----------|
| Data storage | Local Parquet (medallion layers) |
| Database | Supabase PostgreSQL (free tier) |
| Model | LightGBM binary classifier (training) |
| Serving | Go 1.26 — pure-Go GBM inference from a JSON model export |
| API | Go `net/http` (`go/cmd/scoring-api`) |
| Orchestration | Apache Airflow |
| Data quality | Great Expectations |
| Experiment tracking | MLflow (local) |
| Interpretability | TreeSHAP (adverse action reasons) — Python `shap` for analysis, pure-Go implementation at serving time |
| Fairness | Disparate Impact, Equal Opportunity, Statistical Parity |
| Monitoring | Go agents (PSI, CSI, AUC tracking) |

## Project Structure

```
├── pipeline/
│   ├── bronze_ingest.py          # CSV.gz → Bronze Parquet
│   ├── silver_transform.py       # Bronze → Silver (clean, validate, target)
│   ├── gold_features.py          # Silver → Gold (feature engineering, time split)
│   ├── train.py                  # Model training + MLflow logging
│   ├── train_challenger.py       # Challenger training entry point (called by Go)
│   ├── export_model_json.py      # Trained model → model.json for the Go runtime
│   ├── reject_inference.py       # Selection bias correction via parcelling
│   ├── fairness.py               # Fairness analysis (DIR, EOD, SPD)
│   ├── data_quality.py           # Great Expectations validation suites
│   └── supabase_schema.sql       # Database DDL
├── go/                           # Serving layer (see go/README.md)
│   ├── cmd/
│   │   ├── scoring-api/          # /score, /score/batch, /health, /reload
│   │   ├── drift-monitor/        # PSI/CSI drift detection
│   │   ├── performance-monitor/  # AUC/KS degradation tracking
│   │   ├── retrain-orchestrator/ # Champion vs challenger retraining flow
│   │   └── supabase-sync/        # Gold → Supabase feature store
│   └── internal/
│       ├── model/                # GBM inference + TreeSHAP (verified vs sklearn/shap)
│       ├── metrics/              # PSI, CSI, AUC, KS, decile analysis
│       ├── gold/                 # Gold parquet reader
│       ├── db/                   # pgx Supabase access + JSONB audit inserts
│       └── config/               # Shared thresholds and paths
├── backup_python/                # Superseded Python services (reference only)
├── dags/
│   ├── credit_risk_pipeline.py   # Monthly pipeline DAG
│   └── credit_risk_monitoring.py # Weekly monitoring DAG
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_training.ipynb         # Model training + evaluation
│   ├── 03_shap_analysis.ipynb    # SHAP interpretability
│   └── 04_reject_inference.ipynb # Reject inference demonstration
├── tests/
│   ├── test_pipeline.py          # Silver/Gold transform tests
│   ├── test_features.py          # Feature engineering + reject inference tests
│   ├── test_fairness.py          # Fairness metric tests
│   └── test_data_quality.py      # Data quality validation tests
├── requirements/                 # Split dependency sets by runtime
│   ├── api.txt                   # FastAPI scoring image
│   ├── pipeline.txt              # Batch training/sync jobs
│   ├── airflow.txt               # Airflow runtime
│   ├── test.txt                  # Test runner
│   └── dev.txt                   # Full local development install
├── Dockerfile.api                # Go scoring API container (multi-stage)
├── Dockerfile.pipeline           # Batch pipeline/training container (Python)
├── Dockerfile.airflow            # Airflow DAG container (Python + Go binaries)
├── Dockerfile.test               # Containerized Python test runner
├── docker-compose.yml            # Local API + Postgres stack
└── requirements.txt              # Backward-compatible dev install wrapper
```

## Setup

### Prerequisites

- Python 3.11+ (training pipeline)
- Go 1.26+ (serving layer)
- Lending Club dataset files in `data/`:
  - `accepted_2007_to_2018Q4.csv.gz`
  - `rejected_2007_to_2018Q4.csv.gz`
- Supabase project (free tier) with `DATABASE_URL` in `.env`

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build the Go serving binaries
cd go && go build -o bin/ ./cmd/... && cd ..
```

`requirements.txt` installs the full development environment. For narrower installs, use:

| File | Purpose |
|------|---------|
| `requirements/pipeline.txt` | Batch pipeline, MLflow, data quality |
| `requirements/airflow.txt` | Airflow runtime |
| `requirements/test.txt` | Test runner without Airflow |
| `requirements/dev.txt` | Full local development environment (incl. SHAP for notebooks) |

The scoring API has no Python dependencies — it is a Go binary (see `go/README.md`).

### Environment

Create a `.env` file:

```
DATABASE_URL=postgresql://postgres.xxxx:password@aws-0-region.pooler.supabase.com:6543/postgres

# Optional: used only by the Postgres JSONB integration test.
# The test creates temporary tables and skips automatically when unset.
TEST_DATABASE_URL=postgresql://postgres.xxxx:password@aws-0-region.pooler.supabase.com:6543/postgres

# Optional runtime overrides.
CREDIT_RISK_MODELS_DIR=data/models
LOG_LEVEL=INFO
```

### Database Schema

Run the schema against your Supabase instance:

```bash
psql $DATABASE_URL -f pipeline/supabase_schema.sql
```

## Running the Pipeline

Execute each step sequentially:

```bash
# 1. Ingest raw CSVs → Bronze Parquet
python pipeline/bronze_ingest.py

# 2. Clean, validate, create target → Silver
python pipeline/silver_transform.py

# 3. Engineer features, time-aware split → Gold
python pipeline/gold_features.py

# 4. Train champion model
python pipeline/train.py

# 5. Export the champion for the Go runtime
python pipeline/export_model_json.py

# 6. Sync features to Supabase (Go)
./go/bin/supabase-sync

# 7. (Optional) Run reject inference
python pipeline/reject_inference.py
```

## Scoring API

The API is a single Go binary that loads the exported champion (`data/models/champion/model.json`) and computes predictions and TreeSHAP adverse actions natively.

```bash
# Start the API
PORT=8000 ./go/bin/scoring-api

# Score an applicant
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"applicant_id": "LC_0000001"}'

# Batch scoring
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{"applicant_ids": ["LC_0000001", "LC_0000002"]}'

# Health check
curl http://localhost:8000/health

# Hot-reload model (after re-running pipeline/export_model_json.py)
curl -X POST http://localhost:8000/reload
```

**Decision rules:**
| Score Range | Decision |
|-------------|----------|
| PD < 0.15 | Approve |
| 0.15 <= PD < 0.30 | Manual Review |
| PD >= 0.30 | Decline |

### Adverse Action Reasons (ECOA)

For decline and manual-review decisions, the API returns the top 4 SHAP-based adverse action reasons — as required by the Equal Credit Opportunity Act (Regulation B). Each reason includes:

- Human-readable feature name (e.g., "Debt-to-Income Ratio")
- SHAP contribution value (how much the feature pushed the score toward default)
- The applicant's actual value for that feature
- Plain-language direction ("increases risk")

Example response (truncated):
```json
{
  "applicant_id": "LC_0000001",
  "pd_score": 0.42,
  "decision": "decline",
  "adverse_actions": [
    {
      "feature_name": "Debt-to-Income Ratio",
      "shap_value": 0.087,
      "feature_value": 38.2,
      "direction": "increases risk"
    }
  ]
}
```

### Audit Logging

Each scoring request writes an audit row to `scoring_log` with the feature snapshot stored as JSONB. Monitoring agents write PSI/AUC summaries to `drift_log`, also with JSONB details. The Go services use pgx parameterized inserts with explicit `::jsonb` casts so Postgres receives valid JSONB values.

## Monitoring Agents

The agents are Go binaries that print a JSON report to stdout and log results to `drift_log`. When `DATABASE_URL` is unset or `scoring_log` is empty, they fall back to the Gold test set as a production proxy.

```bash
# Drift monitor — PSI on score distribution, CSI on features
./go/bin/drift-monitor

# Performance monitor — AUC/KS tracking vs training baseline
./go/bin/performance-monitor

# Retrain orchestrator — train challenger (via Python), compare, recommend
./go/bin/retrain-orchestrator manual
```

**Thresholds:**
| Metric | Warning | Action |
|--------|---------|--------|
| Score PSI | > 0.10 | > 0.25 → retrain |
| Feature CSI | — | > 0.20 → investigate |
| AUC drop | — | > 0.03 → retrain |

## Airflow Orchestration

Two DAGs automate the pipeline end-to-end:

### `credit_risk_pipeline` (Monthly)

Runs the full medallion pipeline with optional reject inference:

```
bronze_ingest → silver_transform → gold_features → train_model → export_model_json → sync_to_supabase
                                                  ↘ fairness_analysis
                                                  ↘ decide_reject_inference → [reject_inference | skip]
```

Training tasks run in-process (Python); `export_model_json` dumps the new champion for the Go runtime, and `sync_to_supabase` shells out to the Go binary.

Reject inference is disabled by default. Enable via DAG config:
```json
{"run_reject_inference": true}
```

### `credit_risk_monitoring` (Weekly)

Monitors model health and triggers retraining when needed:

```
[drift_monitor, performance_monitor] (parallel) → decide_retrain → [retrain | skip]
```

Retraining triggers automatically when:
- PSI status is CRITICAL (score distribution shift)
- AUC drop exceeds the configured threshold (0.03)

The monitor tasks shell out to the Go binaries, parse their JSON reports, and route them through XCom; the retrain reason is passed via XCom so the orchestrator logs why it was triggered.

## Fairness Analysis

Evaluates model fairness across three proxy-protected attributes:

| Attribute | Groups | Privileged Group |
|-----------|--------|-----------------|
| Home Ownership | MORTGAGE, OTHER, OWN, RENT | MORTGAGE |
| Verification Status | Not Verified, Source Verified, Verified | Verified |
| Employment Length | Reported, Not Reported | Reported |

Three metrics are computed per attribute:

- **Disparate Impact Ratio (DIR)** — Approval rate ratio between unprivileged and privileged groups. Flags violations of the 80% four-fifths rule.
- **Equal Opportunity Difference (EOD)** — Gap in true positive rates (P(approve | non-defaulter)) between groups.
- **Statistical Parity Difference (SPD)** — Gap in raw approval rates between groups.

```bash
python -m pipeline.fairness
```

Produces per-group breakdowns with AUC, default rate, approval rate, and mean PD score.

## Data Quality Validation

Great Expectations validates data at each medallion layer:

| Layer | Key Checks |
|-------|-----------|
| Bronze | Row count > 0, `ingested_at` non-null, `source_file` non-null |
| Silver | Core columns non-null, FICO 300-850, income >= 0, DTI 0-100, binary target, default rate 5-50% |
| Gold | All 33 features present, `grade_numeric` 1-7, `sub_grade_numeric` 1-35, binary flags 0/1, row count > 1000 |

Validation runs automatically at the end of each pipeline step. Great Expectations is an optional dependency — the pipeline still works without it installed.

```bash
# Validate existing Gold data manually
python -m pipeline.data_quality
```

## Model Performance

Trained on Lending Club origination-time features only (no data leakage). Origination-only models on this dataset typically achieve 0.68–0.73 AUC; this pipeline sits at the upper end of that range.

| Split | Period | AUC | KS | Gini |
|-------|--------|-----|-----|------|
| Train | < 2016 | 0.7468 | 0.3588 | 0.4936 |
| Val | 2016 – H1 2017 | 0.7434 | 0.3525 | 0.4868 |
| Test | >= Jul 2017 | 0.7168 | 0.3164 | 0.4335 |

The temporal split reveals natural default rate drift (18.6% → 24.8% → 26.4%), which is expected and demonstrates what the monitoring agents detect.

## Reject Inference

The pipeline includes a parcelling-based reject inference module (`pipeline/reject_inference.py`) that:

1. Scores rejected applicants using the accepted-only champion model
2. Assigns pseudo-default labels assuming rejected applicants default at 2x the accepted rate
3. Retrains on the combined dataset with rejected samples weighted at 0.3
4. Compares the augmented model to the champion and saves as challenger

## Containers

The project has separate images for the API, batch pipeline, Airflow, and tests. Runtime images do not copy `.env`, `.venv`, notebooks, raw data, or model artifacts; models and data should be mounted or supplied by the deployment environment.

### API Image

A multi-stage build compiles the Go scoring API and ships it on a slim Debian base (no Python in the image):

```bash
docker build -f Dockerfile.api -t credit-risk-api .
docker run --rm \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://credit_risk:credit_risk@host.docker.internal:5432/credit_risk \
  -e CREDIT_RISK_MODELS_DIR=/models \
  -v "$PWD/data/models:/models:ro" \
  credit-risk-api
```

The API image runs as a non-root user and exposes `/health` as its Docker healthcheck. It expects an exported champion at `${CREDIT_RISK_MODELS_DIR}/champion/model.json` (run `pipeline/export_model_json.py` after training).

### Local Compose Stack

Start local Postgres, initialize `pipeline/supabase_schema.sql`, and run the API against the mounted local champion model:

```bash
docker compose up --build postgres api
```

### Pipeline Image

Build a batch job image for training/sync/fairness tasks:

```bash
docker build -f Dockerfile.pipeline -t credit-risk-pipeline .
docker run --rm \
  -e DATABASE_URL=postgresql://... \
  -v "$PWD/data:/app/data" \
  credit-risk-pipeline python pipeline/train.py
```

### Airflow Image

`Dockerfile.airflow` packages the DAGs, the Python pipeline modules, and the compiled Go binaries (at `/opt/airflow/bin`) on top of the official Airflow Python 3.11 image. Use it when deploying DAGs into an Airflow stack; mount `data/` at `/opt/airflow/data` if the DAGs should read local artifacts.

## Tests

Python suite (training pipeline):

```bash
source .venv/bin/activate
python -m pytest -q
```

79 tests across 4 modules:

- `test_pipeline.py` — Silver parsing, target mapping, Gold feature engineering
- `test_features.py` — Reject inference alignment, pseudo-labeling, model comparison
- `test_fairness.py` — Disparate impact, equal opportunity, statistical parity metrics
- `test_data_quality.py` — Bronze/Silver/Gold validation with valid and invalid data

Go suite (serving layer):

```bash
cd go && go test ./...
```

The Go tests cross-check against fixtures generated by the Python stack: sklearn `predict_proba` (1e-9), `shap` TreeSHAP values (1e-6), numpy PSI/CSI, sklearn AUC/KS, and pandas decile analysis. API decision logic and adverse-action selection are tested against the real champion model.

## Key Design Decisions

- **No data leakage**: Only 25 origination-time columns are used. Post-origination features (payments, recoveries) are excluded.
- **Time-aware splitting**: Train/val/test split by loan issue date, not random. This mimics real deployment where the model is trained on historical data and evaluated on newer loans.
- **LightGBM**: The de facto standard GBM in credit scoring — fast histogram-based training with explicit validation-set early stopping. Requires OpenMP (`brew install libomp` on macOS; bundled in the Linux containers). The model is exported to a library-agnostic JSON tree format, so the Go serving layer is unaffected by the training-library choice.
- **Sample-weighted reject inference**: Pseudo-labeled rejected applicants are weighted at 0.3 to prevent uncertain labels from dominating the training signal.
- **Human-in-the-loop retraining**: The retrain orchestrator recommends but does not auto-promote models (SR 11-7 compliance).
- **SHAP for adverse actions**: TreeSHAP provides exact, model-consistent explanations — more reliable than surrogate models or generic reason-code tables for ECOA compliance. The serving implementation is pure Go, verified to 1e-6 against the Python `shap` library.
- **Go serving layer**: The model is exported to a JSON tree format once after training; the API and monitors run inference natively. This removes the scientific-Python stack from serving images (smaller containers, static binaries, parallel batch scoring) while keeping training in sklearn. Superseded Python services are kept in `backup_python/` for reference.
- **Fairness via proxy attributes**: Direct protected classes (race, gender) are unavailable in the data. Analysis uses proxy attributes (home ownership, verification status, employment reporting) to detect potential disparate impact.
- **Data quality as optional dependency**: Great Expectations validation is wrapped in `try/except ImportError` so the core pipeline remains functional without it — useful for lightweight dev environments.
- **joblib over pickle**: Faster serialization for numpy-heavy sklearn models, smaller files, and recommended by scikit-learn. Backward-compatible fallback to `.pkl` for pre-migration models.
