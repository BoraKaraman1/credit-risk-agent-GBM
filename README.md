# Credit Risk Scoring Pipeline

[![CI](https://github.com/BoraKaraman1/credit-risk-agent-GBM/actions/workflows/ci.yml/badge.svg)](https://github.com/BoraKaraman1/credit-risk-agent-GBM/actions/workflows/ci.yml)

> **TL;DR:** Credit risk pipeline trained on 2.3M+ Lending Club loans. Python runs training (LightGBM); Go runs scoring and monitoring as static binaries with pure-Go inference and TreeSHAP. Test AUC 0.717 on origination-only features, toward the top of what this dataset supports (typical range: 0.68–0.73).

---

Credit risk pipeline built on Lending Club data (2.3M+ loans). It runs a Bronze/Silver/Gold medallion architecture, trains a LightGBM model, serves scores through a real-time API with SHAP-based adverse action reasons, monitors for drift, runs fairness analysis, and orchestrates everything through Airflow. All on free-tier infrastructure.

**Python** owns everything that needs the scientific stack: data prep, LightGBM training, fairness analysis, data-quality validation, and Airflow DAG definitions. **Go** owns the serving layer (`go/`): the scoring API, drift/performance monitors, retrain orchestration, and feature-store sync. After training, the model is exported to a library-agnostic JSON tree format (`pipeline/export_model_json.py`); Go runs inference and TreeSHAP natively with no Python at serving time. Predictions match LightGBM to 1e-9 and SHAP values match the `shap` library to 1e-6.

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
│   bronze/  (raw)        │──[DQ]──> │ scoring_log     │        │ pure-Go    │
│   silver/  (clean)      │─ sync ──>│ drift_log       │        │ GBM + SHAP │
│   gold/    (features)   │  (Go)    │ training_dist   │        └────────────┘
│   models/  (champion)   │──[DQ]──> └─────────────────┘              │
└─────────────────────────┘                 ▲                         │
         │                          Monitoring Agents (Go) ───────────┘
         └── Fairness Analysis      (drift, performance, retrain)
             (DIR, EOD, SPD)

[DQ] = data quality validation gate at each layer (pipeline/data_quality.py)
```

## Stack

| Component | Technology |
|-----------|-----------|
| Data storage | Local Parquet (medallion layers) |
| Database | Supabase PostgreSQL (free tier) |
| Model | LightGBM binary classifier (training) |
| Serving | Go 1.26, pure-Go GBM inference from a JSON model export |
| API | Go `net/http` (`gbm serve`, `go/inference`) |
| Orchestration | Apache Airflow |
| Data quality | Pure-pandas validation gates (`pipeline/data_quality.py`) |
| Experiment tracking | MLflow (local) |
| Interpretability | TreeSHAP (adverse action reasons): Python `shap` for analysis, pure-Go implementation at serving time |
| Fairness | Disparate Impact, Equal Opportunity, Statistical Parity |
| Monitoring | Go agents (PSI, CSI, AUC tracking) |
| Model risk review | Claude (Anthropic SDK) writing advisory review memos; all gates stay deterministic |

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
│   ├── data_quality.py           # Data-quality validation gates (pure pandas)
│   └── supabase_schema.sql       # Database DDL
├── go/                           # Serving layer (see go/README.md)
│   ├── main.go                   # single `gbm` binary; dispatches subcommands
│   ├── inference/                # gbm serve — /score, /score/batch, /health, /reload, /metrics
│   ├── monitoring/               # gbm drift|performance|backfill|retrain|promote|sync
│   ├── db/                       # pgx Supabase access + JSONB audit inserts
│   └── shared/
│       ├── model/                # GBM inference + TreeSHAP (verified vs sklearn/shap)
│       ├── metrics/              # PSI, CSI, AUC, KS, decile analysis
│       ├── gold/                 # Gold parquet reader
│       └── config/               # Shared thresholds and paths
├── agents/
│   └── review_agent.py           # LLM model-risk review memo (advisory; optional)
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
│   ├── base.txt                  # Core ML libraries (shared)
│   ├── pipeline.txt              # Batch training/sync jobs
│   ├── airflow.txt               # Airflow runtime
│   ├── agent.txt                 # Optional LLM review agent (anthropic SDK)
│   ├── test.txt                  # Test runner
│   ├── dev.txt                   # Full local development install
│   └── constraints.txt           # Pinned versions for reproducible installs
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

# Build the Go serving binary
cd go && go build -o bin/gbm . && cd ..
```

`requirements.txt` installs the full development environment. For narrower installs, use:

| File | Purpose |
|------|---------|
| `requirements/pipeline.txt` | Batch pipeline, MLflow, data quality |
| `requirements/airflow.txt` | Airflow runtime |
| `requirements/test.txt` | Test runner without Airflow |
| `requirements/dev.txt` | Full local development environment (incl. SHAP for notebooks) |

The scoring API has no Python dependencies; it is a Go binary (see `go/README.md`).

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

# 4. Train a challenger (writes data/models/challenger + its model card)
python pipeline/train.py

# 5. Export the challenger for the Go runtime
python pipeline/export_model_json.py

# 6. Review the challenger's model card, then promote — the ONLY door
#    to champion/. The public LendingClub data carries structural
#    disparities, so the first (bootstrap) champion is REVIEW REQUIRED
#    and needs the audited override; once a champion exists, later
#    challengers are judged champion-relative and can be APPROVED.
ALLOW_UNAPPROVED_MODEL=true ./go/bin/gbm promote

# 7. Sync features to Supabase (Go)
./go/bin/gbm sync

# 8. (Optional) Run reject inference
python pipeline/reject_inference.py
```

## Scoring API

The API is a single Go binary that loads the exported champion (`data/models/champion/model.json`) and computes predictions and TreeSHAP adverse actions natively.

```bash
# Start the API (auth required; use ALLOW_UNAUTHENTICATED_DEV=true for
# local development without keys)
API_KEYS=your-key PORT=8000 ./go/bin/gbm serve

# Score an applicant
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"applicant_id": "LC_130956066"}'

# Batch scoring
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"applicant_ids": ["LC_130956066", "LC_130968727"]}'

# Health check (open; reports model + database readiness)
curl http://localhost:8000/health

# Prometheus metrics (authenticated; X-API-Key or Bearer token)
curl http://localhost:8000/metrics -H "X-API-Key: your-key"

# Hot-reload model (gbm promote calls this automatically when
# SCORING_API_URL is set; manual fallback below)
curl -X POST http://localhost:8000/reload -H "X-API-Key: your-key"
```

**Decision rules** (applied to the calibrated PD; pre-calibration models fall back to the raw score):
| Calibrated PD | Decision |
|---------------|----------|
| PD < 0.15 | Approve |
| 0.15 <= PD < 0.30 | Manual Review |
| PD >= 0.30 | Decline |

### Calibration and Scorecard Scaling

Raw GBM scores rank-order well but are not calibrated PDs, and everything downstream of a PD (pricing, provisioning, IFRS 9 / CECL) assumes calibration. The training pipeline fits an isotonic calibrator on the early-stopping holdout (never gradient-fitted), reports Brier score and reliability curves on the test split, and exports the calibrator inside `model.json` as breakpoints the Go runtime interpolates without any Python dependency.

Each response carries three numbers:

- `score`: the raw model probability, logged and monitored (the drift reference distribution is built on raw scores)
- `pd`: the calibrated probability of default — this is what the credit decision is made on (the API falls back to the raw score only for pre-calibration models)
- `scaled_score`: an industry-style scorecard score from points-to-double-odds scaling, anchored at 600 = 30:1 good:bad odds with 20 points to double the odds

Go calibration and scaling match the sklearn calibrator to 1e-9 on the parity fixtures.

### Adverse Action Reasons (ECOA)

For decline and manual-review decisions, the API returns the top 4 SHAP-based adverse action reasons, as required by ECOA (Regulation B). Each reason includes:

- A standardized ECOA / Regulation B principal reason statement and a stable `code`. Several features can map to one reason, since the notice discloses the reason, not the internal feature.
- Human-readable feature name (e.g., "Debt-to-Income Ratio")
- SHAP contribution value (how much the feature pushed the score toward default)
- The applicant's actual value for that feature
- Plain-language direction ("increases risk")

Example response (truncated):
```json
{
  "applicant_id": "LC_130956066",
  "score": 0.42,
  "pd": 0.40562,
  "scaled_score": 512,
  "decision": "decline",
  "adverse_actions": [
    {
      "code": 2,
      "reason": "Excessive obligations in relation to income",
      "feature_name": "Debt-to-Income Ratio",
      "shap_value": 0.087,
      "feature_value": 38.2,
      "direction": "increases risk"
    }
  ]
}
```

### Authentication and Rate Limiting

`/score`, `/score/batch`, `/reload`, and `/metrics` require authentication; `/health` stays open. `/metrics` skips the rate limiter so scrapes are not throttled.

- **API keys**: set `API_KEYS` to a comma-separated list. Callers send one in the `X-API-Key` header (or `Authorization: Bearer <key>`), compared in constant time. The API **fails closed**: with no `API_KEYS` it refuses to start unless `ALLOW_UNAUTHENTICATED_DEV=true` is set for local development.
- **Rate limiting**: a per-client token bucket (`golang.org/x/time/rate`), keyed by API key when authenticated and by client IP otherwise. Defaults are 20 req/s sustained with a burst of 40, tunable via `RATE_LIMIT_RPS` and `RATE_LIMIT_BURST`. Over-budget requests get `429` with a `Retry-After` header.

Every request gets an `X-Request-ID` (generated, or the incoming one echoed back) that ties together the structured access logs.

### Server Lifecycle

The API runs on an `http.Server` with read, write, and idle timeouts instead of the bare `http.ListenAndServe`. On `SIGINT`/`SIGTERM` it stops accepting connections and drains in-flight requests for up to 30 seconds before exiting, so a deploy or `docker compose down` does not cut off a scoring request mid-flight.

### Audit Logging

Each scoring request writes an audit row to `scoring_log` with the feature snapshot stored as JSONB. Monitoring agents write PSI/AUC summaries to `drift_log`, also with JSONB details. The Go services use pgx parameterized inserts with explicit `::jsonb` casts so Postgres receives valid JSONB values.

## Monitoring Agents

These are `gbm` subcommands that print a JSON report to stdout and log results to `drift_log`. When `DATABASE_URL` is unset or `scoring_log` is empty, they fall back to the Gold test set as a production proxy.

```bash
# Drift monitor — PSI on score distribution, CSI on features
./go/bin/gbm drift

# Performance monitor — AUC/KS tracking vs training baseline
./go/bin/gbm performance

# Outcome backfill — mature scoring_log outcomes from Gold test labels
./go/bin/gbm backfill

# Retrain orchestrator — train challenger (via Python), compare, recommend
./go/bin/gbm retrain manual

# Promote challenger — publish as an immutable versioned dir and
# atomically repoint the champion symlink (no no-champion window)
./go/bin/gbm promote
```

The outcome backfill closes the monitoring loop. The scoring API cannot know an applicant's real outcome at decision time, so `scoring_log.actual_default` starts NULL and the performance monitor falls back to the test-set proxy. This job simulates outcomes arriving: for applicants scored at least `OUTCOME_BACKFILL_DELAY_DAYS` ago, it writes back the true label from the Gold test set. Once enough outcomes accumulate, the performance monitor runs on production data.

**Thresholds:**
| Metric | Warning | Action |
|--------|---------|--------|
| Score PSI | > 0.10 | > 0.25 → retrain |
| Feature CSI | — | > 0.20 → investigate |
| AUC drop | — | > 0.03 → retrain |

### LLM-Assisted Model Risk Review

The original V1 design had Claude orchestrating the monitors; V2 replaced that with the deterministic Go loop above, because retrain triggers and promotion gates in a credit model need to be reproducible and auditable. The LLM now sits one level up, where an analyst would: `agents/review_agent.py` reads the weekly monitoring output and writes the review memo a human model-risk reviewer would otherwise assemble by hand.

The division of labor is strict: **the agent advises, deterministic code decides.** PSI/AUC thresholds trigger retraining, the champion-relative fairness gate blocks promotion, and promotion stays human-approved (SR 11-7). The agent summarizes the run, flags trends heading toward thresholds, and drafts follow-ups; it decides nothing, and the loop runs unchanged without it.

It is built on the Anthropic SDK's tool runner with read-only tools: the drift and performance monitors (`gbm drift`, `gbm performance`), the model card, and champion/challenger metadata. It cannot write to the database, touch model directories, or call `gbm promote`.

```bash
pip install -r requirements/agent.txt

# Standalone: runs the monitors itself, writes docs/monitoring_review.md
ANTHROPIC_API_KEY=... python agents/review_agent.py
```

In the weekly monitoring DAG it runs as the final `llm_review` task, fed the drift/performance/retrain reports from XCom. The task is optional by design: a missing `anthropic` install or API key logs a warning and the run still succeeds.

Governance controls: the model ID is pinned via `REVIEW_AGENT_MODEL` (default `claude-opus-4-8`), the memo opens with an advisory disclaimer, and every run appends a full audit record (inputs, memo, token usage) to `data/review_agent_audit.jsonl`.

### Prometheus + Grafana

The API exposes Prometheus metrics at `/metrics`: request counts and latency histograms per endpoint, the predicted-PD distribution, decision counts (approve/review/decline), and the loaded model version as a labeled info gauge. A compose profile brings up Prometheus and a pre-provisioned Grafana dashboard:

```bash
docker compose --profile monitoring up
```

Grafana is at `http://localhost:3000` (anonymous viewer access enabled), Prometheus at `http://localhost:9090`. The default `docker compose up` is unchanged: it runs only Postgres and the API.

## Airflow Orchestration

Two DAGs automate the pipeline end-to-end:

### `credit_risk_pipeline` (Monthly)

Runs the full medallion pipeline with optional reject inference:

```
bronze_ingest → silver_transform → gold_features → train_challenger → export_model_json → sync_to_supabase
                                                  ↘ fairness_analysis  ↘ decide_reject_inference → [reject_inference | skip]
```

Training tasks run in-process (Python) and produce a **gated
challenger** with its model card; the champion is never written by the
DAG. Promotion stays a human step (`gbm promote`), which atomically
publishes the version and tells the API to reload. `export_model_json`
dumps the challenger for the Go runtime, and `sync_to_supabase` shells
out to the Go binary.

Reject inference is disabled by default. Enable via DAG config:
```json
{"run_reject_inference": true}
```

### `credit_risk_monitoring` (Weekly)

Monitors model health and triggers retraining when needed:

```
outcome_backfill → performance_monitor ┐
                   drift_monitor       ┴→ decide_retrain → [retrain | skip] → llm_review
```

Retraining triggers automatically when:
- PSI status is CRITICAL (score distribution shift)
- AUC drop exceeds the configured threshold (0.03)

The monitor tasks shell out to the Go binaries, parse their JSON reports, and route them through XCom; the retrain reason is passed via XCom so the orchestrator logs why it was triggered. The closing `llm_review` task writes the advisory memo described above and never blocks the loop.

### Running Airflow locally

The compose stack includes a single-container Airflow (standalone mode) under the `orchestration` profile. The image ships only code; the medallion data and model registry are mounted from `./data` (the compose file does this for you):

```bash
docker compose --profile orchestration up airflow
# UI at http://localhost:8080 — the admin password is printed in the logs
```

DAG integrity is enforced in CI: `tests/test_dags.py` resolves every lazy import inside task callables (no Airflow needed) and parses both DAGs with a real Airflow install in the `airflow-dags` job.

## Fairness Analysis

Evaluates model fairness across three proxy-protected attributes:

| Attribute | Groups | Privileged Group |
|-----------|--------|-----------------|
| Home Ownership | MORTGAGE, OTHER, OWN, RENT | MORTGAGE |
| Verification Status | Not Verified, Source Verified, Verified | Verified |
| Employment Length | Reported, Not Reported | Reported |

Three metrics are computed per attribute:

- **Disparate Impact Ratio (DIR):** Approval rate ratio between unprivileged and privileged groups. Flags violations of the 80% four-fifths rule.
- **Equal Opportunity Difference (EOD):** Gap in true positive rates (P(approve | non-defaulter)) between groups.
- **Statistical Parity Difference (SPD):** Gap in raw approval rates between groups.

```bash
python -m pipeline.fairness
```

Produces per-group breakdowns with AUC, default rate, approval rate, and mean PD score.

### Fairness Gate in Promotion

The retrain orchestrator will not recommend PROMOTE on accuracy alone. It computes fairness for both the challenger and the current champion on the test set and blocks promotion when the challenger introduces a new DIR violation or worsens one the champion already had (champion-relative, since this dataset already carries inherent disparity on the proxies). A more discriminatory model is held back even if its AUC is higher (SR 11-7 + ECOA). The comparison is champion-relative rather than an absolute 0.80 cutoff, because an absolute cutoff would make every challenger unpromotable here.

## Model Card

Every training run writes a markdown validation report next to the model itself (`data/models/<role>/model_card.md`): version and data window, discrimination metrics, calibration (Brier and reliability), scorecard parameters, the full fairness breakdown, hyperparameters, and an overall validation status. This is the artifact a model risk management team reviews **before promotion**; `gbm promote` carries the reviewed card into the immutable version directory, so the served champion always ships with the card that approved it. [`docs/model_card.md`](docs/model_card.md) is a committed snapshot for browsing the repo.

```bash
# Regenerate the repo snapshot from an existing model without retraining
python pipeline/model_card.py data/models/champion docs/model_card.md
```

## Data Quality Validation

Pure-pandas validation checks (`pipeline/data_quality.py`) gate each medallion layer:

| Layer | Key Checks |
|-------|-----------|
| Bronze | Row count > 0, `ingested_at` non-null, `source_file` non-null |
| Silver | Core columns non-null, FICO 300-850, income >= 0, DTI 0-100, binary target, default rate 5-50% |
| Gold | All 33 features present, `grade_numeric` 1-7, `sub_grade_numeric` 1-35, binary flags 0/1, row count > 1000 |

Validation runs before each layer's artifact is written: in strict mode (`CREDIT_RISK_STRICT_DQ=true`, set in the pipeline and Airflow images) a failure aborts the step and nothing is persisted; otherwise it logs a warning.

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

The API image runs as a non-root user and exposes `/health` as its Docker healthcheck. It expects a champion at `${CREDIT_RISK_MODELS_DIR}/champion/model.json`, created by promoting an exported challenger (`gbm promote`).

### Local Compose Stack

Start local Postgres, initialize `pipeline/supabase_schema.sql`, and run the API against the mounted local champion model. The first (bootstrap) champion is `REVIEW REQUIRED` (structural disparities in the public LendingClub data), so the governance gate fails closed by default; explicitly accept serving an unapproved model to run the demo. Once a champion exists, challenger exports are judged champion-relative and can be `APPROVED`, after which the override is unnecessary:

```bash
ALLOW_UNAPPROVED_MODEL=true docker compose up --build postgres api
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
- **LightGBM**: What credit risk teams actually use. Fast histogram-based training with explicit validation-set early stopping. Requires OpenMP (`brew install libomp` on macOS; bundled in the Linux containers). The model is exported to a library-agnostic JSON tree format, so the Go serving layer is unaffected by the training-library choice.
- **Sample-weighted reject inference**: Pseudo-labeled rejected applicants are weighted at 0.3 to prevent uncertain labels from dominating the training signal.
- **Human-in-the-loop retraining**: The retrain orchestrator recommends but does not auto-promote models (SR 11-7 compliance).
- **SHAP for adverse actions**: TreeSHAP provides exact, model-consistent explanations, more reliable than surrogate models or generic reason-code tables for ECOA compliance. The serving implementation is pure Go, verified to 1e-6 against the Python `shap` library.
- **Go serving layer**: The model is exported to a JSON tree format once after training; the API and monitors run inference natively. Serving images drop the scientific-Python stack entirely (one static binary on a slim base) while training stays in Python. Superseded Python services are kept in `backup_python/` for reference.
- **Fairness via proxy attributes**: Direct protected classes (race, gender) are unavailable in the data. Analysis uses proxy attributes (home ownership, verification status, employment reporting) to detect potential disparate impact.
- **Data quality as a hard gate**: Validation is pure pandas (no extra dependency), always imported, and runs before each layer's artifact is written, so a strict-mode failure can never leave rejected data persisted for downstream consumers.
- **joblib over pickle**: Faster and smaller than pickle for numpy-heavy sklearn models, and the serialization method scikit-learn recommends. Backward-compatible fallback to `.pkl` for pre-migration models.
