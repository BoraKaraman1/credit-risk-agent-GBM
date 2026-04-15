# Credit Risk Scoring Pipeline

> **TL;DR:** Production-grade credit risk pipeline scoring 2.3M+ Lending Club loans — featuring a medallion data architecture, SHAP-based regulatory-compliant adverse action reasons, automated drift monitoring, and fairness analysis. Test AUC of 0.718 on origination-only features (upper end for this dataset, where typical models achieve 0.68–0.73).

---

End-to-end credit risk modeling pipeline built on Lending Club data (2.3M+ loans). Implements a medallion architecture (Bronze/Silver/Gold), gradient boosting model training, a real-time scoring API with SHAP-based adverse action reasons, automated drift monitoring, fairness analysis, data quality validation, and Airflow orchestration — all on free-tier infrastructure.

## Architecture

```
                              APACHE AIRFLOW
                    ┌──────────────────────────────────┐
                    │ credit_risk_pipeline   (@monthly) │
                    │ credit_risk_monitoring (@weekly)  │
                    └──────────────┬───────────────────┘
                                   │ orchestrates
                                   ▼
LOCAL FILESYSTEM (medallion)          SUPABASE POSTGRES           FASTAPI
┌─────────────────────────┐          ┌─────────────────┐        ┌────────────┐
│ data/                   │          │ applicant_feats  │───────>│ POST /score│
│   bronze/  (raw)        │──[GX]──>│ scoring_log      │        │ model +    │
│   silver/  (clean)      │── sync ─>│ drift_log        │        │ SHAP       │
│   gold/    (features)   │──[GX]──>│ training_dist    │        └────────────┘
│   models/  (champion)   │          └─────────────────┘              │
└─────────────────────────┘                 ▲                         │
         │                          Monitoring Agents ────────────────┘
         └── Fairness Analysis      (drift, performance, retrain)
             (DIR, EOD, SPD)

[GX] = Great Expectations data quality validation at each layer
```

## Stack

| Component | Technology |
|-----------|-----------|
| Data storage | Local Parquet (medallion layers) |
| Database | Supabase PostgreSQL (free tier) |
| Model | scikit-learn HistGradientBoostingClassifier |
| API | FastAPI + Uvicorn |
| Orchestration | Apache Airflow |
| Data quality | Great Expectations |
| Experiment tracking | MLflow (local) |
| Interpretability | SHAP (adverse action reasons) |
| Fairness | Disparate Impact, Equal Opportunity, Statistical Parity |
| Monitoring | Automated agents (PSI, CSI, AUC tracking) |

## Project Structure

```
├── pipeline/
│   ├── bronze_ingest.py          # CSV.gz → Bronze Parquet
│   ├── silver_transform.py       # Bronze → Silver (clean, validate, target)
│   ├── gold_features.py          # Silver → Gold (feature engineering, time split)
│   ├── train.py                  # Model training + MLflow logging
│   ├── reject_inference.py       # Selection bias correction via parcelling
│   ├── sync_to_supabase.py       # Gold → Supabase feature store
│   ├── fairness.py               # Fairness analysis (DIR, EOD, SPD)
│   ├── data_quality.py           # Great Expectations validation suites
│   └── supabase_schema.sql       # Database DDL
├── api/
│   ├── scoring_service.py        # FastAPI scoring + SHAP adverse actions
│   └── models.py                 # Pydantic schemas (incl. AdverseAction)
├── agents/
│   ├── drift_monitor.py          # PSI/CSI drift detection
│   ├── performance_monitor.py    # AUC/KS degradation tracking
│   ├── retrain_orchestrator.py   # Automated retraining pipeline
│   └── config.py                 # Shared thresholds and paths
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
│   ├── test_features.py          # Feature engineering + drift metric tests
│   ├── test_api.py               # API endpoint + adverse action tests
│   ├── test_fairness.py          # Fairness metric tests
│   └── test_data_quality.py      # Data quality validation tests
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+
- Lending Club dataset files in `data/`:
  - `accepted_2007_to_2018Q4.csv.gz`
  - `rejected_2007_to_2018Q4.csv.gz`
- Supabase project (free tier) with `DATABASE_URL` in `.env`

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

Create a `.env` file:

```
DATABASE_URL=postgresql://postgres.xxxx:password@aws-0-region.pooler.supabase.com:6543/postgres
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

# 5. Sync features to Supabase
python pipeline/sync_to_supabase.py

# 6. (Optional) Run reject inference
python pipeline/reject_inference.py
```

## Scoring API

```bash
# Start the API
uvicorn api.scoring_service:app --reload --port 8000

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

# Hot-reload model
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

## Monitoring Agents

```bash
# Drift monitor — PSI on score distribution, CSI on features
python -m agents.drift_monitor

# Performance monitor — AUC/KS tracking vs training baseline
python -m agents.performance_monitor

# Retrain orchestrator — train challenger, compare, recommend
python -m agents.retrain_orchestrator manual
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
bronze_ingest → silver_transform → gold_features → train_model → sync_to_supabase
                                                  ↘ fairness_analysis
                                                  ↘ decide_reject_inference → [reject_inference | skip]
```

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

The retrain reason is passed via XCom so the orchestrator logs why it was triggered.

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
| Train | < 2016 | 0.7478 | 0.3612 | 0.4956 |
| Val | 2016 – H1 2017 | 0.7442 | 0.3574 | 0.4884 |
| Test | >= Jul 2017 | 0.7180 | 0.3182 | 0.4359 |

The temporal split reveals natural default rate drift (18.6% → 24.8% → 26.4%), which is expected and demonstrates what the monitoring agents detect.

## Reject Inference

The pipeline includes a parcelling-based reject inference module (`pipeline/reject_inference.py`) that:

1. Scores rejected applicants using the accepted-only champion model
2. Assigns pseudo-default labels assuming rejected applicants default at 2x the accepted rate
3. Retrains on the combined dataset with rejected samples weighted at 0.3
4. Compares the augmented model to the champion and saves as challenger

## Tests

```bash
pytest tests/ -v
```

99 tests across 5 modules:

- `test_pipeline.py` — Silver parsing, target mapping, Gold feature engineering
- `test_features.py` — PSI/CSI metrics, reject inference alignment, pseudo-labeling
- `test_api.py` — API endpoints, decision logic, adverse action reasons, Pydantic schemas
- `test_fairness.py` — Disparate impact, equal opportunity, statistical parity metrics
- `test_data_quality.py` — Bronze/Silver/Gold validation with valid and invalid data

## Key Design Decisions

- **No data leakage**: Only 25 origination-time columns are used. Post-origination features (payments, recoveries) are excluded.
- **Time-aware splitting**: Train/val/test split by loan issue date, not random. This mimics real deployment where the model is trained on historical data and evaluated on newer loans.
- **HistGradientBoosting over XGBoost**: Avoids OpenMP dependency issues on macOS ARM while providing identical statistical properties.
- **Sample-weighted reject inference**: Pseudo-labeled rejected applicants are weighted at 0.3 to prevent uncertain labels from dominating the training signal.
- **Human-in-the-loop retraining**: The retrain orchestrator recommends but does not auto-promote models (SR 11-7 compliance).
- **SHAP for adverse actions**: TreeExplainer provides exact, model-consistent explanations — more reliable than surrogate models or generic reason-code tables for ECOA compliance.
- **Fairness via proxy attributes**: Direct protected classes (race, gender) are unavailable in the data. Analysis uses proxy attributes (home ownership, verification status, employment reporting) to detect potential disparate impact.
- **Data quality as optional dependency**: Great Expectations validation is wrapped in `try/except ImportError` so the core pipeline remains functional without it — useful for lightweight dev environments.
- **joblib over pickle**: Faster serialization for numpy-heavy sklearn models, smaller files, and recommended by scikit-learn. Backward-compatible fallback to `.pkl` for pre-migration models.