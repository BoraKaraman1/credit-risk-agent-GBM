# Credit Risk Scoring Pipeline

End-to-end credit risk modeling pipeline built on Lending Club data (2.3M+ loans). Implements a medallion architecture (Bronze/Silver/Gold), gradient boosting model training, a real-time scoring API, automated drift monitoring, and reject inference — all on free-tier infrastructure.

## Architecture

```
LOCAL FILESYSTEM (medallion)          SUPABASE POSTGRES           FASTAPI
┌─────────────────────────┐          ┌─────────────────┐        ┌────────────┐
│ data/                   │          │ applicant_feats  │───────>│ POST /score│
│   bronze/  (raw)        │          │ scoring_log      │        │ model      │
│   silver/  (clean)      │── sync ─>│ drift_log        │        │ in-memory  │
│   gold/    (features)   │          │ training_dist    │        └────────────┘
│   models/  (champion)   │          └─────────────────┘              │
└─────────────────────────┘                 ▲                         │
                                    Claude Code Agents ───────────────┘
                                    (drift, performance, retrain)
```

## Stack

| Component | Technology |
|-----------|-----------|
| Data storage | Local Parquet (medallion layers) |
| Database | Supabase PostgreSQL (free tier) |
| Model | scikit-learn HistGradientBoostingClassifier |
| API | FastAPI + Uvicorn |
| Experiment tracking | MLflow (local) |
| Interpretability | SHAP |
| Monitoring | Claude Code agents (PSI, CSI, AUC tracking) |

## Project Structure

```
├── pipeline/
│   ├── bronze_ingest.py          # CSV.gz → Bronze Parquet
│   ├── silver_transform.py       # Bronze → Silver (clean, validate, target)
│   ├── gold_features.py          # Silver → Gold (feature engineering, time split)
│   ├── train.py                  # Model training + MLflow logging
│   ├── reject_inference.py       # Selection bias correction via parcelling
│   ├── sync_to_supabase.py       # Gold → Supabase feature store
│   └── supabase_schema.sql       # Database DDL
├── api/
│   ├── scoring_service.py        # FastAPI scoring endpoints
│   └── models.py                 # Pydantic schemas
├── agents/
│   ├── drift_monitor.py          # PSI/CSI drift detection
│   ├── performance_monitor.py    # AUC/KS degradation tracking
│   ├── retrain_orchestrator.py   # Automated retraining pipeline
│   └── config.py                 # Shared thresholds and paths
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_training.ipynb         # Model training + evaluation
│   ├── 03_shap_analysis.ipynb    # SHAP interpretability
│   └── 04_reject_inference.ipynb # Reject inference demonstration
├── tests/
│   ├── test_pipeline.py          # Silver/Gold transform tests
│   ├── test_features.py          # Feature engineering + drift metric tests
│   └── test_api.py               # API endpoint tests
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

## Model Performance

Trained on Lending Club origination-time features only (no data leakage).

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

- `test_pipeline.py` — Silver parsing, target mapping, Gold feature engineering
- `test_features.py` — PSI/CSI metrics, reject inference alignment, pseudo-labeling
- `test_api.py` — API endpoints, decision logic, Pydantic schemas

## Key Design Decisions

- **No data leakage**: Only 25 origination-time columns are used. Post-origination features (payments, recoveries) are excluded.
- **Time-aware splitting**: Train/val/test split by loan issue date, not random. This mimics real deployment where the model is trained on historical data and evaluated on newer loans.
- **HistGradientBoosting over XGBoost**: Avoids OpenMP dependency issues on macOS ARM while providing identical statistical properties.
- **Sample-weighted reject inference**: Pseudo-labeled rejected applicants are weighted at 0.3 to prevent uncertain labels from dominating the training signal.
- **Human-in-the-loop retraining**: The retrain orchestrator recommends but does not auto-promote models (SR 11-7 compliance).
