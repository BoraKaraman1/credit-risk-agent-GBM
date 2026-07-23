# Code Map

A file-by-file guide to this repository: what each code file does and how the
pieces fit together.

**Big picture:** a credit-risk GBM system. A Python medallion pipeline
(Bronze → Silver → Gold) trains a LightGBM default-risk model on Lending Club
data and exports it to a portable `model.json`. A pure-Go layer (`gbm` binary)
serves scores over HTTP and runs deterministic drift / performance / retrain
monitoring. Airflow orchestrates the batch jobs; an advisory LLM agent writes
weekly model-risk memos but never gates anything.

---

## `pipeline/` — Python data & training pipeline

| File | What it does |
|---|---|
| `pipeline/__init__.py` | Empty package marker. |
| `pipeline/config.py` | Central config: path helpers for bronze/silver/gold/models/champion/challenger dirs driven by env vars (`CREDIT_RISK_DATA_DIR`, `CREDIT_RISK_MODELS_DIR`), mirroring `go/shared/config`. Also implements strict data-quality mode (`enforce_data_quality` raises in CI/prod, warns locally). |
| `pipeline/bronze_ingest.py` | Bronze ingestion: reads raw Lending Club accepted/rejected CSVs with pandas and writes immutable Parquet to `data/bronze/` with `ingested_at`/`source_file` metadata. Skips existing outputs; validates before writing. CLI: `python pipeline/bronze_ingest.py`. |
| `pipeline/data_quality.py` | Pure-pandas validation for all three layers — row counts, non-null metadata, FICO 300–850, DTI 0–100, binary target, default-rate sanity (5–50%), grade ranges, Gold feature-schema presence. Runs before each layer's artifact is written; failures flow into `config.enforce_data_quality` (raise in strict mode, warn otherwise). |
| `pipeline/silver_transform.py` | Bronze → Silver: filters accepted loans to resolved `loan_status` values, builds the binary `default` target, keeps only 25 origination-time columns (explicitly no post-origination leakage), parses `term`/`emp_length`/`int_rate`, derives `credit_history_months` and `fico_score`, applies sentinel/median imputation and quality gates. Also normalizes the rejected-loans file (`default = NaN`) for reject inference. Outputs `accepted_clean.parquet` / `rejected_clean.parquet`. |
| `pipeline/gold_features.py` | Silver → Gold: engineers the 33-feature space (`log_annual_inc`, `loan_to_income`, `installment_to_income`, `dti_x_income`, ordinal grade/sub-grade, binary flags like `delinq_ever`/`high_utilization`, deterministic persisted category maps with unseen → −1). Performs the time-aware split (train <2016, val 2016–H1 2017, test ≥2017-07) and writes `features_{train,val,test}.parquet` + `feature_metadata.json` (`FEATURE_VERSION=1` schema contract). |
| `pipeline/train.py` | Challenger training: trains `lightgbm.LGBMClassifier` on train+val with a deterministic stratified early-stopping carve-out (the same carve-out later fits the calibrator), evaluates AUC/KS/Gini, logs to MLflow, then — under the cross-language registry flock — saves versioned `model.joblib` + `model_metadata.json`, calibration, fairness, and the model card into `challenger/`. The champion is only ever created by `gbm promote`. CLI: `python pipeline/train.py`. |
| `pipeline/train_challenger.py` | Challenger trainer invoked by the Go `gbm retrain` orchestrator: trains with deliberately different hyperparameters (deeper, slower LR, more regularization) using the same early-stopping carve-out, fits calibration, computes fairness for both challenger and champion, exports `model.json`, prints a single JSON result on stdout. CLI: `python pipeline/train_challenger.py <version>`. |
| `pipeline/calibrate.py` | Fits an sklearn `IsotonicRegression` calibrator on the early-stopping holdout, reports Brier scores (raw vs calibrated) and quantile-binned reliability tables, and maps calibrated PDs to a scorecard score via points-to-double-odds (base 600 @ 30:1 odds, PDO 20) with `floor(x+0.5)` rounding chosen to match Go's `math.Floor`. Saves `calibrator.joblib`. |
| `pipeline/fairness.py` | Fairness analysis with no external library: Disparate Impact Ratio (80% rule), Equal Opportunity Difference, and Statistical Parity Difference across proxy protected attributes (`home_ownership`, `verification_status`, `emp_length_missing`), using calibrated PDs against the serving thresholds (approve <0.15, review <0.30). `summarize()` feeds the model card and the promotion gate. |
| `pipeline/reject_inference.py` | Reject inference via parcelling: aligns rejected applicants to the 33-feature space (overlap mapped, rest filled with training medians, capped at 500k rows), scores them with the champion, pseudo-labels the top fraction assuming 2× the accepted default rate (capped at 60%), retrains on combined data with 0.3 sample weights, compares champion vs augmented (AUC/KS/Gini + PSI), and saves a `-ri` challenger with a promote/keep recommendation. |
| `pipeline/model_card.py` | Model card / validation report generator (SR 11-7 artifact). Assembles markdown from model + feature metadata: validation status (`_validation_status` derives APPROVED / REVIEW REQUIRED from DIR violations and calibration presence), data window, discrimination metrics, calibration reliability, scorecard scaling, fairness breakdown, hyperparameters. Every training run writes `model_card.md` into the model's own directory; the CLI default (`docs/model_card.md`) regenerates the committed repo snapshot. |
| `pipeline/export_model_json.py` | Exports a trained LightGBM classifier to the portable `model.json` consumed by the Go runtime: columnar node arrays with sklearn-style semantics (value ≤ threshold → left, NaN follows `missing_go_to_left`), verified to reproduce LightGBM raw scores to 1e-9, embedding baseline prediction, validation status, feature schema version, isotonic calibrator breakpoints, and scorecard params. Written atomically (temp file + fsync + rename). |

## `dags/` — Airflow orchestration

- **`dags/credit_risk_pipeline.py`** — Monthly DAG (`credit_risk_pipeline`): bronze_ingest → silver_transform → gold_features → train_challenger → export_model_json → `gbm sync`, plus a parallel fairness_analysis task (skips until a champion exists) and a BranchPythonOperator after export that conditionally runs reject inference when `run_reject_inference` is set in the DAG-run config. The DAG only ever produces a gated challenger; promotion to champion stays a human `gbm promote`.
- **`dags/credit_risk_monitoring.py`** — Weekly DAG (`credit_risk_monitoring`): outcome backfill → performance + drift monitors in parallel → retrain branch reading the monitors' machine `needs_retrain`/`retrain_reasons` verdicts (thresholds live only in Go/contract.json) and invoking `gbm retrain <reason>` → optional advisory LLM review task that always runs and never blocks the loop; a parallel `scoring_log_prune` task enforces audit retention (`gbm prune`). Reports flow via XCom.

## `agents/` — LLM review agent

- **`agents/review_agent.py`** — Advisory model-risk review agent using the Anthropic SDK (model pinned via `REVIEW_AGENT_MODEL`). Reads weekly monitoring outputs (drift/performance/retrain reports, model card, model metadata) through three read-only tools (`run_monitor` executing Go `gbm drift|performance`, `read_model_card`, `read_model_metadata`) and writes a structured markdown memo (Summary / Findings / Risks / Follow-ups, <600 words) to `docs/monitoring_review.md`. Explicitly advisory: all gates stay deterministic; every run appends a full audit record (prompt, memo, token usage) to `data/review_agent_audit.jsonl`; importable without the `anthropic` package. CLI: `ANTHROPIC_API_KEY=... python agents/review_agent.py`.

## `ui/` — Streamlit front end

- **`ui/app.py`** — Entry point (`streamlit run ui/app.py`): two-page `st.navigation` app configured via `API_URL` / `API_KEY` / `DATABASE_URL`.
- **`ui/scoring.py`** — Scoring page: applicant picker (feature-store sample or free text), `POST /score`, then calibrated PD / scorecard score / raw score tiles, the decision band as an icon+label banner, the ECOA adverse-action table, and the raw JSON.
- **`ui/dashboard.py`** — Monitoring & fairness page reading `drift_log`: PSI trend with labeled warning/critical rules, per-feature CSI bars, AUC trend vs training baseline and retrain threshold, decile rank-ordering, and per-attribute DIR bars (four-fifths rule marked) with the full DIR/EOD/SPD table. Single-hue marks; status colors only on labeled threshold rules; every chart has a table view.
- **`ui/core.py`** — Pure data-shaping helpers (frames for fairness/CSI/deciles/history, decision presentation, thresholds loaded from the cross-language contract `go/shared/config/contract.json`); no streamlit import, unit-tested.
- **`ui/services.py`** — The UI's only I/O: REST calls to the scoring API and cached SQLAlchemy reads of `drift_log` / `applicant_features`.

## `go/` — pure-Go serving & monitoring (`gbm` binary)

| File | What it does |
|---|---|
| `go/main.go` | CLI spine: dispatches `serve` → `inference.Serve()` and `drift` / `performance` / `retrain [reason]` / `promote` / `backfill` / `sync` → the corresponding `monitoring.Run*` functions. Unknown commands print usage, exit 2. |
| `go/inference/server.go` | The HTTP scoring API (Go port of the old FastAPI service; handlers live here). Endpoints: `POST /score`, `POST /score/batch` (≤1000), `POST /reload`, `GET /health` (checks DB), `GET /metrics`. Core flow: fetch features from the store → enforce staleness (720 h) + feature-version contract → predict → calibrate PD/scorecard → decide (0.15/0.30 thresholds) → top-4 ECOA/Reg B adverse-action reasons from TreeSHAP → audit-log to `scoring_log`. Enforces the SR 11-7 governance gate (only APPROVED models serve unless `ALLOW_UNAPPROVED_MODEL`) and fails closed without API keys unless `ALLOW_UNAUTHENTICATED_DEV`. |
| `go/inference/metrics.go` | Prometheus metrics: request counter, latency histogram, PD score-distribution histogram, decision counter, `model_info` version gauge. |
| `go/inference/middleware.go` | Middleware chain: request IDs + access logs + HTTP metrics; constant-time API-key auth (`X-API-Key` or Bearer); per-client token-bucket rate limiter keyed by API key or client IP (`X-Forwarded-For` honored only with `TRUST_PROXY_HEADERS`), with idle-bucket eviction. |
| `go/monitoring/drift.go` | `gbm drift`: recomputes training scores from Gold train parquet with the current champion, pulls production scores + feature snapshots from `scoring_log` (test-set fallback only when the DB is unset or empty; a configured-but-unreachable DB fails the run), computes score PSI and per-feature CSI, classifies OK/WARNING/CRITICAL, logs to `drift_log`, emits both a prose recommendation and machine-readable `needs_retrain`/`retrain_reasons`. |
| `go/monitoring/performance.go` | `gbm performance`: AUC/KS/Gini plus decile rank-ordering on cohorts with observed outcomes (from `scoring_log`, test-set proxy fallback under the same fail-closed rule as drift), compares against training metrics, guards single-class cohorts ("INSUFFICIENT OUTCOMES") and missing baselines ("INSUFFICIENT BASELINE"), logs to `drift_log`, emits prose plus `needs_retrain`/`retrain_reasons`. |
| `go/monitoring/retrain.go` | `gbm retrain [reason]`: shells out to Python (`pipeline/train_challenger.py` — the one step that stays in Python), then natively evaluates champion vs challenger on the test set (AUC/KS/Gini, score PSI) and applies a champion-relative four-fifths fairness gate (blocks new or worsened DIR violations; fairness overrides AUC gains). Emits PROMOTE/CONSIDER/DO-NOT-PROMOTE with the `gbm promote` command. |
| `go/monitoring/promote.go` | `gbm promote`: publishes `models/challenger` as an immutable `models/versions/<version>` directory and atomically repoints the `models/champion` symlink (single rename; legacy real champion dirs are archived first). Refuses duplicate versions and non-APPROVED challengers (same governance gate and `ALLOW_UNAPPROVED_MODEL` override as serving). |
| `go/monitoring/backfill.go` | `gbm backfill`: for `scoring_log` rows older than `OUTCOME_BACKFILL_DELAY_DAYS` with NULL `actual_default`, maps applicant IDs (`LC_<loan-id>`, keyed on the stable LendingClub id carried through the medallion layers) to Gold test-set labels and writes outcomes back so the performance monitor can run on real production data. |
| `go/monitoring/prune.go` | `gbm prune`: deletes `scoring_log` rows older than `SCORING_LOG_RETENTION_DAYS` (default 750 days — Reg B's 25-month record retention with margin); run weekly by the monitoring DAG. |
| `go/monitoring/lock.go` | Advisory flock on `models/.registry.lock` taken by `gbm retrain` and `gbm promote` — and, via `pipeline/io_utils.registry_lock()`, by the Python training/export entry points — so concurrent registry mutations exclude each other instead of relying on Airflow scheduling. |
| `go/monitoring/sync.go` | `gbm sync` (port of `sync_to_supabase.py`): bulk-upserts Gold test-set features into the `applicant_features` feature store in 10k-row batches (assigning `LC_<loan-id>` IDs from the Gold `id` column, computing data completeness), and stores the champion's training score histogram in `training_distribution` as the external PSI reference. |
| `go/shared/config/config.go` | Thresholds parsed at init from the embedded cross-language `contract.json` (PSI 0.10/0.25, CSI 0.20, AUC-drop 0.03, DIR 0.80, decision 0.15/0.30) plus env-driven accessors for data/models dirs, `DATABASE_URL`, `API_KEYS`, rate limits, backfill delay, and audited overrides (`ALLOW_UNAUTHENTICATED_DEV`, `ALLOW_UNAPPROVED_MODEL`, `TRUST_PROXY_HEADERS`). Mirrors the Python config. |
| `go/shared/metrics/metrics.go` | Pure-Go monitoring math reproducing numpy/sklearn/pandas exactly: `Histogram`/`Linspace`/`quantile` primitives, PSI, CSI, ROC AUC (Mann-Whitney with midranks), KS, decile analysis with rank-order-break counting. |
| `go/shared/gold/gold.go` | Parquet reader for Gold feature files into a columnar `Frame` (nulls → NaN, ints widened to float64); `Frame.Rows` assembles the row-major matrix in model column order for batch scoring. |
| `go/shared/model/model.go` | Loads the library-agnostic `model.json` and provides pure-Go GBM inference: `PredictProba`/`PredictProbaBatch` (parallel), NaN routing via `missing_go_to_left`, structural tree validation (child-index/acyclicity checks), cover-weighted `ExpectedValue` SHAP base. Carries `ValidationStatus` for the serving governance gate. |
| `go/shared/model/calibration.go` | Isotonic `Calibration.Apply` mirroring sklearn (`out_of_bounds="clip"`, linear interpolation) and the points-to-double-odds `Scorecard.Score` (600 = 30:1 odds, PDO 20, PD clipped to 1e-6). |
| `go/shared/model/shap.go` | Path-dependent TreeSHAP (Lundberg et al., Algorithm 2) over the exported trees, matching `shap.TreeExplainer` in log-odds space including missing-value routing. Powers ECOA adverse-action reasons. |
| `go/db/db.go` | pgx-based Postgres/Supabase layer: connection pool, `InsertScoringLog`/`InsertDriftLog` audit writes, `FetchApplicantFeatures`, `RecentScores`/`RecentFeatureSnapshots` (drift), `ScoredOutcomes`/`PendingOutcomes`/`BackfillOutcomes` (performance/backfill), bulk `UpsertApplicantFeatures` and `InsertTrainingDistribution` (sync). All queries version-filtered so prior models' data never contaminates current evaluations. |

**Go tests** verify parity against Python-generated fixtures (PSI/CSI vs numpy,
AUC/KS vs sklearn, SHAP/calibration/model outputs to 1e-9/1e-6 incl. SHAP local
accuracy), plus unit tests for config parsing, parquet round-trips, tree
validation, API handlers/middleware, Prometheus metrics, backfill ID parsing,
promotion atomicity, and the fairness gate.

## `backup_python/` — superseded reference snapshot

Replaced by the Go services in `go/` (June 2026); nothing imports it. Kept for
reference only. Key difference: this stack loads a serialized `model.joblib`
and uses the `shap` library at serve time, whereas the Go layer consumes
`model.json` and reimplements inference/TreeSHAP natively.

| File | What it did (→ Go replacement) |
|---|---|
| `api/scoring_service.py` | Original FastAPI scoring service: loads champion `model.joblib` + `shap.TreeExplainer`, `/score`, `/score/batch`, `/health`, `/reload`, SHAP-based adverse-action reasons, audit logging → `gbm serve`. |
| `api/models.py` | Pydantic request/response schemas for the API (`ScoreRequest`, `ScoreResponse`, `AdverseAction`, batch, health). |
| `agents/config.py` | Shared thresholds and path/DB config for the monitoring agents. |
| `agents/db_logging.py` | Parameterized JSONB insert SQL for the `drift_log` table. |
| `agents/drift_monitor.py` | PSI/CSI drift monitor → `gbm drift`. |
| `agents/performance_monitor.py` | AUC/KS/Gini + decile rank-ordering monitor → `gbm performance`. |
| `agents/retrain_orchestrator.py` | Challenger-training agent (HistGradientBoostingClassifier) with PROMOTE recommendation, no auto-promotion → `gbm retrain` + `pipeline/train_challenger.py`. |
| `pipeline/sync_to_supabase.py` | Feature-store sync via psycopg2 bulk upserts + training score histogram → `gbm sync`. |
| `tests/test_api.py` | TestClient tests for the FastAPI service with mocked DB. |
| `tests/test_audit_logging_sql.py` | JSONB-binding tests for the audit-insert SQL. |

## `tests/` — Python pytest suite

- **`tests/test_pipeline.py`** — Silver parsing helpers (`parse_term`, `parse_emp_length`), target-status mapping disjointness, leakage exclusion from `ORIGINATION_COLS`, Gold `engineer_features` output.
- **`tests/test_features.py`** — KS/Gini helpers, reject-inference functions (`align_rejected_features`, `assign_pseudo_labels`, `compare_models` incl. PSI), LightGBM dummy-model scoring.
- **`tests/test_data_quality.py`** — `validate_bronze`/`validate_silver`/`validate_gold` expectation logic (pass/fail paths).
- **`tests/test_model_card.py`** — Model card renderer and `_validation_status` verdict logic (APPROVED vs REVIEW REQUIRED).
- **`tests/test_fairness.py`** — Fairness metrics (DIR with 80% threshold, EOD, SPD, `analyze_attribute`, `run`, `summarize`) on synthetic data.
- **`tests/test_review_agent.py`** — Review agent's deterministic plumbing only (tool validation, prompt assembly, memo/audit writing); no API key required.
- **`tests/test_ui.py`** — UI pure helpers (`ui/core.py`): frame shaping for fairness/CSI/deciles/history, decision presentation, PSI status; no streamlit or network.
- **`tests/test_contract.py`** — Pins pipeline/config, fairness, and ui/core threshold values to `go/shared/config/contract.json`, the single cross-language source of truth (Go side covered by `config_test.go` over the embedded copy).
- **`tests/test_dags.py`** — Airflow DAG integrity: AST-resolves every lazy `from X import Y` inside task callables (no Airflow required), and parses both DAGs via `DagBag` with structure assertions (skipped unless Airflow is installed; CI runs it in the `airflow-dags` job).
- **`tests/test_calibration.py`** — Scorecard scaling math (base 600@30:1, PDO 20, monotonicity) and isotonic calibration behavior.

## Repo root & scripts

- **`scripts/draw_architecture.py`** — Renders `docs/architecture.png`, a hand-laid matplotlib diagram of the whole system (color-coded clusters, typed arrows for artifact loads / DB access / control flow).

## `notebooks/` — analysis notebooks

- **`01_eda.ipynb`** — EDA of the Gold feature set: class balance, temporal drift, distributions, correlations, default rate by grade/FICO band.
- **`02_training.ipynb`** — Training walkthrough with MLflow tracking, ROC/PR/calibration curves, decile rank-ordering analysis. (Markdown references `HistGradientBoostingClassifier`; production uses LightGBM.)
- **`03_shap_analysis.ipynb`** — SHAP interpretability (global importance, dependence plots, per-decision waterfall) motivated by SR 11-7 / ECOA explainability.
- **`04_reject_inference.ipynb`** — Reject-inference parcelling demo: alignment, pseudo-labeling, weighted retraining, champion-vs-augmented comparison with PSI.

## Infra

- **`docker-compose.yml`** — Local dev stack: Postgres 16 (schema from `pipeline/supabase_schema.sql`), the Go scoring API (port 8000), the Streamlit UI (port 8501), optional `orchestration` profile with single-container Airflow standalone (port 8080, `./data` mounted in), optional `monitoring` profile with Prometheus + Grafana. Local-only credentials.
- **`Dockerfile.airflow`** — Airflow 2.8.4 image with the compiled `gbm` binary embedded (multi-stage Go build) so DAGs can invoke Go jobs via `CREDIT_RISK_GO_BIN`.
- **`Dockerfile.api`** — Multi-stage build of the `gbm` binary on slim Debian, non-root, `/health` healthcheck; entrypoint `gbm serve` on port 8000.
- **`Dockerfile.pipeline`** — Python 3.11-slim image running `python pipeline/train.py` as non-root.
- **`Dockerfile.test`** — Containerized `pytest -q` runner.
- **`Dockerfile.ui`** — Python 3.11-slim Streamlit image (non-root, `/_stcore/health` healthcheck) serving `ui/app.py` on port 8501.
- **`monitoring/prometheus.yml`** — Scrape config (15 s) for the API's authenticated `/metrics`.
- **`monitoring/grafana/`** — Provisioned Prometheus datasource, dashboard provider, and the "Credit Risk Scoring API" dashboard (request rate, p95 latency, decisions/sec, PD distribution) built on `scoring_api_*` metrics.
- **`.claude/hooks/*.sh`** — Claude Code hooks: `pre-tool-use.sh` (blocks dangerous commands, audit-logs tool calls), `post-tool-use.sh` (audit results, bell on test stderr, prettier auto-format), `stop.sh` (turn-end log rotation + macOS notification), `notification.sh` (osascript/notify-send delivery).
