# Roadmap

Candidate additions to make the project more comprehensive and more
production-like, grouped by theme and ordered by impact-per-effort
within each group.

Suggested sequence: CI first (one afternoon, permanent badge), then
calibration + scorecard scaling (domain signal), then metrics endpoint +
auth + graceful shutdown (production Go signal), then the outcome
backfill loop (makes the existing monitoring stack actually operate).

## 1. Production engineering

- [x] **CI/CD** — GitHub Actions workflow that runs both test suites
  (`pytest`, `go test ./...`), linters (`golangci-lint`, `ruff`,
  `go vet`), and builds all four Docker images on every push. Add a
  status badge to the README. Highest-value single addition: nothing
  currently proves the repo is green except a local machine.
- [ ] **API authentication & rate limiting** — the scoring API is wide
  open; anyone with the URL can score applicants and write to the audit
  log. Add API-key middleware (keys in env, constant-time compare) and
  a token-bucket rate limiter (~30 lines in Go).
- [ ] **Observability** — `/metrics` Prometheus endpoint on the Go API
  (request count/latency histograms, score distribution, decision
  counts, model version label) plus request-ID structured logging.
  A compose profile with Prometheus + Grafana and one dashboard JSON
  makes the monitoring story visual instead of stdout JSON.
- [ ] **Server hardening** — the Go API uses default
  `http.ListenAndServe`: no read/write timeouts, no graceful shutdown.
  Use `http.Server{ReadTimeout, WriteTimeout}` and drain in-flight
  requests on SIGTERM.
- [ ] **DB integration test in CI** — the Postgres round-trip audit test
  moved to `backup_python/` with the Python services. Add a Go
  integration test (gated on `TEST_DATABASE_URL`, run against the
  compose Postgres in CI) covering `internal/db`.
- [ ] **Schema migrations** — `pipeline/supabase_schema.sql` is a
  one-shot DDL file. Move to `golang-migrate` or `atlas` with numbered
  migrations; schemas evolve.
- [ ] **OpenAPI spec** — FastAPI provided `/docs` for free; the Go port
  lost it. Hand-write `openapi.yaml` and serve it at `/docs`.

## 2. Credit-risk domain depth

- [ ] **Probability calibration** — do this first in this group. Raw GBM
  scores are not calibrated PDs, but everything downstream (pricing,
  provisioning, IFRS 9 / CECL) assumes they are. Add isotonic or Platt
  calibration after training, report Brier score and calibration
  curves, and export the calibrator with the model.
- [ ] **Scorecard scaling** — map PD to an industry-style score via
  points-to-double-odds (e.g. 600 = 30:1 odds, PDO 20). Return both PD
  and scaled score from the API.
- [ ] **Outcome backfill loop** — `actual_default` in `scoring_log` is
  always NULL, so the performance monitor permanently falls back to the
  test-set proxy. Add a job that simulates outcomes arriving (backfill
  labels for scored test-set applicants after a delay) so the AUC
  monitor runs on production data and the monitoring loop closes.
- [ ] **Fairness gate in promotion** — the retrain orchestrator
  recommends on AUC alone. Wire in the fairness analysis: refuse to
  recommend PROMOTE if the challenger's DIR drops below 0.8 on any
  proxy attribute (SR 11-7 + ECOA in one feature).
- [ ] **Standardized adverse action codes** — map SHAP reasons to the
  numbered Reg B / FCRA model adverse action codes lenders must return.
- [ ] **Model card / validation report** — auto-generate a markdown
  model card per training run: data window, metrics, calibration,
  fairness, hyperparameters, approval status. This is what model risk
  management teams consume.

## 3. Bigger swings

- [ ] **Shadow scoring** — score every request with both champion and
  challenger, log both, decide on champion only. Promotion decisions
  then use real traffic instead of test-set AUC.
- [ ] **Kubernetes** — manifests or a single-node k3d setup for the full
  stack.
- [ ] **Streaming ingestion** — applications arriving via a queue
  instead of batch parquet.
