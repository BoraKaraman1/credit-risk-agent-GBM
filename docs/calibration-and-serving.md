# Calibration, Scorecard Scaling, and Serving Hardening

Implementation notes for the work done on 2026-06-13. Two pieces landed
together:

1. Probability calibration and scorecard scaling, from the Python
   training pipeline through to the Go serving runtime.
2. Production engineering on the Go scoring API: API-key authentication,
   per-client rate limiting, Prometheus metrics with request-ID logging,
   and server timeouts with graceful shutdown.

This document explains what changed, why, and how each part works.

---

## Part 1: Probability Calibration and Scorecard Scaling

### The problem

A LightGBM classifier outputs a number between 0 and 1, but that number
only ranks applicants. It is not guaranteed that "the model says 0.20"
means "20% of such applicants default". Everything downstream of a
probability of default (risk-based pricing, loss provisioning, IFRS 9 /
CECL) reads the PD as a literal probability, so the gap between "ranks
well" and "is a real probability" matters.

Lenders also expect a score on a familiar scale (something like 300 to
850), not a raw probability. So there are two jobs: make the probability
honest, then translate it to a points scale.

### How calibration works

The calibrator is an isotonic regression: a monotonic step function that
maps raw model scores to observed default rates. Monotonic is the key
property. It can stretch or compress regions of the score range but can
never reverse the ordering, so AUC, KS, and Gini are unchanged. With
2.3M rows there is plenty of data to fit it without overfitting, which is
the usual reason to reach for Platt scaling instead.

The fit happens on the early-stopping holdout. The training pipeline
already carves out a stratified slice of train+val for LightGBM's early
stopping, and that slice never feeds the gradient fitting. Reusing it for
calibration means the calibrator learns on data the model did not train
on, so it corrects genuine optimism rather than memorized noise. That
carve-out was factored into `early_stopping_split()` in
`pipeline/train.py` so both the trainer and the calibrator produce the
identical split (`random_state=42`).

Evidence is reported on the test split:

- **Brier score** before and after calibration (mean squared error
  between predicted PD and the 0/1 outcome).
- A **reliability table**: quantile-binned mean predicted PD versus
  observed default rate, the data behind a calibration curve.

For the current champion (v1.2): 266 isotonic breakpoints, test Brier
0.174172 raw versus 0.174133 calibrated. The tiny change is itself a
finding. LightGBM's logistic objective was already close to calibrated;
the isotonic layer makes that verifiable instead of assumed.

### How scorecard scaling works

Calibrated PDs map to a points score using points-to-double-odds, the
standard credit scorecard convention:

```
factor = PDO / ln(2)
offset = base_score - factor * ln(base_odds)
score  = offset + factor * ln((1 - pd) / pd)        # (1-pd)/pd is the good:bad odds
```

The anchors are 600 points at 30:1 good:bad odds, with 20 points to
double the odds (PDO). That gives `factor` 28.8539 and `offset` 501.862.
An applicant at 60:1 scores 620, at 15:1 scores 580.

Two details exist for cross-language correctness:

- PD is clipped to `[1e-6, 1 - 1e-6]` before the log, because isotonic
  output can be exactly 0 or 1 and the odds transform would be infinite.
- Rounding is `floor(x + 0.5)`, not language-default rounding. Python's
  `round` is half-to-even and Go's `math.Round` is half-away-from-zero;
  `floor(x + 0.5)` is the one expression both compute identically, so
  integer scores match exactly rather than within a tolerance.

### What ships in the model file

`pipeline/export_model_json.py` embeds the calibrator into
`model.json` as two arrays, the isotonic breakpoints `x` (raw scores)
and `y` (calibrated PDs), plus the scorecard constants. The Go runtime
gets a calibrator as plain data, so it keeps its no-Python-at-serving-time
property. `calibrator.joblib` is also written next to the model for the
Python stack.

The export is optional and backward compatible: a `model.json` without a
calibration block loads and serves exactly as before.

### The Go side

`go/shared/model/calibration.go` mirrors scikit-learn at inference:

- `Calibration.Apply(p)` reproduces `IsotonicRegression(out_of_bounds=
  "clip")`. It clips to the breakpoint range, then linearly interpolates
  between the two bracketing breakpoints found by binary search
  (`sort.SearchFloat64s`). Same formula and operation order as scipy,
  which is why parity holds to 1e-9.
- `Scorecard.Score(pd)` runs the same clip, log-odds, `floor(x + 0.5)`
  sequence as the Python side, using the `factor` and `offset` read from
  `model.json` rather than recomputed, so there is one source of truth.

`Model` gained `Calibration *Calibration` and `Scorecard *Scorecard`
pointer fields. Pointers do real work here: an old `model.json` without
these keys unmarshals to `nil`, and every consumer checks for `nil`, so
a pre-calibration model serves byte-for-byte as before. `Load` validates
a present calibration block (non-empty, `len(x) == len(y)`, `x` sorted)
so a corrupt export fails at startup rather than producing garbage PDs at
request time.

### API response

The scoring response now carries three numbers:

- `score`: the raw model probability. It is logged and monitored, so the
  drift monitor's reference distribution stays valid.
- `pd`: the calibrated probability of default. The credit decision is made
  on this — the 0.15/0.30 thresholds are probabilities of default — with a
  fallback to the raw score only for pre-calibration models.
- `scaled_score`: the integer scorecard score.

`pd` and `scaled_score` are omitted (null) when the loaded model has no
calibrator. `/health` reports `calibrated: true|false`.

### Files touched (Part 1)

| File | Change |
|------|--------|
| `pipeline/calibrate.py` | New. Isotonic fit, Brier + reliability, scorecard scaling. |
| `pipeline/train.py` | `early_stopping_split()` extracted; `run()` calibrates after training. |
| `pipeline/train_challenger.py` | Calibrates the challenger before export. |
| `pipeline/reject_inference.py` | Deletes any stale calibrator so the exporter never pairs one with the wrong model. |
| `pipeline/export_model_json.py` | Embeds calibration breakpoints + scorecard params in `model.json`. |
| `go/shared/model/calibration.go` | New. `Calibration.Apply`, `Scorecard.Score`. |
| `go/shared/model/model.go` | Optional `Calibration`/`Scorecard` fields, `Load` validation. |
| `go/inference/server.go` | `pd` and `scaled_score` in the response, `calibrated` in health. |
| `tests/test_calibration.py`, `go/shared/model/calibration_test.go`, `model_test.go` | Unit + sklearn parity tests. |

---

## Part 2: Scoring API Production Engineering

The API was a bare `http.ListenAndServe` with no auth, no rate limiting,
no metrics, and no graceful shutdown. Four things were added.

### Request flow

Every business request passes through this chain:

```
client
  -> instrument            (request ID, access log, HTTP metrics)
     -> auth               (X-API-Key, constant-time compare)
        -> request limit   (per-client HTTP token bucket)
           -> handler      (score / batch / reload)
              -> scoring limit (one token per applicant)
```

`instrument` is outermost so even a 401 or 429 gets a request ID, an
access-log line, and a metrics increment. `/health` is instrumented and
open (liveness/readiness). `/metrics` is instrumented and authenticated —
scrapers present the key as a Bearer token (see Authentication).

### Authentication

`X-API-Key` is compared against the keys in the `API_KEYS` environment
variable (comma-separated). The comparison is constant time
(`crypto/subtle.ConstantTimeCompare`) and does not short-circuit on the
first match, so timing does not reveal which or how many keys exist.

The server fails closed: when `API_KEYS` is unset it refuses to start
unless `ALLOW_UNAUTHENTICATED_DEV=true` is set explicitly for local
development. A key may be presented as `X-API-Key` or as an
`Authorization: Bearer <key>` header (the Bearer form lets Prometheus
scrape `/metrics`).

Authentication applies to `/score`, `/score/batch`, `/reload`, and
`/metrics` (the per-client rate limiter is skipped for `/metrics` so
scrapes are not throttled). `/health` stays open so load balancers can
reach it.

### Rate limiting

Two per-client token buckets (`golang.org/x/time/rate`) keep HTTP
overhead and scoring work separate. The request bucket charges every
authenticated business request once. The scoring bucket charges
`/score` once and `/score/batch` once per valid applicant, so a
two-applicant batch costs one request token and two scoring tokens,
never three scoring tokens.

Request limits default to 50 requests/second with a burst of 100, set by
`REQUEST_RATE_LIMIT_RPS` and `REQUEST_RATE_LIMIT_BURST`. Scoring limits
default to 20 decisions/second with a burst of 40, set by
`SCORING_RATE_LIMIT_RPS` and `SCORING_RATE_LIMIT_BURST`. The older
`RATE_LIMIT_RPS` and `RATE_LIMIT_BURST` variables remain fallbacks for
the scoring bucket. An over-budget request gets a `429` with a
`Retry-After` header.

Both buckets use the API-key digest when authenticated and the client IP
otherwise. Idle client entries are evicted periodically so the limiter
maps stay bounded.

### Observability

`/metrics` exposes Prometheus metrics from `prometheus/client_golang` on
the default registry (which also brings Go runtime and process metrics
for free):

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `scoring_api_requests_total` | counter | endpoint, method, status | HTTP request count |
| `scoring_api_request_duration_seconds` | histogram | endpoint | request latency |
| `scoring_api_score` | histogram | (none) | predicted PD distribution |
| `scoring_api_decisions_total` | counter | decision | approve / review / decline counts |
| `scoring_api_model_info` | gauge | version | loaded model version (value always 1) |

HTTP metrics are recorded by `instrument`. Score and decision metrics are
recorded only after the complete audit row commits successfully.
`scoring_api_model_info` uses the info
pattern: a gauge set to 1 with the version as a label, reset on reload so
only the current version is present.

The endpoint label is a fixed, known string per route rather than the raw
URL path. That keeps label cardinality bounded, which matters because
Prometheus creates one time series per label combination.

Every request also gets an `X-Request-ID` header (generated as a UUID, or
the incoming one echoed back). A request-scoped `slog` logger carrying
that ID is stored in the context, so the access log and any handler log
lines for the same request share an ID.

A compose profile brings up Prometheus and a pre-provisioned Grafana
dashboard:

```bash
docker compose --profile monitoring up
```

Grafana is at `http://localhost:3000` (anonymous viewer access enabled),
Prometheus at `http://localhost:9090`. The dashboard has panels for
request rate by status, p95 latency by endpoint, decisions per second,
the predicted-PD distribution (median and p90), and the loaded model
version. The default `docker compose up` is unchanged: it runs only
Postgres and the API.

### Server hardening

The bare `http.ListenAndServe` was replaced with an `http.Server`
configured with timeouts:

| Timeout | Value | Reason |
|---------|-------|--------|
| ReadHeaderTimeout | 5s | slow-header attacks |
| ReadTimeout | 15s | slow request bodies |
| WriteTimeout | 30s | generous, since batch scoring does many DB round-trips |
| IdleTimeout | 60s | keep-alive cleanup |

On `SIGINT` or `SIGTERM` (`signal.NotifyContext`), the server stops
accepting new connections and drains in-flight requests for up to 30
seconds before exiting. A deploy or `docker compose down` no longer cuts
off a scoring request mid-flight. If `ListenAndServe` fails to bind, the
process exits non-zero instead of silently continuing.

### Configuration reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_KEYS` | unset (required) | comma-separated accepted API keys; the server fails closed if unset |
| `ALLOW_UNAUTHENTICATED_DEV` | unset (false) | allow startup with no `API_KEYS` (local development only) |
| `ALLOW_UNAPPROVED_MODEL` | unset (false) | audited override to serve a non-APPROVED model card (SR 11-7) |
| `REQUEST_RATE_LIMIT_RPS` | 50 | per-client sustained HTTP request rate |
| `REQUEST_RATE_LIMIT_BURST` | 100 | per-client HTTP request bucket depth |
| `SCORING_RATE_LIMIT_RPS` | 20 | per-client sustained scoring rate |
| `SCORING_RATE_LIMIT_BURST` | 40 | per-client scoring bucket depth |
| `RATE_LIMIT_RPS` | unset | legacy fallback for `SCORING_RATE_LIMIT_RPS` |
| `RATE_LIMIT_BURST` | unset | legacy fallback for `SCORING_RATE_LIMIT_BURST` |
| `PORT` | 8000 | listen port |
| `DATABASE_URL` | unset | Postgres / Supabase connection (existing) |
| `CREDIT_RISK_MODELS_DIR` | `data/models` | model location (existing) |

### Files touched (Part 2)

| File | Change |
|------|--------|
| `go/inference/middleware.go` | New. Instrument, auth, rate limiter, client identity. |
| `go/inference/metrics.go` | New. Prometheus metric definitions and helpers. |
| `go/inference/server.go` | Middleware wiring, `/metrics`, server timeouts, graceful shutdown. |
| `go/shared/config/config.go` | API keys plus separate request and scoring limiter configuration. |
| `go/inference/middleware_test.go`, `metrics_test.go`, `config_test.go` | Unit tests. |
| `docker-compose.yml` | Prometheus + Grafana under a `monitoring` profile. |
| `monitoring/` | New. `prometheus.yml`, Grafana provisioning, dashboard JSON. |
| `go/go.mod`, `go/go.sum` | New deps: `prometheus/client_golang`, `client_model`, `x/time`, `google/uuid` promoted to direct. |

---

## Testing

Both parts are verified by unit tests and, for the cross-language
contracts, by parity tests against the real champion model.

Calibration and scorecard:

- Python (`tests/test_calibration.py`): scorecard anchors (600 at 30:1,
  +20 per doubling), isotonic recovery of a known true PD on synthetic
  data, reliability-binning edge cases.
- Go (`calibration_test.go`): interpolation and clipping, scorecard
  anchors, `Load` rejection of malformed calibration blocks.
- Parity (`model_test.go`): Go `pd` matches the sklearn calibrator to
  1e-9 and the integer `scaled_score` matches Python exactly, on the same
  fixture rows used for prediction and SHAP parity.

API:

- `middleware_test.go`: auth pass-through when disabled, 401 without and
  with a wrong key, 200 plus key stashing with a valid key, constant-time
  key match, rate-limit burst then 429, IP and forwarded-for parsing,
  request-ID generation and propagation, status capture in metrics.
- `metrics_test.go`: `recordScore` increments the histogram and decision
  counter, `setModelInfo` clears the previous version on reload.
- `config_test.go`: `API_KEYS` parsing, rate-limit defaults and overrides.

A runtime smoke test confirmed the assembled server: `/health` open and
serving, `/metrics` serving under authentication
(`scoring_api_model_info{version="v1.2"} 1`), 401 without or with a wrong
key, a clean allowed-then-429 transition under a burst-2 limit with
`Retry-After: 1`, and a SIGTERM that logs draining and then a clean stop.

`gofmt`, `go vet`, the full Go suite, and `ruff` are all green.
