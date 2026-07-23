# Go Services

Go port of the serving and monitoring layer: the scoring API, the
monitoring jobs, and the feature-store sync. Model **training** stays
in Python (LightGBM) — the Go retrain orchestrator shells out to it.

Everything ships as a single `gbm` binary with subcommands. The code is
laid out by area: `inference/` (serving), `monitoring/` (the batch jobs),
`db/` (Postgres/Supabase access), and `shared/` (config, model, metrics,
gold-dataset I/O used across both).

## What runs where

| Subcommand (package) | Language | Why |
|----------------------|----------|-----|
| `gbm serve` (`inference/`) | Go | `/score`, `/score/batch`, `/health`, `/reload`, `/metrics` — pure-Go inference + TreeSHAP, API-key auth, per-client rate limiting |
| `gbm drift` (`monitoring/`) | Go | PSI on scores, CSI per feature; also publishes the fairness summary |
| `gbm performance` (`monitoring/`) | Go | AUC / KS / Gini, decile rank-ordering |
| `gbm backfill` (`monitoring/`) | Go | Mature `scoring_log.actual_default` from Gold test labels |
| `gbm retrain` (`monitoring/`) | Go | Orchestrates retraining; calls `pipeline/train_challenger.py` for the LightGBM fit, evaluates champion vs challenger natively, gates promotion on fairness |
| `gbm promote` (`monitoring/`) | Go | Atomically promote the challenger to champion (versioned, symlink swap) |
| `gbm sync` (`monitoring/`) | Go | Bulk upsert of Gold features + training score distribution |
| `gbm prune` (`monitoring/`) | Go | Delete `scoring_log` rows older than the retention window |
| `pipeline/` (bronze/silver/gold, training, fairness, data quality) | Python | LightGBM / SHAP have no Go equivalents |
| `dags/` | Python | Airflow |

## Governance and compliance posture

- **Fail-closed model gate.** `serve` refuses to load a champion whose
  `model.json` validation status is not `APPROVED`; `promote` applies the
  same check so a non-approved challenger can never create a
  restart-time outage. `ALLOW_UNAPPROVED_MODEL=true` overrides, loudly
  logged, for local work.
- **Fail-closed audit trail.** Every decision is written to
  `scoring_log`; if the audit write fails the decision is withheld
  (HTTP 503), because for ECOA/Reg B the decision record is the
  compliance artifact.
- **Retention.** `gbm prune` enforces `SCORING_LOG_RETENTION_DAYS`
  (default 750 days, covering Reg B's 25-month requirement with margin)
  so the audit table cannot grow without bound.
- **Atomic promotion, single door.** `promote` publishes an immutable
  `models/versions/<v>` directory and repoints the `champion` symlink via
  rename — it is the only writer of `models/champion` (the Python
  pipeline always produces challengers and refuses to write through the
  symlink). When `SCORING_API_URL` is set, promote then POSTs
  `/reload` so serving picks up the new champion immediately
  (best-effort: a failed reload is loudly logged, never a failed
  promotion).
- **Stable applicant identity.** Feature-store IDs are keyed on the loan
  ID (`LC_<loan_id>`), never on parquet row position, so regenerating the
  Gold test set cannot re-point applicants or corrupt backfilled labels.

## Cross-language contract

`shared/config/contract.json` is the single source of truth for the
decision thresholds (approve/review), monitoring thresholds (PSI/CSI/AUC
drop), and the fairness DIR threshold. The Go services embed it at
compile time (`go:embed`, sanity-checked at init); the Python pipeline
(`pipeline/config.py`) and the UI load the same file, and
`tests/test_contract.py` pins every consumer in CI. A threshold change
is one edit, not a cross-repo hunt.

## How inference works without Python

`pipeline/export_model_json.py` dumps the champion LightGBM model
(tree nodes, baseline, metadata) to `data/models/champion/model.json`,
normalizing LightGBM's tree dump into a library-agnostic columnar
format (`value <= threshold` goes left, NaN follows
`missing_go_to_left`). `shared/model` loads it and implements:

- **`predict_proba`** — tree traversal verified to 1e-9 against
  LightGBM's own predictions.
- **TreeSHAP** — path-dependent algorithm (Lundberg et al., Algorithm 2)
  for the ECOA adverse action reasons, verified to 1e-6 against the
  Python `shap` library, including rows with missing values.

Tree structure is validated at load (child indexes must exceed their
parent, proving acyclicity), so a malformed export fails fast instead of
hanging a traversal.

The retrain flow exports the challenger automatically; a manually
trained challenger is exported with:

```bash
.venv/bin/python pipeline/export_model_json.py            # challenger (default)
```

A promoted champion's `model.json` is published by `gbm promote` and is
immutable — the exporter refuses to write through the champion symlink.

## Running

All binaries read `.env` (`DATABASE_URL`) and expect to run from the
repository root (or set `CREDIT_RISK_DATA_DIR` / `CREDIT_RISK_MODELS_DIR`).
`SCORING_API_URL` (optional) is the running API's base URL; when set,
`gbm promote` POSTs `/reload` after the champion swap, authenticating
with the first key in `API_KEYS`.
The scoring API requires `API_KEYS` (comma-separated); it refuses to
start unauthenticated unless `ALLOW_UNAUTHENTICATED_DEV=true` is set for
local development. `RATE_LIMIT_RPS` / `RATE_LIMIT_BURST` tune the
per-client token bucket (default 20/40); `/score/batch` charges one
token **per applicant**, so a batch cannot multiply a client's allowance.
`/health` reports 503 (degraded) when the feature store is unreachable,
so load balancers stop routing to an instance that cannot score.

```bash
cd go
go build -o bin/gbm .
cd ..

PORT=8000 ./go/bin/gbm serve        # POST /score {"applicant_id": "LC_123456"}
./go/bin/gbm drift                  # JSON report on stdout, logs to drift_log
./go/bin/gbm performance
./go/bin/gbm backfill               # mature scoring_log outcomes from Gold labels
./go/bin/gbm retrain psi_critical   # reason arg, default "manual"
./go/bin/gbm promote                # gated on the challenger being APPROVED
./go/bin/gbm sync
./go/bin/gbm prune                  # enforce scoring_log retention
```

Every monitor runs under a per-command deadline (`monitoring/timeouts.go`:
30 min drift/sync, 15 min performance/backfill, 10 min prune, 3 h retrain),
so a wedged dependency fails the Airflow task instead of hanging it
forever; retrain's deadline also bounds the Python training subprocess
via `exec.CommandContext`.

The monitors fall back to the Gold test set as a production proxy when
`DATABASE_URL` is unset or `scoring_log` is empty, exactly like the
Python agents. `PYTHON_BIN` overrides the interpreter used for
challenger training (default `.venv/bin/python`, then `python3`).

## Tests

```bash
cd go && go test ./...
```

The test suites cross-check against committed fixtures generated by the
Python stack (`shared/model/testdata/parity_model.json`,
`shared/metrics/testdata`): LightGBM `predict_proba`, `shap` TreeSHAP
values, numpy PSI/CSI, sklearn AUC/KS, and pandas decile analysis. The
parity fixtures are checked in, so the Go↔Python equivalence guarantees
run on every CI build rather than silently skipping when no local model
export exists. CI also runs `go vet`, `gofmt`, and `go test -race`.
