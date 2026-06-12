# Superseded Python Services

These files were replaced by the Go services in `go/` (June 2026) and
are kept for reference only. Nothing imports them.

| Backed up | Replaced by |
|-----------|-------------|
| `api/` (FastAPI scoring service) | `go/cmd/scoring-api` |
| `agents/drift_monitor.py` | `go/cmd/drift-monitor` |
| `agents/performance_monitor.py` | `go/cmd/performance-monitor` |
| `agents/retrain_orchestrator.py` | `go/cmd/retrain-orchestrator` (+ `pipeline/train_challenger.py`) |
| `agents/config.py`, `agents/db_logging.py` | `go/internal/config`, `go/internal/db` |
| `pipeline/sync_to_supabase.py` | `go/cmd/supabase-sync` |
| `tests/test_api.py` | `go/cmd/scoring-api/main_test.go` + Go package tests |
| `tests/test_audit_logging_sql.py` | `go/internal/db` (verified against live Supabase) |
| `requirements-api.txt` | none — the Go API has no Python dependencies |
