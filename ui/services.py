"""
Data access for the Streamlit UI. The UI talks only to the scoring API
(REST) and Postgres; it never reads model files or parquet directly.

Configuration (environment):
    API_URL       scoring API base URL   (default http://localhost:8000)
    API_KEY       X-API-Key for /score   (empty = no header)
    DATABASE_URL  Postgres DSN for drift_log / applicant_features
    CREDIT_RISK_MODELS_DIR  champion artifacts for the governance page
                  (default data/models, read-only)
    MODEL_CARD_PATH         generated model card (default docs/model_card.md)
"""

import json
import os
from pathlib import Path

import requests
import streamlit as st
from sqlalchemy import create_engine, text

API_TIMEOUT_SECONDS = 30


def api_url() -> str:
    return os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def _api_headers() -> dict:
    key = os.getenv("API_KEY", "")
    return {"X-API-Key": key} if key else {}


def api_health() -> dict:
    """GET /health; raises requests exceptions on failure."""
    resp = requests.get(f"{api_url()}/health", timeout=API_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def api_health_full() -> tuple[dict, int]:
    """GET /health, returning (body, status_code). Unlike api_health this
    tolerates the 503 the API returns when degraded — the governance page
    exists precisely to show that state."""
    resp = requests.get(f"{api_url()}/health", timeout=API_TIMEOUT_SECONDS)
    try:
        body = resp.json()
    except ValueError:
        body = {"detail": resp.text}
    return body, resp.status_code


def api_score(applicant_id: str) -> tuple[dict, int]:
    """POST /score; returns (body, status_code) so pages can render the
    API's own error detail (404 unknown applicant, 401 bad key, ...)."""
    resp = requests.post(
        f"{api_url()}/score",
        json={"applicant_id": applicant_id},
        headers=_api_headers(),
        timeout=API_TIMEOUT_SECONDS,
    )
    try:
        body = resp.json()
    except ValueError:
        body = {"detail": resp.text}
    return body, resp.status_code


@st.cache_resource
def _engine():
    url = os.environ["DATABASE_URL"]
    # SQLAlchemy needs the postgresql:// scheme (Supabase URLs may say postgres://)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return create_engine(url, pool_pre_ping=True)


def db_available() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def _parse_details(raw):
    return raw if isinstance(raw, dict) else json.loads(raw or "{}")


@st.cache_data(ttl=60)
def metric_history(metric_name: str, limit: int = 500) -> list[tuple]:
    """drift_log rows for one metric, oldest first, details parsed."""
    with _engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT measured_at, metric_value, model_version, details
            FROM drift_log
            WHERE metric_name = :metric
            ORDER BY measured_at DESC
            LIMIT :limit
        """), {"metric": metric_name, "limit": limit}).fetchall()
    return [(r[0], float(r[1]), r[2], _parse_details(r[3])) for r in reversed(rows)]


@st.cache_data(ttl=300)
def sample_applicant_ids(limit: int = 200) -> list[str]:
    """A stable sample of feature-store applicants for the picker."""
    with _engine().connect() as conn:
        rows = conn.execute(text("""
            SELECT applicant_id FROM applicant_features
            ORDER BY applicant_id
            LIMIT :limit
        """), {"limit": limit}).fetchall()
    return [r[0] for r in rows]


# --- Governance page artifacts (published champion outputs, read-only) ---
#
# The governance page breaks the "API + Postgres only" rule deliberately:
# the model card and champion metadata are published artifacts, not live
# state, and model.json itself is never loaded here (it carries the full
# tree arrays and can be tens of MB).


@st.cache_data(ttl=60)
def champion_metadata() -> dict | None:
    """champion/model_metadata.json; None when no champion is on disk."""
    path = Path(os.getenv("CREDIT_RISK_MODELS_DIR", "data/models")) / "champion" / "model_metadata.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=60)
def model_card_markdown() -> str | None:
    """The generated model card (docs/model_card.md); None when absent."""
    try:
        return Path(os.getenv("MODEL_CARD_PATH", "docs/model_card.md")).read_text()
    except OSError:
        return None
