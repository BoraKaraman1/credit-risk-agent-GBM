"""
Credit Risk Scoring API
FastAPI service that loads the champion model, queries Supabase for features,
and returns a credit decision with logging.
"""

import json
import os
import pickle
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from api.models import (
    ScoreRequest, ScoreResponse,
    BatchScoreRequest, BatchScoreResponse,
    HealthResponse,
)

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

# Decision thresholds
APPROVE_THRESHOLD = 0.15
REVIEW_THRESHOLD = 0.30

app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")

# --- Global state loaded at startup ---
_model = None
_model_meta = None
_model_loaded_at = None
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise HTTPException(500, "DATABASE_URL not configured")
        _engine = create_engine(db_url, pool_pre_ping=True)
    return _engine


def load_model():
    global _model, _model_meta, _model_loaded_at
    champion_dir = MODELS_DIR / "champion"
    model_path = champion_dir / "model.pkl"
    meta_path = champion_dir / "model_metadata.json"

    if not model_path.exists():
        raise RuntimeError(f"No champion model at {model_path}")

    with open(model_path, "rb") as f:
        _model = pickle.load(f)
    with open(meta_path) as f:
        _model_meta = json.load(f)
    _model_loaded_at = datetime.now(timezone.utc).isoformat()

    print(f"[API] Loaded model {_model_meta['version']} ({_model_meta['n_features']} features)")


@app.on_event("startup")
def startup():
    load_model()


def fetch_features(applicant_id: str) -> dict:
    """Fetch feature vector from Supabase by applicant_id."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT features, data_completeness, fico_score, grade, computed_at
                FROM applicant_features
                WHERE applicant_id = :applicant_id
            """),
            {"applicant_id": applicant_id},
        ).fetchone()

    if result is None:
        raise HTTPException(404, f"Applicant {applicant_id} not found in feature store")

    features, completeness, fico, grade, computed_at = result

    # Staleness check (>24h)
    if computed_at:
        age_hours = (datetime.now(timezone.utc) - computed_at.replace(tzinfo=timezone.utc)).total_seconds() / 3600
        if age_hours > 24:
            raise HTTPException(
                409, f"Features for {applicant_id} are {age_hours:.1f}h old (stale). Trigger refresh."
            )

    if isinstance(features, str):
        features = json.loads(features)

    return {
        "features": features,
        "data_completeness": float(completeness) if completeness else None,
        "fico_score": int(fico) if fico else None,
        "grade": int(grade) if grade else None,
    }


def score_applicant(applicant_id: str) -> ScoreResponse:
    """Score a single applicant: fetch features → predict → decide → log."""
    feat_data = fetch_features(applicant_id)
    features = feat_data["features"]

    # Build feature vector in the correct column order
    feature_cols = _model_meta["features"]
    X = np.array([[features.get(col, 0.0) or 0.0 for col in feature_cols]])

    score = float(_model.predict_proba(X)[:, 1][0])

    # Decision rules
    if score < APPROVE_THRESHOLD:
        decision = "approve"
    elif score < REVIEW_THRESHOLD:
        decision = "review"
    else:
        decision = "decline"

    # Log to scoring_log
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO scoring_log
                        (applicant_id, model_version, feature_snapshot, score, decision)
                    VALUES
                        (:applicant_id, :model_version, :feature_snapshot::jsonb, :score, :decision)
                """),
                {
                    "applicant_id": applicant_id,
                    "model_version": _model_meta["version"],
                    "feature_snapshot": json.dumps(features),
                    "score": score,
                    "decision": decision,
                },
            )
    except Exception as e:
        print(f"[API] Warning: failed to log scoring decision: {e}")

    return ScoreResponse(
        applicant_id=applicant_id,
        score=round(score, 5),
        decision=decision,
        model_version=_model_meta["version"],
        fico_score=feat_data["fico_score"],
        grade=feat_data["grade"],
        data_completeness=feat_data["data_completeness"],
    )


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    """Score a single applicant."""
    return score_applicant(request.applicant_id)


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest):
    """Score multiple applicants."""
    results = []
    errors = []
    for aid in request.applicant_ids:
        try:
            results.append(score_applicant(aid))
        except HTTPException as e:
            errors.append({"applicant_id": aid, "error": e.detail})
    return BatchScoreResponse(results=results, errors=errors)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check with model info."""
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    return HealthResponse(
        status="ok",
        model_version=_model_meta["version"],
        model_loaded_at=_model_loaded_at,
        n_features=_model_meta["n_features"],
    )


@app.post("/reload")
def reload_model():
    """Hot-reload the champion model without restarting the service."""
    load_model()
    return {"status": "reloaded", "model_version": _model_meta["version"]}
