"""
Credit Risk Scoring API
FastAPI service that loads the champion model, queries Supabase for features,
and returns a credit decision with SHAP-based adverse action reasons and logging.
"""

import json
import logging
import os
import joblib
import numpy as np
import shap
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from api.models import (
    ScoreRequest, ScoreResponse, AdverseAction,
    BatchScoreRequest, BatchScoreResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

# Decision thresholds
APPROVE_THRESHOLD = 0.15
REVIEW_THRESHOLD = 0.30

# ECOA adverse action: return top 4 reasons
NUM_ADVERSE_ACTIONS = 4

# Human-readable feature name mapping (all 33 Gold features)
FEATURE_DISPLAY_NAMES = {
    "loan_amnt": "Loan Amount",
    "term": "Loan Term",
    "int_rate": "Interest Rate",
    "installment": "Monthly Installment",
    "emp_length": "Employment Length",
    "home_ownership": "Home Ownership Status",
    "annual_inc": "Annual Income",
    "verification_status": "Income Verification Status",
    "purpose": "Loan Purpose",
    "dti": "Debt-to-Income Ratio",
    "delinq_2yrs": "Delinquencies in Last 2 Years",
    "inq_last_6mths": "Credit Inquiries in Last 6 Months",
    "mths_since_last_delinq": "Months Since Last Delinquency",
    "open_acc": "Number of Open Accounts",
    "pub_rec": "Public Records",
    "revol_bal": "Revolving Balance",
    "revol_util": "Revolving Utilization Rate",
    "total_acc": "Total Number of Accounts",
    "mort_acc": "Number of Mortgage Accounts",
    "pub_rec_bankruptcies": "Public Record Bankruptcies",
    "credit_history_months": "Length of Credit History",
    "fico_score": "FICO Score",
    "emp_length_missing": "Employment Length Not Reported",
    "log_annual_inc": "Log Annual Income",
    "loan_to_income": "Loan-to-Income Ratio",
    "installment_to_income": "Installment-to-Income Ratio",
    "dti_x_income": "Absolute Debt Burden",
    "grade_numeric": "Credit Grade",
    "delinq_ever": "Prior Delinquency Flag",
    "high_utilization": "High Credit Utilization Flag",
    "has_mortgage": "Mortgage Account Flag",
    "has_bankruptcy": "Bankruptcy on Record",
    "sub_grade_numeric": "Credit Sub-Grade",
}

# --- Global state loaded at startup ---
_model = None
_model_meta = None
_model_loaded_at = None
_explainer = None
_engine = None


def _model_path(directory):
    """Resolve model path with backward compat."""
    p = directory / "model.joblib"
    if p.exists():
        return p
    return directory / "model.pkl"


def get_engine():
    global _engine
    if _engine is None:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            raise HTTPException(500, "DATABASE_URL not configured")
        _engine = create_engine(db_url, pool_pre_ping=True)
    return _engine


def load_model():
    global _model, _model_meta, _model_loaded_at, _explainer
    champion_dir = MODELS_DIR / "champion"
    model_path = _model_path(champion_dir)
    meta_path = champion_dir / "model_metadata.json"

    if not model_path.exists():
        raise RuntimeError(f"No champion model at {model_path}")

    _model = joblib.load(model_path)
    with open(meta_path) as f:
        _model_meta = json.load(f)
    _model_loaded_at = datetime.now(timezone.utc).isoformat()
    _explainer = shap.TreeExplainer(_model)

    logger.info(f"Loaded model {_model_meta['version']} ({_model_meta['n_features']} features), SHAP explainer ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0", lifespan=lifespan)


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


def _compute_adverse_actions(X: np.ndarray, feature_cols: list[str],
                              features: dict) -> list[AdverseAction]:
    """Compute top adverse action reasons via SHAP for a single applicant."""
    shap_values = _explainer.shap_values(X)
    shap_row = shap_values[0]  # single applicant

    # Positive SHAP values push toward default (class 1) → increase risk
    sorted_indices = np.argsort(-shap_row)

    actions = []
    for idx in sorted_indices:
        if shap_row[idx] <= 0:
            break
        if len(actions) >= NUM_ADVERSE_ACTIONS:
            break
        feat_name = feature_cols[idx]
        actions.append(AdverseAction(
            feature_name=FEATURE_DISPLAY_NAMES.get(feat_name, feat_name),
            shap_value=round(float(shap_row[idx]), 5),
            feature_value=round(float(features.get(feat_name, 0.0) or 0.0), 4),
            direction="increases risk",
        ))
    return actions


def score_applicant(applicant_id: str) -> ScoreResponse:
    """Score a single applicant: fetch features → predict → decide → explain → log."""
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

    # ECOA adverse action reasons (only for decline/review)
    adverse_actions = []
    if decision in ("decline", "review"):
        adverse_actions = _compute_adverse_actions(X, feature_cols, features)

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
        logger.warning(f"Failed to log scoring decision: {e}")

    return ScoreResponse(
        applicant_id=applicant_id,
        score=round(score, 5),
        decision=decision,
        model_version=_model_meta["version"],
        fico_score=feat_data["fico_score"],
        grade=feat_data["grade"],
        data_completeness=feat_data["data_completeness"],
        adverse_actions=adverse_actions,
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
