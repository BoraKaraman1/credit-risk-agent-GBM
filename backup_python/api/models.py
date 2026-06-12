"""Pydantic request/response schemas for the scoring API."""

from pydantic import BaseModel, Field
from typing import Optional


class ScoreRequest(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier (e.g. LC_0000001)")


class AdverseAction(BaseModel):
    feature_name: str = Field(..., description="Human-readable feature name")
    shap_value: float = Field(..., description="SHAP contribution (positive = increases risk)")
    feature_value: float = Field(..., description="Applicant's value for this feature")
    direction: str = Field(..., description="How this feature affects the decision")


class ScoreResponse(BaseModel):
    applicant_id: str
    score: float = Field(..., description="Predicted probability of default (0-1)")
    decision: str = Field(..., description="approve / review / decline")
    model_version: str
    fico_score: Optional[int] = None
    grade: Optional[int] = None
    data_completeness: Optional[float] = None
    adverse_actions: list[AdverseAction] = Field(
        default_factory=list,
        description="Top reasons driving the credit decision (ECOA adverse action reasons)",
    )


class BatchScoreRequest(BaseModel):
    applicant_ids: list[str] = Field(..., description="List of applicant IDs to score")


class BatchScoreResponse(BaseModel):
    results: list[ScoreResponse]
    errors: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded_at: str
    n_features: int
