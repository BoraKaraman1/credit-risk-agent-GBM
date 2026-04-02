"""Pydantic request/response schemas for the scoring API."""

from pydantic import BaseModel, Field
from typing import Optional


class ScoreRequest(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier (e.g. LC_0000001)")


class ScoreResponse(BaseModel):
    applicant_id: str
    score: float = Field(..., description="Predicted probability of default (0-1)")
    decision: str = Field(..., description="approve / review / decline")
    model_version: str
    fico_score: Optional[int] = None
    grade: Optional[int] = None
    data_completeness: Optional[float] = None


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
