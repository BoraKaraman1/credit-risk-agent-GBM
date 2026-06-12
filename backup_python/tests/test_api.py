"""
Tests for the FastAPI scoring API.
Uses TestClient with mocked Supabase — no database required.
"""

import pytest
import json
import numpy as np
import shap
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

from api.models import ScoreResponse, HealthResponse, AdverseAction


# --- Fixtures ---

@pytest.fixture
def dummy_model():
    """Train a small model for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = rng.choice([0, 1], 200, p=[0.8, 0.2])
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def dummy_meta():
    return {
        "version": "v1.0",
        "features": ["f0", "f1", "f2", "f3", "f4"],
        "n_features": 5,
        "metrics": {"test": {"auc": 0.72, "ks": 0.32, "gini": 0.44}},
    }


@pytest.fixture
def client(dummy_model, dummy_meta):
    """Create a test client with mocked model and DB."""
    import api.scoring_service as svc

    svc._model = dummy_model
    svc._model_meta = dummy_meta
    svc._model_loaded_at = datetime.now(timezone.utc).isoformat()
    svc._explainer = shap.TreeExplainer(dummy_model)

    return TestClient(svc.app)


@pytest.fixture
def mock_features():
    """Feature dict matching the dummy model's 5 features."""
    return {
        "f0": 0.5, "f1": -0.3, "f2": 1.2, "f3": -0.1, "f4": 0.8,
    }


# --- Health endpoint ---

class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_version"] == "v1.0"
        assert data["n_features"] == 5

    def test_health_no_model(self):
        import api.scoring_service as svc
        svc._model = None
        client = TestClient(svc.app)
        resp = client.get("/health")
        assert resp.status_code == 503


# --- Score endpoint ---

class TestScoreEndpoint:
    def _mock_fetch(self, mock_features, stale=False):
        """Create a patched fetch_features that returns mock data."""
        computed_at = datetime.now(timezone.utc)
        if stale:
            computed_at -= timedelta(hours=48)

        def fetch(applicant_id):
            return {
                "features": mock_features,
                "data_completeness": 1.0,
                "fico_score": 720,
                "grade": 2,
            }

        return fetch

    @patch("api.scoring_service.get_engine")
    @patch("api.scoring_service.fetch_features")
    def test_score_returns_valid_response(self, mock_fetch_fn, mock_engine,
                                          client, mock_features):
        mock_fetch_fn.side_effect = self._mock_fetch(mock_features)
        mock_engine.return_value = MagicMock()

        resp = client.post("/score", json={"applicant_id": "TEST_001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["applicant_id"] == "TEST_001"
        assert 0 <= data["score"] <= 1
        assert data["decision"] in ("approve", "review", "decline")
        assert data["model_version"] == "v1.0"
        assert "adverse_actions" in data

    @patch("api.scoring_service.get_engine")
    @patch("api.scoring_service.fetch_features")
    def test_adverse_actions_present_for_decline_review(self, mock_fetch_fn, mock_engine,
                                                         client, dummy_model, dummy_meta):
        """Adverse actions should be present when decision is decline or review."""
        import api.scoring_service as svc
        mock_engine.return_value = MagicMock()

        # Try multiple random feature sets to find a decline/review
        found_non_approve = False
        for seed in range(50):
            rng = np.random.RandomState(seed)
            features = {f"f{i}": float(rng.randn()) for i in range(5)}
            mock_fetch_fn.side_effect = self._mock_fetch(features)
            resp = client.post("/score", json={"applicant_id": f"T_{seed}"})
            data = resp.json()

            if data["decision"] in ("decline", "review"):
                found_non_approve = True
                assert len(data["adverse_actions"]) > 0
                for action in data["adverse_actions"]:
                    assert action["shap_value"] > 0
                    assert action["direction"] == "increases risk"
                break

        # If all were approve, verify adverse_actions is empty for approve
        if not found_non_approve:
            assert data["adverse_actions"] == []

    @patch("api.scoring_service.get_engine")
    @patch("api.scoring_service.fetch_features")
    def test_adverse_actions_empty_for_approve(self, mock_fetch_fn, mock_engine,
                                                client, dummy_model, dummy_meta):
        """Adverse actions should be empty when decision is approve."""
        import api.scoring_service as svc
        mock_engine.return_value = MagicMock()

        for seed in range(50):
            rng = np.random.RandomState(seed)
            features = {f"f{i}": float(rng.randn()) for i in range(5)}
            mock_fetch_fn.side_effect = self._mock_fetch(features)
            resp = client.post("/score", json={"applicant_id": f"T_{seed}"})
            data = resp.json()

            if data["decision"] == "approve":
                assert data["adverse_actions"] == []
                break

    @patch("api.scoring_service.get_engine")
    @patch("api.scoring_service.fetch_features")
    def test_decision_thresholds(self, mock_fetch_fn, mock_engine,
                                  client, dummy_model, dummy_meta):
        """Test that decision rules are applied correctly."""
        import api.scoring_service as svc
        mock_engine.return_value = MagicMock()

        # Generate features that produce a known score range
        for _ in range(10):
            rng = np.random.RandomState(_)
            features = {f"f{i}": float(rng.randn()) for i in range(5)}

            mock_fetch_fn.side_effect = self._mock_fetch(features)
            resp = client.post("/score", json={"applicant_id": f"T_{_}"})
            data = resp.json()

            score = data["score"]
            decision = data["decision"]

            if score < svc.APPROVE_THRESHOLD:
                assert decision == "approve"
            elif score < svc.REVIEW_THRESHOLD:
                assert decision == "review"
            else:
                assert decision == "decline"

    def test_missing_applicant_id(self, client):
        resp = client.post("/score", json={})
        assert resp.status_code == 422


# --- Batch score endpoint ---

class TestBatchScoreEndpoint:
    @patch("api.scoring_service.get_engine")
    @patch("api.scoring_service.fetch_features")
    def test_batch_returns_results(self, mock_fetch_fn, mock_engine,
                                    client, mock_features):
        mock_fetch_fn.return_value = {
            "features": mock_features,
            "data_completeness": 1.0,
            "fico_score": 720,
            "grade": 2,
        }
        mock_engine.return_value = MagicMock()

        resp = client.post("/score/batch", json={
            "applicant_ids": ["A1", "A2", "A3"]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3

    def test_empty_batch(self, client):
        resp = client.post("/score/batch", json={"applicant_ids": []})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 0


# --- Reload endpoint ---

class TestReloadEndpoint:
    @patch("api.scoring_service.load_model")
    def test_reload(self, mock_load, client):
        resp = client.post("/reload")
        assert resp.status_code == 200
        mock_load.assert_called_once()


# --- Pydantic model tests ---

class TestPydanticModels:
    def test_score_response_schema(self):
        resp = ScoreResponse(
            applicant_id="LC_001",
            score=0.123,
            decision="approve",
            model_version="v1.0",
        )
        assert resp.score == 0.123
        assert resp.fico_score is None  # optional
        assert resp.adverse_actions == []  # default empty

    def test_score_response_with_adverse_actions(self):
        resp = ScoreResponse(
            applicant_id="LC_002",
            score=0.45,
            decision="decline",
            model_version="v1.0",
            adverse_actions=[
                AdverseAction(
                    feature_name="Debt-to-Income Ratio",
                    shap_value=0.05,
                    feature_value=35.0,
                    direction="increases risk",
                ),
            ],
        )
        assert len(resp.adverse_actions) == 1
        assert resp.adverse_actions[0].feature_name == "Debt-to-Income Ratio"

    def test_health_response_schema(self):
        resp = HealthResponse(
            status="ok",
            model_version="v2.0",
            model_loaded_at="2025-01-01T00:00:00Z",
            n_features=30,
        )
        assert resp.n_features == 30
