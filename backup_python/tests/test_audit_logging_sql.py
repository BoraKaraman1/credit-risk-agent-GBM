"""
Tests for audit-log JSONB inserts.

The Postgres round-trip test is skipped unless TEST_DATABASE_URL points to a
real PostgreSQL database. It uses temporary tables and does not touch permanent
application tables.
"""

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql

from agents.db_logging import DRIFT_LOG_INSERT_SQL, insert_drift_log
from api.scoring_service import SCORING_LOG_INSERT_SQL, insert_scoring_log


def test_scoring_log_jsonb_statement_binds_feature_snapshot():
    compiled = SCORING_LOG_INSERT_SQL.compile(dialect=postgresql.dialect())

    assert "feature_snapshot" in compiled.params
    assert "CAST" in str(compiled)
    assert "::jsonb" not in str(compiled)


def test_drift_log_jsonb_statement_binds_details():
    compiled = DRIFT_LOG_INSERT_SQL.compile(dialect=postgresql.dialect())

    assert "details" in compiled.params
    assert "CAST" in str(compiled)
    assert "::jsonb" not in str(compiled)


def test_postgres_jsonb_audit_logging_round_trip():
    db_url = os.environ.get("TEST_DATABASE_URL")
    if not db_url:
        pytest.skip("Set TEST_DATABASE_URL to run the Postgres JSONB integration test")

    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TEMP TABLE scoring_log (
                id BIGSERIAL PRIMARY KEY,
                applicant_id TEXT NOT NULL,
                scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                model_version TEXT NOT NULL,
                feature_snapshot JSONB NOT NULL,
                score NUMERIC(6,5),
                decision TEXT,
                actual_default BOOLEAN,
                outcome_observed_at TIMESTAMPTZ
            ) ON COMMIT DROP
        """))
        conn.execute(text("""
            CREATE TEMP TABLE drift_log (
                id BIGSERIAL PRIMARY KEY,
                measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metric_name TEXT NOT NULL,
                metric_value NUMERIC(8,5),
                model_version TEXT NOT NULL,
                details JSONB
            ) ON COMMIT DROP
        """))

        insert_scoring_log(
            conn,
            applicant_id="TEST_001",
            model_version="v-test",
            features={"fico_score": 720, "nested": {"ok": True}},
            score=0.12345,
            decision="approve",
        )
        insert_drift_log(
            conn,
            metric_name="psi",
            metric_value=0.01234,
            model_version="v-test",
            details={"csi": {"fico_score": 0.01}, "drifted_features": []},
        )

        fico_score = conn.execute(text("""
            SELECT feature_snapshot->>'fico_score'
            FROM scoring_log
            WHERE applicant_id = 'TEST_001'
        """)).scalar_one()
        csi_value = conn.execute(text("""
            SELECT details->'csi'->>'fico_score'
            FROM drift_log
            WHERE metric_name = 'psi'
        """)).scalar_one()

        assert fico_score == "720"
        assert csi_value == "0.01"
