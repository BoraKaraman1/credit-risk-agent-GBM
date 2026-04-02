-- Credit Risk Feature Store — Supabase PostgreSQL Schema
-- Run this in the Supabase SQL Editor to create all tables.

-- Feature store: latest features per applicant (for real-time scoring)
CREATE TABLE IF NOT EXISTS applicant_features (
    applicant_id       TEXT PRIMARY KEY,
    feature_version    INT NOT NULL DEFAULT 1,
    computed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    features           JSONB NOT NULL,
    data_completeness  NUMERIC(4,3),
    fico_score         SMALLINT,
    grade              SMALLINT
);

CREATE INDEX IF NOT EXISTS ix_applicant_features_computed
    ON applicant_features (computed_at DESC);

CREATE INDEX IF NOT EXISTS ix_applicant_features_fico
    ON applicant_features (fico_score);

-- Decision log: every scoring decision (for monitoring + retraining)
CREATE TABLE IF NOT EXISTS scoring_log (
    id                  BIGSERIAL PRIMARY KEY,
    applicant_id        TEXT NOT NULL,
    scored_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version       TEXT NOT NULL,
    feature_snapshot    JSONB NOT NULL,
    score               NUMERIC(6,5),
    decision            TEXT,
    actual_default      BOOLEAN,
    outcome_observed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_scoring_log_scored_at
    ON scoring_log (scored_at DESC);

CREATE INDEX IF NOT EXISTS ix_scoring_log_model_version
    ON scoring_log (model_version);

-- Drift monitoring log
CREATE TABLE IF NOT EXISTS drift_log (
    id              BIGSERIAL PRIMARY KEY,
    measured_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_name     TEXT NOT NULL,
    metric_value    NUMERIC(8,5),
    model_version   TEXT NOT NULL,
    details         JSONB
);

CREATE INDEX IF NOT EXISTS ix_drift_log_measured_at
    ON drift_log (measured_at DESC);

-- Training score distribution (used as reference for PSI computation)
CREATE TABLE IF NOT EXISTS training_distribution (
    id              BIGSERIAL PRIMARY KEY,
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    bin_edges       JSONB NOT NULL,       -- score bin boundaries
    bin_counts      JSONB NOT NULL,       -- count per bin (training set)
    total_count     INT NOT NULL,
    metadata        JSONB                 -- AUC, KS etc. at training time
);
