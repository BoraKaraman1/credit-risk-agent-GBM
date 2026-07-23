-- Credit Risk Feature Store — Supabase PostgreSQL Schema
-- Idempotent bootstrap plus forward migrations. Docker Compose reruns this
-- through its migration gate before starting database-dependent services.

-- Feature store: immutable-contract snapshots per applicant and version
CREATE TABLE IF NOT EXISTS applicant_features (
    applicant_id       TEXT NOT NULL,
    feature_version    INT NOT NULL DEFAULT 1,
    computed_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    features           JSONB NOT NULL,
    data_completeness  NUMERIC(4,3),
    fico_score         SMALLINT,
    grade              SMALLINT,
    PRIMARY KEY (applicant_id, feature_version)
);

-- Forward migration from the original one-row-per-applicant store. The
-- existing row keeps its feature_version and becomes the first versioned
-- snapshot; later challenger syncs can coexist with champion snapshots.
DO $$
DECLARE
    current_pk TEXT;
    current_pk_columns INT;
BEGIN
    SELECT conname, cardinality(conkey)
      INTO current_pk, current_pk_columns
      FROM pg_constraint
     WHERE conrelid = 'applicant_features'::regclass
       AND contype = 'p';

    IF current_pk IS NOT NULL AND current_pk_columns = 1 THEN
        EXECUTE format(
            'ALTER TABLE applicant_features DROP CONSTRAINT %I',
            current_pk
        );
        ALTER TABLE applicant_features
            ADD PRIMARY KEY (applicant_id, feature_version);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_applicant_features_computed
    ON applicant_features (computed_at DESC);

CREATE INDEX IF NOT EXISTS ix_applicant_features_fico
    ON applicant_features (fico_score);

-- Decision log: every scoring decision (for monitoring + retraining).
-- This is the ECOA/Reg B compliance artifact: the API fails closed if a
-- row cannot be written. feature_snapshot carries applicant-level data,
-- so retention is bounded: `gbm prune` (weekly monitoring DAG) deletes
-- rows older than SCORING_LOG_RETENTION_DAYS (default 750 days, i.e.
-- Reg B's 25-month record-retention requirement with margin).
CREATE TABLE IF NOT EXISTS scoring_log (
    id                  BIGSERIAL PRIMARY KEY,
    applicant_id        TEXT NOT NULL,
    scored_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id          TEXT,
    model_version       TEXT NOT NULL,
    feature_version     INT NOT NULL DEFAULT 0,
    feature_snapshot    JSONB NOT NULL,
    score               NUMERIC(6,5),
    calibrated_pd       NUMERIC(6,5),
    scaled_score        INT,
    decision            TEXT,
    adverse_actions     JSONB NOT NULL DEFAULT '[]'::jsonb,
    actual_default      BOOLEAN,
    outcome_observed_at TIMESTAMPTZ
);

-- Idempotent forward migration for databases created by older versions.
ALTER TABLE scoring_log ADD COLUMN IF NOT EXISTS request_id TEXT;
ALTER TABLE scoring_log ADD COLUMN IF NOT EXISTS feature_version INT NOT NULL DEFAULT 0;
ALTER TABLE scoring_log ADD COLUMN IF NOT EXISTS calibrated_pd NUMERIC(6,5);
ALTER TABLE scoring_log ADD COLUMN IF NOT EXISTS scaled_score INT;
ALTER TABLE scoring_log ADD COLUMN IF NOT EXISTS adverse_actions JSONB NOT NULL DEFAULT '[]'::jsonb;

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
