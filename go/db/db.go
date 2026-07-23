// Package db wraps Supabase PostgreSQL access for the Go services:
// audit logging (scoring_log, drift_log), the applicant feature store,
// and the training_distribution table.
package db

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// ErrNotFound is returned when an applicant is missing from the
// feature store.
var ErrNotFound = errors.New("not found")

type DB struct {
	Pool *pgxpool.Pool
}

func Connect(ctx context.Context, databaseURL string) (*DB, error) {
	if databaseURL == "" {
		return nil, errors.New("DATABASE_URL not configured")
	}
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}
	return &DB{Pool: pool}, nil
}

func (d *DB) Close() { d.Pool.Close() }

// Ping verifies the connection pool can reach the database (readiness).
func (d *DB) Ping(ctx context.Context) error { return d.Pool.Ping(ctx) }

// InsertDriftLog mirrors agents/db_logging.py.
func (d *DB) InsertDriftLog(ctx context.Context, metricName string, metricValue float64,
	modelVersion string, details any) error {
	detailsJSON, err := json.Marshal(details)
	if err != nil {
		return err
	}
	_, err = d.Pool.Exec(ctx, `
		INSERT INTO drift_log (metric_name, metric_value, model_version, details)
		VALUES ($1, $2, $3, $4::jsonb)`,
		metricName, metricValue, modelVersion, string(detailsJSON))
	return err
}

// PruneScoringLog deletes scoring_log rows older than the retention
// window. Retention exists because every row carries a full feature
// snapshot (applicant PII proxies) that must not accumulate forever.
func (d *DB) PruneScoringLog(ctx context.Context, retentionDays int) (int64, error) {
	tag, err := d.Pool.Exec(ctx, `
		DELETE FROM scoring_log
		WHERE scored_at < NOW() - make_interval(days => $1)`, retentionDays)
	if err != nil {
		return 0, err
	}
	return tag.RowsAffected(), nil
}

// HasDriftLogEntry reports whether drift_log already holds a row for
// this metric and model version (used for idempotent one-row-per-version
// publishes like the fairness summary sync).
func (d *DB) HasDriftLogEntry(ctx context.Context, metricName, modelVersion string) (bool, error) {
	var exists bool
	err := d.Pool.QueryRow(ctx, `
		SELECT EXISTS (
			SELECT 1 FROM drift_log
			WHERE metric_name = $1 AND model_version = $2)`,
		metricName, modelVersion).Scan(&exists)
	return exists, err
}

// AdverseAction is one principal reason disclosed with a non-approval.
// It lives in db so the complete decision envelope has one typed audit
// contract instead of being assembled as loosely related SQL arguments.
type AdverseAction struct {
	Code         int     `json:"code"`
	Reason       string  `json:"reason"`
	FeatureName  string  `json:"feature_name"`
	ShapValue    float64 `json:"shap_value"`
	FeatureValue float64 `json:"feature_value"`
	Direction    string  `json:"direction"`
}

// ScoringAudit is the exact decision envelope persisted before a score
// may be returned to the caller.
type ScoringAudit struct {
	RequestID       string
	ApplicantID     string
	ModelVersion    string
	FeatureVersion  int
	FeatureSnapshot map[string]*float64
	RawScore        float64
	CalibratedPD    *float64
	ScaledScore     *int
	Decision        string
	AdverseActions  []AdverseAction
}

// InsertScoringLog writes one complete scoring audit record.
func (d *DB) InsertScoringLog(ctx context.Context, audit ScoringAudit) error {
	featuresJSON, err := json.Marshal(audit.FeatureSnapshot)
	if err != nil {
		return err
	}
	actionsJSON, err := json.Marshal(audit.AdverseActions)
	if err != nil {
		return err
	}
	_, err = d.Pool.Exec(ctx, `
		INSERT INTO scoring_log
			(request_id, applicant_id, model_version, feature_version,
			 feature_snapshot, score, calibrated_pd, scaled_score,
			 decision, adverse_actions)
		VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10::jsonb)`,
		audit.RequestID, audit.ApplicantID, audit.ModelVersion,
		audit.FeatureVersion, string(featuresJSON), audit.RawScore,
		audit.CalibratedPD, audit.ScaledScore, audit.Decision,
		string(actionsJSON))
	return err
}

// ApplicantFeatures is one row of the applicant_features feature store.
type ApplicantFeatures struct {
	Features         map[string]*float64
	DataCompleteness *float64
	FicoScore        *int
	Grade            *int
	FeatureVersion   int
	ComputedAt       *time.Time
}

// FetchApplicantFeatures returns the stored feature vector for one
// applicant, or ErrNotFound.
func (d *DB) FetchApplicantFeatures(ctx context.Context, applicantID string) (*ApplicantFeatures, error) {
	var (
		featuresJSON []byte
		out          ApplicantFeatures
	)
	err := d.Pool.QueryRow(ctx, `
		SELECT features, data_completeness, fico_score, grade, feature_version, computed_at
		FROM applicant_features
		WHERE applicant_id = $1`, applicantID).
		Scan(&featuresJSON, &out.DataCompleteness, &out.FicoScore, &out.Grade,
			&out.FeatureVersion, &out.ComputedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(featuresJSON, &out.Features); err != nil {
		return nil, fmt.Errorf("decode features for %s: %w", applicantID, err)
	}
	return &out, nil
}

// RecentScores returns up to limit most recent production scores from
// scoring_log for one model version. Filtering by version keeps the drift
// monitor from mixing a prior model's scores into the current model's
// distribution after a model change.
func (d *DB) RecentScores(ctx context.Context, modelVersion string, limit int) ([]float64, error) {
	rows, err := d.Pool.Query(ctx,
		`SELECT score FROM scoring_log WHERE model_version = $1 ORDER BY scored_at DESC LIMIT $2`,
		modelVersion, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var scores []float64
	for rows.Next() {
		var s float64
		if err := rows.Scan(&s); err != nil {
			return nil, err
		}
		scores = append(scores, s)
	}
	return scores, rows.Err()
}

// RecentFeatureSnapshots returns the feature_snapshot maps from the most
// recent scoring_log rows for one model version, so the drift monitor can
// compute production CSI on the same window of real scored applicants.
func (d *DB) RecentFeatureSnapshots(ctx context.Context, modelVersion string, limit int) ([]map[string]*float64, error) {
	rows, err := d.Pool.Query(ctx,
		`SELECT feature_snapshot FROM scoring_log WHERE model_version = $1 ORDER BY scored_at DESC LIMIT $2`,
		modelVersion, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []map[string]*float64
	for rows.Next() {
		var raw []byte
		if err := rows.Scan(&raw); err != nil {
			return nil, err
		}
		snap := map[string]*float64{}
		if err := json.Unmarshal(raw, &snap); err != nil {
			return nil, err
		}
		out = append(out, snap)
	}
	return out, rows.Err()
}

// ScoredOutcomes returns scores with observed outcomes for one model
// version. Filtering by version keeps the performance monitor from scoring
// a new champion against a prior model's observed outcomes.
func (d *DB) ScoredOutcomes(ctx context.Context, modelVersion string, limit int) (yScore []float64, yTrue []int, err error) {
	rows, err := d.Pool.Query(ctx, `
		SELECT score, actual_default
		FROM scoring_log
		WHERE model_version = $1 AND actual_default IS NOT NULL
		ORDER BY outcome_observed_at DESC
		LIMIT $2`, modelVersion, limit)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var (
			s float64
			y bool
		)
		if err := rows.Scan(&s, &y); err != nil {
			return nil, nil, err
		}
		yScore = append(yScore, s)
		label := 0
		if y {
			label = 1
		}
		yTrue = append(yTrue, label)
	}
	return yScore, yTrue, rows.Err()
}

// PendingOutcomes returns distinct applicant IDs in scoring_log that
// have no observed outcome yet and were scored at least delayDays ago.
func (d *DB) PendingOutcomes(ctx context.Context, delayDays int) ([]string, error) {
	rows, err := d.Pool.Query(ctx, `
		SELECT DISTINCT applicant_id
		FROM scoring_log
		WHERE actual_default IS NULL
		  AND scored_at < NOW() - make_interval(days => $1)`, delayDays)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// BackfillOutcomes sets actual_default and outcome_observed_at for the
// given applicant IDs in a single statement. ids and labels must be the
// same length. Returns the number of scoring_log rows updated (an
// applicant scored more than once has all its rows updated).
func (d *DB) BackfillOutcomes(ctx context.Context, ids []string, labels []bool) (int64, error) {
	if len(ids) == 0 {
		return 0, nil
	}
	tag, err := d.Pool.Exec(ctx, `
		UPDATE scoring_log s
		SET actual_default = v.label, outcome_observed_at = NOW()
		FROM (SELECT unnest($1::text[]) AS applicant_id,
		             unnest($2::bool[]) AS label) v
		WHERE s.applicant_id = v.applicant_id
		  AND s.actual_default IS NULL`, ids, labels)
	if err != nil {
		return 0, err
	}
	return tag.RowsAffected(), nil
}

// FeatureRow is one applicant_features upsert row for the sync job.
type FeatureRow struct {
	ApplicantID      string
	FeatureVersion   int
	ComputedAt       string
	Features         map[string]*float64
	DataCompleteness float64
	FicoScore        *int
	Grade            *int
}

// UpsertApplicantFeatures bulk-upserts one batch via a single
// multi-row INSERT ... ON CONFLICT statement.
func (d *DB) UpsertApplicantFeatures(ctx context.Context, batch []FeatureRow) error {
	if len(batch) == 0 {
		return nil
	}
	sql := `
		INSERT INTO applicant_features
			(applicant_id, feature_version, computed_at, features, data_completeness, fico_score, grade)
		SELECT * FROM unnest(
			$1::text[], $2::int[], $3::timestamptz[], $4::jsonb[], $5::float8[], $6::int[], $7::int[])
		ON CONFLICT (applicant_id)
		DO UPDATE SET
			feature_version = EXCLUDED.feature_version,
			computed_at = EXCLUDED.computed_at,
			features = EXCLUDED.features,
			data_completeness = EXCLUDED.data_completeness,
			fico_score = EXCLUDED.fico_score,
			grade = EXCLUDED.grade`

	ids := make([]string, len(batch))
	versions := make([]int32, len(batch))
	computedAt := make([]string, len(batch))
	features := make([]string, len(batch))
	completeness := make([]float64, len(batch))
	fico := make([]*int32, len(batch))
	grade := make([]*int32, len(batch))
	for i, r := range batch {
		featJSON, err := json.Marshal(r.Features)
		if err != nil {
			return err
		}
		ids[i] = r.ApplicantID
		versions[i] = int32(r.FeatureVersion)
		computedAt[i] = r.ComputedAt
		features[i] = string(featJSON)
		completeness[i] = r.DataCompleteness
		if r.FicoScore != nil {
			v := int32(*r.FicoScore)
			fico[i] = &v
		}
		if r.Grade != nil {
			v := int32(*r.Grade)
			grade[i] = &v
		}
	}
	_, err := d.Pool.Exec(ctx, sql, ids, versions, computedAt, features, completeness, fico, grade)
	return err
}

// InsertTrainingDistribution stores the training score histogram used
// as the PSI reference distribution.
func (d *DB) InsertTrainingDistribution(ctx context.Context, modelVersion string,
	binEdges, binCounts []float64, totalCount int, metadata any) error {
	edgesJSON, err := json.Marshal(binEdges)
	if err != nil {
		return err
	}
	intCounts := make([]int64, len(binCounts))
	for i, c := range binCounts {
		intCounts[i] = int64(c)
	}
	countsJSON, err := json.Marshal(intCounts)
	if err != nil {
		return err
	}
	metaJSON, err := json.Marshal(metadata)
	if err != nil {
		return err
	}
	_, err = d.Pool.Exec(ctx, `
		INSERT INTO training_distribution
			(model_version, bin_edges, bin_counts, total_count, metadata)
		VALUES ($1, $2, $3, $4, $5)`,
		modelVersion, string(edgesJSON), string(countsJSON), totalCount, string(metaJSON))
	return err
}
