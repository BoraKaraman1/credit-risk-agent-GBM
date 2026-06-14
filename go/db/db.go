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

// InsertScoringLog writes one scoring audit record.
func (d *DB) InsertScoringLog(ctx context.Context, applicantID, modelVersion string,
	features map[string]*float64, score float64, decision string) error {
	featuresJSON, err := json.Marshal(features)
	if err != nil {
		return err
	}
	_, err = d.Pool.Exec(ctx, `
		INSERT INTO scoring_log
			(applicant_id, model_version, feature_snapshot, score, decision)
		VALUES ($1, $2, $3::jsonb, $4, $5)`,
		applicantID, modelVersion, string(featuresJSON), score, decision)
	return err
}

// ApplicantFeatures is one row of the applicant_features feature store.
type ApplicantFeatures struct {
	Features         map[string]*float64
	DataCompleteness *float64
	FicoScore        *int
	Grade            *int
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
		SELECT features, data_completeness, fico_score, grade, computed_at
		FROM applicant_features
		WHERE applicant_id = $1`, applicantID).
		Scan(&featuresJSON, &out.DataCompleteness, &out.FicoScore, &out.Grade, &out.ComputedAt)
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
// scoring_log.
func (d *DB) RecentScores(ctx context.Context, limit int) ([]float64, error) {
	rows, err := d.Pool.Query(ctx,
		`SELECT score FROM scoring_log ORDER BY scored_at DESC LIMIT $1`, limit)
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

// ScoredOutcomes returns scores with observed outcomes for
// performance monitoring.
func (d *DB) ScoredOutcomes(ctx context.Context, limit int) (yScore []float64, yTrue []int, err error) {
	rows, err := d.Pool.Query(ctx, `
		SELECT score, actual_default
		FROM scoring_log
		WHERE actual_default IS NOT NULL
		ORDER BY outcome_observed_at DESC
		LIMIT $1`, limit)
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
