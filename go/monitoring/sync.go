// Supabase Sync (Go port of pipeline/sync_to_supabase.py).
// Bulk-upserts Gold test-set features into the versioned
// applicant_features store and saves the selected model's training
// score distribution.
package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/gold"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/metrics"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
)

const batchSize = 10000

type featureMetadata struct {
	FeatureColumns []string `json:"feature_columns"`
	FeatureVersion int      `json:"feature_version"`
}

func loadFeatureMetadata() (*featureMetadata, error) {
	metaBytes, err := os.ReadFile(filepath.Join(config.GoldDir(), "feature_metadata.json"))
	if err != nil {
		return nil, err
	}
	var meta featureMetadata
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return nil, err
	}
	if meta.FeatureVersion == 0 {
		meta.FeatureVersion = 1 // legacy metadata without the field
	}
	return &meta, nil
}

func syncModelPath(slot string) (string, error) {
	switch slot {
	case "champion":
		return config.ChampionModelPath(), nil
	case "challenger":
		return config.ChallengerModelPath(), nil
	default:
		return "", fmt.Errorf("invalid sync model %q; expected champion or challenger", slot)
	}
}

func validateSyncContract(m *model.Model, meta *featureMetadata) error {
	modelFeatureVersion := m.FeatureVersion
	if modelFeatureVersion == 0 {
		modelFeatureVersion = 1
	}
	if modelFeatureVersion != meta.FeatureVersion {
		return fmt.Errorf(
			"Gold feature version %d does not match model %s feature version %d",
			meta.FeatureVersion, m.Version, modelFeatureVersion,
		)
	}
	if !sameFeatures(meta.FeatureColumns, m.Features) {
		return fmt.Errorf("Gold feature columns do not match model %s", m.Version)
	}
	return nil
}

func loadSyncTarget(slot string) (*model.Model, *featureMetadata, error) {
	path, err := syncModelPath(slot)
	if err != nil {
		return nil, nil, err
	}
	m, err := model.Load(path)
	if err != nil {
		return nil, nil, fmt.Errorf("load %s model: %w", slot, err)
	}
	meta, err := loadFeatureMetadata()
	if err != nil {
		return nil, nil, err
	}
	if err := validateSyncContract(m, meta); err != nil {
		return nil, nil, err
	}
	return m, meta, nil
}

func syncFeatures(
	ctx context.Context,
	d *db.DB,
	m *model.Model,
	meta *featureMetadata,
) error {
	featureCols := meta.FeatureColumns
	featureVersion := meta.FeatureVersion

	// Test set only — simulates active applicants awaiting scoring
	// (training data is historical, not needed in the feature store)
	cols := append(append([]string{}, featureCols...), "fico_score", "grade_numeric", "id")
	frame, err := gold.ReadColumns(filepath.Join(config.GoldDir(), "features_test.parquet"), cols)
	if err != nil {
		return err
	}
	total := frame.NumRows
	slog.Info("preparing feature rows for bulk upsert",
		"rows", total,
		"model_version", m.Version,
		"feature_version", featureVersion,
	)

	now := time.Now().UTC().Format(time.RFC3339Nano)
	synced := 0
	for start := 0; start < total; start += batchSize {
		end := min(start+batchSize, total)
		batch := make([]db.FeatureRow, 0, end-start)
		for i := start; i < end; i++ {
			features := make(map[string]*float64, len(featureCols))
			nonNull := 0
			for _, c := range featureCols {
				v := frame.Columns[c][i]
				if math.IsNaN(v) {
					features[c] = nil
				} else {
					val := v
					features[c] = &val
					nonNull++
				}
			}
			completeness := round(float64(nonNull)/float64(len(featureCols)), 3)

			// Applicant identity is the stable LendingClub loan id
			// carried through the medallion layers, never the row
			// position (which changes across regenerations).
			loanID := frame.Columns["id"][i]
			if math.IsNaN(loanID) {
				return fmt.Errorf("row %d has no loan id", i)
			}
			row := db.FeatureRow{
				ApplicantID:      fmt.Sprintf("LC_%d", int64(loanID)),
				FeatureVersion:   featureVersion,
				ComputedAt:       now,
				Features:         features,
				DataCompleteness: completeness,
			}
			if v := frame.Columns["fico_score"][i]; !math.IsNaN(v) {
				fico := int(v)
				row.FicoScore = &fico
			}
			if v := frame.Columns["grade_numeric"][i]; !math.IsNaN(v) {
				grade := int(v)
				row.Grade = &grade
			}
			batch = append(batch, row)
		}
		if err := d.UpsertApplicantFeatures(ctx, batch); err != nil {
			return fmt.Errorf("upsert batch at %d: %w", start, err)
		}
		synced = end
		slog.Info(fmt.Sprintf("%d/%d (%.1f%%)", synced, total, float64(synced)/float64(total)*100))
	}
	slog.Info(fmt.Sprintf("Feature sync complete. %d rows upserted.", synced))
	return nil
}

func syncTrainingDistribution(ctx context.Context, d *db.DB, m *model.Model) error {
	train, err := gold.ReadColumns(filepath.Join(config.GoldDir(), "features_train.parquet"), m.Features)
	if err != nil {
		return err
	}
	rows, err := train.Rows(m.Features)
	if err != nil {
		return err
	}
	scores := m.PredictProbaBatch(rows)

	binEdges := metrics.Linspace(0, 1, 11)
	counts := metrics.Histogram(scores, binEdges)

	err = d.InsertTrainingDistribution(ctx, m.Version, binEdges, counts, len(scores), m.Metrics)
	if err != nil {
		return err
	}
	slog.Info("Training distribution saved", "model_version", m.Version)
	return nil
}

func round(x float64, decimals int) float64 {
	p := math.Pow(10, float64(decimals))
	return math.Round(x*p) / p
}

func RunSync(slot string) {
	config.LoadEnv()
	ctx, cancel := withDeadline(syncTimeout)
	defer cancel()

	m, meta, err := loadSyncTarget(slot)
	if err != nil {
		slog.Error("sync preflight failed", "model", slot, "error", err)
		os.Exit(1)
	}

	d, err := db.Connect(ctx, config.DatabaseURL())
	if err != nil {
		slog.Error("DATABASE_URL not set or unreachable", "error", err)
		os.Exit(1)
	}
	defer d.Close()

	if err := syncFeatures(ctx, d, m, meta); err != nil {
		slog.Error("feature sync failed", "error", err)
		os.Exit(1)
	}
	if err := syncTrainingDistribution(ctx, d, m); err != nil {
		slog.Error("training distribution sync failed", "error", err)
		os.Exit(1)
	}
	slog.Info("Supabase sync done.")
}
