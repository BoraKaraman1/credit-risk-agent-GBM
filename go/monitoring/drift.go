// Drift Monitor Agent (Go port of agents/drift_monitor.py).
// Computes PSI on score distributions and CSI on individual features,
// prints a JSON report to stdout, and logs results to drift_log.
package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/gold"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/metrics"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
	"path/filepath"
)

type driftReport struct {
	Timestamp              string             `json:"timestamp"`
	ModelVersion           string             `json:"model_version"`
	PSI                    float64            `json:"psi"`
	PSIStatus              string             `json:"psi_status"`
	PSIThresholds          map[string]float64 `json:"psi_thresholds"`
	TrainDistribution      []float64          `json:"train_distribution"`
	ProductionDistribution []float64          `json:"production_distribution"`
	CSIResults             map[string]float64 `json:"csi_results"`
	DriftedFeatures        map[string]float64 `json:"drifted_features"`
	Recommendation         string             `json:"recommendation"`
}

func runDrift(ctx context.Context) (*driftReport, error) {
	m, err := model.Load(config.ChampionModelPath())
	if err != nil {
		return nil, err
	}

	// Training scores (reference distribution). These are recomputed from
	// the local Gold parquet by scoring the train set with the CURRENT
	// champion, so the PSI reference always matches the model under test.
	// (The training_distribution table written by `gbm sync` is a snapshot
	// for external dashboards; it is intentionally not used here because a
	// prior model's stored histogram would not match the current champion.)
	train, err := gold.ReadColumns(filepath.Join(config.GoldDir(), "features_train.parquet"), m.Features)
	if err != nil {
		return nil, err
	}
	trainRows, err := train.Rows(m.Features)
	if err != nil {
		return nil, err
	}
	trainScores := m.PredictProbaBatch(trainRows)

	// Production scores and per-feature columns: Supabase scoring_log first
	// (real scored applicants for the current model version), test set as
	// fallback. prodFeatureCols drives CSI in both modes.
	var (
		productionScores []float64
		prodFeatureCols  map[string][]float64
		database         *db.DB
	)
	if config.DatabaseURL() != "" {
		if d, err := db.Connect(ctx, config.DatabaseURL()); err == nil {
			database = d
			defer d.Close()
			if scores, err := d.RecentScores(ctx, m.Version, 50000); err != nil {
				slog.Warn("could not read scoring_log", "error", err)
			} else if len(scores) > 0 {
				productionScores = scores
				slog.Info("using scores from scoring_log", "n", len(scores))
				// Compute CSI on the same scored applicants' feature
				// snapshots so feature drift is visible in real production
				// mode, not only when falling back to the test set.
				if snaps, err := d.RecentFeatureSnapshots(ctx, m.Version, 50000); err != nil {
					slog.Warn("could not read feature snapshots for CSI", "error", err)
				} else if len(snaps) > 0 {
					prodFeatureCols = featureColumnsFromSnapshots(snaps, m.Features)
				}
			}
		} else {
			slog.Warn("could not connect to Supabase", "error", err)
		}
	}
	if productionScores == nil {
		test, err := gold.ReadColumns(filepath.Join(config.GoldDir(), "features_test.parquet"), m.Features)
		if err != nil {
			return nil, err
		}
		testRows, err := test.Rows(m.Features)
		if err != nil {
			return nil, err
		}
		productionScores = m.PredictProbaBatch(testRows)
		prodFeatureCols = test.Columns
		slog.Info("using test set as production proxy")
	}

	// --- PSI on score distribution ---
	psi, trainPct, prodPct := metrics.PSI(trainScores, productionScores, 10)
	psiStatus := "OK"
	switch {
	case psi > config.PSICritical:
		psiStatus = "CRITICAL"
	case psi > config.PSIWarning:
		psiStatus = "WARNING"
	}
	slog.Info(fmt.Sprintf("Score PSI = %.4f (%s)", psi, psiStatus))

	// --- CSI on individual features ---
	csiResults := map[string]float64{}
	for _, col := range m.Features {
		prodCol, ok := prodFeatureCols[col]
		if !ok {
			continue
		}
		csi := metrics.CSI(train.Columns[col], prodCol, 10)
		csiResults[col] = round(csi, 4)
	}
	drifted := map[string]float64{}
	for k, v := range csiResults {
		if v > config.CSIThreshold {
			drifted[k] = v
		}
	}
	if len(drifted) > 0 {
		slog.Info("features above CSI threshold", "drifted", drifted)
	} else {
		slog.Info(fmt.Sprintf("no individual features above CSI threshold (%.2f)", config.CSIThreshold))
	}

	rep := &driftReport{
		Timestamp:              time.Now().UTC().Format(time.RFC3339),
		ModelVersion:           m.Version,
		PSI:                    round(psi, 4),
		PSIStatus:              psiStatus,
		PSIThresholds:          map[string]float64{"warning": config.PSIWarning, "critical": config.PSICritical},
		TrainDistribution:      trainPct,
		ProductionDistribution: prodPct,
		CSIResults:             csiResults,
		DriftedFeatures:        drifted,
		Recommendation:         driftRecommendation(psi, psiStatus, csiResults),
	}

	if database != nil {
		driftedNames := make([]string, 0, len(drifted))
		for k := range drifted {
			driftedNames = append(driftedNames, k)
		}
		sort.Strings(driftedNames)
		err := database.InsertDriftLog(ctx, "psi", psi, m.Version, map[string]any{
			"csi":              csiResults,
			"drifted_features": driftedNames,
		})
		if err != nil {
			slog.Warn("could not log to Supabase", "error", err)
		} else {
			slog.Info("results logged to drift_log table")
		}
		syncFairnessLog(ctx, database)
	}
	return rep, nil
}

// featureColumnsFromSnapshots pivots scoring_log feature snapshots into
// per-feature columns for CSI. Missing or null values become NaN, which
// metrics.CSI drops — the same treatment as the stored snapshots.
func featureColumnsFromSnapshots(snaps []map[string]*float64, features []string) map[string][]float64 {
	cols := make(map[string][]float64, len(features))
	for _, col := range features {
		cols[col] = make([]float64, 0, len(snaps))
	}
	for _, snap := range snaps {
		for _, col := range features {
			if v, ok := snap[col]; ok && v != nil {
				cols[col] = append(cols[col], *v)
			} else {
				cols[col] = append(cols[col], math.NaN())
			}
		}
	}
	return cols
}

func driftRecommendation(psi float64, psiStatus string, csiResults map[string]float64) string {
	switch psiStatus {
	case "CRITICAL":
		return fmt.Sprintf(
			"RETRAIN RECOMMENDED. Score PSI (%.4f) exceeds critical threshold. "+
				"The population applying for loans has shifted significantly from training data.", psi)
	case "WARNING":
		var drifted []string
		for k, v := range csiResults {
			if v > config.CSIThreshold {
				drifted = append(drifted, k)
			}
		}
		sort.Strings(drifted)
		if len(drifted) > 0 {
			return fmt.Sprintf(
				"MONITOR CLOSELY. Score PSI (%.4f) shows moderate drift. "+
					"Features driving drift: %s. If this persists for 2+ weeks, consider retraining.",
				psi, strings.Join(drifted, ", "))
		}
		return fmt.Sprintf(
			"MONITOR CLOSELY. Score PSI (%.4f) shows moderate drift "+
				"but no individual features exceed CSI threshold.", psi)
	}
	return fmt.Sprintf("No action needed. Score PSI (%.4f) is within normal range.", psi)
}

func RunDrift() {
	config.LoadEnv()
	rep, err := runDrift(context.Background())
	if err != nil {
		slog.Error("drift monitor failed", "error", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rep); err != nil {
		slog.Error("failed to encode drift report", "error", err)
		os.Exit(1)
	}
}
