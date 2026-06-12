// Drift Monitor Agent (Go port of agents/drift_monitor.py).
// Computes PSI on score distributions and CSI on individual features,
// prints a JSON report to stdout, and logs results to drift_log.
package main

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

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/gold"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/metrics"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/model"
	"path/filepath"
)

type report struct {
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

func run(ctx context.Context) (*report, error) {
	m, err := model.Load(config.ChampionModelPath())
	if err != nil {
		return nil, err
	}

	// Training scores (reference distribution)
	train, err := gold.ReadColumns(filepath.Join(config.GoldDir(), "features_train.parquet"), m.Features)
	if err != nil {
		return nil, err
	}
	trainRows, err := train.Rows(m.Features)
	if err != nil {
		return nil, err
	}
	trainScores := m.PredictProbaBatch(trainRows)

	// Production scores: Supabase scoring_log first, test set as fallback
	var (
		productionScores   []float64
		productionFeatures *gold.Frame
		database           *db.DB
	)
	if config.DatabaseURL() != "" {
		if d, err := db.Connect(ctx, config.DatabaseURL()); err == nil {
			database = d
			defer d.Close()
			if scores, err := d.RecentScores(ctx, 50000); err != nil {
				slog.Warn("could not read scoring_log", "error", err)
			} else if len(scores) > 0 {
				productionScores = scores
				slog.Info("using scores from scoring_log", "n", len(scores))
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
		productionFeatures = test
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
	if productionFeatures != nil {
		for _, col := range m.Features {
			prodCol, ok := productionFeatures.Columns[col]
			if !ok {
				continue
			}
			csi := metrics.CSI(train.Columns[col], prodCol, 10)
			csiResults[col] = round(csi, 4)
		}
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

	rep := &report{
		Timestamp:              time.Now().UTC().Format(time.RFC3339),
		ModelVersion:           m.Version,
		PSI:                    round(psi, 4),
		PSIStatus:              psiStatus,
		PSIThresholds:          map[string]float64{"warning": config.PSIWarning, "critical": config.PSICritical},
		TrainDistribution:      trainPct,
		ProductionDistribution: prodPct,
		CSIResults:             csiResults,
		DriftedFeatures:        drifted,
		Recommendation:         makeRecommendation(psi, psiStatus, csiResults),
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
	}
	return rep, nil
}

func makeRecommendation(psi float64, psiStatus string, csiResults map[string]float64) string {
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

func round(x float64, decimals int) float64 {
	p := math.Pow(10, float64(decimals))
	return math.Round(x*p) / p
}

func main() {
	config.LoadEnv()
	rep, err := run(context.Background())
	if err != nil {
		slog.Error("drift monitor failed", "error", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
}
