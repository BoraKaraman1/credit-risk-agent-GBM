// Performance Monitor Agent (Go port of agents/performance_monitor.py).
// Tracks model AUC, KS, and Gini on cohorts with known outcomes and
// compares against training metrics to detect degradation.
package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/gold"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/metrics"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
)

type performanceReport struct {
	Timestamp       string                        `json:"timestamp"`
	ModelVersion    string                        `json:"model_version"`
	OutcomesSource  string                        `json:"outcomes_source"`
	NObservations   int                           `json:"n_observations"`
	CurrentMetrics  map[string]float64            `json:"current_metrics"`
	TrainingMetrics map[string]map[string]float64 `json:"training_metrics"`
	AUCDrop         float64                       `json:"auc_drop"`
	AUCDropThresh   float64                       `json:"auc_drop_threshold"`
	RankOrderBreaks int                           `json:"rank_order_breaks"`
	DecileAnalysis  []metrics.Decile              `json:"decile_analysis"`
	Recommendation  string                        `json:"recommendation"`
}

func runPerformance(ctx context.Context) (*performanceReport, error) {
	m, err := model.Load(config.ChampionModelPath())
	if err != nil {
		return nil, err
	}

	// Try Supabase for real outcomes; fall back to the test set.
	outcomesSource := "test_set_proxy"
	var (
		yTrue    []int
		yScore   []float64
		database *db.DB
	)
	if config.DatabaseURL() != "" {
		if d, err := db.Connect(ctx, config.DatabaseURL()); err == nil {
			database = d
			defer d.Close()
			scores, labels, err := d.ScoredOutcomes(ctx, m.Version, 100000)
			if err != nil {
				slog.Warn("could not read scoring_log", "error", err)
			} else if len(scores) >= 100 {
				yScore, yTrue = scores, labels
				outcomesSource = fmt.Sprintf("scoring_log (%d outcomes)", len(scores))
				slog.Info("using outcomes from scoring_log", "n", len(scores))
			}
		} else {
			slog.Warn("could not connect to Supabase", "error", err)
		}
	}
	if yTrue == nil {
		cols := append(append([]string{}, m.Features...), "default")
		test, err := gold.ReadColumns(filepath.Join(config.GoldDir(), "features_test.parquet"), cols)
		if err != nil {
			return nil, err
		}
		rows, err := test.Rows(m.Features)
		if err != nil {
			return nil, err
		}
		yScore = m.PredictProbaBatch(rows)
		yTrue = make([]int, test.NumRows)
		for i, v := range test.Columns["default"] {
			yTrue[i] = int(v)
		}
		slog.Info("using test set as production proxy")
	}

	// AUC/KS are undefined without both outcome classes, and a NaN metric
	// would break JSON encoding downstream. Report an explicit status
	// instead of emitting NaN/Inf for an early or degenerate cohort.
	if !bothClasses(yTrue) {
		slog.Warn("insufficient outcomes for performance metrics", "n_observations", len(yTrue))
		return &performanceReport{
			Timestamp:       time.Now().UTC().Format(time.RFC3339),
			ModelVersion:    m.Version,
			OutcomesSource:  outcomesSource,
			NObservations:   len(yTrue),
			CurrentMetrics:  map[string]float64{},
			TrainingMetrics: m.Metrics,
			AUCDropThresh:   config.AUCDropThreshold,
			DecileAnalysis:  []metrics.Decile{},
			Recommendation: "INSUFFICIENT OUTCOMES. Both default and non-default outcomes " +
				"are required to compute AUC/KS; skipping this cohort.",
		}, nil
	}

	currentAUC := metrics.ROCAUC(yTrue, yScore)
	currentKS := metrics.KS(yTrue, yScore)
	currentMetrics := map[string]float64{
		"auc":  round(currentAUC, 4),
		"ks":   round(currentKS, 4),
		"gini": round(2*currentAUC-1, 4),
	}

	// Compare against the cleanest generalization estimate: the temporal
	// test holdout (test is never used for fitting or early stopping).
	trainAUC := 0.0
	if test, ok := m.Metrics["test"]; ok {
		trainAUC = test["auc"]
	} else if val, ok := m.Metrics["val"]; ok {
		trainAUC = val["auc"]
	}
	aucDrop := trainAUC - currentAUC

	decileStats, rankOrderBreaks := metrics.DecileAnalysis(yTrue, yScore)

	slog.Info(fmt.Sprintf("Current AUC=%.4f  Training AUC=%.4f  Drop=%.4f", currentAUC, trainAUC, aucDrop))
	slog.Info(fmt.Sprintf("Rank-ordering breaks: %d", rankOrderBreaks))

	rep := &performanceReport{
		Timestamp:       time.Now().UTC().Format(time.RFC3339),
		ModelVersion:    m.Version,
		OutcomesSource:  outcomesSource,
		NObservations:   len(yTrue),
		CurrentMetrics:  currentMetrics,
		TrainingMetrics: m.Metrics,
		AUCDrop:         round(aucDrop, 4),
		AUCDropThresh:   config.AUCDropThreshold,
		RankOrderBreaks: rankOrderBreaks,
		DecileAnalysis:  decileStats,
		Recommendation:  performanceRecommendation(aucDrop, rankOrderBreaks, currentMetrics),
	}

	if database != nil {
		err := database.InsertDriftLog(ctx, "auc", currentAUC, m.Version, map[string]any{
			"auc_drop":          round(aucDrop, 4),
			"ks":                currentMetrics["ks"],
			"rank_order_breaks": rankOrderBreaks,
		})
		if err != nil {
			slog.Warn("could not log to Supabase", "error", err)
		}
	}
	return rep, nil
}

// bothClasses reports whether the outcome labels contain at least one
// default and one non-default, the precondition for AUC and KS.
func bothClasses(yTrue []int) bool {
	var pos, neg bool
	for _, y := range yTrue {
		if y == 1 {
			pos = true
		} else {
			neg = true
		}
		if pos && neg {
			return true
		}
	}
	return false
}

func performanceRecommendation(aucDrop float64, rankBreaks int, current map[string]float64) string {
	if aucDrop > config.AUCDropThreshold {
		return fmt.Sprintf(
			"RETRAIN RECOMMENDED. AUC has dropped %.4f from training (threshold: %g). Current AUC: %g.",
			aucDrop, config.AUCDropThreshold, current["auc"])
	}
	if rankBreaks > 2 {
		return fmt.Sprintf(
			"INVESTIGATE. Rank ordering has %d breaks across deciles. "+
				"The model's discrimination is degrading even though AUC drop (%.4f) "+
				"hasn't crossed the threshold yet.", rankBreaks, aucDrop)
	}
	return fmt.Sprintf(
		"Model performance stable. AUC drop: %.4f (within %g threshold). Current KS: %g.",
		aucDrop, config.AUCDropThreshold, current["ks"])
}

func RunPerformance() {
	config.LoadEnv()
	ctx, cancel := withDeadline(performanceTimeout)
	defer cancel()
	rep, err := runPerformance(ctx)
	if err != nil {
		slog.Error("performance monitor failed", "error", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rep); err != nil {
		slog.Error("failed to encode performance report", "error", err)
		os.Exit(1)
	}
}
