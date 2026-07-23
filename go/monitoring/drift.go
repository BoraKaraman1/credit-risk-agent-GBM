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
	ScoresSource           string             `json:"scores_source"`
	NObservations          int                `json:"n_observations"`
	MinimumObservations    int                `json:"minimum_observations"`
	MonitoringEligible     bool               `json:"monitoring_eligible"`
	PSI                    float64            `json:"psi"`
	PSIStatus              string             `json:"psi_status"`
	PSIThresholds          map[string]float64 `json:"psi_thresholds"`
	TrainDistribution      []float64          `json:"train_distribution"`
	ProductionDistribution []float64          `json:"production_distribution"`
	CSIResults             map[string]float64 `json:"csi_results"`
	DriftedFeatures        map[string]float64 `json:"drifted_features"`
	Recommendation         string             `json:"recommendation"`
	// Machine-readable verdict consumed by the monitoring DAG's branch
	// (dags/credit_risk_monitoring.py); the prose above is for humans.
	NeedsRetrain   bool     `json:"needs_retrain"`
	RetrainReasons []string `json:"retrain_reasons"`
}

// driftObservationState prevents either a tiny real sample or the local
// test-set proxy from becoming an automated governance signal.
func driftObservationState(realProduction bool, n int, diagnosticStatus string) (string, bool) {
	if !realProduction {
		return diagnosticStatus, false
	}
	if n < config.MinDriftScores {
		return "INSUFFICIENT_DATA", false
	}
	return diagnosticStatus, true
}

// driftRetrainSignal is the machine half of the drift verdict: retrain
// on an eligible CRITICAL score-distribution shift.
func driftRetrainSignal(monitoringEligible bool, psiStatus string, psi float64) (bool, []string) {
	if monitoringEligible && psiStatus == "CRITICAL" {
		return true, []string{fmt.Sprintf("psi_critical (%.4f)", psi)}
	}
	return false, []string{}
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
	// fallback when no database is configured or the log is genuinely
	// empty. A CONFIGURED database that cannot be reached or read fails
	// the run instead: a silent test-set fallback would report "no
	// drift" for exactly as long as the monitoring data source is down.
	scoresSource := "test_set_proxy"
	var (
		productionScores []float64
		prodFeatureCols  map[string][]float64
		database         *db.DB
		realProduction   bool
	)
	if config.DatabaseURL() != "" {
		d, err := db.Connect(ctx, config.DatabaseURL())
		if err != nil {
			return nil, fmt.Errorf("DATABASE_URL is configured but unreachable: %w", err)
		}
		database = d
		defer d.Close()
		scores, err := d.RecentScores(ctx, m.Version, 50000)
		if err != nil {
			return nil, fmt.Errorf("read scoring_log: %w", err)
		}
		if len(scores) > 0 {
			productionScores = scores
			realProduction = true
			scoresSource = fmt.Sprintf("scoring_log (%d scores)", len(scores))
			slog.Info("using scores from scoring_log", "n", len(scores))
			// Compute CSI on the same scored applicants' feature
			// snapshots so feature drift is visible in real production
			// mode, not only when falling back to the test set.
			snaps, err := d.RecentFeatureSnapshots(ctx, m.Version, 50000)
			if err != nil {
				return nil, fmt.Errorf("read feature snapshots for CSI: %w", err)
			}
			if len(snaps) > 0 {
				prodFeatureCols = featureColumnsFromSnapshots(snaps, m.Features)
			}
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
	diagnosticStatus := "OK"
	switch {
	case psi > config.PSICritical:
		diagnosticStatus = "CRITICAL"
	case psi > config.PSIWarning:
		diagnosticStatus = "WARNING"
	}
	psiStatus, monitoringEligible := driftObservationState(
		realProduction, len(productionScores), diagnosticStatus,
	)
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

	needsRetrain, retrainReasons := driftRetrainSignal(monitoringEligible, psiStatus, psi)
	rep := &driftReport{
		Timestamp:              time.Now().UTC().Format(time.RFC3339),
		ModelVersion:           m.Version,
		ScoresSource:           scoresSource,
		NObservations:          len(productionScores),
		MinimumObservations:    config.MinDriftScores,
		MonitoringEligible:     monitoringEligible,
		PSI:                    round(psi, 4),
		PSIStatus:              psiStatus,
		PSIThresholds:          map[string]float64{"warning": config.PSIWarning, "critical": config.PSICritical},
		TrainDistribution:      trainPct,
		ProductionDistribution: prodPct,
		CSIResults:             csiResults,
		DriftedFeatures:        drifted,
		Recommendation: driftRecommendation(
			psi, psiStatus, csiResults, realProduction, len(productionScores),
		),
		NeedsRetrain:   needsRetrain,
		RetrainReasons: retrainReasons,
	}

	if database != nil {
		driftedNames := make([]string, 0, len(drifted))
		for k := range drifted {
			driftedNames = append(driftedNames, k)
		}
		sort.Strings(driftedNames)
		err := database.InsertDriftLog(ctx, "psi", psi, m.Version, map[string]any{
			"csi":                  csiResults,
			"drifted_features":     driftedNames,
			"scores_source":        scoresSource,
			"n_observations":       len(productionScores),
			"minimum_observations": config.MinDriftScores,
			"monitoring_eligible":  monitoringEligible,
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

func driftRecommendation(
	psi float64,
	psiStatus string,
	csiResults map[string]float64,
	realProduction bool,
	n int,
) string {
	if !realProduction {
		return fmt.Sprintf(
			"DIAGNOSTIC ONLY. Score PSI (%.4f) uses the test-set proxy and cannot trigger automated retraining.",
			psi,
		)
	}
	if psiStatus == "INSUFFICIENT_DATA" {
		return fmt.Sprintf(
			"WAIT FOR MORE DATA. Only %d production scores are available; at least %d are required for automated drift action.",
			n, config.MinDriftScores,
		)
	}
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
	ctx, cancel := withDeadline(driftTimeout)
	defer cancel()
	rep, err := runDrift(ctx)
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
