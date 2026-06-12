// Retrain Orchestrator Agent (Go port of agents/retrain_orchestrator.py).
// Shells out to the Python stack for sklearn training (the one step that
// must stay in Python), then evaluates challenger vs champion natively:
// AUC/KS/Gini on the test set plus PSI between score distributions.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/gold"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/metrics"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/model"
)

type testMetrics struct {
	AUC  float64 `json:"auc"`
	KS   float64 `json:"ks"`
	Gini float64 `json:"gini"`
}

type trainResult struct {
	Version     string            `json:"version"`
	NIterations int               `json:"n_iterations"`
	Params      map[string]string `json:"params"`
	TestMetrics testMetrics       `json:"test_metrics"`
}

type report struct {
	Timestamp string `json:"timestamp"`
	Reason    string `json:"reason"`
	Champion  struct {
		Version     *string      `json:"version"`
		TestMetrics *testMetrics `json:"test_metrics"`
	} `json:"champion"`
	Challenger struct {
		Version     string            `json:"version"`
		TestMetrics testMetrics       `json:"test_metrics"`
		Params      map[string]string `json:"params"`
		NIterations int               `json:"n_iterations"`
	} `json:"challenger"`
	Comparison struct {
		ScorePSI       *float64 `json:"score_psi"`
		AUCImprovement float64  `json:"auc_improvement"`
		KSImprovement  float64  `json:"ks_improvement"`
	} `json:"comparison"`
	Recommendation string `json:"recommendation"`
	ActionRequired string `json:"action_required"`
	PromoteCommand string `json:"promote_command"`
}

func pythonBin() string {
	if p := os.Getenv("PYTHON_BIN"); p != "" {
		return p
	}
	if venv := filepath.Join(".venv", "bin", "python"); fileExists(venv) {
		return venv
	}
	return "python3"
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// nextVersion bumps the champion's minor version (v1.2 -> v1.3).
func nextVersion(championVersion string) string {
	parts := strings.SplitN(strings.TrimPrefix(championVersion, "v"), ".", 2)
	if len(parts) == 2 {
		if minor, err := strconv.Atoi(parts[1]); err == nil {
			return fmt.Sprintf("v%s.%d", parts[0], minor+1)
		}
	}
	return "v1.0"
}

func evaluate(scores []float64, yTrue []int) testMetrics {
	auc := metrics.ROCAUC(yTrue, scores)
	return testMetrics{
		AUC:  round(auc, 4),
		KS:   round(metrics.KS(yTrue, scores), 4),
		Gini: round(2*auc-1, 4),
	}
}

func run(ctx context.Context, reason string) (*report, error) {
	slog.Info("starting retraining", "reason", reason)

	// Load current champion for comparison (may not exist yet)
	var champion *model.Model
	version := "v1.0"
	if m, err := model.Load(config.ChampionModelPath()); err == nil {
		champion = m
		version = nextVersion(m.Version)
		slog.Info("current champion", "version", m.Version)
	} else {
		slog.Info("no champion model found", "detail", err)
	}

	// Train challenger via the Python stack
	slog.Info("training challenger model via Python", "python", pythonBin(), "version", version)
	cmd := exec.CommandContext(ctx, pythonBin(), filepath.Join("pipeline", "train_challenger.py"), version)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("challenger training failed: %w", err)
	}
	var trained trainResult
	if err := json.Unmarshal(out, &trained); err != nil {
		return nil, fmt.Errorf("parse training result: %w (output: %.200s)", err, out)
	}
	slog.Info("challenger trained", "iterations", trained.NIterations)

	challenger, err := model.Load(config.ChallengerModelPath())
	if err != nil {
		return nil, fmt.Errorf("load challenger: %w", err)
	}

	// Evaluate both on the test set
	cols := append(append([]string{}, challenger.Features...), "default")
	test, err := gold.ReadColumns(filepath.Join(config.GoldDir(), "features_test.parquet"), cols)
	if err != nil {
		return nil, err
	}
	rows, err := test.Rows(challenger.Features)
	if err != nil {
		return nil, err
	}
	yTrue := make([]int, test.NumRows)
	for i, v := range test.Columns["default"] {
		yTrue[i] = int(v)
	}

	challengerScores := challenger.PredictProbaBatch(rows)
	challengerMetrics := evaluate(challengerScores, yTrue)

	rep := &report{
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
		Reason:         reason,
		ActionRequired: "Human review required before promoting challenger to champion (SR 11-7).",
		PromoteCommand: "cp -r data/models/challenger/. data/models/champion/",
	}
	rep.Challenger.Version = version
	rep.Challenger.TestMetrics = challengerMetrics
	rep.Challenger.Params = trained.Params
	rep.Challenger.NIterations = trained.NIterations

	var championMetrics *testMetrics
	if champion != nil {
		championRows := rows
		if !sameFeatures(champion.Features, challenger.Features) {
			champFrame, err := gold.ReadColumns(
				filepath.Join(config.GoldDir(), "features_test.parquet"), champion.Features)
			if err != nil {
				return nil, err
			}
			if championRows, err = champFrame.Rows(champion.Features); err != nil {
				return nil, err
			}
		}
		championScores := champion.PredictProbaBatch(championRows)
		cm := evaluate(championScores, yTrue)
		championMetrics = &cm
		rep.Champion.Version = &champion.Version
		rep.Champion.TestMetrics = championMetrics

		// PSI between champion and challenger score distributions
		psi, _, _ := metrics.PSI(championScores, challengerScores, 10)
		psi = round(psi, 4)
		rep.Comparison.ScorePSI = &psi

		slog.Info("=== Comparison ===")
		slog.Info(fmt.Sprintf("  Champion (%s): AUC=%g  KS=%g", champion.Version, cm.AUC, cm.KS))
	}

	championAUC, championKS := 0.0, 0.0
	if championMetrics != nil {
		championAUC, championKS = championMetrics.AUC, championMetrics.KS
	}
	rep.Comparison.AUCImprovement = round(challengerMetrics.AUC-championAUC, 4)
	rep.Comparison.KSImprovement = round(challengerMetrics.KS-championKS, 4)
	rep.Recommendation = makeRecommendation(challengerMetrics, championMetrics, rep.Comparison.ScorePSI)

	slog.Info(fmt.Sprintf("  Challenger (%s): AUC=%g  KS=%g", version, challengerMetrics.AUC, challengerMetrics.KS))
	if rep.Comparison.ScorePSI != nil {
		slog.Info(fmt.Sprintf("  Score PSI between models: %g", *rep.Comparison.ScorePSI))
	}
	slog.Info("  Recommendation: " + rep.Recommendation)
	slog.Info("Challenger saved. Awaiting human review for promotion.")

	return rep, nil
}

func makeRecommendation(challenger testMetrics, champion *testMetrics, scorePSI *float64) string {
	if champion == nil {
		return "PROMOTE. No existing champion — challenger becomes the first production model."
	}
	psi := math.NaN()
	if scorePSI != nil {
		psi = *scorePSI
	}
	aucDiff := challenger.AUC - champion.AUC
	switch {
	case aucDiff > 0.005:
		return fmt.Sprintf(
			"PROMOTE. Challenger AUC (%g) exceeds champion (%g) by %.4f. Score PSI: %g.",
			challenger.AUC, champion.AUC, aucDiff, psi)
	case aucDiff > -0.005:
		return fmt.Sprintf(
			"CONSIDER. Challenger AUC (%g) is within 0.5%% of champion (%g). "+
				"Promotion optional — may prefer stability. Score PSI: %g.",
			challenger.AUC, champion.AUC, psi)
	default:
		return fmt.Sprintf(
			"DO NOT PROMOTE. Challenger AUC (%g) is worse than champion (%g) by %.4f. "+
				"Investigate hyperparameters or data quality.",
			challenger.AUC, champion.AUC, math.Abs(aucDiff))
	}
}

func sameFeatures(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func round(x float64, decimals int) float64 {
	p := math.Pow(10, float64(decimals))
	return math.Round(x*p) / p
}

func main() {
	config.LoadEnv()
	reason := "manual"
	if len(os.Args) > 1 {
		reason = os.Args[1]
	}
	rep, err := run(context.Background(), reason)
	if err != nil {
		slog.Error("retrain orchestrator failed", "error", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
}
