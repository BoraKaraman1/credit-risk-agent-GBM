// Retrain Orchestrator Agent (Go port of agents/retrain_orchestrator.py).
// Shells out to the Python stack for sklearn training (the one step that
// must stay in Python), then evaluates challenger vs champion natively:
// AUC/KS/Gini on the test set plus PSI between score distributions.
package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/gold"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/metrics"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
)

// A challenger may worsen an already-violating group by up to this much
// before the gate treats it as a regression (guards against retrain
// noise in the DIR estimate).
const fairnessWorsenTolerance = 0.01

type testMetrics struct {
	AUC  float64 `json:"auc"`
	KS   float64 `json:"ks"`
	Gini float64 `json:"gini"`
}

// fairnessSummary is the subset of pipeline/fairness.py's output the gate
// needs: per attribute, each group's Disparate Impact Ratio.
type fairnessSummary struct {
	DIRThreshold float64 `json:"dir_threshold"`
	Attributes   map[string]struct {
		Groups map[string]struct {
			DIR float64 `json:"dir"`
		} `json:"groups"`
	} `json:"attributes"`
}

type trainResult struct {
	Version          string            `json:"version"`
	NIterations      int               `json:"n_iterations"`
	Params           map[string]string `json:"params"`
	TestMetrics      testMetrics       `json:"test_metrics"`
	Fairness         *fairnessSummary  `json:"fairness"`
	ChampionFairness *fairnessSummary  `json:"champion_fairness"`
}

type retrainReport struct {
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
	Fairness struct {
		Gate         string   `json:"gate"` // passed | blocked | skipped
		DIRThreshold float64  `json:"dir_threshold"`
		Reasons      []string `json:"reasons"`
	} `json:"fairness"`
	Recommendation string `json:"recommendation"`
	ActionRequired string `json:"action_required"`
	PromoteCommand string `json:"promote_command"`
}

// fairnessGate applies the champion-relative four-fifths rule: a
// challenger is blocked only if it introduces a new DIR violation or
// worsens one the champion already had. A group the champion already
// fails, left no worse, does not block (the dataset's inherent disparity
// should not make every challenger unpromotable). Returns the blocking
// reasons, sorted for deterministic output.
func fairnessGate(champ, chal *fairnessSummary) (bool, []string) {
	if chal == nil {
		return false, nil
	}
	thr := chal.DIRThreshold
	if thr <= 0 {
		thr = config.FairnessDIRThreshold
	}
	var reasons []string
	for attr, a := range chal.Attributes {
		for group, g := range a.Groups {
			if g.DIR >= thr {
				continue
			}
			champDIR, had := 0.0, false
			if champ != nil {
				if ca, ok := champ.Attributes[attr]; ok {
					if cg, ok2 := ca.Groups[group]; ok2 {
						champDIR, had = cg.DIR, true
					}
				}
			}
			switch {
			case !had || champDIR >= thr:
				reasons = append(reasons, fmt.Sprintf(
					"%s/%s introduces a DIR violation (%.2f < %.2f)", attr, group, g.DIR, thr))
			case g.DIR < champDIR-fairnessWorsenTolerance:
				reasons = append(reasons, fmt.Sprintf(
					"%s/%s worsens an existing DIR violation (champion %.2f -> challenger %.2f)",
					attr, group, champDIR, g.DIR))
			}
		}
	}
	sort.Strings(reasons)
	return len(reasons) > 0, reasons
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

func runRetrain(ctx context.Context, reason string) (*retrainReport, error) {
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

	rep := &retrainReport{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Reason:    reason,
		ActionRequired: "Human review required before promoting challenger to champion (SR 11-7); " +
			"the challenger model card must be APPROVED (see docs/model_card.md) or the serving " +
			"gate will refuse it.",
		// `gbm promote` publishes the challenger as an immutable versioned
		// directory and atomically repoints the champion symlink at it, so
		// the serving runtime never observes a missing or partial champion.
		PromoteCommand: "gbm promote",
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

	// Champion-relative fairness gate (skipped when there is no champion).
	gateBlocked, gateReasons := false, []string(nil)
	rep.Fairness.Gate = "skipped"
	if trained.Fairness != nil {
		rep.Fairness.DIRThreshold = trained.Fairness.DIRThreshold
	}
	if champion != nil {
		gateBlocked, gateReasons = fairnessGate(trained.ChampionFairness, trained.Fairness)
		rep.Fairness.Reasons = gateReasons
		if gateBlocked {
			rep.Fairness.Gate = "blocked"
		} else {
			rep.Fairness.Gate = "passed"
		}
	}
	rep.Recommendation = retrainRecommendation(
		challengerMetrics, championMetrics, rep.Comparison.ScorePSI, gateBlocked, gateReasons)

	slog.Info(fmt.Sprintf("  Challenger (%s): AUC=%g  KS=%g", version, challengerMetrics.AUC, challengerMetrics.KS))
	if rep.Comparison.ScorePSI != nil {
		slog.Info(fmt.Sprintf("  Score PSI between models: %g", *rep.Comparison.ScorePSI))
	}
	slog.Info("  Recommendation: " + rep.Recommendation)
	slog.Info("Challenger saved. Awaiting human review for promotion.")

	return rep, nil
}

func retrainRecommendation(challenger testMetrics, champion *testMetrics, scorePSI *float64,
	fairnessBlocked bool, fairnessReasons []string) string {
	if champion == nil {
		return "PROMOTE. No existing champion — challenger becomes the first production model."
	}
	// The fairness gate overrides AUC: a model that is more discriminatory
	// must not be promoted on accuracy alone (SR 11-7 + ECOA).
	if fairnessBlocked {
		return "DO NOT PROMOTE. Fairness gate failed: " + strings.Join(fairnessReasons, "; ") + "."
	}
	psi := math.NaN()
	if scorePSI != nil {
		psi = *scorePSI
	}
	aucDiff := challenger.AUC - champion.AUC
	switch {
	case aucDiff > 0.005:
		return fmt.Sprintf(
			"PROMOTE. Challenger AUC (%g) exceeds champion (%g) by %.4f. Score PSI: %g. Fairness gate passed.",
			challenger.AUC, champion.AUC, aucDiff, psi)
	case aucDiff > -0.005:
		return fmt.Sprintf(
			"CONSIDER. Challenger AUC (%g) is within 0.5%% of champion (%g). "+
				"Promotion optional — may prefer stability. Score PSI: %g. Fairness gate passed.",
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

func RunRetrain(reason string) {
	config.LoadEnv()
	rep, err := runRetrain(context.Background(), reason)
	if err != nil {
		slog.Error("retrain orchestrator failed", "error", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(rep); err != nil {
		slog.Error("failed to encode retrain report", "error", err)
		os.Exit(1)
	}
}
