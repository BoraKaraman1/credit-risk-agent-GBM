package metrics

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

type metricsFixtures struct {
	PSI struct {
		Expected    []float64 `json:"expected"`
		Actual      []float64 `json:"actual"`
		PSI         float64   `json:"psi"`
		ExpectedPct []float64 `json:"expected_pct"`
		ActualPct   []float64 `json:"actual_pct"`
	} `json:"psi"`
	CSI struct {
		Train []*float64 `json:"train"`
		Prod  []*float64 `json:"prod"`
		CSI   float64    `json:"csi"`
	} `json:"csi"`
	CSIDiscrete struct {
		Train []*float64 `json:"train"`
		Prod  []*float64 `json:"prod"`
		CSI   float64    `json:"csi"`
	} `json:"csi_discrete"`
	Perf struct {
		YTrue           []int     `json:"y_true"`
		YScore          []float64 `json:"y_score"`
		AUC             float64   `json:"auc"`
		KS              float64   `json:"ks"`
		Deciles         []Decile  `json:"deciles"`
		RankOrderBreaks int       `json:"rank_order_breaks"`
	} `json:"perf"`
}

func toFloats(vals []*float64) []float64 {
	out := make([]float64, len(vals))
	for i, v := range vals {
		if v == nil {
			out[i] = math.NaN()
		} else {
			out[i] = *v
		}
	}
	return out
}

func loadMetricsFixtures(t *testing.T) *metricsFixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/fixtures.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx metricsFixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}
	return &fx
}

func almostEqual(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s: got %v want %v (diff %g)", name, got, want, math.Abs(got-want))
	}
}

func TestPSIMatchesNumpy(t *testing.T) {
	fx := loadMetricsFixtures(t)
	psi, ep, ap := PSI(fx.PSI.Expected, fx.PSI.Actual, 10)
	almostEqual(t, "psi", psi, fx.PSI.PSI, 1e-10)
	for i := range ep {
		almostEqual(t, "expected_pct", ep[i], fx.PSI.ExpectedPct[i], 1e-12)
		almostEqual(t, "actual_pct", ap[i], fx.PSI.ActualPct[i], 1e-12)
	}
}

func TestCSIMatchesNumpy(t *testing.T) {
	fx := loadMetricsFixtures(t)
	csi := CSI(toFloats(fx.CSI.Train), toFloats(fx.CSI.Prod), 10)
	almostEqual(t, "csi", csi, fx.CSI.CSI, 1e-10)

	csiDisc := CSI(toFloats(fx.CSIDiscrete.Train), toFloats(fx.CSIDiscrete.Prod), 10)
	almostEqual(t, "csi_discrete", csiDisc, fx.CSIDiscrete.CSI, 1e-10)
}

func TestROCAUCMatchesSklearn(t *testing.T) {
	fx := loadMetricsFixtures(t)
	almostEqual(t, "auc", ROCAUC(fx.Perf.YTrue, fx.Perf.YScore), fx.Perf.AUC, 1e-10)
}

func TestKSMatchesSklearn(t *testing.T) {
	fx := loadMetricsFixtures(t)
	almostEqual(t, "ks", KS(fx.Perf.YTrue, fx.Perf.YScore), fx.Perf.KS, 1e-10)
}

func TestDecileAnalysisMatchesPandas(t *testing.T) {
	fx := loadMetricsFixtures(t)
	stats, breaks := DecileAnalysis(fx.Perf.YTrue, fx.Perf.YScore)
	if breaks != fx.Perf.RankOrderBreaks {
		t.Errorf("rank_order_breaks: got %d want %d", breaks, fx.Perf.RankOrderBreaks)
	}
	if len(stats) != len(fx.Perf.Deciles) {
		t.Fatalf("deciles: got %d want %d", len(stats), len(fx.Perf.Deciles))
	}
	for i, d := range stats {
		want := fx.Perf.Deciles[i]
		if d.Count != want.Count {
			t.Errorf("decile %d count: got %d want %d", i, d.Count, want.Count)
		}
		almostEqual(t, "default_rate", d.DefaultRate, want.DefaultRate, 1e-12)
		almostEqual(t, "avg_score", d.AvgScore, want.AvgScore, 1e-12)
	}
}
