package metrics

import (
	"math"
	"testing"
)

// White-box unit tests for the metric primitives. The fixture tests in
// metrics_test.go pin exact numpy/sklearn/pandas parity; these pin the
// hand-checkable properties and edge cases.

func TestHistogram(t *testing.T) {
	edges := []float64{0, 1, 2, 3}
	cases := []struct {
		name   string
		values []float64
		want   []float64
	}{
		{"empty input", nil, []float64{0, 0, 0}},
		{"interior values", []float64{0.5, 1.5, 2.5}, []float64{1, 1, 1}},
		{"left edge inclusive", []float64{0, 1, 2}, []float64{1, 1, 1}},
		{"max edge closed into last bin", []float64{3, 3}, []float64{0, 0, 2}},
		{"out of range dropped", []float64{-0.1, 3.5}, []float64{0, 0, 0}},
		{"NaN dropped", []float64{math.NaN(), 0.5}, []float64{1, 0, 0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Histogram(tc.values, edges)
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Errorf("bin %d: got %v want %v (full: %v)", i, got[i], tc.want[i], got)
				}
			}
		})
	}
}

func TestLinspace(t *testing.T) {
	t.Run("eleven points on unit interval", func(t *testing.T) {
		got := Linspace(0, 1, 11)
		if len(got) != 11 || got[0] != 0 || got[10] != 1 {
			t.Fatalf("got %v", got)
		}
		if math.Abs(got[3]-0.3) > 1e-12 {
			t.Errorf("got[3] = %v, want 0.3", got[3])
		}
	})
	t.Run("two points are the endpoints", func(t *testing.T) {
		got := Linspace(2, 5, 2)
		if got[0] != 2 || got[1] != 5 {
			t.Fatalf("got %v", got)
		}
	})
	t.Run("endpoint is exact", func(t *testing.T) {
		got := Linspace(0, 0.3, 4)
		if got[3] != 0.3 {
			t.Errorf("last = %v, want exactly 0.3", got[3])
		}
	})
}

func TestQuantile(t *testing.T) {
	sorted := []float64{1, 2, 3, 4, 5}
	cases := []struct {
		name string
		q    float64
		want float64
	}{
		{"q0 is min", 0, 1},
		{"q1 is max", 1, 5},
		{"median", 0.5, 3},
		{"exact position", 0.25, 2},
		{"linear interpolation", 0.1, 1.4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := quantile(sorted, tc.q)
			if math.Abs(got-tc.want) > 1e-12 {
				t.Errorf("q=%v: got %v want %v", tc.q, got, tc.want)
			}
		})
	}
	t.Run("single element", func(t *testing.T) {
		if got := quantile([]float64{7}, 0.9); got != 7 {
			t.Errorf("got %v want 7", got)
		}
	})
}

func TestSmoothedPct(t *testing.T) {
	t.Run("empty counts give uniform", func(t *testing.T) {
		got := smoothedPct([]float64{0, 0})
		if got[0] != 0.5 || got[1] != 0.5 {
			t.Fatalf("got %v", got)
		}
	})
	t.Run("laplace smoothing", func(t *testing.T) {
		got := smoothedPct([]float64{8, 0})
		if math.Abs(got[0]-0.9) > 1e-12 || math.Abs(got[1]-0.1) > 1e-12 {
			t.Fatalf("got %v, want [0.9 0.1]", got)
		}
	})
	t.Run("sums to one", func(t *testing.T) {
		got := smoothedPct([]float64{3, 1, 4, 1, 5})
		sum := 0.0
		for _, p := range got {
			sum += p
		}
		if math.Abs(sum-1) > 1e-12 {
			t.Errorf("sum = %v", sum)
		}
	})
}

func TestPSIProperties(t *testing.T) {
	t.Run("identical distributions give zero", func(t *testing.T) {
		scores := []float64{0.1, 0.2, 0.3, 0.5, 0.8, 0.9}
		psi, _, _ := PSI(scores, scores, 10)
		if psi != 0 {
			t.Errorf("psi = %v, want exactly 0", psi)
		}
	})
	t.Run("shifted distribution is positive", func(t *testing.T) {
		low := make([]float64, 100)
		high := make([]float64, 100)
		for i := range low {
			low[i] = 0.1 + 0.001*float64(i)
			high[i] = 0.8 + 0.001*float64(i)
		}
		psi, _, _ := PSI(low, high, 10)
		if psi <= 0.25 {
			t.Errorf("psi = %v, want a large shift", psi)
		}
	})
	t.Run("empty actual does not divide by zero", func(t *testing.T) {
		psi, _, _ := PSI([]float64{0.5, 0.6}, nil, 10)
		if math.IsNaN(psi) || math.IsInf(psi, 0) {
			t.Errorf("psi = %v", psi)
		}
	})
	t.Run("returns one pct per bin", func(t *testing.T) {
		_, ep, ap := PSI([]float64{0.5}, []float64{0.6}, 10)
		if len(ep) != 10 || len(ap) != 10 {
			t.Errorf("lengths %d, %d", len(ep), len(ap))
		}
	})
}

func TestCSIProperties(t *testing.T) {
	t.Run("identical features give zero", func(t *testing.T) {
		col := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		if csi := CSI(col, col, 10); csi != 0 {
			t.Errorf("csi = %v, want exactly 0", csi)
		}
	})
	t.Run("shifted feature is positive", func(t *testing.T) {
		train := make([]float64, 200)
		prod := make([]float64, 200)
		for i := range train {
			train[i] = float64(i)
			prod[i] = float64(i) + 150
		}
		if csi := CSI(train, prod, 10); csi <= 0.2 {
			t.Errorf("csi = %v, want a large shift", csi)
		}
	})
	t.Run("all NaN gives zero", func(t *testing.T) {
		nan := []float64{math.NaN(), math.NaN()}
		if csi := CSI(nan, nan, 10); csi != 0 {
			t.Errorf("csi = %v", csi)
		}
	})
	t.Run("constant feature gives zero", func(t *testing.T) {
		c := []float64{5, 5, 5, 5}
		if csi := CSI(c, c, 10); csi != 0 {
			t.Errorf("csi = %v", csi)
		}
	})
	t.Run("NaNs are ignored not counted", func(t *testing.T) {
		clean := []float64{1, 2, 3, 4, 5, 6, 7, 8}
		withNaN := append([]float64{math.NaN(), math.NaN()}, clean...)
		if csi := CSI(clean, withNaN, 10); csi != 0 {
			t.Errorf("csi = %v, want 0 (NaNs ignored)", csi)
		}
	})
}

func TestROCAUCCases(t *testing.T) {
	cases := []struct {
		name  string
		yTrue []int
		score []float64
		want  float64
	}{
		{"perfect separation", []int{0, 0, 1, 1}, []float64{0.1, 0.2, 0.8, 0.9}, 1.0},
		{"inverted scores", []int{0, 0, 1, 1}, []float64{0.9, 0.8, 0.2, 0.1}, 0.0},
		{"all tied is chance", []int{0, 1, 0, 1}, []float64{0.5, 0.5, 0.5, 0.5}, 0.5},
		{"partial ties use midranks", []int{0, 0, 1, 1}, []float64{0.3, 0.5, 0.5, 0.9}, 0.875},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ROCAUC(tc.yTrue, tc.score)
			if math.Abs(got-tc.want) > 1e-12 {
				t.Errorf("auc = %v, want %v", got, tc.want)
			}
		})
	}
	t.Run("single class is NaN", func(t *testing.T) {
		if got := ROCAUC([]int{1, 1}, []float64{0.1, 0.9}); !math.IsNaN(got) {
			t.Errorf("auc = %v, want NaN", got)
		}
	})
}

func TestKSCases(t *testing.T) {
	cases := []struct {
		name  string
		yTrue []int
		score []float64
		want  float64
	}{
		{"perfect separation", []int{0, 0, 1, 1}, []float64{0.1, 0.2, 0.8, 0.9}, 1.0},
		{"all tied", []int{1, 0}, []float64{0.5, 0.5}, 0.0},
		{"interleaved", []int{0, 1, 0, 1}, []float64{0.2, 0.4, 0.6, 0.8}, 0.5},
		{"no discrimination floor", []int{1, 0, 1, 0}, []float64{0.1, 0.9, 0.2, 0.8}, 0.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := KS(tc.yTrue, tc.score)
			if math.Abs(got-tc.want) > 1e-12 {
				t.Errorf("ks = %v, want %v", got, tc.want)
			}
		})
	}
}

// buildDecileData creates 100 observations in 10 score deciles where
// decile d has the given default rate (out of 10 observations each).
func buildDecileData(ratesPerDecile []int) ([]int, []float64) {
	var yTrue []int
	var yScore []float64
	for d, defaults := range ratesPerDecile {
		for i := 0; i < 10; i++ {
			yScore = append(yScore, float64(d)/10+float64(i)/100+0.001)
			label := 0
			if i < defaults {
				label = 1
			}
			yTrue = append(yTrue, label)
		}
	}
	return yTrue, yScore
}

func TestDecileAnalysis(t *testing.T) {
	t.Run("monotone rates have zero breaks", func(t *testing.T) {
		yTrue, yScore := buildDecileData([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
		stats, breaks := DecileAnalysis(yTrue, yScore)
		if breaks != 0 {
			t.Errorf("breaks = %d, want 0", breaks)
		}
		if len(stats) != 10 {
			t.Fatalf("deciles = %d, want 10", len(stats))
		}
	})
	t.Run("one inversion is one break", func(t *testing.T) {
		yTrue, yScore := buildDecileData([]int{0, 1, 2, 4, 3, 5, 6, 7, 8, 9})
		_, breaks := DecileAnalysis(yTrue, yScore)
		if breaks != 1 {
			t.Errorf("breaks = %d, want 1", breaks)
		}
	})
	t.Run("counts and rates per decile", func(t *testing.T) {
		yTrue, yScore := buildDecileData([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
		stats, _ := DecileAnalysis(yTrue, yScore)
		for d, s := range stats {
			if s.Count != 10 {
				t.Errorf("decile %d count = %d, want 10", d, s.Count)
			}
			if math.Abs(s.DefaultRate-float64(d)/10) > 1e-12 {
				t.Errorf("decile %d rate = %v, want %v", d, s.DefaultRate, float64(d)/10)
			}
		}
	})
	t.Run("constant scores degrade gracefully", func(t *testing.T) {
		// All quantile edges collapse: no usable buckets, no panic
		// (pandas qcut raises on this input; we return empty).
		yTrue := make([]int, 100)
		yScore := make([]float64, 100)
		for i := range yScore {
			yScore[i] = 0.5
		}
		stats, breaks := DecileAnalysis(yTrue, yScore)
		if len(stats) != 0 || breaks != 0 {
			t.Errorf("stats = %d buckets, breaks = %d, want 0/0", len(stats), breaks)
		}
	})
	t.Run("two distinct scores yield one bucket per edge gap", func(t *testing.T) {
		yTrue := make([]int, 100)
		yScore := make([]float64, 100)
		for i := range yScore {
			if i >= 50 {
				yScore[i] = 0.9
				yTrue[i] = 1
			} else {
				yScore[i] = 0.1
			}
		}
		stats, breaks := DecileAnalysis(yTrue, yScore)
		if len(stats) != 2 || breaks != 0 {
			t.Fatalf("stats = %d buckets, breaks = %d, want 2/0", len(stats), breaks)
		}
		if stats[0].Count != 50 || stats[0].DefaultRate != 0 || stats[1].DefaultRate != 1 {
			t.Errorf("stats = %+v", stats)
		}
	})
	t.Run("avg score is within decile bounds", func(t *testing.T) {
		yTrue, yScore := buildDecileData([]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
		stats, _ := DecileAnalysis(yTrue, yScore)
		prev := math.Inf(-1)
		for d, s := range stats {
			if s.AvgScore <= prev {
				t.Errorf("decile %d avg score %v not increasing", d, s.AvgScore)
			}
			prev = s.AvgScore
		}
	})
}
