package model

import (
	"math"
	"testing"
)

// Unit tests for isotonic calibration and scorecard scaling on
// hand-built breakpoints; the fixture tests in model_test.go pin
// sklearn parity on the real champion calibrator.

func testCalibration() *Calibration {
	return &Calibration{
		Method: "isotonic",
		X:      []float64{0.1, 0.2, 0.4},
		Y:      []float64{0.05, 0.1, 0.3},
	}
}

// testScorecard mirrors the constants in pipeline/calibrate.py:
// 600 = 30:1 odds, PDO 20.
func testScorecard() *Scorecard {
	factor := 20.0 / math.Ln2
	return &Scorecard{
		BaseScore: 600,
		BaseOdds:  30,
		PDO:       20,
		Factor:    factor,
		Offset:    600 - factor*math.Log(30),
	}
}

func TestCalibrationApply(t *testing.T) {
	c := testCalibration()
	cases := []struct {
		name string
		p    float64
		want float64
	}{
		{"below range clips to first", 0.01, 0.05},
		{"at first breakpoint", 0.1, 0.05},
		{"midpoint interpolates", 0.15, 0.075},
		{"at middle breakpoint", 0.2, 0.1},
		{"interior interpolates", 0.3, 0.2},
		{"at last breakpoint", 0.4, 0.3},
		{"above range clips to last", 0.9, 0.3},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := c.Apply(tc.p); math.Abs(got-tc.want) > 1e-15 {
				t.Errorf("Apply(%v) = %v, want %v", tc.p, got, tc.want)
			}
		})
	}
	t.Run("single breakpoint is constant", func(t *testing.T) {
		c := &Calibration{Method: "isotonic", X: []float64{0.5}, Y: []float64{0.2}}
		for _, p := range []float64{0, 0.5, 1} {
			if got := c.Apply(p); got != 0.2 {
				t.Errorf("Apply(%v) = %v, want 0.2", p, got)
			}
		}
	})
	t.Run("monotonic over the full range", func(t *testing.T) {
		prev := math.Inf(-1)
		for p := 0.0; p <= 1.0; p += 0.001 {
			got := c.Apply(p)
			if got < prev {
				t.Fatalf("Apply(%v) = %v < previous %v", p, got, prev)
			}
			prev = got
		}
	})
}

func TestScorecardScore(t *testing.T) {
	s := testScorecard()
	cases := []struct {
		name string
		pd   float64
		want int
	}{
		{"base odds anchor", 1.0 / 31, 600}, // 30:1 odds
		{"double odds adds PDO", 1.0 / 61, 620},
		{"quadruple odds adds 2*PDO", 1.0 / 121, 640},
		{"half odds subtracts PDO", 1.0 / 16, 580},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := s.Score(tc.pd); got != tc.want {
				t.Errorf("Score(%v) = %d, want %d", tc.pd, got, tc.want)
			}
		})
	}
	t.Run("extreme PDs stay finite via clipping", func(t *testing.T) {
		lo, hi := s.Score(1), s.Score(0)
		if lo >= hi {
			t.Errorf("Score(1) = %d should be below Score(0) = %d", lo, hi)
		}
		if hi > 1500 || lo < -1500 {
			t.Errorf("clipped scores out of plausible range: [%d, %d]", lo, hi)
		}
	})
	t.Run("monotonic decreasing in PD", func(t *testing.T) {
		prev := math.MaxInt
		for pd := 0.001; pd < 1; pd += 0.001 {
			got := s.Score(pd)
			if got > prev {
				t.Fatalf("Score(%v) = %d > previous %d", pd, got, prev)
			}
			prev = got
		}
	})
}

func TestLoadCalibrationValidation(t *testing.T) {
	validTree := `{"value":[0,-1,1],"count":[2,1,1],"feature_idx":[0,0,0],` +
		`"num_threshold":[0.5,0,0],"missing_go_to_left":[1,0,0],` +
		`"left":[1,0,0],"right":[2,0,0],"is_leaf":[0,1,1]}`
	base := `"format_version":1,"model_version":"v1","n_features":1,` +
		`"features":["a"],"baseline_prediction":0,"trees":[` + validTree + `]`

	cases := []struct {
		name    string
		content string
		wantErr bool
	}{
		{"length mismatch", `{` + base + `,"calibration":{"method":"isotonic","x":[0.1,0.2],"y":[0.05]}}`, true},
		{"empty breakpoints", `{` + base + `,"calibration":{"method":"isotonic","x":[],"y":[]}}`, true},
		{"unsorted breakpoints", `{` + base + `,"calibration":{"method":"isotonic","x":[0.2,0.1],"y":[0.05,0.1]}}`, true},
		{"valid calibration loads", `{` + base + `,"calibration":{"method":"isotonic","x":[0.1,0.2],"y":[0.05,0.1]},` +
			`"scorecard":{"base_score":600,"base_odds":30,"pdo":20,"factor":28.85,"offset":501.86}}`, false},
		{"no calibration still loads", `{` + base + `}`, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, err := Load(writeModelFile(t, tc.content))
			if tc.wantErr && err == nil {
				t.Error("Load succeeded, want error")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("Load failed: %v", err)
			}
			if !tc.wantErr && err == nil && m.Calibration != nil && m.Scorecard == nil {
				t.Error("scorecard missing on calibrated model")
			}
		})
	}
}
