package model

import (
	"math"
	"sort"
)

// Calibration is an isotonic calibrator exported as breakpoints by
// pipeline/export_model_json.py. Apply mirrors sklearn's
// IsotonicRegression(out_of_bounds="clip"): inputs are clipped to the
// breakpoint range, values between breakpoints interpolate linearly.
type Calibration struct {
	Method string    `json:"method"`
	X      []float64 `json:"x"`
	Y      []float64 `json:"y"`
}

// Apply maps a raw model probability to a calibrated PD.
func (c *Calibration) Apply(p float64) float64 {
	n := len(c.X)
	if p <= c.X[0] {
		return c.Y[0]
	}
	if p >= c.X[n-1] {
		return c.Y[n-1]
	}
	i := sort.SearchFloat64s(c.X, p) // first index with X[i] >= p
	if c.X[i] == p {
		return c.Y[i]
	}
	slope := (c.Y[i] - c.Y[i-1]) / (c.X[i] - c.X[i-1])
	return slope*(p-c.X[i-1]) + c.Y[i-1]
}

// Scorecard maps calibrated PDs to points-to-double-odds scores,
// mirroring pipeline/calibrate.py (600 = 30:1 good:bad odds, PDO 20).
type Scorecard struct {
	BaseScore float64 `json:"base_score"`
	BaseOdds  float64 `json:"base_odds"`
	PDO       float64 `json:"pdo"`
	Factor    float64 `json:"factor"`
	Offset    float64 `json:"offset"`
}

// pdClip mirrors PD_CLIP in pipeline/calibrate.py: isotonic output can
// be exactly 0 or 1, which would make the odds transform infinite.
const pdClip = 1e-6

// Score maps a PD to an integer scorecard score. floor(x + 0.5)
// matches the Python exporter's rounding exactly.
func (s *Scorecard) Score(pd float64) int {
	pd = math.Min(math.Max(pd, pdClip), 1-pdClip)
	score := s.Offset + s.Factor*math.Log((1-pd)/pd)
	return int(math.Floor(score + 0.5))
}
