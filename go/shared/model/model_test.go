package model

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// fixtures.json is generated from the Python stack (sklearn + shap) by
// pipeline/export_model_json.py's sibling snippet; rows 0-1 contain NaNs
// to exercise missing-value routing.
type fixtures struct {
	Features          []string     `json:"features"`
	Rows              [][]*float64 `json:"rows"`
	Proba             []float64    `json:"proba"`
	ShapValues        [][]float64  `json:"shap_values"`
	ShapExpectedValue float64      `json:"shap_expected_value"`
	CalibratedPD      []float64    `json:"calibrated_pd"`
	ScaledScore       []int        `json:"scaled_score"`
}

func loadFixtures(t *testing.T) (*Model, *fixtures, [][]float64) {
	t.Helper()
	// The committed parity model (scripts/generate_model_fixtures.py):
	// a hard failure, never a skip, so CI always verifies Go<->Python
	// parity instead of silently going green without it.
	m, err := Load("testdata/parity_model.json")
	if err != nil {
		t.Fatalf("parity model missing or unreadable (regenerate with "+
			"scripts/generate_model_fixtures.py): %v", err)
	}

	data, err := os.ReadFile("testdata/fixtures.json")
	if err != nil {
		t.Fatalf("read fixtures: %v", err)
	}
	var fx fixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatalf("parse fixtures: %v", err)
	}

	rows := make([][]float64, len(fx.Rows))
	for i, r := range fx.Rows {
		rows[i] = make([]float64, len(r))
		for j, v := range r {
			if v == nil {
				rows[i][j] = math.NaN()
			} else {
				rows[i][j] = *v
			}
		}
	}
	return m, &fx, rows
}

func TestPredictProbaMatchesSklearn(t *testing.T) {
	m, fx, rows := loadFixtures(t)
	for i, row := range rows {
		got := m.PredictProba(row)
		if diff := math.Abs(got - fx.Proba[i]); diff > 1e-9 {
			t.Errorf("row %d: proba %v != sklearn %v (diff %g)", i, got, fx.Proba[i], diff)
		}
	}
}

func TestPredictProbaBatch(t *testing.T) {
	m, fx, rows := loadFixtures(t)
	got := m.PredictProbaBatch(rows)
	for i := range got {
		if math.Abs(got[i]-fx.Proba[i]) > 1e-9 {
			t.Errorf("row %d: batch proba %v != %v", i, got[i], fx.Proba[i])
		}
	}
}

func TestCalibratedPDMatchesSklearn(t *testing.T) {
	m, fx, rows := loadFixtures(t)
	if m.Calibration == nil || len(fx.CalibratedPD) == 0 {
		t.Fatal("parity model or fixtures lack calibration (regenerate with scripts/generate_model_fixtures.py)")
	}
	for i, row := range rows {
		got := m.Calibration.Apply(m.PredictProba(row))
		if diff := math.Abs(got - fx.CalibratedPD[i]); diff > 1e-9 {
			t.Errorf("row %d: calibrated pd %v != sklearn %v (diff %g)", i, got, fx.CalibratedPD[i], diff)
		}
	}
}

func TestScaledScoreMatchesPython(t *testing.T) {
	m, fx, rows := loadFixtures(t)
	if m.Calibration == nil || m.Scorecard == nil || len(fx.ScaledScore) == 0 {
		t.Fatal("parity model or fixtures lack scorecard (regenerate with scripts/generate_model_fixtures.py)")
	}
	for i, row := range rows {
		pd := m.Calibration.Apply(m.PredictProba(row))
		if got := m.Scorecard.Score(pd); got != fx.ScaledScore[i] {
			t.Errorf("row %d: scaled score %d != python %d", i, got, fx.ScaledScore[i])
		}
	}
}

func TestExpectedValueMatchesShap(t *testing.T) {
	m, fx, _ := loadFixtures(t)
	if diff := math.Abs(m.ExpectedValue() - fx.ShapExpectedValue); diff > 1e-6 {
		t.Errorf("expected value %v != shap %v (diff %g)", m.ExpectedValue(), fx.ShapExpectedValue, diff)
	}
}

func TestShapValuesMatchPython(t *testing.T) {
	m, fx, rows := loadFixtures(t)
	for i, row := range rows {
		phi := m.ShapValues(row)
		for j := range phi {
			if diff := math.Abs(phi[j] - fx.ShapValues[i][j]); diff > 1e-6 {
				t.Errorf("row %d feature %s: shap %v != python %v (diff %g)",
					i, fx.Features[j], phi[j], fx.ShapValues[i][j], diff)
			}
		}
	}
}

// Local accuracy: attributions plus base value must reproduce the raw
// model output exactly (the defining TreeSHAP property).
func TestShapLocalAccuracy(t *testing.T) {
	m, _, rows := loadFixtures(t)
	for i, row := range rows {
		phi := m.ShapValues(row)
		sum := m.ExpectedValue()
		for _, p := range phi {
			sum += p
		}
		raw := m.RawPredict(row)
		if diff := math.Abs(sum - raw); diff > 1e-8 {
			t.Errorf("row %d: sum(phi)+EV = %v but raw = %v (diff %g)", i, sum, raw, diff)
		}
	}
}
