package inference

import (
	"math"
	"testing"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
)

// Verifies adverse-action selection against the champion model: only
// positive SHAP contributions, descending, at most 4, with display
// names applied.
func TestComputeAdverseActions(t *testing.T) {
	m, err := model.Load("../shared/model/testdata/parity_model.json")
	if err != nil {
		t.Fatalf("parity model missing (regenerate with "+
			"scripts/generate_model_fixtures.py): %v", err)
	}

	// A deliberately risky applicant profile (raw feature order).
	features := map[string]*float64{}
	x := make([]float64, len(m.Features))
	risky := map[string]float64{
		"loan_amnt": 35000, "term": 60, "int_rate": 28.5, "installment": 1100,
		"annual_inc": 30000, "dti": 39, "fico_score": 640, "revol_util": 98,
		"inq_last_6mths": 6, "delinq_2yrs": 3, "grade_numeric": 7, "sub_grade_numeric": 35,
		"loan_to_income": 1.17, "installment_to_income": 0.44, "high_utilization": 1,
	}
	for i, col := range m.Features {
		v := risky[col]
		x[i] = v
		val := v
		features[col] = &val
	}

	actions := computeAdverseActions(m, x, features)
	if len(actions) == 0 || len(actions) > numAdverseActions {
		t.Fatalf("got %d adverse actions, want 1..%d", len(actions), numAdverseActions)
	}
	prev := math.Inf(1)
	seenCodes := map[int]bool{}
	for _, a := range actions {
		if a.ShapValue <= 0 {
			t.Errorf("%s: non-positive shap value %v", a.FeatureName, a.ShapValue)
		}
		if a.ShapValue > prev {
			t.Errorf("actions not sorted descending: %v after %v", a.ShapValue, prev)
		}
		if a.Direction != "increases risk" {
			t.Errorf("unexpected direction %q", a.Direction)
		}
		if seenCodes[a.Code] {
			t.Errorf("duplicate adverse reason code %d in %+v", a.Code, actions)
		}
		seenCodes[a.Code] = true
		prev = a.ShapValue
	}

	// Display names must come from the mapping, e.g. int_rate -> Interest Rate.
	for _, a := range actions {
		for raw := range featureDisplayNames {
			if a.FeatureName == raw {
				t.Errorf("raw feature name %q leaked into response", raw)
			}
		}
	}
}
