package monitoring

import (
	"encoding/json"
	"testing"
)

func TestMinDIR(t *testing.T) {
	fairness := []byte(`{
		"dir_threshold": 0.8,
		"attributes": {
			"home_ownership": {"groups": {
				"MORTGAGE": {"dir": 1.0},
				"RENT": {"dir": 0.634}
			}},
			"emp_length_missing": {"groups": {
				"Reported": {"dir": 1.0},
				"Not Reported": {"dir": 0.499}
			}}
		}
	}`)
	if got := minDIR(fairness); got != 0.499 {
		t.Errorf("minDIR = %v, want 0.499", got)
	}

	t.Run("empty or malformed defaults to parity", func(t *testing.T) {
		if got := minDIR([]byte(`{}`)); got != 1.0 {
			t.Errorf("minDIR(empty) = %v, want 1.0", got)
		}
		if got := minDIR([]byte(`not json`)); got != 1.0 {
			t.Errorf("minDIR(garbage) = %v, want 1.0", got)
		}
	})
}

func TestChampionFairnessParsing(t *testing.T) {
	raw := []byte(`{"version": "v1.4", "metrics": {}, "fairness": {"dir_threshold": 0.8}}`)
	var meta championFairness
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	if meta.Version != "v1.4" || len(meta.Fairness) == 0 {
		t.Errorf("parsed %+v", meta)
	}

	t.Run("missing fairness block yields empty raw message", func(t *testing.T) {
		var m2 championFairness
		if err := json.Unmarshal([]byte(`{"version": "v1.0"}`), &m2); err != nil {
			t.Fatal(err)
		}
		if len(m2.Fairness) != 0 {
			t.Errorf("expected empty fairness, got %s", m2.Fairness)
		}
	})
}
