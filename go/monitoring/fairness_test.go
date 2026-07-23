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
	got, err := minDIR(fairness)
	if err != nil {
		t.Fatalf("minDIR: %v", err)
	}
	if got != 0.499 {
		t.Errorf("minDIR = %v, want 0.499", got)
	}

	t.Run("empty or malformed fails closed", func(t *testing.T) {
		if _, err := minDIR([]byte(`{}`)); err == nil {
			t.Error("minDIR(empty) should error, not default to parity")
		}
		if _, err := minDIR([]byte(`not json`)); err == nil {
			t.Error("minDIR(garbage) should error, not default to parity")
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
