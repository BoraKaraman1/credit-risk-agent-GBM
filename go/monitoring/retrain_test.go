package monitoring

import (
	"strings"
	"testing"
)

// fair builds a one-attribute, one-group fairness summary at the given DIR.
func fair(attr, group string, dir float64) *fairnessSummary {
	s := &fairnessSummary{DIRThreshold: 0.80}
	s.Attributes = map[string]struct {
		Groups map[string]struct {
			DIR float64 `json:"dir"`
		} `json:"groups"`
	}{}
	g := map[string]struct {
		DIR float64 `json:"dir"`
	}{group: {DIR: dir}}
	s.Attributes[attr] = struct {
		Groups map[string]struct {
			DIR float64 `json:"dir"`
		} `json:"groups"`
	}{Groups: g}
	return s
}

func TestFairnessGate(t *testing.T) {
	t.Run("nil challenger does not block", func(t *testing.T) {
		if blocked, _ := fairnessGate(nil, nil); blocked {
			t.Error("nil challenger should not block")
		}
	})
	t.Run("challenger within threshold passes", func(t *testing.T) {
		champ := fair("home_ownership", "RENT", 0.95)
		chal := fair("home_ownership", "RENT", 0.93)
		if blocked, _ := fairnessGate(champ, chal); blocked {
			t.Error("compliant challenger should pass")
		}
	})
	t.Run("new violation blocks", func(t *testing.T) {
		champ := fair("home_ownership", "RENT", 0.90) // champion fine
		chal := fair("home_ownership", "RENT", 0.70)  // challenger violates
		blocked, reasons := fairnessGate(champ, chal)
		if !blocked {
			t.Fatal("new violation should block")
		}
		if len(reasons) != 1 || !strings.Contains(reasons[0], "introduces") {
			t.Errorf("reasons = %v", reasons)
		}
	})
	t.Run("pre-existing violation left no worse passes", func(t *testing.T) {
		champ := fair("home_ownership", "RENT", 0.70)
		chal := fair("home_ownership", "RENT", 0.71) // slightly better
		if blocked, r := fairnessGate(champ, chal); blocked {
			t.Errorf("inherited violation should pass, got %v", r)
		}
	})
	t.Run("worsened existing violation blocks", func(t *testing.T) {
		champ := fair("home_ownership", "RENT", 0.70)
		chal := fair("home_ownership", "RENT", 0.60) // worse by 0.10
		blocked, reasons := fairnessGate(champ, chal)
		if !blocked {
			t.Fatal("worsened violation should block")
		}
		if !strings.Contains(reasons[0], "worsens") {
			t.Errorf("reasons = %v", reasons)
		}
	})
	t.Run("tiny worsening within tolerance passes", func(t *testing.T) {
		champ := fair("home_ownership", "RENT", 0.70)
		chal := fair("home_ownership", "RENT", 0.695) // worse by 0.005 < tol
		if blocked, _ := fairnessGate(champ, chal); blocked {
			t.Error("sub-tolerance noise should not block")
		}
	})
	t.Run("missing champion baseline treats violation as new", func(t *testing.T) {
		chal := fair("home_ownership", "RENT", 0.70)
		blocked, reasons := fairnessGate(nil, chal)
		if !blocked || !strings.Contains(reasons[0], "introduces") {
			t.Errorf("blocked=%v reasons=%v", blocked, reasons)
		}
	})
}

func TestRetrainRecommendationFairnessOverride(t *testing.T) {
	champion := &testMetrics{AUC: 0.70, KS: 0.30}
	challenger := testMetrics{AUC: 0.99, KS: 0.50} // far better AUC

	t.Run("fairness block overrides a strong AUC", func(t *testing.T) {
		rec := retrainRecommendation(challenger, champion, nil, true, []string{"home_ownership/RENT introduces a DIR violation (0.70 < 0.80)"})
		if !strings.HasPrefix(rec, "DO NOT PROMOTE. Fairness gate failed") {
			t.Errorf("rec = %q", rec)
		}
	})
	t.Run("passing gate keeps AUC recommendation", func(t *testing.T) {
		rec := retrainRecommendation(challenger, champion, nil, false, nil)
		if !strings.HasPrefix(rec, "PROMOTE.") || !strings.Contains(rec, "Fairness gate passed") {
			t.Errorf("rec = %q", rec)
		}
	})
	t.Run("first model promotes regardless", func(t *testing.T) {
		rec := retrainRecommendation(challenger, nil, nil, false, nil)
		if !strings.Contains(rec, "first production model") {
			t.Errorf("rec = %q", rec)
		}
	})
}
