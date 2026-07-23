package inference

import (
	"math"
	"os"
	"path/filepath"
	"strings"
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

func TestLoadModelRejectsStaleChampionAndSerializesReload(t *testing.T) {
	modelsDir := t.TempDir()
	t.Setenv("CREDIT_RISK_MODELS_DIR", modelsDir)

	versionsDir := filepath.Join(modelsDir, "versions")
	for _, version := range []string{"v1", "v2"} {
		dir := filepath.Join(versionsDir, version)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "model.json"), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	champion := filepath.Join(modelsDir, "champion")
	if err := os.Symlink(filepath.Join("versions", "v2"), champion); err != nil {
		t.Fatal(err)
	}

	v2Started := make(chan struct{})
	releaseV2 := make(chan struct{})
	v1Started := make(chan struct{})
	approved := &model.ValidationStatus{Status: "APPROVED"}
	s := &server{
		model: &model.Model{Version: "v1"},
		loadModelFile: func(path string) (*model.Model, error) {
			switch {
			case strings.Contains(path, filepath.Join("versions", "v2")):
				close(v2Started)
				<-releaseV2
				return &model.Model{Version: "v2", ValidationStatus: approved}, nil
			case strings.Contains(path, filepath.Join("versions", "v1")):
				close(v1Started)
				return &model.Model{Version: "v1", ValidationStatus: approved}, nil
			default:
				return nil, &os.PathError{Op: "load", Path: path, Err: os.ErrNotExist}
			}
		},
	}

	firstErr := make(chan error, 1)
	go func() { firstErr <- s.loadModel() }()
	<-v2Started

	next := filepath.Join(modelsDir, ".champion.rollback")
	if err := os.Symlink(filepath.Join("versions", "v1"), next); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(next, champion); err != nil {
		t.Fatal(err)
	}

	secondErr := make(chan error, 1)
	go func() { secondErr <- s.loadModel() }()

	// The rollback load cannot enter the model loader while the activation
	// load owns reloadMu.
	select {
	case <-v1Started:
		t.Fatal("rollback reload ran concurrently with activation reload")
	default:
	}

	close(releaseV2)
	if err := <-firstErr; err == nil || !strings.Contains(err.Error(), "champion changed during reload") {
		t.Fatalf("stale activation error = %v", err)
	}
	if err := <-secondErr; err != nil {
		t.Fatalf("rollback reload: %v", err)
	}
	if got, _ := s.currentModel(); got.Version != "v1" {
		t.Fatalf("served model = %q, want rolled-back v1", got.Version)
	}
}
