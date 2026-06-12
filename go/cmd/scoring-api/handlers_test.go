package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/model"
)

// Handler and helper tests that need no database and no champion
// model on disk — they run on a synthetic two-feature model.

func syntheticAPIModel() *model.Model {
	tree := func(f int32, leftVal, rightVal float64) model.Tree {
		return model.Tree{
			Value:           []float64{0, leftVal, rightVal},
			Count:           []float64{100, 50, 50},
			FeatureIdx:      []int32{f, 0, 0},
			NumThreshold:    []float64{0.5, 0, 0},
			MissingGoToLeft: []uint8{1, 0, 0},
			Left:            []uint32{1, 0, 0},
			Right:           []uint32{2, 0, 0},
			IsLeaf:          []uint8{0, 1, 1},
		}
	}
	return &model.Model{
		FormatVersion:      1,
		Version:            "vtest",
		NFeatures:          2,
		Features:           []string{"dti", "fico_score"},
		BaselinePrediction: 0,
		Trees:              []model.Tree{tree(0, -1, 1), tree(1, -1, 1)},
	}
}

func TestRound(t *testing.T) {
	cases := []struct {
		name     string
		x        float64
		decimals int
		want     float64
	}{
		{"five decimals", 0.123456789, 5, 0.12346},
		{"four decimals", 38.20004, 4, 38.2},
		{"negative value", -0.000015, 5, -0.00002},
		{"integer unchanged", 42, 3, 42},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := round(tc.x, tc.decimals); got != tc.want {
				t.Errorf("round(%v, %d) = %v, want %v", tc.x, tc.decimals, got, tc.want)
			}
		})
	}
}

func TestFeatureDisplayNames(t *testing.T) {
	t.Run("covers all 33 gold features", func(t *testing.T) {
		if len(featureDisplayNames) != 33 {
			t.Errorf("mapping has %d entries, want 33", len(featureDisplayNames))
		}
	})
	t.Run("no empty display names", func(t *testing.T) {
		for raw, display := range featureDisplayNames {
			if display == "" {
				t.Errorf("feature %q has empty display name", raw)
			}
		}
	})
}

func TestComputeAdverseActionsSynthetic(t *testing.T) {
	m := syntheticAPIModel()
	val := func(v float64) *float64 { return &v }

	t.Run("positive contributions only", func(t *testing.T) {
		// dti high (risky), fico routes left (protective)
		x := []float64{1, 0}
		actions := computeAdverseActions(m, x, map[string]*float64{
			"dti": val(1), "fico_score": val(0),
		})
		if len(actions) != 1 {
			t.Fatalf("got %d actions, want 1: %+v", len(actions), actions)
		}
		if actions[0].FeatureName != "Debt-to-Income Ratio" {
			t.Errorf("feature = %q", actions[0].FeatureName)
		}
		if actions[0].ShapValue <= 0 {
			t.Errorf("shap = %v, want positive", actions[0].ShapValue)
		}
	})
	t.Run("no positive contributions yields empty list", func(t *testing.T) {
		x := []float64{0, 0} // both features protective
		actions := computeAdverseActions(m, x, map[string]*float64{
			"dti": val(0), "fico_score": val(0),
		})
		if len(actions) != 0 {
			t.Errorf("got %d actions, want 0", len(actions))
		}
	})
	t.Run("sorted by contribution descending", func(t *testing.T) {
		x := []float64{1, 1}
		actions := computeAdverseActions(m, x, map[string]*float64{
			"dti": val(1), "fico_score": val(1),
		})
		if len(actions) != 2 {
			t.Fatalf("got %d actions, want 2", len(actions))
		}
		if actions[0].ShapValue < actions[1].ShapValue {
			t.Error("actions not sorted descending")
		}
	})
	t.Run("nil feature value reported as zero", func(t *testing.T) {
		x := []float64{1, 0}
		actions := computeAdverseActions(m, x, map[string]*float64{"fico_score": val(0)})
		if len(actions) != 1 || actions[0].FeatureValue != 0 {
			t.Errorf("actions = %+v", actions)
		}
	})
}

func TestAdverseActionCap(t *testing.T) {
	// Five single-split trees on five features, all pushed risky:
	// five positive contributions must be capped at numAdverseActions.
	var trees []model.Tree
	features := []string{"a", "b", "c", "d", "e"}
	for i := range features {
		trees = append(trees, model.Tree{
			Value:           []float64{0, -1, 1},
			Count:           []float64{100, 50, 50},
			FeatureIdx:      []int32{int32(i), 0, 0},
			NumThreshold:    []float64{0.5, 0, 0},
			MissingGoToLeft: []uint8{1, 0, 0},
			Left:            []uint32{1, 0, 0},
			Right:           []uint32{2, 0, 0},
			IsLeaf:          []uint8{0, 1, 1},
		})
	}
	m := &model.Model{
		FormatVersion: 1, Version: "vtest", NFeatures: 5,
		Features: features, Trees: trees,
	}
	x := []float64{1, 1, 1, 1, 1}
	feats := map[string]*float64{}
	for _, f := range features {
		v := 1.0
		feats[f] = &v
	}
	actions := computeAdverseActions(m, x, feats)
	if len(actions) != numAdverseActions {
		t.Errorf("got %d actions, want cap of %d", len(actions), numAdverseActions)
	}
}

func postJSON(t *testing.T, handler http.HandlerFunc, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(body))
	rec := httptest.NewRecorder()
	handler(rec, req)
	return rec
}

func decodeDetail(t *testing.T, rec *httptest.ResponseRecorder) string {
	t.Helper()
	var body map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("response is not a JSON object: %s", rec.Body.String())
	}
	return body["detail"]
}

func TestHandleHealth(t *testing.T) {
	t.Run("503 when model not loaded", func(t *testing.T) {
		s := &server{}
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		rec := httptest.NewRecorder()
		s.handleHealth(rec, req)
		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want 503", rec.Code)
		}
	})
	t.Run("200 with model info", func(t *testing.T) {
		s := &server{model: syntheticAPIModel(), modelLoadedAt: "2026-01-01T00:00:00Z"}
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		rec := httptest.NewRecorder()
		s.handleHealth(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d", rec.Code)
		}
		var resp healthResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.ModelVersion != "vtest" || resp.NFeatures != 2 || resp.Status != "ok" {
			t.Errorf("resp = %+v", resp)
		}
	})
}

func TestHandleScoreValidation(t *testing.T) {
	s := &server{model: syntheticAPIModel()}
	cases := []struct {
		name string
		body string
	}{
		{"invalid json", `{nope`},
		{"missing applicant_id", `{}`},
		{"empty applicant_id", `{"applicant_id": ""}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := postJSON(t, s.handleScore, tc.body)
			if rec.Code != http.StatusUnprocessableEntity {
				t.Errorf("status = %d, want 422", rec.Code)
			}
			if decodeDetail(t, rec) == "" {
				t.Error("error response has no detail")
			}
		})
	}
	t.Run("503 when model not loaded", func(t *testing.T) {
		rec := postJSON(t, (&server{}).handleScore, `{"applicant_id": "LC_1"}`)
		if rec.Code != http.StatusServiceUnavailable {
			t.Errorf("status = %d, want 503", rec.Code)
		}
	})
	t.Run("500 without DATABASE_URL", func(t *testing.T) {
		t.Setenv("DATABASE_URL", "")
		rec := postJSON(t, s.handleScore, `{"applicant_id": "LC_1"}`)
		if rec.Code != http.StatusInternalServerError {
			t.Errorf("status = %d, want 500", rec.Code)
		}
		if !strings.Contains(decodeDetail(t, rec), "DATABASE_URL") {
			t.Errorf("detail = %q", decodeDetail(t, rec))
		}
	})
}

func TestHandleScoreBatchValidation(t *testing.T) {
	s := &server{model: syntheticAPIModel()}
	t.Run("invalid json", func(t *testing.T) {
		rec := postJSON(t, s.handleScoreBatch, `{nope`)
		if rec.Code != http.StatusUnprocessableEntity {
			t.Errorf("status = %d, want 422", rec.Code)
		}
	})
	t.Run("failures land in errors not results", func(t *testing.T) {
		t.Setenv("DATABASE_URL", "")
		rec := postJSON(t, s.handleScoreBatch, `{"applicant_ids": ["LC_1", "LC_2"]}`)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d", rec.Code)
		}
		var resp batchScoreResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if len(resp.Results) != 0 || len(resp.Errors) != 2 {
			t.Errorf("results=%d errors=%d, want 0/2", len(resp.Results), len(resp.Errors))
		}
		if resp.Errors[0].ApplicantID != "LC_1" {
			t.Errorf("errors[0] = %+v", resp.Errors[0])
		}
	})
}

func TestHandleReloadFailure(t *testing.T) {
	t.Setenv("CREDIT_RISK_MODELS_DIR", t.TempDir()) // no model.json there
	s := &server{}
	req := httptest.NewRequest(http.MethodPost, "/reload", nil)
	rec := httptest.NewRecorder()
	s.handleReload(rec, req)
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", rec.Code)
	}
	if detail := decodeDetail(t, rec); !strings.Contains(detail, "export_model_json") {
		t.Errorf("detail should point at the export script, got %q", detail)
	}
}

func TestDecide(t *testing.T) {
	cases := []struct {
		name  string
		score float64
		want  string
	}{
		{"low risk approves", 0.05, "approve"},
		{"just under approve threshold", 0.1499, "approve"},
		{"approve threshold reviews", 0.15, "review"},
		{"just under review threshold", 0.2999, "review"},
		{"review threshold declines", 0.30, "decline"},
		{"high risk declines", 0.95, "decline"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := decide(tc.score); got != tc.want {
				t.Errorf("decide(%v) = %s, want %s", tc.score, got, tc.want)
			}
		})
	}
}
