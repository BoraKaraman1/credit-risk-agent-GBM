package inference

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
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

func TestFeatureAdverseReason(t *testing.T) {
	t.Run("covers all 33 gold features", func(t *testing.T) {
		for raw := range featureDisplayNames {
			if _, ok := featureAdverseReason[raw]; !ok {
				t.Errorf("feature %q has no adverse action reason", raw)
			}
		}
		if len(featureAdverseReason) != len(featureDisplayNames) {
			t.Errorf("adverse reason map has %d entries, display names %d",
				len(featureAdverseReason), len(featureDisplayNames))
		}
	})
	t.Run("every reason has a code and text", func(t *testing.T) {
		for raw, r := range featureAdverseReason {
			if r.Code <= 0 || r.Reason == "" {
				t.Errorf("feature %q has invalid reason %+v", raw, r)
			}
		}
	})
}

func TestComputeAdverseActionsSynthetic(t *testing.T) {
	m := syntheticAPIModel()
	val := func(v float64) *float64 { return &v }

	t.Run("attaches ECOA code and reason", func(t *testing.T) {
		x := []float64{1, 0}
		actions := computeAdverseActions(m, x, map[string]*float64{"dti": val(1), "fico_score": val(0)})
		if len(actions) != 1 {
			t.Fatalf("got %d actions, want 1", len(actions))
		}
		if actions[0].Code != featureAdverseReason["dti"].Code {
			t.Errorf("code = %d, want %d", actions[0].Code, featureAdverseReason["dti"].Code)
		}
		if actions[0].Reason != featureAdverseReason["dti"].Reason {
			t.Errorf("reason = %q", actions[0].Reason)
		}
	})

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
	t.Run("degraded without a feature store", func(t *testing.T) {
		// A loaded model with no DATABASE_URL cannot score, so health
		// must not answer 200 — a load balancer would route to it.
		s := &server{model: syntheticAPIModel(), modelLoadedAt: "2026-01-01T00:00:00Z"}
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		rec := httptest.NewRecorder()
		s.handleHealth(rec, req)
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("status = %d, want 503", rec.Code)
		}
		var resp healthResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if resp.ModelVersion != "vtest" || resp.NFeatures != 2 ||
			resp.Status != "degraded" || resp.Database != "not_configured" {
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
	t.Run("500 without DATABASE_URL returns a generic message", func(t *testing.T) {
		t.Setenv("DATABASE_URL", "")
		rec := postJSON(t, s.handleScore, `{"applicant_id": "LC_1"}`)
		if rec.Code != http.StatusInternalServerError {
			t.Errorf("status = %d, want 500", rec.Code)
		}
		detail := decodeDetail(t, rec)
		if strings.Contains(detail, "DATABASE_URL") {
			t.Errorf("internal detail leaked to client: %q", detail)
		}
		if detail != "internal server error" {
			t.Errorf("detail = %q, want generic message", detail)
		}
	})
}

func TestHandleScoreBatchValidation(t *testing.T) {
	s := &server{model: syntheticAPIModel(), limiter: newRateLimiter(100, 100)}
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
	// The export-script hint goes to the server logs; the client gets a
	// generic message (no filesystem paths leaked).
	if detail := decodeDetail(t, rec); detail != "internal server error" {
		t.Errorf("reload failure should return a generic message, got %q", detail)
	}
}

func TestValidateApplicantID(t *testing.T) {
	for _, id := range []string{"LC_1", "LC_0000001", "abc-123", "A_b-9"} {
		if err := validateApplicantID(id); err != nil {
			t.Errorf("validateApplicantID(%q) = %v, want nil", id, err)
		}
	}
	for _, id := range []string{"", "has space", "semi;colon", "drop'table", strings.Repeat("x", maxApplicantIDLen+1)} {
		if err := validateApplicantID(id); err == nil {
			t.Errorf("validateApplicantID(%q) = nil, want error", id)
		}
	}
}

func TestHandleScoreUnknownField(t *testing.T) {
	s := &server{model: syntheticAPIModel()}
	rec := postJSON(t, s.handleScore, `{"applicant_id":"LC_1","surprise":1}`)
	if rec.Code != http.StatusUnprocessableEntity {
		t.Errorf("status = %d, want 422 for unknown field", rec.Code)
	}
}

func TestHandleScoreTrailingData(t *testing.T) {
	s := &server{model: syntheticAPIModel()}
	rec := postJSON(t, s.handleScore, `{"applicant_id":"LC_1"} {"applicant_id":"LC_2"}`)
	if rec.Code != http.StatusUnprocessableEntity {
		t.Errorf("status = %d, want 422 for trailing data after the JSON object", rec.Code)
	}
}

func TestHandleScoreBatchLimits(t *testing.T) {
	s := &server{model: syntheticAPIModel(), limiter: newRateLimiter(100, 100)}
	t.Run("empty list is 422", func(t *testing.T) {
		rec := postJSON(t, s.handleScoreBatch, `{"applicant_ids":[]}`)
		if rec.Code != http.StatusUnprocessableEntity {
			t.Errorf("status = %d, want 422", rec.Code)
		}
	})
	t.Run("oversized batch is 413", func(t *testing.T) {
		ids := make([]string, maxBatchSize+1)
		for i := range ids {
			ids[i] = "LC_1"
		}
		body, _ := json.Marshal(map[string][]string{"applicant_ids": ids})
		rec := postJSON(t, s.handleScoreBatch, string(body))
		if rec.Code != http.StatusRequestEntityTooLarge {
			t.Errorf("status = %d, want 413", rec.Code)
		}
	})
}

// A batch is N scoring decisions, not one request: each scored applicant
// must consume a rate-limit token, or /score/batch is a maxBatchSize
// multiplier on every client's allowance.
func TestHandleScoreBatchChargesPerApplicant(t *testing.T) {
	t.Setenv("DATABASE_URL", "") // scoring itself fails fast; tokens still burn
	// burst=2, no refill: the first two applicants consume the bucket.
	s := &server{model: syntheticAPIModel(), limiter: newRateLimiter(0.0001, 2)}
	rec := postJSON(t, s.handleScoreBatch, `{"applicant_ids": ["LC_1", "LC_2", "LC_3", "LC_4"]}`)
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429 once the bucket empties", rec.Code)
	}
	if rec.Header().Get("Retry-After") == "" {
		t.Error("429 without Retry-After header")
	}
	var resp batchScoreResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	// First two attempted (DB error), remaining two failed fast on the limiter.
	if len(resp.Errors) != 4 {
		t.Fatalf("errors = %d, want 4 (2 DB + 2 rate-limited)", len(resp.Errors))
	}
	if resp.Errors[2].Error != "rate limit exceeded" || resp.Errors[3].Error != "rate limit exceeded" {
		t.Errorf("tail errors = %+v, want rate-limit failures for LC_3/LC_4", resp.Errors[2:])
	}
	if resp.Errors[2].ApplicantID != "LC_3" || resp.Errors[3].ApplicantID != "LC_4" {
		t.Errorf("rate-limited IDs = %+v, want LC_3/LC_4 in order", resp.Errors[2:])
	}
}

func TestCheckModelGovernance(t *testing.T) {
	approved := &model.Model{Version: "v1", ValidationStatus: &model.ValidationStatus{Status: "APPROVED"}}
	if err := checkModelGovernance(approved); err != nil {
		t.Errorf("APPROVED model blocked: %v", err)
	}
	// A model without an embedded validation_status fails closed.
	missing := &model.Model{Version: "v1"}
	if err := checkModelGovernance(missing); err == nil {
		t.Error("model with missing validation_status should be blocked without override")
	}
	review := &model.Model{Version: "v1",
		ValidationStatus: &model.ValidationStatus{Status: "REVIEW REQUIRED", Rationale: "DIR violations"}}
	if err := checkModelGovernance(review); err == nil {
		t.Error("REVIEW REQUIRED model should be blocked without override")
	}
	t.Run("audited override allows unapproved and missing", func(t *testing.T) {
		t.Setenv("ALLOW_UNAPPROVED_MODEL", "true")
		if err := checkModelGovernance(review); err != nil {
			t.Errorf("override should allow REVIEW REQUIRED: %v", err)
		}
		if err := checkModelGovernance(missing); err != nil {
			t.Errorf("override should allow missing status: %v", err)
		}
	})
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
