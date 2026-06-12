// Credit Risk Scoring API (Go port of api/scoring_service.py).
// Loads the exported champion model, queries Supabase for features,
// and returns a credit decision with TreeSHAP-based adverse action
// reasons and audit logging.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/internal/model"
)

// ECOA adverse action: return top 4 reasons
const numAdverseActions = 4

// Feature staleness cutoff, mirroring the Python service.
const stalenessHours = 720

// Human-readable feature name mapping (all 33 Gold features)
var featureDisplayNames = map[string]string{
	"loan_amnt":              "Loan Amount",
	"term":                   "Loan Term",
	"int_rate":               "Interest Rate",
	"installment":            "Monthly Installment",
	"emp_length":             "Employment Length",
	"home_ownership":         "Home Ownership Status",
	"annual_inc":             "Annual Income",
	"verification_status":    "Income Verification Status",
	"purpose":                "Loan Purpose",
	"dti":                    "Debt-to-Income Ratio",
	"delinq_2yrs":            "Delinquencies in Last 2 Years",
	"inq_last_6mths":         "Credit Inquiries in Last 6 Months",
	"mths_since_last_delinq": "Months Since Last Delinquency",
	"open_acc":               "Number of Open Accounts",
	"pub_rec":                "Public Records",
	"revol_bal":              "Revolving Balance",
	"revol_util":             "Revolving Utilization Rate",
	"total_acc":              "Total Number of Accounts",
	"mort_acc":               "Number of Mortgage Accounts",
	"pub_rec_bankruptcies":   "Public Record Bankruptcies",
	"credit_history_months":  "Length of Credit History",
	"fico_score":             "FICO Score",
	"emp_length_missing":     "Employment Length Not Reported",
	"log_annual_inc":         "Log Annual Income",
	"loan_to_income":         "Loan-to-Income Ratio",
	"installment_to_income":  "Installment-to-Income Ratio",
	"dti_x_income":           "Absolute Debt Burden",
	"grade_numeric":          "Credit Grade",
	"delinq_ever":            "Prior Delinquency Flag",
	"high_utilization":       "High Credit Utilization Flag",
	"has_mortgage":           "Mortgage Account Flag",
	"has_bankruptcy":         "Bankruptcy on Record",
	"sub_grade_numeric":      "Credit Sub-Grade",
}

type adverseAction struct {
	FeatureName  string  `json:"feature_name"`
	ShapValue    float64 `json:"shap_value"`
	FeatureValue float64 `json:"feature_value"`
	Direction    string  `json:"direction"`
}

type scoreResponse struct {
	ApplicantID      string          `json:"applicant_id"`
	Score            float64         `json:"score"`
	Decision         string          `json:"decision"`
	ModelVersion     string          `json:"model_version"`
	FicoScore        *int            `json:"fico_score"`
	Grade            *int            `json:"grade"`
	DataCompleteness *float64        `json:"data_completeness"`
	AdverseActions   []adverseAction `json:"adverse_actions"`
}

type batchError struct {
	ApplicantID string `json:"applicant_id"`
	Error       string `json:"error"`
}

type batchScoreResponse struct {
	Results []scoreResponse `json:"results"`
	Errors  []batchError    `json:"errors"`
}

type healthResponse struct {
	Status        string `json:"status"`
	ModelVersion  string `json:"model_version"`
	ModelLoadedAt string `json:"model_loaded_at"`
	NFeatures     int    `json:"n_features"`
}

// httpError mirrors FastAPI's HTTPException ({"detail": ...} body).
type httpError struct {
	status int
	detail string
}

func (e *httpError) Error() string { return e.detail }

type server struct {
	mu            sync.RWMutex
	model         *model.Model
	modelLoadedAt string

	dbMu sync.Mutex
	db   *db.DB
}

func (s *server) loadModel() error {
	m, err := model.Load(config.ChampionModelPath())
	if err != nil {
		return fmt.Errorf("%w (run pipeline/export_model_json.py to export the champion model)", err)
	}
	s.mu.Lock()
	s.model = m
	s.modelLoadedAt = time.Now().UTC().Format(time.RFC3339)
	s.mu.Unlock()
	slog.Info("model loaded", "version", m.Version, "n_features", m.NFeatures, "trees", len(m.Trees))
	return nil
}

func (s *server) currentModel() (*model.Model, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.model, s.modelLoadedAt
}

// getDB lazily connects, like the Python service's get_engine.
func (s *server) getDB(ctx context.Context) (*db.DB, error) {
	s.dbMu.Lock()
	defer s.dbMu.Unlock()
	if s.db != nil {
		return s.db, nil
	}
	d, err := db.Connect(ctx, config.DatabaseURL())
	if err != nil {
		return nil, &httpError{500, err.Error()}
	}
	s.db = d
	return d, nil
}

func round(x float64, decimals int) float64 {
	p := math.Pow(10, float64(decimals))
	return math.Round(x*p) / p
}

func computeAdverseActions(m *model.Model, x []float64, features map[string]*float64) []adverseAction {
	shapValues := m.ShapValues(x)
	order := make([]int, len(shapValues))
	for i := range order {
		order[i] = i
	}
	// Positive SHAP values push toward default (class 1) -> increase risk
	sort.Slice(order, func(a, b int) bool { return shapValues[order[a]] > shapValues[order[b]] })

	actions := []adverseAction{}
	for _, idx := range order {
		if shapValues[idx] <= 0 || len(actions) >= numAdverseActions {
			break
		}
		featName := m.Features[idx]
		display, ok := featureDisplayNames[featName]
		if !ok {
			display = featName
		}
		featValue := 0.0
		if v := features[featName]; v != nil {
			featValue = *v
		}
		actions = append(actions, adverseAction{
			FeatureName:  display,
			ShapValue:    round(shapValues[idx], 5),
			FeatureValue: round(featValue, 4),
			Direction:    "increases risk",
		})
	}
	return actions
}

// scoreApplicant: fetch features -> predict -> decide -> explain -> log.
func (s *server) scoreApplicant(ctx context.Context, applicantID string) (*scoreResponse, error) {
	m, _ := s.currentModel()
	if m == nil {
		return nil, &httpError{503, "Model not loaded"}
	}

	d, err := s.getDB(ctx)
	if err != nil {
		return nil, err
	}
	feat, err := d.FetchApplicantFeatures(ctx, applicantID)
	if errors.Is(err, db.ErrNotFound) {
		return nil, &httpError{404, fmt.Sprintf("Applicant %s not found in feature store", applicantID)}
	}
	if err != nil {
		return nil, &httpError{500, err.Error()}
	}

	if feat.ComputedAt != nil {
		ageHours := time.Since(*feat.ComputedAt).Hours()
		if ageHours > stalenessHours {
			return nil, &httpError{409, fmt.Sprintf(
				"Features for %s are %.1fh old (stale). Trigger refresh.", applicantID, ageHours)}
		}
	}

	// Build feature vector in the model's column order; missing/null -> 0.0
	x := make([]float64, len(m.Features))
	for i, col := range m.Features {
		if v := feat.Features[col]; v != nil {
			x[i] = *v
		}
	}

	score := m.PredictProba(x)

	var decision string
	switch {
	case score < config.ApproveThreshold:
		decision = "approve"
	case score < config.ReviewThreshold:
		decision = "review"
	default:
		decision = "decline"
	}

	// ECOA adverse action reasons (only for decline/review)
	actions := []adverseAction{}
	if decision == "decline" || decision == "review" {
		actions = computeAdverseActions(m, x, feat.Features)
	}

	// Audit log; failures are logged but never block the decision.
	logCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := d.InsertScoringLog(logCtx, applicantID, m.Version, feat.Features, score, decision); err != nil {
		slog.Warn("failed to log scoring decision", "error", err)
	}

	return &scoreResponse{
		ApplicantID:      applicantID,
		Score:            round(score, 5),
		Decision:         decision,
		ModelVersion:     m.Version,
		FicoScore:        feat.FicoScore,
		Grade:            feat.Grade,
		DataCompleteness: feat.DataCompleteness,
		AdverseActions:   actions,
	}, nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, err error) {
	var he *httpError
	if errors.As(err, &he) {
		writeJSON(w, he.status, map[string]string{"detail": he.detail})
		return
	}
	writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": err.Error()})
}

func (s *server) handleScore(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ApplicantID string `json:"applicant_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.ApplicantID == "" {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]string{"detail": "applicant_id is required"})
		return
	}
	resp, err := s.scoreApplicant(r.Context(), req.ApplicantID)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *server) handleScoreBatch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ApplicantIDs []string `json:"applicant_ids"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]string{"detail": "applicant_ids is required"})
		return
	}
	out := batchScoreResponse{Results: []scoreResponse{}, Errors: []batchError{}}
	for _, aid := range req.ApplicantIDs {
		resp, err := s.scoreApplicant(r.Context(), aid)
		if err != nil {
			out.Errors = append(out.Errors, batchError{ApplicantID: aid, Error: err.Error()})
			continue
		}
		out.Results = append(out.Results, *resp)
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	m, loadedAt := s.currentModel()
	if m == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"detail": "Model not loaded"})
		return
	}
	writeJSON(w, http.StatusOK, healthResponse{
		Status:        "ok",
		ModelVersion:  m.Version,
		ModelLoadedAt: loadedAt,
		NFeatures:     m.NFeatures,
	})
}

func (s *server) handleReload(w http.ResponseWriter, r *http.Request) {
	if err := s.loadModel(); err != nil {
		writeError(w, err)
		return
	}
	m, _ := s.currentModel()
	writeJSON(w, http.StatusOK, map[string]string{"status": "reloaded", "model_version": m.Version})
}

func main() {
	config.LoadEnv()

	s := &server{}
	if err := s.loadModel(); err != nil {
		slog.Error("startup failed", "error", err)
		os.Exit(1)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /score", s.handleScore)
	mux.HandleFunc("POST /score/batch", s.handleScoreBatch)
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("POST /reload", s.handleReload)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}
	addr := ":" + port
	slog.Info("credit risk scoring API listening", "addr", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		slog.Error("server stopped", "error", err)
		os.Exit(1)
	}
}
