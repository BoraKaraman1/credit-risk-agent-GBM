// Credit Risk Scoring API (Go port of api/scoring_service.py).
// Loads the exported champion model, queries Supabase for features,
// and returns a credit decision with TreeSHAP-based adverse action
// reasons and audit logging.
package inference

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"sync"
	"syscall"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Server timeouts. WriteTimeout is generous so large batch requests
// (many applicants, each a DB round-trip) are not cut off mid-response.
const (
	readHeaderTimeout = 5 * time.Second
	readTimeout       = 15 * time.Second
	writeTimeout      = 30 * time.Second
	idleTimeout       = 60 * time.Second
	shutdownTimeout   = 30 * time.Second
)

// ECOA adverse action: return top 4 reasons
const numAdverseActions = 4

// Feature staleness cutoff, mirroring the Python service.
const stalenessHours = 720

// Request limits. A caller must not be able to exhaust memory, goroutines,
// or DB connections cheaply: bodies are capped, batches are bounded, and
// applicant IDs are length- and charset-checked.
const (
	maxRequestBytes   = 1 << 20 // 1 MiB request body
	maxBatchSize      = 1000    // applicants per /score/batch call
	maxApplicantIDLen = 64
)

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

// ECOA / Regulation B adverse action reasons. Each Gold feature maps to
// the closest principal reason statement a lender may disclose on an
// adverse action notice (Reg B Appendix C model forms). The code is a
// stable internal numbering; several features can share one reason, since
// the notice discloses the reason, not the internal feature.
type adverseReason struct {
	Code   int
	Reason string
}

var featureAdverseReason = map[string]adverseReason{
	"loan_amnt":              {22, "Loan amount or terms relative to credit profile"},
	"term":                   {22, "Loan amount or terms relative to credit profile"},
	"int_rate":               {20, "Assigned credit risk grade"},
	"installment":            {23, "Monthly payment obligation too high"},
	"emp_length":             {15, "Length of employment"},
	"home_ownership":         {18, "Type of residence"},
	"annual_inc":             {1, "Income insufficient for amount of credit requested"},
	"verification_status":    {17, "Income or employment could not be verified"},
	"purpose":                {19, "Purpose of the requested credit"},
	"dti":                    {2, "Excessive obligations in relation to income"},
	"delinq_2yrs":            {5, "Delinquent past or present credit obligations"},
	"inq_last_6mths":         {7, "Number of recent inquiries on credit bureau report"},
	"mths_since_last_delinq": {6, "Time since most recent delinquency too short"},
	"open_acc":               {11, "Number of open or established accounts"},
	"pub_rec":                {13, "Public record information on file"},
	"revol_bal":              {10, "Level of revolving account balances"},
	"revol_util":             {9, "Proportion of revolving balances to credit limits too high"},
	"total_acc":              {11, "Number of open or established accounts"},
	"mort_acc":               {12, "Number of mortgage accounts"},
	"pub_rec_bankruptcies":   {14, "Bankruptcy reported on credit file"},
	"credit_history_months":  {8, "Length of credit history insufficient"},
	"fico_score":             {21, "Credit bureau score reflects elevated risk"},
	"emp_length_missing":     {16, "Length of employment not reported"},
	"log_annual_inc":         {1, "Income insufficient for amount of credit requested"},
	"loan_to_income":         {4, "Amount of credit requested high relative to income"},
	"installment_to_income":  {3, "Payment burden high relative to income"},
	"dti_x_income":           {2, "Excessive obligations in relation to income"},
	"grade_numeric":          {20, "Assigned credit risk grade"},
	"delinq_ever":            {5, "Delinquent past or present credit obligations"},
	"high_utilization":       {9, "Proportion of revolving balances to credit limits too high"},
	"has_mortgage":           {12, "Number of mortgage accounts"},
	"has_bankruptcy":         {14, "Bankruptcy reported on credit file"},
	"sub_grade_numeric":      {20, "Assigned credit risk grade"},
}

type adverseAction = db.AdverseAction

type scoreResponse struct {
	ApplicantID      string          `json:"applicant_id"`
	Score            float64         `json:"score"`
	PD               *float64        `json:"pd"`
	ScaledScore      *int            `json:"scaled_score"`
	Decision         string          `json:"decision"`
	ModelVersion     string          `json:"model_version"`
	FeatureVersion   int             `json:"feature_version"`
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
	Calibrated    bool   `json:"calibrated"`
	Database      string `json:"database"`
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
	db   scoringStore

	apiKeys        [][]byte
	requestLimiter *rateLimiter
	scoringLimiter *rateLimiter
	trustProxy     bool
}

// scoringStore is the serving boundary over Postgres. Keeping it narrow
// makes the fail-closed audit behavior testable without a live database.
type scoringStore interface {
	FetchApplicantFeatures(context.Context, string) (*db.ApplicantFeatures, error)
	InsertScoringLog(context.Context, db.ScoringAudit) error
	Ping(context.Context) error
	Close()
}

func (s *server) loadModel() error {
	m, err := model.Load(config.ChampionModelPath())
	if err != nil {
		return fmt.Errorf("%w (run pipeline/export_model_json.py to export the champion model)", err)
	}
	if err := checkModelGovernance(m); err != nil {
		return err
	}
	s.mu.Lock()
	s.model = m
	s.modelLoadedAt = time.Now().UTC().Format(time.RFC3339)
	s.mu.Unlock()
	setModelInfo(m.Version)
	slog.Info("model loaded", "version", m.Version, "n_features", m.NFeatures, "trees", len(m.Trees))
	return nil
}

// checkModelGovernance enforces the model-card verdict (SR 11-7): a
// champion whose validation status is not APPROVED must not serve
// production traffic unless an audited override (ALLOW_UNAPPROVED_MODEL)
// is supplied. A model exported without any validation_status (legacy or
// tampered) is treated as unapproved and fails closed for the same reason.
func checkModelGovernance(m *model.Model) error {
	if m.ValidationStatus != nil && m.ValidationStatus.Status == "APPROVED" {
		return nil
	}
	status, rationale := "missing", "model exported without a validation_status; re-run pipeline/export_model_json.py"
	if m.ValidationStatus != nil {
		status, rationale = m.ValidationStatus.Status, m.ValidationStatus.Rationale
	}
	if config.AllowUnapprovedModel() {
		slog.Warn("serving a non-APPROVED model under audited override (ALLOW_UNAPPROVED_MODEL=true)",
			"version", m.Version, "status", status, "rationale", rationale)
		return nil
	}
	return fmt.Errorf("model %s validation status is %q, not APPROVED: %s "+
		"(set ALLOW_UNAPPROVED_MODEL=true to serve it under documented sign-off)",
		m.Version, status, rationale)
}

func (s *server) currentModel() (*model.Model, string) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.model, s.modelLoadedAt
}

// getDB lazily connects, like the Python service's get_engine.
func (s *server) getDB(ctx context.Context) (scoringStore, error) {
	s.dbMu.Lock()
	defer s.dbMu.Unlock()
	if s.db != nil {
		return s.db, nil
	}
	d, err := db.Connect(ctx, config.DatabaseURL())
	if err != nil {
		logger(ctx).Error("database connection failed", "error", err)
		return nil, &httpError{500, "internal server error"}
	}
	s.db = d
	return d, nil
}

func round(x float64, decimals int) float64 {
	p := math.Pow(10, float64(decimals))
	return math.Round(x*p) / p
}

// decide maps a probability of default to a credit decision.
func decide(score float64) string {
	switch {
	case score < config.ApproveThreshold:
		return "approve"
	case score < config.ReviewThreshold:
		return "review"
	default:
		return "decline"
	}
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
	seenCodes := make(map[int]struct{}, numAdverseActions)
	for _, idx := range order {
		if shapValues[idx] <= 0 || len(actions) >= numAdverseActions {
			break
		}
		featName := m.Features[idx]
		display, ok := featureDisplayNames[featName]
		if !ok {
			display = featName
		}
		reason, ok := featureAdverseReason[featName]
		if !ok {
			reason = adverseReason{Code: 0, Reason: display}
		}
		if _, duplicate := seenCodes[reason.Code]; duplicate {
			continue
		}
		seenCodes[reason.Code] = struct{}{}
		featValue := 0.0
		if v := features[featName]; v != nil {
			featValue = *v
		}
		actions = append(actions, adverseAction{
			Code:         reason.Code,
			Reason:       reason.Reason,
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
		logger(ctx).Error("feature fetch failed", "applicant_id", applicantID, "error", err)
		return nil, &httpError{500, "internal server error"}
	}

	if feat.ComputedAt != nil {
		ageHours := time.Since(*feat.ComputedAt).Hours()
		if ageHours > stalenessHours {
			return nil, &httpError{409, fmt.Sprintf(
				"Features for %s are %.1fh old (stale). Trigger refresh.", applicantID, ageHours)}
		}
	}

	// Feature-store schema contract: the stored snapshot's feature_version
	// must match the version the model was exported against. A mismatch
	// means the feature definitions changed under the model, so reject
	// rather than score on incompatible inputs. Skipped when either side
	// predates the version field (0) for backward compatibility.
	if m.FeatureVersion != 0 && feat.FeatureVersion != 0 && m.FeatureVersion != feat.FeatureVersion {
		logger(ctx).Error("feature version mismatch", "applicant_id", applicantID,
			"store_version", feat.FeatureVersion, "model_version", m.Version, "model_feature_version", m.FeatureVersion)
		return nil, &httpError{409, fmt.Sprintf(
			"feature version mismatch for %s (store v%d, model expects v%d)",
			applicantID, feat.FeatureVersion, m.FeatureVersion)}
	}

	// Build the feature vector in the model's column order. Missing values
	// are preserved as NaN so the model's per-split missing routing fires
	// exactly as in training (both prediction and SHAP route NaN via
	// missing_go_to_left). A column absent from the snapshot — as opposed
	// to present-but-null — signals a feature-schema mismatch and is
	// rejected rather than silently scored as missing.
	x := make([]float64, len(m.Features))
	for i, col := range m.Features {
		v, present := feat.Features[col]
		if !present {
			logger(ctx).Error("feature schema mismatch", "applicant_id", applicantID,
				"missing_feature", col, "model_version", m.Version)
			return nil, &httpError{500, "internal server error"}
		}
		if v == nil {
			x[i] = math.NaN()
		} else {
			x[i] = *v
		}
	}

	// The raw score is logged and monitored (the drift reference is built
	// from raw scores). Credit decisions use the calibrated PD when the
	// model carries a calibrator — the 0.15/0.30 thresholds are
	// probabilities of default — and fall back to the raw score otherwise.
	score := m.PredictProba(x)
	decisionBasis := score

	var calibratedPD *float64
	var scaledScore *int
	if m.Calibration != nil {
		pd := m.Calibration.Apply(score)
		decisionBasis = pd
		rounded := round(pd, 5)
		calibratedPD = &rounded
		if m.Scorecard != nil {
			sc := m.Scorecard.Score(pd)
			scaledScore = &sc
		}
	}

	decision := decide(decisionBasis)

	// ECOA adverse action reasons (only for decline/review)
	actions := []adverseAction{}
	if decision == "decline" || decision == "review" {
		actions = computeAdverseActions(m, x, feat.Features)
	}

	// The scoring_log row is the ECOA/Reg B compliance artifact: a
	// decision that cannot be recorded is not rendered. Fail closed —
	// the caller gets a retryable 503, never an unaudited decision.
	logCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	resp := &scoreResponse{
		ApplicantID:      applicantID,
		Score:            round(score, 5),
		PD:               calibratedPD,
		ScaledScore:      scaledScore,
		Decision:         decision,
		ModelVersion:     m.Version,
		FeatureVersion:   feat.FeatureVersion,
		FicoScore:        feat.FicoScore,
		Grade:            feat.Grade,
		DataCompleteness: feat.DataCompleteness,
		AdverseActions:   actions,
	}
	audit := db.ScoringAudit{
		RequestID:       requestID(ctx),
		ApplicantID:     resp.ApplicantID,
		ModelVersion:    resp.ModelVersion,
		FeatureVersion:  resp.FeatureVersion,
		FeatureSnapshot: feat.Features,
		RawScore:        resp.Score,
		CalibratedPD:    resp.PD,
		ScaledScore:     resp.ScaledScore,
		Decision:        resp.Decision,
		AdverseActions:  resp.AdverseActions,
	}
	if err := d.InsertScoringLog(logCtx, audit); err != nil {
		logger(ctx).Error("audit write failed; decision withheld", "error", err)
		return nil, &httpError{http.StatusServiceUnavailable,
			"decision could not be audit-logged; retry"}
	}

	// Only audited decisions are observable as successful business
	// events. Failed audit writes must not inflate score or decision
	// metrics for decisions that were withheld.
	recordScore(resp.Score, resp.Decision)
	return resp, nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func writeError(w http.ResponseWriter, r *http.Request, err error) {
	var he *httpError
	if errors.As(err, &he) {
		writeJSON(w, he.status, map[string]string{"detail": he.detail})
		return
	}
	// Unexpected non-httpError: log the detail server-side and return a
	// generic message so internal state never reaches the client. The
	// X-Request-ID header (set by instrument) ties it to the server logs.
	logger(r.Context()).Error("unhandled request error", "error", err)
	writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "internal server error"})
}

// decodeBody reads a capped, strictly-typed JSON request body. Unknown
// fields are rejected, the body is limited to maxRequestBytes, and any
// trailing data after the first JSON value is an error.
func decodeBody(w http.ResponseWriter, r *http.Request, dst any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	if dec.More() {
		return errors.New("body must contain a single JSON object")
	}
	return nil
}

// validateApplicantID bounds the length and charset of an applicant ID so
// it cannot smuggle oversized or unexpected input into queries and logs.
func validateApplicantID(id string) error {
	if id == "" {
		return errors.New("applicant_id is required")
	}
	if len(id) > maxApplicantIDLen {
		return fmt.Errorf("applicant_id exceeds %d characters", maxApplicantIDLen)
	}
	for _, c := range id {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_', c == '-':
		default:
			return errors.New("applicant_id may contain only letters, digits, '_' and '-'")
		}
	}
	return nil
}

func (s *server) handleScore(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ApplicantID string `json:"applicant_id"`
	}
	if err := decodeBody(w, r, &req); err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]string{"detail": "applicant_id is required"})
		return
	}
	if err := validateApplicantID(req.ApplicantID); err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]string{"detail": err.Error()})
		return
	}
	if !s.allowScoring(w, r, 1) {
		writeJSON(w, http.StatusTooManyRequests, map[string]string{"detail": "rate limit exceeded"})
		return
	}
	resp, err := s.scoreApplicant(r.Context(), req.ApplicantID)
	if err != nil {
		writeError(w, r, err)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *server) handleScoreBatch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ApplicantIDs []string `json:"applicant_ids"`
	}
	if err := decodeBody(w, r, &req); err != nil {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]string{"detail": "applicant_ids is required"})
		return
	}
	if len(req.ApplicantIDs) == 0 {
		writeJSON(w, http.StatusUnprocessableEntity, map[string]string{"detail": "applicant_ids must not be empty"})
		return
	}
	if len(req.ApplicantIDs) > maxBatchSize {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{
			"detail": fmt.Sprintf("batch size %d exceeds maximum %d", len(req.ApplicantIDs), maxBatchSize)})
		return
	}
	out := batchScoreResponse{Results: []scoreResponse{}, Errors: []batchError{}}
	// Charge one rate-limit token per scored applicant: a batch is N
	// scoring decisions, not one request, so it must not multiply a
	// client's allowance by up to maxBatchSize. When the bucket empties
	// mid-batch the remainder fails fast instead of queueing behind a
	// slow database.
	for i, aid := range req.ApplicantIDs {
		if err := validateApplicantID(aid); err != nil {
			out.Errors = append(out.Errors, batchError{ApplicantID: aid, Error: err.Error()})
			continue
		}
		if !s.allowScoring(w, r, 1) {
			for _, rest := range req.ApplicantIDs[i:] {
				out.Errors = append(out.Errors, batchError{ApplicantID: rest, Error: "rate limit exceeded"})
			}
			writeJSON(w, http.StatusTooManyRequests, out)
			return
		}
		resp, err := s.scoreApplicant(r.Context(), aid)
		if err != nil {
			out.Errors = append(out.Errors, batchError{ApplicantID: aid, Error: err.Error()})
			continue
		}
		out.Results = append(out.Results, *resp)
	}
	writeJSON(w, http.StatusOK, out)
}

// allowScoring charges the business-operation limiter independently of
// the request limiter. A single score costs one token and a batch costs
// one token per attempted valid applicant, never N plus one.
func (s *server) allowScoring(w http.ResponseWriter, r *http.Request, n int) bool {
	if s.scoringLimiter == nil {
		return true
	}
	client := clientID(r, s.trustProxy)
	now := time.Now()
	limiter := s.scoringLimiter.get(client, now)
	if limiter.AllowN(now, n) {
		return true
	}
	logger(r.Context()).Warn("scoring rate limit exceeded", "client", client, "tokens", n)
	w.Header().Set("Retry-After", "1")
	return false
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	m, loadedAt := s.currentModel()
	if m == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"detail": "Model not loaded"})
		return
	}
	resp := healthResponse{
		Status:        "ok",
		ModelVersion:  m.Version,
		ModelLoadedAt: loadedAt,
		NFeatures:     m.NFeatures,
		Calibrated:    m.Calibration != nil,
		Database:      "not_configured",
	}
	// Readiness includes the feature store: a healthy model with a
	// missing or unreachable database cannot actually score, so never
	// answer 200 unless a /score could succeed.
	status := http.StatusOK
	if config.DatabaseURL() == "" {
		resp.Database, resp.Status, status = "not_configured", "degraded", http.StatusServiceUnavailable
	} else {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if d, err := s.getDB(ctx); err != nil {
			resp.Database, resp.Status, status = "unreachable", "degraded", http.StatusServiceUnavailable
		} else if err := d.Ping(ctx); err != nil {
			resp.Database, resp.Status, status = "unreachable", "degraded", http.StatusServiceUnavailable
		} else {
			resp.Database = "ok"
		}
	}
	writeJSON(w, status, resp)
}

func (s *server) handleReload(w http.ResponseWriter, r *http.Request) {
	if err := s.loadModel(); err != nil {
		writeError(w, r, err)
		return
	}
	m, _ := s.currentModel()
	writeJSON(w, http.StatusOK, map[string]string{"status": "reloaded", "model_version": m.Version})
}

func parseAPIKeys(keys []string) [][]byte {
	out := make([][]byte, len(keys))
	for i, k := range keys {
		out[i] = []byte(k)
	}
	return out
}

func Serve() {
	config.LoadEnv()

	s := &server{
		apiKeys: parseAPIKeys(config.APIKeys()),
		requestLimiter: newRateLimiter(
			config.RequestRateLimitRPS(), config.RequestRateLimitBurst(),
		),
		scoringLimiter: newRateLimiter(
			config.ScoringRateLimitRPS(), config.ScoringRateLimitBurst(),
		),
		trustProxy: config.TrustProxyHeaders(),
	}
	if err := s.loadModel(); err != nil {
		slog.Error("startup failed", "error", err)
		os.Exit(1)
	}
	// Fail fast on the feature store too: pgxpool connects lazily, so
	// without this probe a misconfigured instance would pass its health
	// check while every /score request fails.
	{
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		d, err := db.Connect(ctx, config.DatabaseURL())
		if err == nil {
			err = d.Ping(ctx)
		}
		cancel()
		if err != nil {
			slog.Error("startup failed: feature store unreachable", "error", err)
			os.Exit(1)
		}
		s.db = d
	}
	// Fail closed: refuse to start unauthenticated unless explicitly
	// allowed for local development.
	if len(s.apiKeys) == 0 {
		if !config.AllowUnauthenticatedDev() {
			slog.Error("refusing to start without authentication: set API_KEYS, " +
				"or ALLOW_UNAUTHENTICATED_DEV=true for local development only")
			os.Exit(1)
		}
		slog.Warn("API authentication DISABLED via ALLOW_UNAUTHENTICATED_DEV — do not use in production")
	} else {
		slog.Info("API authentication enabled", "keys", len(s.apiKeys))
	}

	mux := http.NewServeMux()
	mux.Handle("POST /score", instrument("/score", s.protect(http.HandlerFunc(s.handleScore))))
	mux.Handle("POST /score/batch", instrument("/score/batch", s.protect(http.HandlerFunc(s.handleScoreBatch))))
	mux.Handle("POST /reload", instrument("/reload", s.protect(http.HandlerFunc(s.handleReload))))
	mux.Handle("GET /health", instrument("/health", http.HandlerFunc(s.handleHealth)))
	// /metrics carries operational detail (score distribution, model
	// version), so it sits behind authentication too — but not the
	// per-client rate limiter, which would throttle Prometheus scrapes.
	mux.Handle("GET /metrics", instrument("/metrics", s.authMiddleware(promhttp.Handler())))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}
	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: readHeaderTimeout,
		ReadTimeout:       readTimeout,
		WriteTimeout:      writeTimeout,
		IdleTimeout:       idleTimeout,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		slog.Info("credit risk scoring API listening", "addr", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
	}()

	select {
	case err := <-errCh:
		slog.Error("server failed", "error", err)
		os.Exit(1)
	case <-ctx.Done():
		slog.Info("shutdown signal received, draining in-flight requests")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			slog.Error("graceful shutdown failed", "error", err)
			os.Exit(1)
		}
		// Release the feature-store pool after in-flight requests drain so
		// Postgres isn't left holding idle connections until they time out.
		if s.db != nil {
			s.db.Close()
		}
		slog.Info("server stopped cleanly")
	}
}
