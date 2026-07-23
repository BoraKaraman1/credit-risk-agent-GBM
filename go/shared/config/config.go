// Package config mirrors agents/config.py: shared paths, thresholds,
// and environment loading for the Go services.
package config

import (
	_ "embed"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
)

// contract.json is the single source of truth for the thresholds shared
// with the Python pipeline (pipeline/config.py) and the UI (ui/core.py).
// It is embedded at compile time, so the binary stays self-contained and
// a malformed contract fails the process at init, never at decision time.
//
//go:embed contract.json
var contractJSON []byte

type thresholdContract struct {
	Decision struct {
		ApproveBelow float64 `json:"approve_below"`
		ReviewBelow  float64 `json:"review_below"`
	} `json:"decision"`
	Monitoring struct {
		PSIWarning       float64 `json:"psi_warning"`
		PSICritical      float64 `json:"psi_critical"`
		CSIThreshold     float64 `json:"csi_threshold"`
		AUCDropThreshold float64 `json:"auc_drop_threshold"`
		MinDriftScores   int     `json:"min_drift_scores"`
	} `json:"monitoring"`
	Fairness struct {
		DIRThreshold       float64 `json:"dir_threshold"`
		DIRWorsenTolerance float64 `json:"dir_worsen_tolerance"`
	} `json:"fairness"`
}

func mustParseContract(b []byte) thresholdContract {
	var c thresholdContract
	if err := json.Unmarshal(b, &c); err != nil {
		panic("config: invalid contract.json: " + err.Error())
	}
	if !(0 < c.Decision.ApproveBelow && c.Decision.ApproveBelow < c.Decision.ReviewBelow && c.Decision.ReviewBelow < 1) ||
		!(0 < c.Monitoring.PSIWarning && c.Monitoring.PSIWarning < c.Monitoring.PSICritical) ||
		c.Monitoring.CSIThreshold <= 0 || c.Monitoring.AUCDropThreshold <= 0 ||
		c.Monitoring.MinDriftScores <= 0 ||
		!(0 < c.Fairness.DIRThreshold && c.Fairness.DIRThreshold <= 1) ||
		!(0 <= c.Fairness.DIRWorsenTolerance && c.Fairness.DIRWorsenTolerance < c.Fairness.DIRThreshold) {
		panic("config: contract.json thresholds fail sanity checks")
	}
	return c
}

var contract = mustParseContract(contractJSON)

// Drift thresholds
var (
	PSIWarning     = contract.Monitoring.PSIWarning     // score distribution drift warning
	PSICritical    = contract.Monitoring.PSICritical    // score distribution drift -> retrain
	CSIThreshold   = contract.Monitoring.CSIThreshold   // per-feature characteristic stability index
	MinDriftScores = contract.Monitoring.MinDriftScores // minimum real scores before automated drift action
)

// Performance thresholds
var AUCDropThreshold = contract.Monitoring.AUCDropThreshold // AUC drop from training -> retrain

// Fairness: 80% rule (four-fifths) for the Disparate Impact Ratio, and
// how much a champion-relative gate lets an inherited violation worsen
// before treating it as a regression (guards against retrain noise).
var (
	FairnessDIRThreshold    = contract.Fairness.DIRThreshold
	FairnessWorsenTolerance = contract.Fairness.DIRWorsenTolerance
)

// Outcome backfill: a scored applicant's outcome is treated as observed
// only after this many days have passed since scoring (loan maturation).
const defaultOutcomeBackfillDelayDays = 0

// Decision thresholds
var (
	ApproveThreshold = contract.Decision.ApproveBelow
	ReviewThreshold  = contract.Decision.ReviewBelow
)

// Per-client request and scoring rate-limit defaults. The first bounds
// authenticated HTTP request overhead; the second charges decisions,
// including one token for every applicant in a batch.
const (
	defaultRequestRateLimitRPS   = 50.0
	defaultRequestRateLimitBurst = 100
	defaultScoringRateLimitRPS   = 20.0
	defaultScoringRateLimitBurst = 40
)

// LoadEnv loads .env if present (like python-dotenv, existing
// environment variables win).
func LoadEnv() {
	_ = godotenv.Load()
}

func DataDir() string {
	if d := os.Getenv("CREDIT_RISK_DATA_DIR"); d != "" {
		return d
	}
	return "data"
}

func GoldDir() string { return filepath.Join(DataDir(), "gold") }

func ModelsDir() string {
	if d := os.Getenv("CREDIT_RISK_MODELS_DIR"); d != "" {
		return d
	}
	return filepath.Join(DataDir(), "models")
}

func ChampionDir() string   { return filepath.Join(ModelsDir(), "champion") }
func ChallengerDir() string { return filepath.Join(ModelsDir(), "challenger") }

func ChampionModelPath() string   { return filepath.Join(ChampionDir(), "model.json") }
func ChallengerModelPath() string { return filepath.Join(ChallengerDir(), "model.json") }

func DatabaseURL() string { return os.Getenv("DATABASE_URL") }

// ScoringAPIURL is the base URL of the running scoring API. Online
// promotion preflights and verifies /reload against this endpoint, and
// rolls the registry back if activation fails.
func ScoringAPIURL() string { return os.Getenv("SCORING_API_URL") }

// boolEnv reports whether an environment variable is set to a truthy
// value (1/true/yes, case-insensitive).
func boolEnv(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes":
		return true
	}
	return false
}

// AllowUnauthenticatedDev permits the scoring API to start without any
// API_KEYS configured. Off by default so the API fails closed in
// production; intended only for local development.
func AllowUnauthenticatedDev() bool { return boolEnv("ALLOW_UNAUTHENTICATED_DEV") }

// AllowUnapprovedModel is the audited override that lets the serving
// governance gate load a champion whose model card is not APPROVED
// (SR 11-7). Off by default so non-APPROVED models fail closed.
func AllowUnapprovedModel() bool { return boolEnv("ALLOW_UNAPPROVED_MODEL") }

// TrustProxyHeaders reports whether the X-Forwarded-For header should be
// trusted to determine the client IP. Enable only when the API sits behind
// a trusted reverse proxy / load balancer; off by default so a direct
// client cannot spoof its identity to evade per-client rate limiting.
func TrustProxyHeaders() bool { return boolEnv("TRUST_PROXY_HEADERS") }

// APIKeys returns the accepted API keys from API_KEYS (comma-separated).
// An empty slice means authentication is disabled.
func APIKeys() []string {
	raw := os.Getenv("API_KEYS")
	if raw == "" {
		return nil
	}
	var keys []string
	for _, k := range strings.Split(raw, ",") {
		if k = strings.TrimSpace(k); k != "" {
			keys = append(keys, k)
		}
	}
	return keys
}

func positiveFloatEnv(key string, fallback float64) float64 {
	if f, err := strconv.ParseFloat(os.Getenv(key), 64); err == nil && f > 0 {
		return f
	}
	return fallback
}

func positiveIntEnv(key string, fallback int) int {
	if n, err := strconv.Atoi(os.Getenv(key)); err == nil && n > 0 {
		return n
	}
	return fallback
}

func RequestRateLimitRPS() float64 {
	return positiveFloatEnv("REQUEST_RATE_LIMIT_RPS", defaultRequestRateLimitRPS)
}

func RequestRateLimitBurst() int {
	return positiveIntEnv("REQUEST_RATE_LIMIT_BURST", defaultRequestRateLimitBurst)
}

// ScoringRateLimitRPS is the per-client decision rate. RATE_LIMIT_RPS
// remains a compatibility fallback for existing deployments.
func ScoringRateLimitRPS() float64 {
	if os.Getenv("SCORING_RATE_LIMIT_RPS") != "" {
		return positiveFloatEnv("SCORING_RATE_LIMIT_RPS", defaultScoringRateLimitRPS)
	}
	return positiveFloatEnv("RATE_LIMIT_RPS", defaultScoringRateLimitRPS)
}

// ScoringRateLimitBurst is the decision bucket depth. RATE_LIMIT_BURST
// remains a compatibility fallback for existing deployments.
func ScoringRateLimitBurst() int {
	if os.Getenv("SCORING_RATE_LIMIT_BURST") != "" {
		return positiveIntEnv("SCORING_RATE_LIMIT_BURST", defaultScoringRateLimitBurst)
	}
	return positiveIntEnv("RATE_LIMIT_BURST", defaultScoringRateLimitBurst)
}

// OutcomeBackfillDelayDays is how long after scoring an outcome is
// considered observed. Defaults to 0 (backfill all scored applicants).
func OutcomeBackfillDelayDays() int {
	if v := os.Getenv("OUTCOME_BACKFILL_DELAY_DAYS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			return n
		}
	}
	return defaultOutcomeBackfillDelayDays
}
