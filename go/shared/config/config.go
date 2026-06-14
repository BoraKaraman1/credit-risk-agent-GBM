// Package config mirrors agents/config.py: shared paths, thresholds,
// and environment loading for the Go services.
package config

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
)

// Drift thresholds
const (
	PSIWarning   = 0.10 // score distribution drift warning
	PSICritical  = 0.25 // score distribution drift -> retrain
	CSIThreshold = 0.20 // per-feature characteristic stability index
)

// Performance thresholds
const AUCDropThreshold = 0.03 // 3-point AUC drop from training -> retrain

// Fairness: 80% rule (four-fifths) for the Disparate Impact Ratio.
const FairnessDIRThreshold = 0.80

// Outcome backfill: a scored applicant's outcome is treated as observed
// only after this many days have passed since scoring (loan maturation).
const defaultOutcomeBackfillDelayDays = 0

// Decision thresholds (mirrored from the Python API)
const (
	ApproveThreshold = 0.15
	ReviewThreshold  = 0.30
)

// Per-client rate limit defaults (token bucket). RATE_LIMIT_RPS is the
// sustained rate, RATE_LIMIT_BURST the bucket depth.
const (
	defaultRateLimitRPS   = 20.0
	defaultRateLimitBurst = 40
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

// RateLimitRPS is the per-client sustained request rate.
func RateLimitRPS() float64 {
	if v := os.Getenv("RATE_LIMIT_RPS"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil && f > 0 {
			return f
		}
	}
	return defaultRateLimitRPS
}

// RateLimitBurst is the per-client token-bucket depth.
func RateLimitBurst() int {
	if v := os.Getenv("RATE_LIMIT_BURST"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return defaultRateLimitBurst
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
