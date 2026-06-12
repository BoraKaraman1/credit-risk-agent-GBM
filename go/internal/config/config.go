// Package config mirrors agents/config.py: shared paths, thresholds,
// and environment loading for the Go services.
package config

import (
	"os"
	"path/filepath"

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

// Decision thresholds (mirrored from the Python API)
const (
	ApproveThreshold = 0.15
	ReviewThreshold  = 0.30
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
