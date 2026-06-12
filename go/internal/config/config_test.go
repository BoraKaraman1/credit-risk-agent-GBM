package config

import (
	"path/filepath"
	"testing"
)

func TestDataDir(t *testing.T) {
	t.Run("defaults to data", func(t *testing.T) {
		t.Setenv("CREDIT_RISK_DATA_DIR", "")
		if got := DataDir(); got != "data" {
			t.Errorf("DataDir() = %q, want \"data\"", got)
		}
	})
	t.Run("env override", func(t *testing.T) {
		t.Setenv("CREDIT_RISK_DATA_DIR", "/srv/data")
		if got := DataDir(); got != "/srv/data" {
			t.Errorf("DataDir() = %q", got)
		}
	})
}

func TestGoldDir(t *testing.T) {
	t.Setenv("CREDIT_RISK_DATA_DIR", "/srv/data")
	if got := GoldDir(); got != filepath.Join("/srv/data", "gold") {
		t.Errorf("GoldDir() = %q", got)
	}
}

func TestModelsDir(t *testing.T) {
	t.Run("follows data dir by default", func(t *testing.T) {
		t.Setenv("CREDIT_RISK_DATA_DIR", "/srv/data")
		t.Setenv("CREDIT_RISK_MODELS_DIR", "")
		if got := ModelsDir(); got != filepath.Join("/srv/data", "models") {
			t.Errorf("ModelsDir() = %q", got)
		}
	})
	t.Run("env override wins", func(t *testing.T) {
		t.Setenv("CREDIT_RISK_MODELS_DIR", "/models")
		if got := ModelsDir(); got != "/models" {
			t.Errorf("ModelsDir() = %q", got)
		}
	})
}

func TestModelPaths(t *testing.T) {
	t.Setenv("CREDIT_RISK_MODELS_DIR", "/models")
	t.Run("champion", func(t *testing.T) {
		want := filepath.Join("/models", "champion", "model.json")
		if got := ChampionModelPath(); got != want {
			t.Errorf("ChampionModelPath() = %q, want %q", got, want)
		}
	})
	t.Run("challenger", func(t *testing.T) {
		want := filepath.Join("/models", "challenger", "model.json")
		if got := ChallengerModelPath(); got != want {
			t.Errorf("ChallengerModelPath() = %q, want %q", got, want)
		}
	})
}

func TestDatabaseURL(t *testing.T) {
	t.Setenv("DATABASE_URL", "postgres://x")
	if got := DatabaseURL(); got != "postgres://x" {
		t.Errorf("DatabaseURL() = %q", got)
	}
}

func TestThresholdOrdering(t *testing.T) {
	// The decision and drift thresholds must keep their relative order;
	// the monitoring logic assumes warning < critical and approve < review.
	if !(PSIWarning < PSICritical) {
		t.Error("PSI warning must be below critical")
	}
	if !(ApproveThreshold < ReviewThreshold) {
		t.Error("approve threshold must be below review threshold")
	}
	if AUCDropThreshold <= 0 || CSIThreshold <= 0 {
		t.Error("thresholds must be positive")
	}
}
