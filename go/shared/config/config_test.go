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

func TestAPIKeys(t *testing.T) {
	t.Run("empty when unset", func(t *testing.T) {
		t.Setenv("API_KEYS", "")
		if got := APIKeys(); len(got) != 0 {
			t.Errorf("APIKeys() = %v, want empty", got)
		}
	})
	t.Run("splits and trims", func(t *testing.T) {
		t.Setenv("API_KEYS", " k1 , k2,, k3 ")
		got := APIKeys()
		want := []string{"k1", "k2", "k3"}
		if len(got) != len(want) {
			t.Fatalf("APIKeys() = %v, want %v", got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("APIKeys()[%d] = %q, want %q", i, got[i], want[i])
			}
		}
	})
}

func TestRateLimit(t *testing.T) {
	t.Run("defaults when unset", func(t *testing.T) {
		t.Setenv("RATE_LIMIT_RPS", "")
		t.Setenv("RATE_LIMIT_BURST", "")
		if got := RateLimitRPS(); got != defaultRateLimitRPS {
			t.Errorf("RateLimitRPS() = %v, want %v", got, defaultRateLimitRPS)
		}
		if got := RateLimitBurst(); got != defaultRateLimitBurst {
			t.Errorf("RateLimitBurst() = %v, want %v", got, defaultRateLimitBurst)
		}
	})
	t.Run("env override", func(t *testing.T) {
		t.Setenv("RATE_LIMIT_RPS", "5.5")
		t.Setenv("RATE_LIMIT_BURST", "12")
		if got := RateLimitRPS(); got != 5.5 {
			t.Errorf("RateLimitRPS() = %v, want 5.5", got)
		}
		if got := RateLimitBurst(); got != 12 {
			t.Errorf("RateLimitBurst() = %v, want 12", got)
		}
	})
	t.Run("invalid and non-positive values fall back to defaults", func(t *testing.T) {
		t.Setenv("RATE_LIMIT_RPS", "abc")
		t.Setenv("RATE_LIMIT_BURST", "-3")
		if got := RateLimitRPS(); got != defaultRateLimitRPS {
			t.Errorf("RateLimitRPS() = %v, want default", got)
		}
		if got := RateLimitBurst(); got != defaultRateLimitBurst {
			t.Errorf("RateLimitBurst() = %v, want default", got)
		}
	})
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
	if !(0 <= FairnessWorsenTolerance && FairnessWorsenTolerance < FairnessDIRThreshold) {
		t.Error("worsen tolerance must be non-negative and below the DIR threshold")
	}
}
