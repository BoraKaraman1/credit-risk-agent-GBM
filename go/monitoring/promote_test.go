package monitoring

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
)

// writeMinimalModel writes a loadable single-leaf model.json (plus a
// sidecar file) into dir, to exercise promotion without a real model.
func writeMinimalModel(t *testing.T, dir, version string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	payload := map[string]any{
		"format_version":      1,
		"model_version":       version,
		"feature_version":     1,
		"n_features":          1,
		"features":            []string{"f"},
		"baseline_prediction": 0.0,
		"trees": []map[string]any{{
			"value":              []float64{0.5},
			"count":              []float64{1},
			"feature_idx":        []int{0},
			"num_threshold":      []float64{0},
			"missing_go_to_left": []int{0},
			"left":               []int{0},
			"right":              []int{0},
			"is_leaf":            []int{1},
		}},
		"validation_status": map[string]string{"status": "APPROVED", "rationale": "test"},
	}
	b, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	// A sidecar (e.g. the joblib) to confirm the whole directory is copied.
	if err := os.WriteFile(filepath.Join(dir, "model.joblib"), []byte("blob"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestPromoteRefusesUnapproved(t *testing.T) {
	modelsDir := t.TempDir()
	challenger := filepath.Join(modelsDir, "challenger")
	writeMinimalModel(t, challenger, "v9.9")

	// Rewrite the challenger's verdict to REVIEW REQUIRED.
	raw, err := os.ReadFile(filepath.Join(challenger, "model.json"))
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatal(err)
	}
	payload["validation_status"] = map[string]string{
		"status": "REVIEW REQUIRED", "rationale": "DIR violations"}
	b, _ := json.Marshal(payload)
	if err := os.WriteFile(filepath.Join(challenger, "model.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := promoteChallenger(modelsDir); err == nil {
		t.Fatal("promote should refuse a REVIEW REQUIRED challenger")
	}

	t.Run("audited override allows it", func(t *testing.T) {
		t.Setenv("ALLOW_UNAPPROVED_MODEL", "true")
		version, err := promoteChallenger(modelsDir)
		if err != nil {
			t.Fatalf("override should allow promotion: %v", err)
		}
		if version != "v9.9" {
			t.Errorf("version = %q, want v9.9", version)
		}
	})
}

func TestPromoteChallenger(t *testing.T) {
	modelsDir := t.TempDir()
	writeMinimalModel(t, filepath.Join(modelsDir, "challenger"), "v1.3")
	// Legacy real champion directory (not yet a symlink).
	writeMinimalModel(t, filepath.Join(modelsDir, "champion"), "v1.2")

	version, err := promoteChallenger(modelsDir)
	if err != nil {
		t.Fatalf("promote: %v", err)
	}
	if version != "v1.3" {
		t.Errorf("version = %q, want v1.3", version)
	}

	// champion is now a symlink to versions/v1.3.
	champion := filepath.Join(modelsDir, "champion")
	fi, err := os.Lstat(champion)
	if err != nil {
		t.Fatal(err)
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		t.Fatal("champion should be a symlink after promotion")
	}
	if tgt, _ := os.Readlink(champion); tgt != filepath.Join("versions", "v1.3") {
		t.Errorf("symlink target = %q, want versions/v1.3", tgt)
	}

	// The symlink resolves to the promoted model, sidecar included.
	m, err := model.Load(filepath.Join(champion, "model.json"))
	if err != nil {
		t.Fatalf("load via symlink: %v", err)
	}
	if m.Version != "v1.3" {
		t.Errorf("served version = %q, want v1.3", m.Version)
	}
	if _, err := os.Stat(filepath.Join(champion, "model.joblib")); err != nil {
		t.Errorf("sidecar not copied: %v", err)
	}

	// The legacy champion was archived, not lost.
	if _, err := os.Stat(filepath.Join(modelsDir, "versions", "v1.2", "model.json")); err != nil {
		t.Errorf("legacy champion v1.2 should be archived: %v", err)
	}

	// Versions are immutable: re-publishing the same version fails.
	if _, err := promoteChallenger(modelsDir); err == nil {
		t.Error("re-promoting an existing version should fail (immutable)")
	}
}

func TestNotifyReload(t *testing.T) {
	t.Setenv("API_KEYS", "k1,k2")
	var gotMethod, gotPath, gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod, gotPath, gotKey = r.Method, r.URL.Path, r.Header.Get("X-API-Key")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := notifyReload(srv.URL); err != nil {
		t.Fatalf("notifyReload: %v", err)
	}
	if gotMethod != http.MethodPost || gotPath != "/reload" {
		t.Errorf("request = %s %s, want POST /reload", gotMethod, gotPath)
	}
	if gotKey != "k1" {
		t.Errorf("X-API-Key = %q, want first configured key", gotKey)
	}

	t.Run("non-200 is an error", func(t *testing.T) {
		bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "no challenger", http.StatusServiceUnavailable)
		}))
		defer bad.Close()
		if err := notifyReload(bad.URL); err == nil {
			t.Error("want error on non-200 reload response")
		}
	})

	t.Run("api down is an error", func(t *testing.T) {
		down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
		down.Close()
		if err := notifyReload(down.URL); err == nil {
			t.Error("want error when the API is unreachable")
		}
	})
}

func TestPromoteAtomicSwapOverSymlink(t *testing.T) {
	// Once champion is a symlink, a subsequent promotion swaps it with a
	// single rename and leaves a valid champion (no migration window).
	modelsDir := t.TempDir()
	writeMinimalModel(t, filepath.Join(modelsDir, "challenger"), "v2.0")
	if err := os.MkdirAll(filepath.Join(modelsDir, "versions", "v1.0"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join("versions", "v1.0"), filepath.Join(modelsDir, "champion")); err != nil {
		t.Fatal(err)
	}
	if _, err := promoteChallenger(modelsDir); err != nil {
		t.Fatalf("promote over symlink: %v", err)
	}
	tgt, _ := os.Readlink(filepath.Join(modelsDir, "champion"))
	if tgt != filepath.Join("versions", "v2.0") {
		t.Errorf("symlink target = %q, want versions/v2.0", tgt)
	}
}
