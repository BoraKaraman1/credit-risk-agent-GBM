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

	// An exact retry reuses the immutable version and succeeds.
	if retryVersion, err := promoteChallenger(modelsDir); err != nil || retryVersion != "v1.3" {
		t.Errorf("identical retry = (%q, %v), want v1.3 success", retryVersion, err)
	}

	// Reusing the version for different bytes still fails.
	if err := os.WriteFile(
		filepath.Join(modelsDir, "challenger", "model.joblib"),
		[]byte("different"),
		0o644,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := promoteChallenger(modelsDir); err == nil {
		t.Error("conflicting artifacts under an immutable version should fail")
	}
}

func TestNotifyReload(t *testing.T) {
	t.Setenv("API_KEYS", "k1,k2")
	var gotMethod, gotPath, gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod, gotPath, gotKey = r.Method, r.URL.Path, r.Header.Get("X-API-Key")
		_ = json.NewEncoder(w).Encode(map[string]string{"model_version": "v1.3"})
	}))
	defer srv.Close()

	if err := notifyReload(srv.URL, "v1.3"); err != nil {
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
		if err := notifyReload(bad.URL, "v1.3"); err == nil {
			t.Error("want error on non-200 reload response")
		}
	})

	t.Run("wrong acknowledged version is an error", func(t *testing.T) {
		wrong := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_ = json.NewEncoder(w).Encode(map[string]string{"model_version": "v1.2"})
		}))
		defer wrong.Close()
		if err := notifyReload(wrong.URL, "v1.3"); err == nil {
			t.Error("want error when reload acknowledges the wrong model")
		}
	})

	t.Run("api down is an error", func(t *testing.T) {
		down := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
		down.Close()
		if err := notifyReload(down.URL, "v1.3"); err == nil {
			t.Error("want error when the API is unreachable")
		}
	})
}

func TestPromoteRequiresExplicitOfflineOverride(t *testing.T) {
	modelsDir := t.TempDir()
	writeMinimalModel(t, filepath.Join(modelsDir, "challenger"), "v1.0")

	if _, err := promoteAndActivate(modelsDir, "", false); err == nil {
		t.Fatal("promotion without an API URL or offline override should fail")
	}
	if _, err := os.Stat(filepath.Join(modelsDir, "versions", "v1.0")); !os.IsNotExist(err) {
		t.Fatalf("failed preflight must not publish artifacts, stat err=%v", err)
	}

	version, err := promoteAndActivate(modelsDir, "", true)
	if err != nil {
		t.Fatalf("explicit offline promotion failed: %v", err)
	}
	if version != "v1.0" {
		t.Errorf("version = %q, want v1.0", version)
	}
}

func TestPromotionRollsBackFailedActivationAndRetries(t *testing.T) {
	modelsDir := t.TempDir()
	writeMinimalModel(t, filepath.Join(modelsDir, "versions", "v1.2"), "v1.2")
	if err := os.Symlink(
		filepath.Join("versions", "v1.2"),
		filepath.Join(modelsDir, "champion"),
	); err != nil {
		t.Fatal(err)
	}
	writeMinimalModel(t, filepath.Join(modelsDir, "challenger"), "v1.3")

	failNew := true
	var acknowledged []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m, err := model.Load(filepath.Join(modelsDir, "champion", "model.json"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		acknowledged = append(acknowledged, m.Version)
		if failNew && m.Version == "v1.3" {
			http.Error(w, "activation refused", http.StatusServiceUnavailable)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"model_version": m.Version})
	}))
	defer srv.Close()

	if _, err := promoteAndActivate(modelsDir, srv.URL, false); err == nil {
		t.Fatal("failed activation should fail promotion")
	}
	if got := championVersion(t, modelsDir); got != "v1.2" {
		t.Fatalf("champion after rollback = %q, want v1.2", got)
	}
	wantFirst := []string{"v1.2", "v1.3", "v1.2"}
	if !equalStrings(acknowledged, wantFirst) {
		t.Fatalf("activation sequence = %v, want %v", acknowledged, wantFirst)
	}

	// The immutable v1.3 publication remains safe to reuse. A retry with
	// the same bytes can activate it after the serving issue is fixed.
	failNew = false
	acknowledged = nil
	version, err := promoteAndActivate(modelsDir, srv.URL, false)
	if err != nil {
		t.Fatalf("idempotent retry failed: %v", err)
	}
	if version != "v1.3" || championVersion(t, modelsDir) != "v1.3" {
		t.Fatalf("retry version=%q champion=%q", version, championVersion(t, modelsDir))
	}
	wantRetry := []string{"v1.2", "v1.3"}
	if !equalStrings(acknowledged, wantRetry) {
		t.Fatalf("retry sequence = %v, want %v", acknowledged, wantRetry)
	}
}

func championVersion(t *testing.T, modelsDir string) string {
	t.Helper()
	m, err := model.Load(filepath.Join(modelsDir, "champion", "model.json"))
	if err != nil {
		t.Fatal(err)
	}
	return m.Version
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
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
