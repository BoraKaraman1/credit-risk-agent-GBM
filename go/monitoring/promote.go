// Challenger promotion (gbm promote). Publishes the challenger as an
// immutable versioned directory and atomically repoints the champion
// symlink at it, so the scoring runtime never observes a missing or
// partially-copied champion. In steady state (champion already a symlink)
// the swap is a single rename(2); only the one-time migration from a
// legacy real champion directory has a brief window, and it archives the
// old champion as a version rather than deleting it.
package monitoring

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/model"
)

// promoteChallenger publishes models/challenger as models/versions/<version>
// and atomically points models/champion at it. Returns the promoted
// version.
func promoteChallenger(modelsDir string) (string, error) {
	challengerDir := filepath.Join(modelsDir, "challenger")
	m, err := model.Load(filepath.Join(challengerDir, "model.json"))
	if err != nil {
		return "", fmt.Errorf("load challenger: %w", err)
	}
	version := m.Version
	if version == "" {
		return "", fmt.Errorf("challenger model has no version")
	}
	if _, err := directoryDigest(challengerDir); err != nil {
		return "", fmt.Errorf("validate challenger artifacts: %w", err)
	}

	// The serving gate refuses non-APPROVED models, so promoting one
	// would schedule an outage for the next restart. Enforce the same
	// governance gate here, with the same audited override.
	status := ""
	if m.ValidationStatus != nil {
		status = m.ValidationStatus.Status
	}
	if status != "APPROVED" && !config.AllowUnapprovedModel() {
		rationale := "no validation status recorded"
		if m.ValidationStatus != nil {
			rationale = m.ValidationStatus.Rationale
		}
		return "", fmt.Errorf(
			"challenger %s validation status is %q, not APPROVED: %s "+
				"(the serving gate would refuse it; set ALLOW_UNAPPROVED_MODEL=true "+
				"to promote under documented sign-off)", version, status, rationale)
	}

	versionsDir := filepath.Join(modelsDir, "versions")
	if err := os.MkdirAll(versionsDir, 0o755); err != nil {
		return "", err
	}
	versionDir := filepath.Join(versionsDir, version)
	if fi, err := os.Stat(versionDir); err == nil {
		if !fi.IsDir() {
			return "", fmt.Errorf("published version path %s is not a directory", versionDir)
		}
		same, err := sameDirectoryContents(challengerDir, versionDir)
		if err != nil {
			return "", fmt.Errorf("compare existing version %s: %w", version, err)
		}
		if !same {
			return "", fmt.Errorf(
				"version %s already exists with different artifacts; bump the challenger version",
				version,
			)
		}
		slog.Info("reusing identical immutable version", "version", version)
	} else if !os.IsNotExist(err) {
		return "", fmt.Errorf("inspect version %s: %w", version, err)
	} else {
		// Stage a full copy on the same filesystem, then atomically publish it
		// as the immutable version directory.
		staging, err := os.MkdirTemp(versionsDir, ".staging-")
		if err != nil {
			return "", err
		}
		defer os.RemoveAll(staging) // no-op once renamed away
		if err := copyDir(challengerDir, staging); err != nil {
			return "", fmt.Errorf("stage challenger: %w", err)
		}
		if err := os.Rename(staging, versionDir); err != nil {
			return "", fmt.Errorf("publish version %s: %w", version, err)
		}
	}

	if err := swapChampionPointer(modelsDir, version); err != nil {
		return "", err
	}
	return version, nil
}

// swapChampionPointer atomically repoints models/champion at
// versions/<version> using a relative symlink target (so the model tree
// stays relocatable). A legacy real champion directory is archived as a
// version on first run, then replaced by the symlink.
func swapChampionPointer(modelsDir, version string) error {
	champion := filepath.Join(modelsDir, "champion")
	target := filepath.Join("versions", version)

	if fi, err := os.Lstat(champion); err == nil && fi.Mode()&os.ModeSymlink == 0 {
		// Legacy non-symlink champion: archive a real directory as its own
		// version (preserving it) before the path can be replaced.
		if fi.IsDir() {
			if err := archiveLegacyChampion(modelsDir, champion); err != nil {
				return err
			}
		} else {
			return fmt.Errorf("champion path is neither a directory nor a symlink")
		}
	}

	return swapChampionTarget(modelsDir, target)
}

func swapChampionTarget(modelsDir, target string) error {
	champion := filepath.Join(modelsDir, "champion")
	tmp := filepath.Join(modelsDir, ".champion.next")
	_ = os.Remove(tmp)
	if err := os.Symlink(target, tmp); err != nil {
		return err
	}
	if err := os.Rename(tmp, champion); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("swap champion pointer: %w", err)
	}
	return nil
}

// archiveLegacyChampion moves a legacy real champion directory under
// versions/<its-version>. It refuses ambiguous or conflicting state
// instead of deleting the only copy of a model.
func archiveLegacyChampion(modelsDir, champion string) error {
	cm, err := model.Load(filepath.Join(champion, "model.json"))
	if err != nil {
		return fmt.Errorf("cannot archive legacy champion safely: %w", err)
	}
	if cm.Version == "" {
		return fmt.Errorf("cannot archive legacy champion safely: model has no version")
	}
	archived := filepath.Join(modelsDir, "versions", cm.Version)
	if fi, err := os.Stat(archived); err == nil {
		if !fi.IsDir() {
			return fmt.Errorf("legacy champion archive path is not a directory: %s", archived)
		}
		same, err := sameDirectoryContents(champion, archived)
		if err != nil {
			return err
		}
		if !same {
			return fmt.Errorf(
				"legacy champion %s conflicts with the existing immutable version",
				cm.Version,
			)
		}
		return os.RemoveAll(champion)
	} else if !os.IsNotExist(err) {
		return err
	}
	return os.Rename(champion, archived)
}

func copyDir(src, dst string) error {
	if err := os.MkdirAll(dst, 0o755); err != nil {
		return err
	}
	entries, err := os.ReadDir(src)
	if err != nil {
		return err
	}
	for _, e := range entries {
		sp, dp := filepath.Join(src, e.Name()), filepath.Join(dst, e.Name())
		if e.IsDir() {
			if err := copyDir(sp, dp); err != nil {
				return err
			}
			continue
		}
		if err := copyFile(sp, dp); err != nil {
			return err
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		_ = out.Close()
		return err
	}
	return out.Close()
}

func directoryDigest(root string) ([sha256.Size]byte, error) {
	h := sha256.New()
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if _, err := io.WriteString(h, rel); err != nil {
			return err
		}
		if _, err := h.Write([]byte{0}); err != nil {
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("artifact directory contains symlink %s", rel)
		}
		if entry.IsDir() {
			_, err = h.Write([]byte("dir\x00"))
			return err
		}
		if _, err := h.Write([]byte("file\x00")); err != nil {
			return err
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		_, copyErr := io.Copy(h, f)
		closeErr := f.Close()
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	})
	var digest [sha256.Size]byte
	if err != nil {
		return digest, err
	}
	copy(digest[:], h.Sum(nil))
	return digest, nil
}

func sameDirectoryContents(left, right string) (bool, error) {
	leftDigest, err := directoryDigest(left)
	if err != nil {
		return false, err
	}
	rightDigest, err := directoryDigest(right)
	if err != nil {
		return false, err
	}
	return leftDigest == rightDigest, nil
}

type championState struct {
	Exists  bool
	Target  string
	Version string
}

func captureChampionState(modelsDir string) (championState, error) {
	champion := filepath.Join(modelsDir, "champion")
	fi, err := os.Lstat(champion)
	if os.IsNotExist(err) {
		return championState{}, nil
	}
	if err != nil {
		return championState{}, err
	}
	if !fi.IsDir() && fi.Mode()&os.ModeSymlink == 0 {
		return championState{}, fmt.Errorf("champion path is neither a directory nor a symlink")
	}
	m, err := model.Load(filepath.Join(champion, "model.json"))
	if err != nil {
		return championState{}, fmt.Errorf("load current champion: %w", err)
	}
	target := filepath.Join("versions", m.Version)
	if fi.Mode()&os.ModeSymlink != 0 {
		target, err = os.Readlink(champion)
		if err != nil {
			return championState{}, err
		}
	}
	return championState{Exists: true, Target: target, Version: m.Version}, nil
}

func restoreChampionState(modelsDir string, previous championState) error {
	champion := filepath.Join(modelsDir, "champion")
	if !previous.Exists {
		if err := os.Remove(champion); err != nil && !os.IsNotExist(err) {
			return err
		}
		return nil
	}
	return swapChampionTarget(modelsDir, previous.Target)
}

// notifyReload tells a running scoring API to hot-load the newly
// promoted champion and verifies that the API reports the exact expected
// model version.
func notifyReload(baseURL, expectedVersion string) error {
	req, err := http.NewRequest(http.MethodPost, strings.TrimRight(baseURL, "/")+"/reload", nil)
	if err != nil {
		return err
	}
	if keys := config.APIKeys(); len(keys) > 0 {
		req.Header.Set("X-API-Key", keys[0])
	}
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("reload returned %s: %s", resp.Status, body)
	}
	var payload struct {
		ModelVersion string `json:"model_version"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&payload); err != nil {
		return fmt.Errorf("decode reload response: %w", err)
	}
	if payload.ModelVersion != expectedVersion {
		return fmt.Errorf(
			"reload reported model version %q, expected %q",
			payload.ModelVersion, expectedVersion,
		)
	}
	return nil
}

func promoteAndActivate(
	modelsDir string,
	scoringAPIURL string,
	allowOffline bool,
) (string, error) {
	previous, err := captureChampionState(modelsDir)
	if err != nil {
		return "", err
	}

	if scoringAPIURL == "" {
		if !allowOffline {
			return "", fmt.Errorf(
				"SCORING_API_URL is required for online promotion; use --offline only when the API is stopped",
			)
		}
		return promoteChallenger(modelsDir)
	}

	if !previous.Exists {
		return "", fmt.Errorf(
			"online promotion requires a current champion for rollback; use --offline for bootstrap",
		)
	}
	// Reloading the current champion is a non-destructive preflight that
	// verifies API reachability, credentials, artifact readability, and
	// the version acknowledgment before any registry mutation.
	if err := notifyReload(scoringAPIURL, previous.Version); err != nil {
		return "", fmt.Errorf("serving preflight failed: %w", err)
	}

	version, err := promoteChallenger(modelsDir)
	if err != nil {
		return "", err
	}
	if err := notifyReload(scoringAPIURL, version); err == nil {
		return version, nil
	} else {
		activationErr := err
		if rollbackErr := restoreChampionState(modelsDir, previous); rollbackErr != nil {
			return "", fmt.Errorf(
				"activate %s: %v; rollback pointer failed: %w",
				version, activationErr, rollbackErr,
			)
		}
		if rollbackReloadErr := notifyReload(scoringAPIURL, previous.Version); rollbackReloadErr != nil {
			return "", fmt.Errorf(
				"activate %s: %v; registry rolled back to %s but API rollback reload failed: %w",
				version, activationErr, previous.Version, rollbackReloadErr,
			)
		}
		return "", fmt.Errorf(
			"activate %s: %w; rolled back registry and API to %s",
			version, activationErr, previous.Version,
		)
	}
}

func RunPromote(allowOffline bool) {
	config.LoadEnv()
	release, err := acquireModelsLock()
	if err != nil {
		slog.Error("promotion failed", "error", err)
		os.Exit(1)
	}
	defer release()
	version, err := promoteAndActivate(
		config.ModelsDir(),
		config.ScoringAPIURL(),
		allowOffline,
	)
	if err != nil {
		slog.Error("promotion failed", "error", err)
		os.Exit(1)
	}
	slog.Info("challenger promoted and activated",
		"version", version, "champion", config.ChampionDir())
	if config.ScoringAPIURL() == "" {
		slog.Warn("offline promotion completed; restart the stopped scoring API before serving traffic")
	}
}
