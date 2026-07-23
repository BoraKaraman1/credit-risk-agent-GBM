// Challenger promotion (gbm promote). Publishes the challenger as an
// immutable versioned directory and atomically repoints the champion
// symlink at it, so the scoring runtime never observes a missing or
// partially-copied champion. In steady state (champion already a symlink)
// the swap is a single rename(2); only the one-time migration from a
// legacy real champion directory has a brief window, and it archives the
// old champion as a version rather than deleting it.
package monitoring

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"

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
	if _, err := os.Stat(versionDir); err == nil {
		return "", fmt.Errorf("version %s already published (immutable); bump the challenger version", version)
	}

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
		} else if err := os.Remove(champion); err != nil {
			return err
		}
	}

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
// versions/<its-version> so no model is lost, or removes it if that
// version already exists or its version cannot be read.
func archiveLegacyChampion(modelsDir, champion string) error {
	if cm, err := model.Load(filepath.Join(champion, "model.json")); err == nil && cm.Version != "" {
		archived := filepath.Join(modelsDir, "versions", cm.Version)
		if _, err := os.Stat(archived); os.IsNotExist(err) {
			return os.Rename(champion, archived)
		}
	}
	return os.RemoveAll(champion)
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

func RunPromote() {
	config.LoadEnv()
	release, err := acquireModelsLock()
	if err != nil {
		slog.Error("promotion failed", "error", err)
		os.Exit(1)
	}
	defer release()
	version, err := promoteChallenger(config.ModelsDir())
	if err != nil {
		slog.Error("promotion failed", "error", err)
		os.Exit(1)
	}
	slog.Info("challenger promoted to champion (atomic pointer swap)",
		"version", version, "champion", config.ChampionDir())
}
