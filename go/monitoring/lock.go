// Model-registry mutation lock.
// Retrain and promote both mutate the models directory; without a lock,
// Airflow scheduling is the only thing preventing a concurrent retrain
// from racing a promotion mid-copy. An advisory flock on a lockfile in
// the models dir makes the exclusion explicit, cross-process, and
// self-releasing on crash (the kernel drops flocks with the process).
package monitoring

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
)

// acquireModelsLock takes the exclusive registry lock, returning a
// release func, or an error if another retrain/promote holds it.
func acquireModelsLock() (func(), error) {
	if err := os.MkdirAll(config.ModelsDir(), 0o755); err != nil {
		return nil, err
	}
	path := filepath.Join(config.ModelsDir(), ".registry.lock")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	if err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		f.Close()
		return nil, fmt.Errorf("another retrain/promote holds %s: %w", path, err)
	}
	return func() {
		_ = syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
		_ = f.Close()
	}, nil
}
