package monitoring

import (
	"context"
	"time"
)

// Per-command deadlines. The monitors run as one-shot commands driven by
// Airflow; without a deadline a wedged Postgres, a stalled network call,
// or a hung Python training subprocess would block the Airflow task
// forever with no external watchdog. The values are generous on purpose:
// they bound a wedged dependency, not healthy runtime. Retrain's deadline
// also bounds the train_challenger.py subprocess via exec.CommandContext.
const (
	driftTimeout       = 30 * time.Minute // re-reads and re-scores the full train parquet
	performanceTimeout = 15 * time.Minute
	backfillTimeout    = 15 * time.Minute
	pruneTimeout       = 10 * time.Minute
	syncTimeout        = 30 * time.Minute // bulk-upserts the full test set
	retrainTimeout     = 3 * time.Hour    // full challenger training + fairness + eval
)

// withDeadline returns a background context bounded by one of the
// per-command deadlines above.
func withDeadline(d time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), d)
}
