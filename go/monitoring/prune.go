// Retention job for scoring_log.
// Every decision writes a full JSONB feature snapshot, so the audit
// table grows without bound on a free-tier Postgres. This job deletes
// rows older than the retention window — default 750 days, covering
// Reg B's 25-month record-retention requirement with margin. Retention
// is enforced by an explicit, logged run (weekly monitoring DAG), never
// silently at write time.
package monitoring

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"strconv"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
)

const defaultRetentionDays = 750

type pruneReport struct {
	Timestamp     string `json:"timestamp"`
	RetentionDays int    `json:"retention_days"`
	RowsDeleted   int64  `json:"rows_deleted"`
}

func retentionDays() (int, error) {
	v := os.Getenv("SCORING_LOG_RETENTION_DAYS")
	if v == "" {
		return defaultRetentionDays, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		return 0, &strconv.NumError{Func: "SCORING_LOG_RETENTION_DAYS", Num: v, Err: strconv.ErrSyntax}
	}
	return n, nil
}

func runPrune(ctx context.Context) (*pruneReport, error) {
	days, err := retentionDays()
	if err != nil {
		return nil, err
	}

	d, err := db.Connect(ctx, config.DatabaseURL())
	if err != nil {
		return nil, err
	}
	defer d.Close()

	deleted, err := d.PruneScoringLog(ctx, days)
	if err != nil {
		return nil, err
	}
	slog.Info("scoring_log pruned", "retention_days", days, "rows_deleted", deleted)

	return &pruneReport{
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
		RetentionDays: days,
		RowsDeleted:   deleted,
	}, nil
}

func RunPrune() {
	config.LoadEnv()
	ctx, cancel := withDeadline(pruneTimeout)
	defer cancel()
	rep, err := runPrune(ctx)
	if err != nil {
		slog.Error("prune failed", "error", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
}
