// Outcome Backfill Job.
// The scoring API logs every decision to scoring_log but cannot know the
// real outcome at decision time, so actual_default stays NULL and the
// performance monitor falls back to the test-set proxy. This job
// simulates outcomes arriving: for applicants scored at least
// OUTCOME_BACKFILL_DELAY_DAYS ago, it looks up the true label from the
// Gold test set (applicant_id LC_<row-index>) and writes it back. Once
// enough outcomes accumulate, the performance monitor runs on production
// data instead of the proxy.
package monitoring

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/gold"
)

type backfillReport struct {
	Timestamp         string `json:"timestamp"`
	DelayDays         int    `json:"delay_days"`
	PendingApplicants int    `json:"pending_applicants"`
	MatchedApplicants int    `json:"matched_applicants"`
	SkippedUnmatched  int    `json:"skipped_unmatched"`
	RowsUpdated       int64  `json:"rows_updated"`
}

// indexFromApplicantID parses the test-set row index out of an applicant
// ID of the form "LC_0000042". The synthetic feature store assigns these
// IDs by Gold test-set row position (see syncFeatures in sync.go).
func indexFromApplicantID(id string) (int, bool) {
	const prefix = "LC_"
	if !strings.HasPrefix(id, prefix) {
		return 0, false
	}
	n, err := strconv.Atoi(id[len(prefix):])
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

// buildBackfill pairs each pending applicant ID with its true label.
// IDs that do not map to a Gold test-set row are skipped.
func buildBackfill(pendingIDs []string, labels []bool) (ids []string, outLabels []bool, skipped int) {
	for _, id := range pendingIDs {
		idx, ok := indexFromApplicantID(id)
		if !ok || idx >= len(labels) {
			skipped++
			continue
		}
		ids = append(ids, id)
		outLabels = append(outLabels, labels[idx])
	}
	return ids, outLabels, skipped
}

// testLabels reads the Gold test-set default column as booleans indexed
// by row position.
func testLabels() ([]bool, error) {
	frame, err := gold.ReadColumns(
		filepath.Join(config.GoldDir(), "features_test.parquet"), []string{"default"})
	if err != nil {
		return nil, err
	}
	labels := make([]bool, frame.NumRows)
	for i, v := range frame.Columns["default"] {
		labels[i] = v != 0
	}
	return labels, nil
}

func runBackfill(ctx context.Context) (*backfillReport, error) {
	delayDays := config.OutcomeBackfillDelayDays()

	labels, err := testLabels()
	if err != nil {
		return nil, err
	}

	d, err := db.Connect(ctx, config.DatabaseURL())
	if err != nil {
		return nil, err
	}
	defer d.Close()

	pending, err := d.PendingOutcomes(ctx, delayDays)
	if err != nil {
		return nil, err
	}

	ids, outLabels, skipped := buildBackfill(pending, labels)
	updated, err := d.BackfillOutcomes(ctx, ids, outLabels)
	if err != nil {
		return nil, err
	}

	slog.Info("outcome backfill complete",
		"pending", len(pending), "matched", len(ids), "skipped", skipped, "rows_updated", updated)

	return &backfillReport{
		Timestamp:         time.Now().UTC().Format(time.RFC3339),
		DelayDays:         delayDays,
		PendingApplicants: len(pending),
		MatchedApplicants: len(ids),
		SkippedUnmatched:  skipped,
		RowsUpdated:       updated,
	}, nil
}

func RunBackfill() {
	config.LoadEnv()
	rep, err := runBackfill(context.Background())
	if err != nil {
		slog.Error("outcome backfill failed", "error", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
}
