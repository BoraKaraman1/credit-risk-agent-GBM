// Outcome Backfill Job.
// The scoring API logs every decision to scoring_log but cannot know the
// real outcome at decision time, so actual_default stays NULL and the
// performance monitor falls back to the test-set proxy. This job
// simulates outcomes arriving: for applicants scored at least
// OUTCOME_BACKFILL_DELAY_DAYS ago, it looks up the true label from the
// Gold test set (applicant_id LC_<loan-id>) and writes it back. Once
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

// loanIDFromApplicantID parses the LendingClub loan id out of an
// applicant ID of the form "LC_68407277" (assigned from the Gold id
// column by syncFeatures in sync.go).
func loanIDFromApplicantID(id string) (int64, bool) {
	const prefix = "LC_"
	if !strings.HasPrefix(id, prefix) {
		return 0, false
	}
	n, err := strconv.ParseInt(id[len(prefix):], 10, 64)
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

// buildBackfill pairs each pending applicant ID with its true label,
// keyed by stable loan id — never by row position, which changes when
// the Gold parquet is regenerated. Unknown IDs are skipped.
func buildBackfill(pendingIDs []string, labels map[int64]bool) (ids []string, outLabels []bool, skipped int) {
	for _, id := range pendingIDs {
		loanID, ok := loanIDFromApplicantID(id)
		if !ok {
			skipped++
			continue
		}
		label, ok := labels[loanID]
		if !ok {
			skipped++
			continue
		}
		ids = append(ids, id)
		outLabels = append(outLabels, label)
	}
	return ids, outLabels, skipped
}

// testLabels reads the Gold test set as a loan-id -> default map.
func testLabels() (map[int64]bool, error) {
	frame, err := gold.ReadColumns(
		filepath.Join(config.GoldDir(), "features_test.parquet"), []string{"id", "default"})
	if err != nil {
		return nil, err
	}
	labels := make(map[int64]bool, frame.NumRows)
	for i, v := range frame.Columns["default"] {
		labels[int64(frame.Columns["id"][i])] = v != 0
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
	ctx, cancel := withDeadline(backfillTimeout)
	defer cancel()
	rep, err := runBackfill(ctx)
	if err != nil {
		slog.Error("outcome backfill failed", "error", err)
		os.Exit(1)
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(rep)
}
