// Fairness sync: publishes the champion's fairness summary (computed by
// pipeline/fairness.py and stored in model_metadata.json) into drift_log
// so dashboards can read it from Postgres alongside PSI and AUC history.
// One row per model version; re-runs are no-ops.
package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"

	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/db"
	"github.com/BoraKaraman1/credit-risk-agent-GBM/go/shared/config"
)

// championFairness is the slice of model_metadata.json the dashboard
// needs: the version plus the whole fairness block, kept as raw JSON so
// Python remains the schema owner.
type championFairness struct {
	Version  string          `json:"version"`
	Fairness json.RawMessage `json:"fairness"`
}

// minDIR extracts the lowest Disparate Impact Ratio across all
// attributes and groups; it is the row's scalar metric_value (worst-case
// four-fifths position, 1.0 = parity). It fails closed: malformed or
// empty summaries return an error instead of degrading to "parity".
func minDIR(fairness []byte) (float64, error) {
	var parsed fairnessSummary
	if err := json.Unmarshal(fairness, &parsed); err != nil {
		return 0, fmt.Errorf("malformed fairness summary: %w", err)
	}
	if len(parsed.Attributes) == 0 {
		return 0, fmt.Errorf("fairness summary has no attributes")
	}
	min := 1.0
	for _, attr := range parsed.Attributes {
		for _, g := range attr.Groups {
			if g.DIR < min {
				min = g.DIR
			}
		}
	}
	return min, nil
}

// syncFairnessLog inserts the champion's fairness summary into drift_log
// once per model version. Missing metadata or a missing fairness block
// is not an error: older champions predate the fairness summary.
func syncFairnessLog(ctx context.Context, database *db.DB) {
	raw, err := os.ReadFile(filepath.Join(config.ChampionDir(), "model_metadata.json"))
	if err != nil {
		slog.Warn("fairness sync: no champion metadata", "error", err)
		return
	}
	var meta championFairness
	if err := json.Unmarshal(raw, &meta); err != nil {
		slog.Warn("fairness sync: bad champion metadata", "error", err)
		return
	}
	if len(meta.Fairness) == 0 || meta.Version == "" {
		slog.Info("fairness sync: champion metadata has no fairness block, skipping")
		return
	}

	// Fail closed: a summary that cannot be parsed is not published as
	// "parity" — it is not published at all, loudly.
	worst, err := minDIR(meta.Fairness)
	if err != nil {
		slog.Error("fairness sync: refusing to publish", "error", err,
			"model_version", meta.Version)
		return
	}

	exists, err := database.HasDriftLogEntry(ctx, "fairness", meta.Version)
	if err != nil {
		slog.Warn("fairness sync: could not check drift_log", "error", err)
		return
	}
	if exists {
		return
	}
	if err := database.InsertDriftLog(ctx, "fairness", worst,
		meta.Version, meta.Fairness); err != nil {
		slog.Warn("fairness sync: insert failed", "error", err)
		return
	}
	slog.Info("fairness summary logged to drift_log", "model_version", meta.Version)
}
