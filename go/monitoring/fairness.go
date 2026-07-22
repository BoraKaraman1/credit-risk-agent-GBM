// Fairness sync: publishes the champion's fairness summary (computed by
// pipeline/fairness.py and stored in model_metadata.json) into drift_log
// so dashboards can read it from Postgres alongside PSI and AUC history.
// One row per model version; re-runs are no-ops.
package monitoring

import (
	"context"
	"encoding/json"
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
// four-fifths position, 1.0 = parity).
func minDIR(fairness []byte) float64 {
	var parsed struct {
		Attributes map[string]struct {
			Groups map[string]struct {
				DIR float64 `json:"dir"`
			} `json:"groups"`
		} `json:"attributes"`
	}
	min := 1.0
	if err := json.Unmarshal(fairness, &parsed); err != nil {
		return min
	}
	for _, attr := range parsed.Attributes {
		for _, g := range attr.Groups {
			if g.DIR < min {
				min = g.DIR
			}
		}
	}
	return min
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

	exists, err := database.HasDriftLogEntry(ctx, "fairness", meta.Version)
	if err != nil {
		slog.Warn("fairness sync: could not check drift_log", "error", err)
		return
	}
	if exists {
		return
	}
	if err := database.InsertDriftLog(ctx, "fairness", minDIR(meta.Fairness),
		meta.Version, meta.Fairness); err != nil {
		slog.Warn("fairness sync: insert failed", "error", err)
		return
	}
	slog.Info("fairness summary logged to drift_log", "model_version", meta.Version)
}
