package inference

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"
)

func histSampleCount(t *testing.T, h prometheus.Histogram) uint64 {
	t.Helper()
	var m dto.Metric
	if err := h.Write(&m); err != nil {
		t.Fatalf("histogram write: %v", err)
	}
	return m.GetHistogram().GetSampleCount()
}

func TestRecordScore(t *testing.T) {
	beforeCount := histSampleCount(t, scoreDistribution)
	beforeDecision := testutil.ToFloat64(decisionsTotal.WithLabelValues("decline"))

	recordScore(0.42, "decline")

	if got := histSampleCount(t, scoreDistribution) - beforeCount; got != 1 {
		t.Errorf("score observations delta = %d, want 1", got)
	}
	if got := testutil.ToFloat64(decisionsTotal.WithLabelValues("decline")) - beforeDecision; got != 1 {
		t.Errorf("decline counter delta = %v, want 1", got)
	}
}

func TestSetModelInfo(t *testing.T) {
	setModelInfo("v9.9")
	if got := testutil.ToFloat64(modelInfo.WithLabelValues("v9.9")); got != 1 {
		t.Errorf("model_info{v9.9} = %v, want 1", got)
	}

	// Reload clears the previous version so only one series remains.
	setModelInfo("v9.10")
	if got := testutil.CollectAndCount(modelInfo); got != 1 {
		t.Errorf("model_info series = %d, want 1 (old version not cleared)", got)
	}
	if got := testutil.ToFloat64(modelInfo.WithLabelValues("v9.10")); got != 1 {
		t.Errorf("model_info{v9.10} = %v, want 1", got)
	}
}
