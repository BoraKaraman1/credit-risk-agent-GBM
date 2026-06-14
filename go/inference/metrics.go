package inference

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Prometheus metrics for the scoring API, registered on the default
// registry and exposed at /metrics. HTTP-level metrics are recorded by
// the instrument middleware; score and decision metrics by the scoring
// handler.
var (
	httpRequests = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "scoring_api_requests_total",
			Help: "HTTP requests by endpoint, method, and status code.",
		},
		[]string{"endpoint", "method", "status"},
	)
	httpDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "scoring_api_request_duration_seconds",
			Help:    "HTTP request latency by endpoint.",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"endpoint"},
	)
	scoreDistribution = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "scoring_api_score",
			Help:    "Distribution of predicted probability of default.",
			Buckets: prometheus.LinearBuckets(0, 0.05, 20),
		},
	)
	decisionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "scoring_api_decisions_total",
			Help: "Credit decisions returned, by outcome.",
		},
		[]string{"decision"},
	)
	modelInfo = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "scoring_api_model_info",
			Help: "Loaded model version (value is always 1; read the version label).",
		},
		[]string{"version"},
	)
)

// recordScore tracks the score distribution and decision counts.
func recordScore(score float64, decision string) {
	scoreDistribution.Observe(score)
	decisionsTotal.WithLabelValues(decision).Inc()
}

// setModelInfo publishes the loaded model version as an info gauge,
// clearing any previous version on reload.
func setModelInfo(version string) {
	modelInfo.Reset()
	modelInfo.WithLabelValues(version).Set(1)
}
