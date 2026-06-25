// Package model loads a gradient-boosted tree model exported by
// pipeline/export_model_json.py and provides pure-Go inference plus
// TreeSHAP explanations. The JSON format is library-agnostic (the
// exporter currently normalizes LightGBM dumps); only binary
// classification with numeric splits is supported.
package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"sync"
)

// Tree holds one decision tree in columnar form, mirroring the node
// array of sklearn's TreePredictor.
type Tree struct {
	Value           []float64 `json:"value"`
	Count           []float64 `json:"count"`
	FeatureIdx      []int32   `json:"feature_idx"`
	NumThreshold    []float64 `json:"num_threshold"`
	MissingGoToLeft []uint8   `json:"missing_go_to_left"`
	Left            []uint32  `json:"left"`
	Right           []uint32  `json:"right"`
	IsLeaf          []uint8   `json:"is_leaf"`
}

type Model struct {
	FormatVersion      int                           `json:"format_version"`
	Version            string                        `json:"model_version"`
	FeatureVersion     int                           `json:"feature_version"`
	NFeatures          int                           `json:"n_features"`
	Features           []string                      `json:"features"`
	Metrics            map[string]map[string]float64 `json:"metrics"`
	BaselinePrediction float64                       `json:"baseline_prediction"`
	Trees              []Tree                        `json:"trees"`
	Calibration        *Calibration                  `json:"calibration"`
	Scorecard          *Scorecard                    `json:"scorecard"`
	ValidationStatus   *ValidationStatus             `json:"validation_status"`

	expectedValue     float64
	expectedValueOnce sync.Once
}

// ValidationStatus is the model-card verdict embedded by the exporter
// (pipeline/export_model_json.py, derived from model_card._validation_status).
// The serving governance gate refuses to load a champion whose status is
// not APPROVED unless an audited override is supplied. Models exported
// before this field existed unmarshal to nil.
type ValidationStatus struct {
	Status    string `json:"status"`
	Rationale string `json:"rationale"`
}

// Load reads a model.json produced by pipeline/export_model_json.py.
func Load(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read model: %w", err)
	}
	var m Model
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse model %s: %w", path, err)
	}
	if m.FormatVersion != 1 {
		return nil, fmt.Errorf("unsupported model format_version %d", m.FormatVersion)
	}
	if len(m.Trees) == 0 || len(m.Features) != m.NFeatures {
		return nil, fmt.Errorf("model %s is malformed", path)
	}
	if c := m.Calibration; c != nil {
		if len(c.X) == 0 || len(c.X) != len(c.Y) || !sort.Float64sAreSorted(c.X) {
			return nil, fmt.Errorf("model %s has malformed calibration", path)
		}
	}
	if err := m.validateTrees(); err != nil {
		return nil, fmt.Errorf("model %s: %w", path, err)
	}
	return &m, nil
}

// validateTrees checks every tree's node arrays are mutually consistent so
// that leafValue and shap can never index out of bounds or loop. For each
// internal node the child indexes must be in range and strictly greater
// than the parent — the exporter appends children after their parent, an
// invariant that also guarantees the graph is acyclic and traversal
// terminates — and the split feature index must be within range.
func (m *Model) validateTrees() error {
	for ti := range m.Trees {
		t := &m.Trees[ti]
		n := len(t.IsLeaf)
		if n == 0 {
			return fmt.Errorf("tree %d is empty", ti)
		}
		if len(t.Value) != n || len(t.FeatureIdx) != n || len(t.NumThreshold) != n ||
			len(t.MissingGoToLeft) != n || len(t.Left) != n || len(t.Right) != n {
			return fmt.Errorf("tree %d has inconsistent node array lengths", ti)
		}
		for i := 0; i < n; i++ {
			if t.IsLeaf[i] == 1 {
				continue
			}
			l, r := int(t.Left[i]), int(t.Right[i])
			if l <= i || r <= i || l >= n || r >= n {
				return fmt.Errorf("tree %d node %d has invalid child indexes (%d, %d)", ti, i, l, r)
			}
			if fi := int(t.FeatureIdx[i]); fi < 0 || fi >= m.NFeatures {
				return fmt.Errorf("tree %d node %d feature index %d out of range", ti, i, fi)
			}
		}
	}
	return nil
}

// leafValue walks one tree for a single row. Missing values (NaN)
// follow the split's missing_go_to_left direction, exactly as
// sklearn's _predict_one_from_raw_data.
func (t *Tree) leafValue(x []float64) float64 {
	i := uint32(0)
	for t.IsLeaf[i] == 0 {
		v := x[t.FeatureIdx[i]]
		if math.IsNaN(v) {
			if t.MissingGoToLeft[i] == 1 {
				i = t.Left[i]
			} else {
				i = t.Right[i]
			}
		} else if v <= t.NumThreshold[i] {
			i = t.Left[i]
		} else {
			i = t.Right[i]
		}
	}
	return t.Value[i]
}

// RawPredict returns the log-odds prediction for one row.
func (m *Model) RawPredict(x []float64) float64 {
	raw := m.BaselinePrediction
	for i := range m.Trees {
		raw += m.Trees[i].leafValue(x)
	}
	return raw
}

func sigmoid(z float64) float64 { return 1.0 / (1.0 + math.Exp(-z)) }

// PredictProba returns P(default=1) for one row.
func (m *Model) PredictProba(x []float64) float64 {
	return sigmoid(m.RawPredict(x))
}

// PredictProbaBatch scores many rows in parallel.
func (m *Model) PredictProbaBatch(rows [][]float64) []float64 {
	out := make([]float64, len(rows))
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	chunk := (len(rows) + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= len(rows) {
			break
		}
		hi := min(lo+chunk, len(rows))
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				out[i] = m.PredictProba(rows[i])
			}
		}(lo, hi)
	}
	wg.Wait()
	return out
}

// treeMean is the cover-weighted average leaf value, i.e. the
// expected output of one tree over the training distribution.
func (t *Tree) treeMean() float64 {
	var node func(i uint32) float64
	node = func(i uint32) float64 {
		if t.IsLeaf[i] == 1 {
			return t.Value[i]
		}
		l, r := t.Left[i], t.Right[i]
		total := t.Count[l] + t.Count[r]
		return (t.Count[l]*node(l) + t.Count[r]*node(r)) / total
	}
	return node(0)
}

// ExpectedValue is the SHAP base value in log-odds space:
// baseline + sum of each tree's cover-weighted mean.
func (m *Model) ExpectedValue() float64 {
	m.expectedValueOnce.Do(func() {
		ev := m.BaselinePrediction
		for i := range m.Trees {
			ev += m.Trees[i].treeMean()
		}
		m.expectedValue = ev
	})
	return m.expectedValue
}
