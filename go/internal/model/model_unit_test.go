package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// Synthetic-tree unit tests. The fixture tests in model_test.go pin
// exact LightGBM/shap parity on the real champion; these pin the tree
// mechanics on tiny hand-built models where every value is derivable
// by hand.

// singleSplitTree builds: root splits feature f at threshold; left
// leaf carries leftVal (cover leftCount), right leaf rightVal.
func singleSplitTree(f int32, threshold, leftVal, rightVal, leftCount, rightCount float64, missingLeft uint8) Tree {
	return Tree{
		Value:           []float64{0, leftVal, rightVal},
		Count:           []float64{leftCount + rightCount, leftCount, rightCount},
		FeatureIdx:      []int32{f, 0, 0},
		NumThreshold:    []float64{threshold, 0, 0},
		MissingGoToLeft: []uint8{missingLeft, 0, 0},
		Left:            []uint32{1, 0, 0},
		Right:           []uint32{2, 0, 0},
		IsLeaf:          []uint8{0, 1, 1},
	}
}

func syntheticModel(baseline float64, trees ...Tree) *Model {
	return &Model{
		FormatVersion:      1,
		Version:            "vtest",
		NFeatures:          2,
		Features:           []string{"f0", "f1"},
		BaselinePrediction: baseline,
		Trees:              trees,
	}
}

func TestLeafValueRouting(t *testing.T) {
	tree := singleSplitTree(0, 0.5, -1, +1, 60, 40, 1)
	cases := []struct {
		name string
		x    []float64
		want float64
	}{
		{"below threshold goes left", []float64{0.2, 9}, -1},
		{"at threshold goes left", []float64{0.5, 9}, -1},
		{"above threshold goes right", []float64{0.7, 9}, +1},
		{"NaN follows missing_go_to_left", []float64{math.NaN(), 9}, -1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tree.leafValue(tc.x); got != tc.want {
				t.Errorf("leafValue(%v) = %v, want %v", tc.x, got, tc.want)
			}
		})
	}
	t.Run("NaN goes right when missing_go_to_left is 0", func(t *testing.T) {
		right := singleSplitTree(0, 0.5, -1, +1, 60, 40, 0)
		if got := right.leafValue([]float64{math.NaN(), 9}); got != +1 {
			t.Errorf("got %v, want +1", got)
		}
	})
}

func TestTreeMean(t *testing.T) {
	t.Run("cover-weighted average", func(t *testing.T) {
		tree := singleSplitTree(0, 0.5, -1, +1, 60, 40, 1)
		want := (60.0*-1 + 40.0*1) / 100.0
		if got := tree.treeMean(); math.Abs(got-want) > 1e-15 {
			t.Errorf("treeMean = %v, want %v", got, want)
		}
	})
	t.Run("balanced leaves cancel", func(t *testing.T) {
		tree := singleSplitTree(0, 0.5, -2, +2, 50, 50, 1)
		if got := tree.treeMean(); got != 0 {
			t.Errorf("treeMean = %v, want 0", got)
		}
	})
}

func TestRawPredictSynthetic(t *testing.T) {
	m := syntheticModel(0.25,
		singleSplitTree(0, 0.5, -1, +1, 50, 50, 1),
		singleSplitTree(1, 10, -0.5, +0.5, 50, 50, 1),
	)
	cases := []struct {
		name string
		x    []float64
		want float64
	}{
		{"both left", []float64{0, 5}, 0.25 - 1 - 0.5},
		{"both right", []float64{1, 20}, 0.25 + 1 + 0.5},
		{"mixed", []float64{1, 5}, 0.25 + 1 - 0.5},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := m.RawPredict(tc.x); math.Abs(got-tc.want) > 1e-15 {
				t.Errorf("RawPredict = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestPredictProbaSynthetic(t *testing.T) {
	m := syntheticModel(0, singleSplitTree(0, 0.5, -2, 2, 50, 50, 1))
	t.Run("is sigmoid of raw", func(t *testing.T) {
		want := 1 / (1 + math.Exp(-2))
		if got := m.PredictProba([]float64{1, 0}); math.Abs(got-want) > 1e-15 {
			t.Errorf("proba = %v, want %v", got, want)
		}
	})
	t.Run("symmetric leaves are complementary", func(t *testing.T) {
		left := m.PredictProba([]float64{0, 0})
		right := m.PredictProba([]float64{1, 0})
		if math.Abs(left+right-1) > 1e-15 {
			t.Errorf("P(left)+P(right) = %v, want 1", left+right)
		}
	})
	t.Run("batch matches single", func(t *testing.T) {
		rows := [][]float64{{0, 0}, {1, 0}, {math.NaN(), 0}}
		batch := m.PredictProbaBatch(rows)
		for i, row := range rows {
			if batch[i] != m.PredictProba(row) {
				t.Errorf("row %d: batch %v != single %v", i, batch[i], m.PredictProba(row))
			}
		}
	})
}

func TestExpectedValueSynthetic(t *testing.T) {
	t.Run("baseline plus tree means", func(t *testing.T) {
		m := syntheticModel(0.25, singleSplitTree(0, 0.5, -1, +1, 60, 40, 1))
		want := 0.25 + (60.0*-1+40.0*1)/100.0
		if got := m.ExpectedValue(); math.Abs(got-want) > 1e-15 {
			t.Errorf("EV = %v, want %v", got, want)
		}
	})
	t.Run("cached value is stable", func(t *testing.T) {
		m := syntheticModel(1, singleSplitTree(0, 0.5, -1, +1, 50, 50, 1))
		if m.ExpectedValue() != m.ExpectedValue() {
			t.Error("ExpectedValue changed between calls")
		}
	})
}

func TestShapSingleSplitExact(t *testing.T) {
	// With a single feature in play, the SHAP value must be exactly
	// f(x) - E[f] and all other features must get zero.
	tree := singleSplitTree(0, 0.5, -1, +1, 60, 40, 1)
	m := syntheticModel(0, tree)
	mean := tree.treeMean()

	cases := []struct {
		name string
		x    []float64
		want float64
	}{
		{"right leaf", []float64{0.9, 5}, 1 - mean},
		{"left leaf", []float64{0.1, 5}, -1 - mean},
		{"missing routed left", []float64{math.NaN(), 5}, -1 - mean},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			phi := m.ShapValues(tc.x)
			if math.Abs(phi[0]-tc.want) > 1e-12 {
				t.Errorf("phi[0] = %v, want %v", phi[0], tc.want)
			}
			if phi[1] != 0 {
				t.Errorf("phi[1] = %v, want 0 (feature unused)", phi[1])
			}
		})
	}
}

func TestShapLocalAccuracySynthetic(t *testing.T) {
	// Efficiency property on a model with two interacting trees and a
	// depth-2 tree that splits the same feature twice (exercising the
	// duplicate-feature unwind path).
	deep := Tree{ // f0 <= 0.5 -> leaf; else f0 <= 0.75 -> two leaves
		Value:           []float64{0, -1, 0, 0.5, 2},
		Count:           []float64{100, 50, 50, 30, 20},
		FeatureIdx:      []int32{0, 0, 0, 0, 0},
		NumThreshold:    []float64{0.5, 0, 0.75, 0, 0},
		MissingGoToLeft: []uint8{1, 0, 1, 0, 0},
		Left:            []uint32{1, 0, 3, 0, 0},
		Right:           []uint32{2, 0, 4, 0, 0},
		IsLeaf:          []uint8{0, 1, 0, 1, 1},
	}
	m := syntheticModel(0.1,
		deep,
		singleSplitTree(1, 10, -0.3, 0.7, 25, 75, 0),
	)

	rows := [][]float64{
		{0.2, 5}, {0.6, 5}, {0.9, 20}, {math.NaN(), 20}, {0.6, math.NaN()},
	}
	for i, x := range rows {
		phi := m.ShapValues(x)
		sum := m.ExpectedValue()
		for _, p := range phi {
			sum += p
		}
		raw := m.RawPredict(x)
		if math.Abs(sum-raw) > 1e-12 {
			t.Errorf("row %d: sum(phi)+EV = %v, raw = %v", i, sum, raw)
		}
	}
}

func writeModelFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.json")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadErrors(t *testing.T) {
	validTree := `{"value":[0,-1,1],"count":[2,1,1],"feature_idx":[0,0,0],` +
		`"num_threshold":[0.5,0,0],"missing_go_to_left":[1,0,0],` +
		`"left":[1,0,0],"right":[2,0,0],"is_leaf":[0,1,1]}`

	cases := []struct {
		name    string
		content string
	}{
		{"invalid json", `{not json`},
		{"wrong format version", `{"format_version":2,"model_version":"v1","n_features":1,"features":["a"],"baseline_prediction":0,"trees":[` + validTree + `]}`},
		{"no trees", `{"format_version":1,"model_version":"v1","n_features":1,"features":["a"],"baseline_prediction":0,"trees":[]}`},
		{"feature count mismatch", `{"format_version":1,"model_version":"v1","n_features":3,"features":["a"],"baseline_prediction":0,"trees":[` + validTree + `]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := Load(writeModelFile(t, tc.content)); err == nil {
				t.Error("Load succeeded, want error")
			}
		})
	}
	t.Run("missing file", func(t *testing.T) {
		if _, err := Load(filepath.Join(t.TempDir(), "nope.json")); err == nil {
			t.Error("Load succeeded, want error")
		}
	})
	t.Run("valid minimal model loads", func(t *testing.T) {
		path := writeModelFile(t, `{"format_version":1,"model_version":"v9","n_features":1,"features":["a"],"baseline_prediction":0.5,"trees":[`+validTree+`]}`)
		m, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		if m.Version != "v9" || len(m.Trees) != 1 {
			t.Errorf("loaded %s with %d trees", m.Version, len(m.Trees))
		}
		if got := m.PredictProba([]float64{0.9}); math.Abs(got-1/(1+math.Exp(-1.5))) > 1e-15 {
			t.Errorf("proba = %v", got)
		}
	})
}
