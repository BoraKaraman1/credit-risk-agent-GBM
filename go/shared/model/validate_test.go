package model

import "testing"

// A single internal node splitting into two leaves: the minimal valid tree.
func validStump() Tree {
	return Tree{
		Value:           []float64{0, -1, 1},
		Count:           []float64{100, 50, 50},
		FeatureIdx:      []int32{0, 0, 0},
		NumThreshold:    []float64{0.5, 0, 0},
		MissingGoToLeft: []uint8{1, 0, 0},
		Left:            []uint32{1, 0, 0},
		Right:           []uint32{2, 0, 0},
		IsLeaf:          []uint8{0, 1, 1},
	}
}

func TestValidateTrees(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		m := &Model{NFeatures: 1, Trees: []Tree{validStump()}}
		if err := m.validateTrees(); err != nil {
			t.Errorf("valid tree rejected: %v", err)
		}
	})
	t.Run("child index out of range", func(t *testing.T) {
		tr := validStump()
		tr.Right[0] = 9 // beyond the node array
		m := &Model{NFeatures: 1, Trees: []Tree{tr}}
		if err := m.validateTrees(); err == nil {
			t.Error("out-of-range child index should be rejected")
		}
	})
	t.Run("back-edge child index (cycle)", func(t *testing.T) {
		tr := validStump()
		tr.Left[0] = 0 // points back to itself
		m := &Model{NFeatures: 1, Trees: []Tree{tr}}
		if err := m.validateTrees(); err == nil {
			t.Error("child index <= parent should be rejected")
		}
	})
	t.Run("feature index out of range", func(t *testing.T) {
		tr := validStump()
		tr.FeatureIdx[0] = 5 // model has only 1 feature
		m := &Model{NFeatures: 1, Trees: []Tree{tr}}
		if err := m.validateTrees(); err == nil {
			t.Error("out-of-range feature index should be rejected")
		}
	})
	t.Run("inconsistent array lengths", func(t *testing.T) {
		tr := validStump()
		tr.Left = []uint32{1, 0} // shorter than the rest
		m := &Model{NFeatures: 1, Trees: []Tree{tr}}
		if err := m.validateTrees(); err == nil {
			t.Error("inconsistent node array lengths should be rejected")
		}
	})
}
