package model

import "math"

// Path-dependent TreeSHAP (Lundberg et al., "Consistent Individualized
// Feature Attribution for Tree Ensembles", Algorithm 2). Produces the
// same attributions as shap.TreeExplainer(model).shap_values(X) in
// log-odds space: sum(phi) + ExpectedValue() == RawPredict(x).

type pathElement struct {
	featureIndex int32
	zeroFraction float64 // fraction of unconditioned paths flowing through
	oneFraction  float64 // 1 when x follows this split, else 0
	pweight      float64 // permutation weight
}

func extendPath(path []pathElement, uniqueDepth int, zeroFraction, oneFraction float64, featureIndex int32) {
	path[uniqueDepth] = pathElement{featureIndex, zeroFraction, oneFraction, 0}
	if uniqueDepth == 0 {
		path[0].pweight = 1
	}
	for i := uniqueDepth - 1; i >= 0; i-- {
		path[i+1].pweight += oneFraction * path[i].pweight * float64(i+1) / float64(uniqueDepth+1)
		path[i].pweight = zeroFraction * path[i].pweight * float64(uniqueDepth-i) / float64(uniqueDepth+1)
	}
}

func unwindPath(path []pathElement, uniqueDepth, pathIndex int) {
	oneFraction := path[pathIndex].oneFraction
	zeroFraction := path[pathIndex].zeroFraction
	nextOnePortion := path[uniqueDepth].pweight

	for j := uniqueDepth - 1; j >= 0; j-- {
		if oneFraction != 0 {
			tmp := path[j].pweight
			path[j].pweight = nextOnePortion * float64(uniqueDepth+1) / (float64(j+1) * oneFraction)
			nextOnePortion = tmp - path[j].pweight*zeroFraction*float64(uniqueDepth-j)/float64(uniqueDepth+1)
		} else {
			path[j].pweight = path[j].pweight * float64(uniqueDepth+1) / (zeroFraction * float64(uniqueDepth-j))
		}
	}
	for j := pathIndex; j < uniqueDepth; j++ {
		path[j].featureIndex = path[j+1].featureIndex
		path[j].zeroFraction = path[j+1].zeroFraction
		path[j].oneFraction = path[j+1].oneFraction
	}
}

func unwoundPathSum(path []pathElement, uniqueDepth, pathIndex int) float64 {
	oneFraction := path[pathIndex].oneFraction
	zeroFraction := path[pathIndex].zeroFraction
	nextOnePortion := path[uniqueDepth].pweight
	total := 0.0

	for j := uniqueDepth - 1; j >= 0; j-- {
		if oneFraction != 0 {
			tmp := nextOnePortion * float64(uniqueDepth+1) / (float64(j+1) * oneFraction)
			total += tmp
			nextOnePortion = path[j].pweight - tmp*zeroFraction*float64(uniqueDepth-j)/float64(uniqueDepth+1)
		} else {
			total += path[j].pweight / (zeroFraction * float64(uniqueDepth-j) / float64(uniqueDepth+1))
		}
	}
	return total
}

// shap accumulates this tree's SHAP values for row x into phi.
func (t *Tree) shap(x []float64, phi []float64) {
	var recurse func(node uint32, uniqueDepth int, parentPath []pathElement, parentZero, parentOne float64, parentFeature int32)
	recurse = func(node uint32, uniqueDepth int, parentPath []pathElement, parentZero, parentOne float64, parentFeature int32) {
		path := make([]pathElement, uniqueDepth+1)
		copy(path, parentPath[:uniqueDepth])
		extendPath(path, uniqueDepth, parentZero, parentOne, parentFeature)

		if t.IsLeaf[node] == 1 {
			for i := 1; i <= uniqueDepth; i++ {
				w := unwoundPathSum(path, uniqueDepth, i)
				el := path[i]
				phi[el.featureIndex] += w * (el.oneFraction - el.zeroFraction) * t.Value[node]
			}
			return
		}

		splitFeature := t.FeatureIdx[node]
		left, right := t.Left[node], t.Right[node]
		v := x[splitFeature]

		var hot, cold uint32
		switch {
		case math.IsNaN(v):
			if t.MissingGoToLeft[node] == 1 {
				hot, cold = left, right
			} else {
				hot, cold = right, left
			}
		case v <= t.NumThreshold[node]:
			hot, cold = left, right
		default:
			hot, cold = right, left
		}

		hotZeroFraction := t.Count[hot] / t.Count[node]
		coldZeroFraction := t.Count[cold] / t.Count[node]
		incomingZero, incomingOne := 1.0, 1.0

		// If this feature was already split on, undo its previous
		// contribution and fold it into the new fractions.
		pathIndex := 0
		for ; pathIndex <= uniqueDepth; pathIndex++ {
			if path[pathIndex].featureIndex == splitFeature {
				break
			}
		}
		if pathIndex != uniqueDepth+1 {
			incomingZero = path[pathIndex].zeroFraction
			incomingOne = path[pathIndex].oneFraction
			unwindPath(path, uniqueDepth, pathIndex)
			uniqueDepth--
		}

		recurse(hot, uniqueDepth+1, path, hotZeroFraction*incomingZero, incomingOne, splitFeature)
		recurse(cold, uniqueDepth+1, path, coldZeroFraction*incomingZero, 0, splitFeature)
	}

	recurse(0, 0, nil, 1, 1, -1)
}

// ShapValues returns per-feature SHAP attributions (log-odds space)
// for a single row.
func (m *Model) ShapValues(x []float64) []float64 {
	phi := make([]float64, m.NFeatures)
	for i := range m.Trees {
		m.Trees[i].shap(x, phi)
	}
	return phi
}
