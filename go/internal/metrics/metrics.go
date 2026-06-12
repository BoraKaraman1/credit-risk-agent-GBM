// Package metrics ports the monitoring math from agents/drift_monitor.py
// and agents/performance_monitor.py: PSI, CSI, ROC AUC, KS, and decile
// analysis. All functions reproduce the numpy/sklearn/pandas results.
package metrics

import (
	"math"
	"sort"
)

// Histogram counts values into the given monotonically increasing bin
// edges, matching np.histogram: bins are half-open [e[i], e[i+1])
// except the last, which is closed; out-of-range values are dropped.
func Histogram(values, edges []float64) []float64 {
	counts := make([]float64, len(edges)-1)
	last := len(edges) - 1
	for _, v := range values {
		if math.IsNaN(v) || v < edges[0] || v > edges[last] {
			continue
		}
		if v == edges[last] {
			counts[last-1]++
			continue
		}
		i := sort.SearchFloat64s(edges, v)
		if i == len(edges) || edges[i] != v {
			i-- // v falls inside bin i
		}
		counts[i]++
	}
	return counts
}

// Linspace returns n evenly spaced values from lo to hi inclusive,
// like np.linspace.
func Linspace(lo, hi float64, n int) []float64 {
	out := make([]float64, n)
	step := (hi - lo) / float64(n-1)
	for i := range out {
		out[i] = lo + float64(i)*step
	}
	out[n-1] = hi
	return out
}

// quantile returns the q-th quantile (0..1) of sorted data using
// linear interpolation, matching np.percentile's default.
func quantile(sorted []float64, q float64) float64 {
	n := len(sorted)
	if n == 1 {
		return sorted[0]
	}
	pos := q * float64(n-1)
	lo := int(math.Floor(pos))
	if lo >= n-1 {
		return sorted[n-1]
	}
	frac := pos - float64(lo)
	return sorted[lo] + frac*(sorted[lo+1]-sorted[lo])
}

func dropNaN(values []float64) []float64 {
	out := make([]float64, 0, len(values))
	for _, v := range values {
		if !math.IsNaN(v) {
			out = append(out, v)
		}
	}
	return out
}

// smoothedPct applies the same Laplace smoothing as the Python agents:
// (count + 1) / (total + nBins).
func smoothedPct(counts []float64) []float64 {
	total := 0.0
	for _, c := range counts {
		total += c
	}
	out := make([]float64, len(counts))
	denom := total + float64(len(counts))
	for i, c := range counts {
		out[i] = (c + 1) / denom
	}
	return out
}

// PSI computes the Population Stability Index between two score
// distributions over equal-width bins on [0, 1].
//
//	PSI < 0.10: no shift; 0.10-0.25: moderate; > 0.25: retrain
func PSI(expected, actual []float64, bins int) (psi float64, expectedPct, actualPct []float64) {
	edges := Linspace(0, 1, bins+1)
	expectedPct = smoothedPct(Histogram(expected, edges))
	actualPct = smoothedPct(Histogram(actual, edges))
	for i := range expectedPct {
		psi += (actualPct[i] - expectedPct[i]) * math.Log(actualPct[i]/expectedPct[i])
	}
	return psi, expectedPct, actualPct
}

// CSI computes the Characteristic Stability Index for one feature:
// PSI math over percentile bins of the combined distribution.
func CSI(train, production []float64, bins int) float64 {
	combined := dropNaN(append(append([]float64{}, train...), production...))
	if len(combined) == 0 {
		return 0
	}
	sort.Float64s(combined)

	edges := make([]float64, 0, bins+1)
	for i := 0; i <= bins; i++ {
		e := quantile(combined, float64(i)/float64(bins))
		if len(edges) == 0 || e != edges[len(edges)-1] {
			edges = append(edges, e) // np.unique on sorted percentiles
		}
	}
	if len(edges) < 2 {
		return 0
	}

	trainPct := smoothedPct(Histogram(dropNaN(train), edges))
	prodPct := smoothedPct(Histogram(dropNaN(production), edges))

	csi := 0.0
	for i := range trainPct {
		csi += (prodPct[i] - trainPct[i]) * math.Log(prodPct[i]/trainPct[i])
	}
	return csi
}

// ROCAUC computes the area under the ROC curve via the Mann-Whitney U
// statistic with midranks for ties (equivalent to sklearn's
// roc_auc_score for binary labels).
func ROCAUC(yTrue []int, yScore []float64) float64 {
	n := len(yScore)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return yScore[idx[a]] < yScore[idx[b]] })

	ranks := make([]float64, n)
	for i := 0; i < n; {
		j := i
		for j < n && yScore[idx[j]] == yScore[idx[i]] {
			j++
		}
		mid := float64(i+j+1) / 2 // average 1-based rank for the tie group
		for k := i; k < j; k++ {
			ranks[idx[k]] = mid
		}
		i = j
	}

	var nPos, nNeg, rankSum float64
	for i, y := range yTrue {
		if y == 1 {
			nPos++
			rankSum += ranks[i]
		} else {
			nNeg++
		}
	}
	if nPos == 0 || nNeg == 0 {
		return math.NaN()
	}
	return (rankSum - nPos*(nPos+1)/2) / (nPos * nNeg)
}

// KS computes the Kolmogorov-Smirnov statistic: max(TPR - FPR) over
// all score thresholds.
func KS(yTrue []int, yScore []float64) float64 {
	n := len(yScore)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return yScore[idx[a]] > yScore[idx[b]] })

	var nPos, nNeg float64
	for _, y := range yTrue {
		if y == 1 {
			nPos++
		} else {
			nNeg++
		}
	}

	best, tp, fp := 0.0, 0.0, 0.0
	for i := 0; i < n; {
		j := i
		for j < n && yScore[idx[j]] == yScore[idx[i]] {
			if yTrue[idx[j]] == 1 {
				tp++
			} else {
				fp++
			}
			j++
		}
		if d := tp/nPos - fp/nNeg; d > best {
			best = d
		}
		i = j
	}
	return best
}

// Decile summarizes one score decile (pd.qcut with duplicates="drop").
type Decile struct {
	Decile      int     `json:"decile"`
	Count       int     `json:"count"`
	DefaultRate float64 `json:"default_rate"`
	AvgScore    float64 `json:"avg_score"`
}

// DecileAnalysis buckets observations into score deciles and reports
// per-decile default rates, plus the number of rank-ordering breaks
// (deciles whose default rate is below the previous decile's).
func DecileAnalysis(yTrue []int, yScore []float64) (stats []Decile, rankOrderBreaks int) {
	sorted := append([]float64{}, yScore...)
	sort.Float64s(sorted)

	edges := make([]float64, 0, 11)
	for i := 0; i <= 10; i++ {
		e := quantile(sorted, float64(i)/10)
		if len(edges) == 0 || e != edges[len(edges)-1] {
			edges = append(edges, e)
		}
	}
	nBins := len(edges) - 1
	if nBins < 1 {
		return nil, 0
	}

	counts := make([]int, nBins)
	defaults := make([]float64, nBins)
	scoreSum := make([]float64, nBins)
	for i, v := range yScore {
		// qcut intervals are right-closed with the lowest edge included
		b := sort.SearchFloat64s(edges[1:], v)
		if b >= nBins {
			b = nBins - 1
		}
		counts[b]++
		defaults[b] += float64(yTrue[i])
		scoreSum[b] += v
	}

	prev := math.Inf(-1)
	for b := 0; b < nBins; b++ {
		if counts[b] == 0 {
			continue
		}
		rate := defaults[b] / float64(counts[b])
		stats = append(stats, Decile{
			Decile:      len(stats),
			Count:       counts[b],
			DefaultRate: rate,
			AvgScore:    scoreSum[b] / float64(counts[b]),
		})
		if rate < prev {
			rankOrderBreaks++
		}
		prev = rate
	}
	return stats, rankOrderBreaks
}
