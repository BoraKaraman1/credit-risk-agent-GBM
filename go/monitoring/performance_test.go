package monitoring

import "testing"

func TestBaselineAUC(t *testing.T) {
	cases := []struct {
		name    string
		metrics map[string]map[string]float64
		want    float64
		wantOK  bool
	}{
		{"test preferred", map[string]map[string]float64{
			"test": {"auc": 0.72}, "val": {"auc": 0.74}}, 0.72, true},
		{"val fallback", map[string]map[string]float64{
			"val": {"auc": 0.74}}, 0.74, true},
		{"missing auc key", map[string]map[string]float64{
			"test": {"ks": 0.3}}, 0, false},
		{"no splits at all", map[string]map[string]float64{}, 0, false},
		{"nil metrics", nil, 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := baselineAUC(tc.metrics)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("baselineAUC() = (%v, %v), want (%v, %v)", got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

func TestBothClasses(t *testing.T) {
	cases := []struct {
		name string
		y    []int
		want bool
	}{
		{"both classes", []int{0, 1, 0}, true},
		{"all non-default", []int{0, 0, 0}, false},
		{"all default", []int{1, 1}, false},
		{"empty", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := bothClasses(tc.y); got != tc.want {
				t.Errorf("bothClasses(%v) = %v, want %v", tc.y, got, tc.want)
			}
		})
	}
}
