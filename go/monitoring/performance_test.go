package monitoring

import "testing"

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
