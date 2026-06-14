package monitoring

import "testing"

func TestIndexFromApplicantID(t *testing.T) {
	cases := []struct {
		id     string
		want   int
		wantOK bool
	}{
		{"LC_0000042", 42, true},
		{"LC_0000000", 0, true},
		{"LC_0145840", 145840, true},
		{"LC_", 0, false},
		{"XX_0000042", 0, false},
		{"LC_notanumber", 0, false},
		{"LC_-5", 0, false},
		{"", 0, false},
	}
	for _, tc := range cases {
		got, ok := indexFromApplicantID(tc.id)
		if ok != tc.wantOK || (ok && got != tc.want) {
			t.Errorf("indexFromApplicantID(%q) = (%d, %v), want (%d, %v)",
				tc.id, got, ok, tc.want, tc.wantOK)
		}
	}
}

func TestBuildBackfill(t *testing.T) {
	labels := []bool{false, true, false, true} // test set has 4 rows

	t.Run("pairs ids with labels by index", func(t *testing.T) {
		ids, out, skipped := buildBackfill([]string{"LC_0000001", "LC_0000003"}, labels)
		if len(ids) != 2 || skipped != 0 {
			t.Fatalf("ids=%v skipped=%d", ids, skipped)
		}
		if out[0] != true || out[1] != true {
			t.Errorf("labels = %v, want [true true]", out)
		}
	})
	t.Run("skips out-of-range and malformed ids", func(t *testing.T) {
		ids, out, skipped := buildBackfill(
			[]string{"LC_0000000", "LC_9999999", "garbage"}, labels)
		if len(ids) != 1 || len(out) != 1 {
			t.Fatalf("ids=%v out=%v", ids, out)
		}
		if ids[0] != "LC_0000000" || out[0] != false {
			t.Errorf("kept %v=%v, want LC_0000000=false", ids, out)
		}
		if skipped != 2 {
			t.Errorf("skipped = %d, want 2", skipped)
		}
	})
	t.Run("empty input", func(t *testing.T) {
		ids, _, skipped := buildBackfill(nil, labels)
		if len(ids) != 0 || skipped != 0 {
			t.Errorf("ids=%v skipped=%d, want empty", ids, skipped)
		}
	})
}
