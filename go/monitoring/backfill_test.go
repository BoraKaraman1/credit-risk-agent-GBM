package monitoring

import "testing"

func TestLoanIDFromApplicantID(t *testing.T) {
	cases := []struct {
		id     string
		want   int64
		wantOK bool
	}{
		{"LC_68407277", 68407277, true},
		{"LC_0", 0, true},
		{"LC_0000042", 42, true},
		{"LC_", 0, false},
		{"XX_68407277", 0, false},
		{"LC_notanumber", 0, false},
		{"LC_-5", 0, false},
		{"", 0, false},
	}
	for _, tc := range cases {
		got, ok := loanIDFromApplicantID(tc.id)
		if ok != tc.wantOK || (ok && got != tc.want) {
			t.Errorf("loanIDFromApplicantID(%q) = (%d, %v), want (%d, %v)",
				tc.id, got, ok, tc.want, tc.wantOK)
		}
	}
}

func TestBuildBackfill(t *testing.T) {
	labels := map[int64]bool{
		68407277: false,
		68355089: true,
		68341763: false,
		66310712: true,
	}

	t.Run("pairs ids with labels by loan id", func(t *testing.T) {
		ids, out, skipped := buildBackfill([]string{"LC_68355089", "LC_66310712"}, labels)
		if len(ids) != 2 || skipped != 0 {
			t.Fatalf("ids=%v skipped=%d", ids, skipped)
		}
		if out[0] != true || out[1] != true {
			t.Errorf("labels = %v, want [true true]", out)
		}
	})
	t.Run("skips unknown and malformed ids", func(t *testing.T) {
		ids, out, skipped := buildBackfill(
			[]string{"LC_68407277", "LC_9999999", "garbage"}, labels)
		if len(ids) != 1 || len(out) != 1 {
			t.Fatalf("ids=%v out=%v", ids, out)
		}
		if ids[0] != "LC_68407277" || out[0] != false {
			t.Errorf("kept %v=%v, want LC_68407277=false", ids, out)
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
