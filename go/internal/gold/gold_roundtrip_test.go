package gold

import (
	"math"
	"path/filepath"
	"testing"

	"github.com/parquet-go/parquet-go"
)

// Round-trip tests against a parquet file written in-test, covering
// nulls, integer widening, duplicate requests, and error paths without
// needing the real Gold data.

type testRow struct {
	A float64  `parquet:"a"`
	B *float64 `parquet:"b"` // optional: nil becomes NaN on read
	C int32    `parquet:"c"`
}

func writeTestParquet(t *testing.T) string {
	t.Helper()
	f := func(v float64) *float64 { return &v }
	rows := []testRow{
		{A: 1.5, B: f(10), C: 7},
		{A: -2.25, B: nil, C: -3},
		{A: 0, B: f(-0.5), C: 0},
	}
	path := filepath.Join(t.TempDir(), "test.parquet")
	if err := parquet.WriteFile(path, rows); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadColumnsRoundTrip(t *testing.T) {
	path := writeTestParquet(t)
	frame, err := ReadColumns(path, []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("row count", func(t *testing.T) {
		if frame.NumRows != 3 {
			t.Errorf("NumRows = %d, want 3", frame.NumRows)
		}
	})
	t.Run("float column", func(t *testing.T) {
		want := []float64{1.5, -2.25, 0}
		for i, v := range want {
			if frame.Columns["a"][i] != v {
				t.Errorf("a[%d] = %v, want %v", i, frame.Columns["a"][i], v)
			}
		}
	})
	t.Run("null becomes NaN", func(t *testing.T) {
		b := frame.Columns["b"]
		if b[0] != 10 || !math.IsNaN(b[1]) || b[2] != -0.5 {
			t.Errorf("b = %v", b)
		}
	})
	t.Run("int32 widened to float64", func(t *testing.T) {
		c := frame.Columns["c"]
		if c[0] != 7 || c[1] != -3 || c[2] != 0 {
			t.Errorf("c = %v", c)
		}
	})
}

func TestReadColumnsDuplicatesAndErrors(t *testing.T) {
	path := writeTestParquet(t)

	t.Run("duplicate columns read once", func(t *testing.T) {
		frame, err := ReadColumns(path, []string{"a", "a", "c", "a"})
		if err != nil {
			t.Fatal(err)
		}
		if len(frame.Columns["a"]) != frame.NumRows {
			t.Errorf("a has %d values, want %d", len(frame.Columns["a"]), frame.NumRows)
		}
	})
	t.Run("unknown column errors", func(t *testing.T) {
		if _, err := ReadColumns(path, []string{"a", "nope"}); err == nil {
			t.Error("want error for unknown column")
		}
	})
	t.Run("missing file errors", func(t *testing.T) {
		if _, err := ReadColumns(filepath.Join(t.TempDir(), "x.parquet"), []string{"a"}); err == nil {
			t.Error("want error for missing file")
		}
	})
}

func TestFrameRows(t *testing.T) {
	path := writeTestParquet(t)
	frame, err := ReadColumns(path, []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("respects column order", func(t *testing.T) {
		rows, err := frame.Rows([]string{"c", "a"})
		if err != nil {
			t.Fatal(err)
		}
		if rows[0][0] != 7 || rows[0][1] != 1.5 {
			t.Errorf("row 0 = %v, want [7 1.5]", rows[0])
		}
		if rows[1][0] != -3 || rows[1][1] != -2.25 {
			t.Errorf("row 1 = %v", rows[1])
		}
	})
	t.Run("propagates NaN", func(t *testing.T) {
		rows, err := frame.Rows([]string{"b"})
		if err != nil {
			t.Fatal(err)
		}
		if !math.IsNaN(rows[1][0]) {
			t.Errorf("rows[1][0] = %v, want NaN", rows[1][0])
		}
	})
	t.Run("unknown column errors", func(t *testing.T) {
		if _, err := frame.Rows([]string{"a", "missing"}); err == nil {
			t.Error("want error for unknown column")
		}
	})
}
