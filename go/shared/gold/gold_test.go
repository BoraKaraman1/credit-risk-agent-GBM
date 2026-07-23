package gold

import "testing"

// Reads the committed Gold sample (200 rows of the real test parquet,
// written by scripts/generate_model_fixtures.py) and sanity-checks
// shapes and values against known properties of the dataset.
func TestReadColumns(t *testing.T) {
	frame, err := ReadColumns("testdata/sample.parquet",
		[]string{"loan_amnt", "fico_score", "default", "emp_length"})
	if err != nil {
		t.Fatalf("gold sample missing (regenerate with "+
			"scripts/generate_model_fixtures.py): %v", err)
	}
	if frame.NumRows != 200 {
		t.Errorf("rows: got %d want 200", frame.NumRows)
	}
	for name, col := range frame.Columns {
		if len(col) != frame.NumRows {
			t.Errorf("column %s: %d values", name, len(col))
		}
	}
	for i, v := range frame.Columns["default"] {
		if v != 0 && v != 1 {
			t.Fatalf("default[%d] = %v, want 0/1", i, v)
		}
	}
	fico := frame.Columns["fico_score"]
	for i, v := range fico {
		if v < 300 || v > 900 {
			t.Fatalf("fico_score[%d] = %v out of range", i, v)
		}
	}

	rows, err := frame.Rows([]string{"loan_amnt", "fico_score"})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != frame.NumRows || rows[0][1] != fico[0] {
		t.Errorf("Rows() misaligned")
	}
}
