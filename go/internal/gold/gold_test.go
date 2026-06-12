package gold

import "testing"

// Reads the real Gold test parquet and sanity-checks shapes and values
// against known properties of the dataset.
func TestReadColumns(t *testing.T) {
	frame, err := ReadColumns("../../../data/gold/features_test.parquet",
		[]string{"loan_amnt", "fico_score", "default", "emp_length"})
	if err != nil {
		t.Skipf("gold parquet not available: %v", err)
	}
	if frame.NumRows != 145841 {
		t.Errorf("rows: got %d want 145841", frame.NumRows)
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
