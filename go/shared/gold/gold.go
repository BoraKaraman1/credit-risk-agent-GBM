// Package gold reads the Gold-layer parquet feature files
// (data/gold/features_*.parquet) into float64 columns.
package gold

import (
	"fmt"
	"io"
	"math"
	"os"

	"github.com/parquet-go/parquet-go"
)

// Frame holds a column-oriented slice of the Gold feature data.
type Frame struct {
	NumRows int
	Columns map[string][]float64
}

// Rows assembles a row-major matrix in the given column order, the
// shape the model expects. Missing columns yield an error.
func (f *Frame) Rows(cols []string) ([][]float64, error) {
	colData := make([][]float64, len(cols))
	for i, c := range cols {
		data, ok := f.Columns[c]
		if !ok {
			return nil, fmt.Errorf("column %q not in frame", c)
		}
		colData[i] = data
	}
	rows := make([][]float64, f.NumRows)
	flat := make([]float64, f.NumRows*len(cols))
	for r := range rows {
		row := flat[r*len(cols) : (r+1)*len(cols)]
		for i := range cols {
			row[i] = colData[i][r]
		}
		rows[r] = row
	}
	return rows, nil
}

// ReadColumns reads the requested columns from a parquet file.
// Nulls become NaN; integer columns are widened to float64.
// Duplicate column names are read once.
func ReadColumns(path string, cols []string) (*Frame, error) {
	seen := make(map[string]bool, len(cols))
	unique := make([]string, 0, len(cols))
	for _, c := range cols {
		if !seen[c] {
			seen[c] = true
			unique = append(unique, c)
		}
	}
	cols = unique

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}
	pf, err := parquet.OpenFile(f, stat.Size())
	if err != nil {
		return nil, fmt.Errorf("open parquet %s: %w", path, err)
	}

	colIndex := map[string]int{}
	for i, field := range pf.Schema().Fields() {
		colIndex[field.Name()] = i
	}

	frame := &Frame{
		NumRows: int(pf.NumRows()),
		Columns: make(map[string][]float64, len(cols)),
	}
	for _, c := range cols {
		if _, ok := colIndex[c]; !ok {
			return nil, fmt.Errorf("column %q not in %s", c, path)
		}
		frame.Columns[c] = make([]float64, 0, frame.NumRows)
	}

	buf := make([]parquet.Value, 4096)
	for _, rg := range pf.RowGroups() {
		chunks := rg.ColumnChunks()
		for _, c := range cols {
			out := frame.Columns[c]
			pages := chunks[colIndex[c]].Pages()
			for {
				page, err := pages.ReadPage()
				if err == io.EOF {
					break
				}
				if err != nil {
					pages.Close()
					return nil, fmt.Errorf("read %s column %s: %w", path, c, err)
				}
				reader := page.Values()
				for {
					n, err := reader.ReadValues(buf)
					for _, v := range buf[:n] {
						out = append(out, toFloat(v))
					}
					if err == io.EOF {
						break
					}
					if err != nil {
						pages.Close()
						return nil, fmt.Errorf("read %s column %s: %w", path, c, err)
					}
				}
			}
			pages.Close()
			frame.Columns[c] = out
		}
	}

	for _, c := range cols {
		if len(frame.Columns[c]) != frame.NumRows {
			return nil, fmt.Errorf("column %q: read %d values, expected %d rows",
				c, len(frame.Columns[c]), frame.NumRows)
		}
	}
	return frame, nil
}

func toFloat(v parquet.Value) float64 {
	if v.IsNull() {
		return math.NaN()
	}
	switch v.Kind() {
	case parquet.Double:
		return v.Double()
	case parquet.Float:
		return float64(v.Float())
	case parquet.Int64:
		return float64(v.Int64())
	case parquet.Int32:
		return float64(v.Int32())
	case parquet.Boolean:
		if v.Boolean() {
			return 1
		}
		return 0
	default:
		return math.NaN()
	}
}
