import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))


def code_cells(path):
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell.get("source", "")
        yield index, "".join(source) if isinstance(source, list) else source


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_code_cells_compile(path):
    for index, source in code_cells(path):
        compile(source, f"{path.name}:cell-{index}", "exec")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_use_current_pipeline_contract(path):
    source = "\n".join(source for _, source in code_cells(path))

    assert "../data" not in source
    assert "model.pkl" not in source
    assert "HistGradientBoostingClassifier" not in source
