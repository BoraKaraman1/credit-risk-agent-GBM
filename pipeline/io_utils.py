"""
Atomic file writes for pipeline artifacts.

Every medallion layer uses skip-if-exists semantics, so a partially written
file must never be visible at its final path: a crash mid-write would
otherwise leave a truncated artifact that all later runs treat as complete
and immutable. Writes go to a temp file in the same directory (same
filesystem, so os.replace is atomic) and are fsync'd before the rename.
On failure the temp file is removed and the destination is left untouched.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def _replace_atomic(tmp_path: Path, dest: Path) -> None:
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp_path, dest)


def _tmp_path_for(dest: Path) -> Path:
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".tmp")
    os.close(fd)
    return Path(tmp)


def atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Write a DataFrame to parquet atomically (tmp file + fsync + rename)."""
    dest = Path(dest)
    tmp_path = _tmp_path_for(dest)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")
        _replace_atomic(tmp_path, dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(obj, dest: Path) -> None:
    """Write a JSON document atomically (tmp file + fsync + rename)."""
    dest = Path(dest)
    tmp_path = _tmp_path_for(dest)
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2)
        _replace_atomic(tmp_path, dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
