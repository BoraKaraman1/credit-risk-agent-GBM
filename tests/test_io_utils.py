"""Tests for pipeline.io_utils atomic writes.

The medallion layers use skip-if-exists semantics, so a partially written
file must never appear at its final path: later runs would treat the
truncated artifact as complete and immutable. These tests pin the
tmp+fsync+rename behavior and the cleanup-on-failure contract.
"""

import json

import pandas as pd
import pytest

from pipeline.io_utils import atomic_write_json, atomic_write_parquet


def test_atomic_write_parquet_roundtrip(tmp_path):
    dest = tmp_path / "out.parquet"
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    atomic_write_parquet(df, dest)
    assert dest.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(dest), df)
    # No temp files left behind
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_parquet_overwrites_atomically(tmp_path):
    dest = tmp_path / "out.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1]}), dest)
    atomic_write_parquet(pd.DataFrame({"a": [2, 3]}), dest)
    assert len(pd.read_parquet(dest)) == 2


def test_atomic_write_parquet_failure_leaves_dest_untouched(tmp_path):
    dest = tmp_path / "out.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1]}), dest)
    before = dest.read_bytes()

    class Boom:
        def to_parquet(self, *args, **kwargs):
            raise RuntimeError("simulated crash mid-write")

    with pytest.raises(RuntimeError):
        atomic_write_parquet(Boom(), dest)

    assert dest.read_bytes() == before  # original preserved
    assert list(tmp_path.glob("*.tmp")) == []  # temp cleaned up


def test_atomic_write_parquet_failure_with_no_prior_dest(tmp_path):
    dest = tmp_path / "out.parquet"

    class Boom:
        def to_parquet(self, *args, **kwargs):
            raise RuntimeError("simulated crash mid-write")

    with pytest.raises(RuntimeError):
        atomic_write_parquet(Boom(), dest)

    assert not dest.exists()  # no partial artifact visible
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_roundtrip(tmp_path):
    dest = tmp_path / "meta.json"
    payload = {"feature_version": 1, "cols": ["a", "b"]}
    atomic_write_json(payload, dest)
    assert json.loads(dest.read_text()) == payload
    assert list(tmp_path.glob("*.tmp")) == []
