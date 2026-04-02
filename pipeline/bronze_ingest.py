"""
Bronze Ingestion Layer
Reads raw CSV.gz files and writes them as Parquet with metadata columns.
Bronze is immutable — data lands exactly as received, never mutated.
"""

import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BRONZE_DIR = DATA_DIR / "bronze"


def ingest_accepted():
    """Ingest accepted loans CSV.gz → Bronze Parquet."""
    source = DATA_DIR / "accepted_2007_to_2018Q4.csv.gz"
    dest = BRONZE_DIR / "accepted_2007_2018.parquet"

    if dest.exists():
        print(f"[SKIP] {dest} already exists. Bronze is immutable.")
        return

    print(f"[BRONZE] Reading {source} ...")
    df = pd.read_csv(source, low_memory=False)

    # Drop the two completely empty trailing rows that Lending Club files sometimes have
    df = df.dropna(how="all")

    # Metadata columns — when and where this data came from
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    df["source_file"] = source.name

    print(f"[BRONZE] Writing {len(df):,} rows × {len(df.columns)} cols → {dest}")
    df.to_parquet(dest, index=False, engine="pyarrow")
    print(f"[BRONZE] Accepted loans ingested. Size: {dest.stat().st_size / 1e6:.1f} MB")


def ingest_rejected():
    """Ingest rejected loans CSV.gz → Bronze Parquet."""
    source = DATA_DIR / "rejected_2007_to_2018Q4.csv.gz"
    dest = BRONZE_DIR / "rejected_2007_2018.parquet"

    if dest.exists():
        print(f"[SKIP] {dest} already exists. Bronze is immutable.")
        return

    print(f"[BRONZE] Reading {source} ...")
    df = pd.read_csv(source, low_memory=False)
    df = df.dropna(how="all")

    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    df["source_file"] = source.name

    print(f"[BRONZE] Writing {len(df):,} rows × {len(df.columns)} cols → {dest}")
    df.to_parquet(dest, index=False, engine="pyarrow")
    print(f"[BRONZE] Rejected loans ingested. Size: {dest.stat().st_size / 1e6:.1f} MB")


def run():
    """Run full Bronze ingestion."""
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    ingest_accepted()
    ingest_rejected()
    print("[BRONZE] Done.")


if __name__ == "__main__":
    run()
