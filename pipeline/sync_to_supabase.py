"""
Supabase Sync
Syncs Gold features and training distribution to Supabase PostgreSQL.
Uses psycopg2 execute_values for fast bulk inserts.
"""

import json
import os
import pickle
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"
MODELS_DIR = DATA_DIR / "models"


def get_conn():
    """Create psycopg2 connection from .env DATABASE_URL."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not set in .env.")
    return psycopg2.connect(db_url)


def sync_features(batch_size=10000):
    """Sync Gold features to applicant_features table via bulk upsert."""
    with open(GOLD_DIR / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]

    # Load test set only — simulates active applicants awaiting scoring
    # (training data is historical, not needed in the feature store)
    df = pd.read_parquet(GOLD_DIR / "features_test.parquet")
    print(f"[SYNC] Preparing {len(df):,} rows (test set) for bulk upsert ...")

    df["applicant_id"] = ["LC_" + str(i).zfill(7) for i in range(len(df))]
    now = datetime.now(timezone.utc).isoformat()

    conn = get_conn()
    cur = conn.cursor()

    total = len(df)
    synced = 0

    for start in range(0, total, batch_size):
        batch = df.iloc[start:start + batch_size]
        values = []
        for _, row in batch.iterrows():
            feat_dict = {col: float(row[col]) if pd.notna(row[col]) else None for col in feature_cols}
            non_null = sum(1 for v in feat_dict.values() if v is not None)
            completeness = round(non_null / len(feat_dict), 3)

            values.append((
                row["applicant_id"],
                1,
                now,
                json.dumps(feat_dict),
                completeness,
                int(row.get("fico_score", 0)) if pd.notna(row.get("fico_score")) else None,
                int(row.get("grade_numeric", 0)) if pd.notna(row.get("grade_numeric")) else None,
            ))

        execute_values(
            cur,
            """
            INSERT INTO applicant_features
                (applicant_id, feature_version, computed_at, features, data_completeness, fico_score, grade)
            VALUES %s
            ON CONFLICT (applicant_id)
            DO UPDATE SET
                feature_version = EXCLUDED.feature_version,
                computed_at = EXCLUDED.computed_at,
                features = EXCLUDED.features,
                data_completeness = EXCLUDED.data_completeness,
                fico_score = EXCLUDED.fico_score,
                grade = EXCLUDED.grade
            """,
            values,
            page_size=batch_size,
        )
        conn.commit()

        synced += len(batch)
        print(f"[SYNC] {synced:,}/{total:,} ({synced/total*100:.1f}%)")

    cur.close()
    conn.close()
    print(f"[SYNC] Feature sync complete. {synced:,} rows upserted.")


def sync_training_distribution():
    """Save training score distribution to Supabase for PSI monitoring."""
    model_path = MODELS_DIR / "champion" / "model.pkl"
    meta_path = MODELS_DIR / "champion" / "model_metadata.json"

    if not model_path.exists():
        print("[SYNC] No champion model found, skipping distribution sync.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    train = pd.read_parquet(GOLD_DIR / "features_train.parquet")
    feature_cols = meta["features"]
    X_train = train[feature_cols]
    scores = model.predict_proba(X_train)[:, 1]

    bin_edges = np.linspace(0, 1, 11).tolist()
    counts, _ = np.histogram(scores, bins=bin_edges)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO training_distribution
            (model_version, bin_edges, bin_counts, total_count, metadata)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            meta["version"],
            json.dumps(bin_edges),
            json.dumps(counts.tolist()),
            len(scores),
            json.dumps(meta["metrics"]),
        ),
    )
    conn.commit()
    cur.close()
    conn.close()
    print(f"[SYNC] Training distribution saved for model {meta['version']}.")


def run():
    """Run full Supabase sync."""
    sync_features()
    sync_training_distribution()
    print("[SYNC] Done.")


if __name__ == "__main__":
    run()
