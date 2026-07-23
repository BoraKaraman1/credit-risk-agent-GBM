"""
Gold Feature Layer
Silver → Gold: engineer derived features, encode categoricals, time-aware split.
Produces train/val/test Parquet files ready for model training.
"""

import json
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import config
from pipeline.data_quality import validate_gold

logger = logging.getLogger(__name__)

DATA_DIR = config.data_dir()
SILVER_DIR = config.silver_dir()
GOLD_DIR = config.gold_dir()

# Grade ordinal mapping
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

# Home ownership categories to keep (rare ones → OTHER)
HOME_KEEP = {"RENT", "OWN", "MORTGAGE"}

# Categorical columns label-encoded by sorted category order. The maps are
# persisted (build_category_maps) so offline, future, and online feature
# generation all encode categories the same way.
CATEGORICAL_COLUMNS = ["purpose", "verification_status", "home_ownership"]

# Feature-store schema version. Bump whenever feature definitions or
# encodings change; serving rejects applicant snapshots whose stored
# feature_version does not match the model's, so incompatible inputs are
# never scored.
FEATURE_VERSION = 1


def build_category_maps(df: pd.DataFrame, columns) -> dict:
    """Map each category to an integer code by sorted order. This
    reproduces pandas cat.codes deterministically, so the resulting
    encoding can be persisted and reapplied identically."""
    maps = {}
    for col in columns:
        categories = sorted(df[col].dropna().unique().tolist())
        maps[col] = {str(c): i for i, c in enumerate(categories)}
    return maps


def apply_category_maps(df: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """Apply persisted category maps. An unseen category encodes to -1,
    matching cat.codes' sentinel for missing values."""
    for col, mapping in maps.items():
        df[col] = df[col].map(lambda v, m=mapping: m.get(str(v), -1)).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from the 25 Silver base columns."""

    # --- Derived numeric features ---
    monthly_inc = df["annual_inc"] / 12
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["installment_to_income"] = df["installment"] / monthly_inc.replace(0, np.nan)
    df["dti_x_income"] = df["dti"] * df["annual_inc"] / 1000  # absolute debt burden (in $k)

    # --- Binary flags ---
    df["grade_numeric"] = df["grade"].map(GRADE_MAP)
    df["delinq_ever"] = (df["delinq_2yrs"] > 0).astype(int)
    df["high_utilization"] = (df["revol_util"] > 75).astype(int)
    df["has_mortgage"] = (df["mort_acc"] > 0).astype(int)
    df["has_bankruptcy"] = (df["pub_rec_bankruptcies"] > 0).astype(int)

    # --- Categorical encoding ---
    # home_ownership: consolidate rare categories
    df["home_ownership"] = df["home_ownership"].apply(
        lambda x: x if x in HOME_KEEP else "OTHER"
    )

    # Sub-grade numeric: A1=1, A2=2, ..., G5=35
    df["sub_grade_numeric"] = df["sub_grade"].apply(
        lambda x: (GRADE_MAP.get(x[0], 0) - 1) * 5 + int(x[1]) if pd.notna(x) and len(x) == 2 else np.nan
    )

    # Purpose, verification_status, home_ownership → label encode via
    # explicit, persisted maps (same ordering as the previous cat.codes,
    # but stable across runs and reproducible for online scoring).
    category_maps = build_category_maps(df, CATEGORICAL_COLUMNS)
    df = apply_category_maps(df, category_maps)

    # Drop columns replaced by engineered versions
    df = df.drop(columns=["grade", "sub_grade"])

    # Fill any remaining NaN from division by zero
    df["loan_to_income"] = df["loan_to_income"].fillna(0)
    df["installment_to_income"] = df["installment_to_income"].fillna(0)

    df.attrs["category_maps"] = category_maps
    return df


def time_aware_split(df: pd.DataFrame):
    """
    Split by issue date:
    - Train: everything before 2016-01-01
    - Validation: 2016-01-01 to 2017-06-30
    - Test: 2017-07-01 onward

    This mimics real deployment: train on historical, validate on recent, test on newest.
    """
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="mixed")

    train = df[df["issue_date"] < "2016-01-01"].copy()
    val = df[(df["issue_date"] >= "2016-01-01") & (df["issue_date"] < "2017-07-01")].copy()
    test = df[df["issue_date"] >= "2017-07-01"].copy()

    # Drop issue_d and issue_date — not a feature
    for split in [train, val, test]:
        split.drop(columns=["issue_d", "issue_date"], inplace=True)

    return train, val, test


def run():
    """Run Gold feature engineering and splitting."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Reading Silver accepted ...")
    df = pd.read_parquet(SILVER_DIR / "accepted_clean.parquet")
    logger.info(f"Silver rows: {len(df):,}")

    logger.info("Engineering features ...")
    df = engineer_features(df)
    category_maps = df.attrs.get("category_maps", {})

    logger.info("Time-aware splitting ...")
    train, val, test = time_aware_split(df)

    logger.info(f"Train: {len(train):,} (default rate: {train['default'].mean():.4f})")
    logger.info(f"Val:   {len(val):,} (default rate: {val['default'].mean():.4f})")
    logger.info(f"Test:  {len(test):,} (default rate: {test['default'].mean():.4f})")

    # Validate before writing: a strict-mode failure must not leave a
    # rejected artifact behind for downstream consumers. (The feature-
    # schema check compares against the previous run's metadata, so it
    # doubles as a schema-drift alarm.)
    for name, split_df in [("train", train), ("val", val), ("test", test)]:
        result = validate_gold(split_df, split_name=name)
        if not result["success"]:
            config.enforce_data_quality(f"Gold ({name})", str(result))

    # Save splits
    train.to_parquet(GOLD_DIR / "features_train.parquet", index=False)
    val.to_parquet(GOLD_DIR / "features_val.parquet", index=False)
    test.to_parquet(GOLD_DIR / "features_test.parquet", index=False)

    # Feature columns (everything except target)
    feature_cols = [c for c in train.columns if c != "default"]

    # Save metadata
    metadata = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "feature_version": FEATURE_VERSION,
        "target": "default",
        "splits": {
            "train": {"rows": len(train), "default_rate": round(train["default"].mean(), 4)},
            "val": {"rows": len(val), "default_rate": round(val["default"].mean(), 4)},
            "test": {"rows": len(test), "default_rate": round(test["default"].mean(), 4)},
        },
        "split_method": "time-aware (train <2016, val 2016-H1 2017, test >=2017-07)",
        "categorical_encodings": category_maps,
    }
    meta_path = GOLD_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Features: {feature_cols}")
    logger.info(f"Written to {GOLD_DIR}")
    logger.info("Gold feature engineering done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
