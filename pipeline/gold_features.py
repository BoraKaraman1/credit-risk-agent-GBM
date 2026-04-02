"""
Gold Feature Layer
Silver → Gold: engineer derived features, encode categoricals, time-aware split.
Produces train/val/test Parquet files ready for model training.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Grade ordinal mapping
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

# Home ownership categories to keep (rare ones → OTHER)
HOME_KEEP = {"RENT", "OWN", "MORTGAGE"}


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

    # Purpose, verification_status, home_ownership → label encode
    for col in ["purpose", "verification_status", "home_ownership"]:
        df[col] = df[col].astype("category").cat.codes

    # Drop columns replaced by engineered versions
    df = df.drop(columns=["grade", "sub_grade"])

    # Fill any remaining NaN from division by zero
    df["loan_to_income"] = df["loan_to_income"].fillna(0)
    df["installment_to_income"] = df["installment_to_income"].fillna(0)

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

    print("[GOLD] Reading Silver accepted ...")
    df = pd.read_parquet(SILVER_DIR / "accepted_clean.parquet")
    print(f"[GOLD] Silver rows: {len(df):,}")

    print("[GOLD] Engineering features ...")
    df = engineer_features(df)

    print("[GOLD] Time-aware splitting ...")
    train, val, test = time_aware_split(df)

    print(f"[GOLD] Train: {len(train):,} (default rate: {train['default'].mean():.4f})")
    print(f"[GOLD] Val:   {len(val):,} (default rate: {val['default'].mean():.4f})")
    print(f"[GOLD] Test:  {len(test):,} (default rate: {test['default'].mean():.4f})")

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
        "target": "default",
        "splits": {
            "train": {"rows": len(train), "default_rate": round(train["default"].mean(), 4)},
            "val": {"rows": len(val), "default_rate": round(val["default"].mean(), 4)},
            "test": {"rows": len(test), "default_rate": round(test["default"].mean(), 4)},
        },
        "split_method": "time-aware (train <2016, val 2016-H1 2017, test >=2017-07)",
    }
    meta_path = GOLD_DIR / "feature_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[GOLD] Features: {feature_cols}")
    print(f"[GOLD] Written to {GOLD_DIR}")
    print("[GOLD] Done.")


if __name__ == "__main__":
    run()
