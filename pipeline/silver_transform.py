"""
Silver Transform Layer
Bronze → Silver: clean, type-cast, validate, create target variable.
Only origination-time columns are kept (no leakage).
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"

# Columns available at origination — no post-origination leakage
ORIGINATION_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low",
    "fico_range_high", "inq_last_6mths", "mths_since_last_delinq",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies",
]

# Target variable mapping
DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
}
NON_DEFAULT_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
# Dropped: "Current", "In Grace Period", "Late (16-30 days)" — ambiguous or unknown outcome


def parse_term(s):
    """' 36 months' → 36"""
    if pd.isna(s):
        return np.nan
    return int(str(s).strip().replace(" months", ""))


def parse_emp_length(s):
    """'10+ years' → 10, '< 1 year' → 0, 'n/a' → NaN"""
    if pd.isna(s) or str(s).strip().lower() == "n/a":
        return np.nan
    s = str(s).strip()
    if s.startswith("< 1"):
        return 0
    if s.startswith("10+"):
        return 10
    return int(s.split()[0])


def transform_accepted():
    """Transform accepted loans from Bronze to Silver."""
    source = BRONZE_DIR / "accepted_2007_2018.parquet"
    dest = SILVER_DIR / "accepted_clean.parquet"

    logger.info(f"Reading {source} ...")
    df = pd.read_parquet(source)
    logger.info(f"Bronze rows: {len(df):,}")

    # --- Target variable ---
    df = df[df["loan_status"].isin(DEFAULT_STATUSES | NON_DEFAULT_STATUSES)].copy()
    df["default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    logger.info(f"After filtering to resolved loans: {len(df):,}")
    logger.info(f"Default rate: {df['default'].mean():.4f} ({df['default'].sum():,} defaults)")

    # --- Keep issue_d for time-aware splitting in Gold, then origination cols ---
    keep_cols = ORIGINATION_COLS + ["default"]
    if "issue_d" in df.columns:
        keep_cols = ["issue_d"] + keep_cols
    df = df[keep_cols].copy()

    # --- Drop rows with systemic NaN (the ~33 junk rows) ---
    core_cols = ["loan_amnt", "int_rate", "grade", "fico_range_low"]
    df = df.dropna(subset=core_cols, how="any")

    # --- Type casting ---
    df["term"] = df["term"].apply(parse_term)
    df["emp_length"] = df["emp_length"].apply(parse_emp_length)

    # int_rate: strip "%" if present (some Lending Club files have it as string)
    if df["int_rate"].dtype == object:
        df["int_rate"] = df["int_rate"].str.replace("%", "", regex=False).astype(float)

    # earliest_cr_line → credit_history_months
    if "issue_d" in df.columns:
        ref_date = pd.to_datetime(df["issue_d"], format="mixed")
    else:
        ref_date = pd.Timestamp.now()
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="mixed")
    df["credit_history_months"] = (
        (ref_date - df["earliest_cr_line"]).dt.days / 30.44
    ).round().astype("Int64")
    df = df.drop(columns=["earliest_cr_line"])

    # FICO: average of range
    df["fico_score"] = ((df["fico_range_low"] + df["fico_range_high"]) / 2).round().astype(int)
    df = df.drop(columns=["fico_range_low", "fico_range_high"])

    # --- Missingness handling ---
    # mths_since_last_delinq: NaN = never delinquent → 999
    df["mths_since_last_delinq"] = df["mths_since_last_delinq"].fillna(999)

    # emp_length: NaN → -1 + missing flag
    df["emp_length_missing"] = df["emp_length"].isna().astype(int)
    df["emp_length"] = df["emp_length"].fillna(-1)

    # Low-missingness columns: median impute
    for col in ["mort_acc", "pub_rec_bankruptcies", "revol_util", "dti"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # delinq_2yrs, inq_last_6mths, open_acc, pub_rec, total_acc: very few missing → median
    for col in ["delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc"]:
        df[col] = df[col].fillna(df[col].median())

    # --- Data quality gates ---
    n_before = len(df)
    df = df[df["annual_inc"] >= 0]
    df = df[df["dti"].between(0, 100)]
    df = df[df["fico_score"].between(300, 850)]
    n_after = len(df)
    if n_before != n_after:
        logger.info(f"Quality gate removed {n_before - n_after:,} rows")

    # Assert non-null completeness
    null_pcts = df.isnull().mean()
    failing = null_pcts[null_pcts > 0.05]
    if len(failing) > 0:
        logger.warning(f"Columns with >5% null: {failing.to_dict()}")
    else:
        logger.info("Quality gate passed: all columns <5% null")

    logger.info(f"Final accepted: {len(df):,} rows × {len(df.columns)} cols")
    df.to_parquet(dest, index=False, engine="pyarrow")
    logger.info(f"Written to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    try:
        from pipeline.data_quality import validate_silver
        result = validate_silver(df)
        if not result["success"]:
            logger.warning(f"Silver validation failed: {result}")
    except ImportError:
        pass


def transform_rejected():
    """Transform rejected loans from Bronze to Silver."""
    source = BRONZE_DIR / "rejected_2007_2018.parquet"
    dest = SILVER_DIR / "rejected_clean.parquet"

    logger.info(f"Reading {source} ...")
    df = pd.read_parquet(source)
    logger.info(f"Bronze rejected rows: {len(df):,}")

    # Rename to match accepted column conventions
    df = df.rename(columns={
        "Amount Requested": "loan_amnt",
        "Risk_Score": "fico_score",
        "Debt-To-Income Ratio": "dti",
        "Employment Length": "emp_length",
        "Application Date": "application_date",
        "State": "state",
    })

    # Parse DTI: strip "%"
    if df["dti"].dtype == object:
        df["dti"] = df["dti"].str.replace("%", "", regex=False)
        df["dti"] = pd.to_numeric(df["dti"], errors="coerce")

    # Parse emp_length
    df["emp_length"] = df["emp_length"].apply(parse_emp_length)
    df["emp_length_missing"] = df["emp_length"].isna().astype(int)
    df["emp_length"] = df["emp_length"].fillna(-1)

    # No outcome — this is the point of reject inference
    df["default"] = np.nan

    # Keep only columns useful for reject inference
    keep_cols = ["loan_amnt", "fico_score", "dti", "emp_length",
                 "emp_length_missing", "state", "application_date", "default"]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Drop rows with no FICO or loan amount
    df = df.dropna(subset=["loan_amnt", "fico_score"])

    logger.info(f"Final rejected: {len(df):,} rows × {len(df.columns)} cols")
    df.to_parquet(dest, index=False, engine="pyarrow")
    logger.info(f"Written to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def run():
    """Run full Silver transformation."""
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    transform_accepted()
    transform_rejected()
    logger.info("Silver transformation done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
