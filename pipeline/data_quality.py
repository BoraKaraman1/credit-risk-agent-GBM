"""
Data Quality Module
Defines Great Expectations validation suites for Bronze, Silver, and Gold layers.
Returns structured validation results for pipeline integration.
"""

import json
import logging
import great_expectations as gx
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLD_DIR = DATA_DIR / "gold"

# Expected Gold feature columns (loaded from metadata at validation time)
BINARY_COLUMNS = [
    "delinq_ever", "high_utilization", "has_mortgage",
    "has_bankruptcy", "emp_length_missing",
]


def _run_expectations(df: pd.DataFrame, expectations: list[dict]) -> list[dict]:
    """Run a list of expectations against a DataFrame using GX."""
    context = gx.get_context()
    data_source = context.data_sources.add_or_update_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="validation_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch_def")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    results = []
    for exp in expectations:
        try:
            expectation = gx.expectations.registry.get_expectation_impl(
                exp["type"]
            )(**exp.get("kwargs", {}))
            validation = batch.validate(expectation)
            results.append({
                "expectation": exp["type"],
                "kwargs": exp.get("kwargs", {}),
                "success": validation.success,
                "observed_value": getattr(validation, "result", {}).get("observed_value"),
            })
        except Exception as e:
            results.append({
                "expectation": exp["type"],
                "kwargs": exp.get("kwargs", {}),
                "success": False,
                "error": str(e),
            })
    return results


def validate_bronze(df: pd.DataFrame, source_name: str = "accepted") -> dict:
    """
    Validate Bronze layer data.
    Expectations:
    - Row count > 0
    - ingested_at column exists and is non-null
    - source_file column exists and is non-null
    """
    results = []

    # Row count
    row_ok = len(df) > 0
    results.append({
        "expectation": "row_count_gt_zero",
        "success": row_ok,
        "observed_value": len(df),
    })

    # ingested_at exists and non-null
    if "ingested_at" in df.columns:
        null_pct = df["ingested_at"].isnull().mean()
        results.append({
            "expectation": "ingested_at_not_null",
            "success": null_pct == 0,
            "observed_value": f"{null_pct:.4f} null fraction",
        })
    else:
        results.append({
            "expectation": "ingested_at_exists",
            "success": False,
            "observed_value": "column missing",
        })

    # source_file exists and non-null
    if "source_file" in df.columns:
        null_pct = df["source_file"].isnull().mean()
        results.append({
            "expectation": "source_file_not_null",
            "success": null_pct == 0,
            "observed_value": f"{null_pct:.4f} null fraction",
        })
    else:
        results.append({
            "expectation": "source_file_exists",
            "success": False,
            "observed_value": "column missing",
        })

    success = all(r["success"] for r in results)
    if success:
        logger.info(f"Bronze ({source_name}) validation passed ({len(results)} checks)")
    else:
        failed = [r for r in results if not r["success"]]
        logger.warning(f"Bronze ({source_name}) validation failed: {failed}")

    return {"success": success, "results": results, "layer": "bronze", "source": source_name}


def validate_silver(df: pd.DataFrame) -> dict:
    """
    Validate Silver layer data.
    Expectations:
    - No nulls in core columns: loan_amnt, int_rate, fico_score
    - fico_score between 300-850
    - annual_inc >= 0
    - dti between 0-100
    - default is binary (0 or 1)
    - Default rate between 5%-50% (sanity check)
    """
    results = []

    # Row count
    results.append({
        "expectation": "row_count_gt_zero",
        "success": len(df) > 0,
        "observed_value": len(df),
    })

    # Non-null core columns
    for col in ["loan_amnt", "int_rate", "fico_score"]:
        if col in df.columns:
            null_pct = df[col].isnull().mean()
            results.append({
                "expectation": f"{col}_not_null",
                "success": null_pct == 0,
                "observed_value": f"{null_pct:.6f} null fraction",
            })

    # fico_score range
    if "fico_score" in df.columns:
        in_range = df["fico_score"].between(300, 850).all()
        results.append({
            "expectation": "fico_score_between_300_850",
            "success": bool(in_range),
            "observed_value": f"min={df['fico_score'].min()}, max={df['fico_score'].max()}",
        })

    # annual_inc >= 0
    if "annual_inc" in df.columns:
        non_neg = (df["annual_inc"] >= 0).all()
        results.append({
            "expectation": "annual_inc_non_negative",
            "success": bool(non_neg),
            "observed_value": f"min={df['annual_inc'].min():.2f}",
        })

    # dti 0-100
    if "dti" in df.columns:
        in_range = df["dti"].between(0, 100).all()
        results.append({
            "expectation": "dti_between_0_100",
            "success": bool(in_range),
            "observed_value": f"min={df['dti'].min():.2f}, max={df['dti'].max():.2f}",
        })

    # default is binary
    if "default" in df.columns:
        unique_vals = set(df["default"].dropna().unique())
        is_binary = unique_vals.issubset({0, 1})
        results.append({
            "expectation": "default_is_binary",
            "success": is_binary,
            "observed_value": f"unique values: {unique_vals}",
        })

        # Default rate sanity check
        default_rate = df["default"].mean()
        rate_ok = 0.05 <= default_rate <= 0.50
        results.append({
            "expectation": "default_rate_between_5_50_pct",
            "success": rate_ok,
            "observed_value": f"{default_rate:.4f}",
        })

    success = all(r["success"] for r in results)
    if success:
        logger.info(f"Silver validation passed ({len(results)} checks)")
    else:
        failed = [r for r in results if not r["success"]]
        logger.warning(f"Silver validation failed: {failed}")

    return {"success": success, "results": results, "layer": "silver"}


def validate_gold(df: pd.DataFrame, split_name: str = "train") -> dict:
    """
    Validate Gold layer data.
    Expectations:
    - All feature columns present (loaded from metadata or checked against known list)
    - grade_numeric between 1-7
    - sub_grade_numeric between 1-35
    - Binary flags are 0/1
    - Row count > 1000
    - Default rate reasonable (5%-50%)
    """
    results = []

    # Row count
    row_ok = len(df) > 1000
    results.append({
        "expectation": "row_count_gt_1000",
        "success": row_ok,
        "observed_value": len(df),
    })

    # Load expected feature columns from metadata if available
    meta_path = GOLD_DIR / "feature_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        expected_cols = meta["feature_columns"]
        missing = [c for c in expected_cols if c not in df.columns]
        results.append({
            "expectation": "all_feature_columns_present",
            "success": len(missing) == 0,
            "observed_value": f"missing: {missing}" if missing else "all present",
        })

    # grade_numeric range
    if "grade_numeric" in df.columns:
        in_range = df["grade_numeric"].dropna().between(1, 7).all()
        results.append({
            "expectation": "grade_numeric_between_1_7",
            "success": bool(in_range),
            "observed_value": f"min={df['grade_numeric'].min()}, max={df['grade_numeric'].max()}",
        })

    # sub_grade_numeric range
    if "sub_grade_numeric" in df.columns:
        valid = df["sub_grade_numeric"].dropna()
        in_range = valid.between(1, 35).all() if len(valid) > 0 else True
        results.append({
            "expectation": "sub_grade_numeric_between_1_35",
            "success": bool(in_range),
            "observed_value": f"min={valid.min() if len(valid) > 0 else 'N/A'}, max={valid.max() if len(valid) > 0 else 'N/A'}",
        })

    # Binary flags
    for col in BINARY_COLUMNS:
        if col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            is_binary = unique_vals.issubset({0, 1})
            results.append({
                "expectation": f"{col}_is_binary",
                "success": is_binary,
                "observed_value": f"unique: {unique_vals}",
            })

    # Default rate
    if "default" in df.columns:
        default_rate = df["default"].mean()
        rate_ok = 0.05 <= default_rate <= 0.50
        results.append({
            "expectation": "default_rate_between_5_50_pct",
            "success": rate_ok,
            "observed_value": f"{default_rate:.4f}",
        })

    success = all(r["success"] for r in results)
    if success:
        logger.info(f"Gold ({split_name}) validation passed ({len(results)} checks)")
    else:
        failed = [r for r in results if not r["success"]]
        logger.warning(f"Gold ({split_name}) validation failed: {failed}")

    return {"success": success, "results": results, "layer": "gold", "split": split_name}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate existing Gold data
    import pandas as pd
    for split in ["train", "val", "test"]:
        path = GOLD_DIR / f"features_{split}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            result = validate_gold(df, split_name=split)
            status = "PASS" if result["success"] else "FAIL"
            logger.info(f"Gold {split}: {status}")
