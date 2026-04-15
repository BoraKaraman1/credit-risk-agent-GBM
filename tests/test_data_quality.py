"""
Tests for the data quality validation module.
Uses synthetic DataFrames — does not require real data files.
"""

import pytest
import pandas as pd
import numpy as np

from pipeline.data_quality import (
    validate_bronze,
    validate_silver,
    validate_gold,
    BINARY_COLUMNS,
)


# --- Helpers ---

def _make_bronze_df(n=100):
    """Create a valid Bronze-like DataFrame."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 40000, n),
        "int_rate": rng.uniform(5, 30, n),
        "grade": rng.choice(["A", "B", "C", "D"], n),
        "ingested_at": "2025-01-01T00:00:00Z",
        "source_file": "test.csv.gz",
    })


def _make_silver_df(n=200):
    """Create a valid Silver-like DataFrame."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 40000, n),
        "int_rate": rng.uniform(5, 30, n),
        "fico_score": rng.randint(620, 800, n),
        "annual_inc": rng.uniform(20000, 200000, n),
        "dti": rng.uniform(0, 50, n),
        "default": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })


def _make_gold_df(n=2000):
    """Create a valid Gold-like DataFrame with all expected features."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 40000, n),
        "term": rng.choice([36, 60], n),
        "int_rate": rng.uniform(5, 30, n),
        "installment": rng.uniform(50, 1500, n),
        "emp_length": rng.choice([0, 1, 3, 5, 10, -1], n),
        "home_ownership": rng.choice([0, 1, 2, 3], n),
        "annual_inc": rng.uniform(20000, 200000, n),
        "verification_status": rng.choice([0, 1, 2], n),
        "purpose": rng.choice(range(14), n),
        "dti": rng.uniform(0, 50, n),
        "delinq_2yrs": rng.choice([0, 0, 0, 1, 2], n),
        "inq_last_6mths": rng.choice([0, 1, 2, 3], n),
        "mths_since_last_delinq": rng.choice([999, 10, 24, 36], n),
        "open_acc": rng.randint(2, 30, n),
        "pub_rec": rng.choice([0, 0, 0, 1], n),
        "revol_bal": rng.uniform(0, 50000, n),
        "revol_util": rng.uniform(0, 100, n),
        "total_acc": rng.randint(4, 60, n),
        "mort_acc": rng.choice([0, 0, 1, 2, 3], n),
        "pub_rec_bankruptcies": rng.choice([0, 0, 0, 1], n),
        "credit_history_months": rng.randint(12, 400, n),
        "fico_score": rng.randint(620, 800, n),
        "emp_length_missing": rng.choice([0, 1], n),
        "log_annual_inc": rng.uniform(10, 13, n),
        "loan_to_income": rng.uniform(0, 2, n),
        "installment_to_income": rng.uniform(0, 0.5, n),
        "dti_x_income": rng.uniform(0, 5000, n),
        "grade_numeric": rng.randint(1, 8, n),
        "delinq_ever": rng.choice([0, 1], n),
        "high_utilization": rng.choice([0, 1], n),
        "has_mortgage": rng.choice([0, 1], n),
        "has_bankruptcy": rng.choice([0, 1], n),
        "sub_grade_numeric": rng.randint(1, 36, n),
        "default": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })


# --- Bronze Validation ---

class TestBronzeValidation:
    def test_valid_bronze(self):
        df = _make_bronze_df()
        result = validate_bronze(df, source_name="test")
        assert result["success"] is True
        assert result["layer"] == "bronze"

    def test_empty_dataframe_fails(self):
        df = pd.DataFrame(columns=["ingested_at", "source_file"])
        result = validate_bronze(df, source_name="test")
        assert result["success"] is False

    def test_missing_ingested_at_fails(self):
        df = _make_bronze_df()
        df = df.drop(columns=["ingested_at"])
        result = validate_bronze(df, source_name="test")
        assert result["success"] is False

    def test_missing_source_file_fails(self):
        df = _make_bronze_df()
        df = df.drop(columns=["source_file"])
        result = validate_bronze(df, source_name="test")
        assert result["success"] is False

    def test_null_ingested_at_fails(self):
        df = _make_bronze_df()
        df.loc[0, "ingested_at"] = None
        result = validate_bronze(df, source_name="test")
        assert result["success"] is False


# --- Silver Validation ---

class TestSilverValidation:
    def test_valid_silver(self):
        df = _make_silver_df()
        result = validate_silver(df)
        assert result["success"] is True
        assert result["layer"] == "silver"

    def test_fico_out_of_range_fails(self):
        df = _make_silver_df()
        df.loc[0, "fico_score"] = 200  # below 300
        result = validate_silver(df)
        assert result["success"] is False

    def test_negative_income_fails(self):
        df = _make_silver_df()
        df.loc[0, "annual_inc"] = -1000
        result = validate_silver(df)
        assert result["success"] is False

    def test_non_binary_default_fails(self):
        df = _make_silver_df()
        df.loc[0, "default"] = 2  # not 0 or 1
        result = validate_silver(df)
        assert result["success"] is False

    def test_null_core_columns_fails(self):
        df = _make_silver_df()
        df.loc[0, "loan_amnt"] = None
        result = validate_silver(df)
        assert result["success"] is False

    def test_dti_out_of_range_fails(self):
        df = _make_silver_df()
        df.loc[0, "dti"] = 150  # above 100
        result = validate_silver(df)
        assert result["success"] is False

    def test_extreme_default_rate_fails(self):
        df = _make_silver_df(n=200)
        df["default"] = 1  # 100% default rate
        result = validate_silver(df)
        assert result["success"] is False


# --- Gold Validation ---

class TestGoldValidation:
    def test_valid_gold(self):
        df = _make_gold_df()
        result = validate_gold(df, split_name="test")
        assert result["success"] is True
        assert result["layer"] == "gold"

    def test_too_few_rows_fails(self):
        df = _make_gold_df(n=100)  # < 1000
        result = validate_gold(df, split_name="test")
        assert result["success"] is False

    def test_grade_numeric_out_of_range_fails(self):
        df = _make_gold_df()
        df.loc[0, "grade_numeric"] = 10  # > 7
        result = validate_gold(df, split_name="test")
        assert result["success"] is False

    def test_binary_column_non_binary_fails(self):
        df = _make_gold_df()
        df.loc[0, "delinq_ever"] = 3  # not 0 or 1
        result = validate_gold(df, split_name="test")
        assert result["success"] is False

    def test_sub_grade_out_of_range_fails(self):
        df = _make_gold_df()
        df.loc[0, "sub_grade_numeric"] = 40  # > 35
        result = validate_gold(df, split_name="test")
        assert result["success"] is False

    def test_extreme_default_rate_fails(self):
        df = _make_gold_df()
        df["default"] = 0  # 0% default rate
        result = validate_gold(df, split_name="test")
        assert result["success"] is False
