"""
Tests for the data pipeline: Bronze → Silver → Gold transformations.
Uses small synthetic DataFrames — does not require real data files.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

# --- Silver transform unit tests ---

from pipeline.silver_transform import (
    parse_term,
    parse_emp_length,
    DEFAULT_STATUSES,
    NON_DEFAULT_STATUSES,
    ORIGINATION_COLS,
)


class TestParseTerm:
    def test_standard(self):
        assert parse_term(" 36 months") == 36
        assert parse_term(" 60 months") == 60

    def test_no_leading_space(self):
        assert parse_term("36 months") == 36

    def test_nan(self):
        assert np.isnan(parse_term(np.nan))


class TestParseEmpLength:
    def test_years(self):
        assert parse_emp_length("5 years") == 5
        assert parse_emp_length("1 year") == 1

    def test_ten_plus(self):
        assert parse_emp_length("10+ years") == 10

    def test_less_than_one(self):
        assert parse_emp_length("< 1 year") == 0

    def test_nan(self):
        assert np.isnan(parse_emp_length(np.nan))

    def test_na_string(self):
        assert np.isnan(parse_emp_length("n/a"))


class TestTargetMapping:
    def test_default_statuses_are_distinct(self):
        assert DEFAULT_STATUSES.isdisjoint(NON_DEFAULT_STATUSES)

    def test_known_defaults(self):
        assert "Charged Off" in DEFAULT_STATUSES
        assert "Late (31-120 days)" in DEFAULT_STATUSES

    def test_known_non_defaults(self):
        assert "Fully Paid" in NON_DEFAULT_STATUSES

    def test_ambiguous_excluded(self):
        assert "Current" not in DEFAULT_STATUSES
        assert "Current" not in NON_DEFAULT_STATUSES
        assert "In Grace Period" not in DEFAULT_STATUSES


class TestOriginationCols:
    def test_no_leakage_columns(self):
        leakage = [
            "total_pymnt", "total_rec_prncp", "total_rec_int",
            "recoveries", "last_pymnt_amnt", "collection_recovery_fee",
        ]
        for col in leakage:
            assert col not in ORIGINATION_COLS, f"Leakage column {col} in ORIGINATION_COLS"

    def test_expected_columns_present(self):
        expected = ["loan_amnt", "int_rate", "grade", "dti", "fico_range_low", "annual_inc"]
        for col in expected:
            assert col in ORIGINATION_COLS


# --- Gold feature engineering tests ---

from pipeline.gold_features import engineer_features, GRADE_MAP


def _make_silver_df(n=100):
    """Create a minimal Silver-like DataFrame for testing feature engineering."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "loan_amnt": rng.uniform(1000, 40000, n),
        "term": rng.choice([36, 60], n),
        "int_rate": rng.uniform(5, 30, n),
        "installment": rng.uniform(50, 1500, n),
        "grade": rng.choice(list(GRADE_MAP.keys()), n),
        "sub_grade": [f"{g}{rng.randint(1,6)}" for g in rng.choice(list(GRADE_MAP.keys()), n)],
        "emp_length": rng.choice([0, 1, 3, 5, 10, -1], n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "ANY"], n),
        "annual_inc": rng.uniform(20000, 200000, n),
        "verification_status": rng.choice(["Verified", "Not Verified", "Source Verified"], n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "home_improvement"], n),
        "dti": rng.uniform(0, 50, n),
        "delinq_2yrs": rng.choice([0, 0, 0, 1, 2], n),
        "credit_history_months": rng.randint(12, 400, n),
        "fico_score": rng.randint(620, 820, n),
        "inq_last_6mths": rng.choice([0, 1, 2, 3], n),
        "mths_since_last_delinq": rng.choice([999, 10, 24, 36], n),
        "open_acc": rng.randint(2, 30, n),
        "pub_rec": rng.choice([0, 0, 0, 1], n),
        "revol_bal": rng.uniform(0, 50000, n),
        "revol_util": rng.uniform(0, 100, n),
        "total_acc": rng.randint(4, 60, n),
        "mort_acc": rng.choice([0, 0, 1, 2, 3], n),
        "pub_rec_bankruptcies": rng.choice([0, 0, 0, 1], n),
        "emp_length_missing": rng.choice([0, 1], n),
        "default": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })


class TestEngineerFeatures:
    def setup_method(self):
        self.df = _make_silver_df()
        self.result = engineer_features(self.df.copy())

    def test_log_annual_inc_created(self):
        assert "log_annual_inc" in self.result.columns
        assert (self.result["log_annual_inc"] >= 0).all()

    def test_loan_to_income_created(self):
        assert "loan_to_income" in self.result.columns
        assert not self.result["loan_to_income"].isna().any()

    def test_grade_numeric(self):
        assert "grade_numeric" in self.result.columns
        assert self.result["grade_numeric"].min() >= 1
        assert self.result["grade_numeric"].max() <= 7

    def test_binary_flags(self):
        for col in ["delinq_ever", "high_utilization", "has_mortgage", "has_bankruptcy"]:
            assert col in self.result.columns
            assert set(self.result[col].unique()).issubset({0, 1})

    def test_grade_and_sub_grade_dropped(self):
        assert "grade" not in self.result.columns
        assert "sub_grade" not in self.result.columns

    def test_sub_grade_numeric_range(self):
        assert "sub_grade_numeric" in self.result.columns
        valid = self.result["sub_grade_numeric"].dropna()
        assert valid.min() >= 1
        assert valid.max() <= 35

    def test_home_ownership_rare_collapsed(self):
        # After encoding, original "ANY" should have been mapped to "OTHER" then encoded
        assert "home_ownership" in self.result.columns

    def test_no_nulls_in_derived(self):
        for col in ["loan_to_income", "installment_to_income"]:
            assert not self.result[col].isna().any(), f"NaN found in {col}"

    def test_dti_x_income(self):
        assert "dti_x_income" in self.result.columns
        assert (self.result["dti_x_income"] >= 0).all()


class TestGradeMap:
    def test_complete(self):
        assert set(GRADE_MAP.keys()) == {"A", "B", "C", "D", "E", "F", "G"}

    def test_ordinal(self):
        assert GRADE_MAP["A"] < GRADE_MAP["G"]
        values = sorted(GRADE_MAP.values())
        assert values == list(range(1, 8))
