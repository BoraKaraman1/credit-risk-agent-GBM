"""
Tests for the fairness analysis module.
Uses synthetic data — does not require real data files or trained models.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.fairness import (
    compute_disparate_impact,
    compute_equal_opportunity_diff,
    compute_statistical_parity_diff,
    compute_group_metrics,
    analyze_attribute,
    run as run_fairness,
    format_report,
    DIR_THRESHOLD,
    APPROVE_THRESHOLD,
    PROTECTED_ATTRIBUTES,
)


# --- Helpers ---

def _make_fairness_data(n=600, seed=42):
    """Create synthetic data with protected attributes for fairness testing."""
    rng = np.random.RandomState(seed)
    feature_cols = ["feat_0", "feat_1", "feat_2", "home_ownership",
                    "verification_status", "emp_length_missing"]
    X = pd.DataFrame({
        "feat_0": rng.randn(n),
        "feat_1": rng.randn(n),
        "feat_2": rng.randn(n),
        "home_ownership": rng.choice([0, 1, 2, 3], n),
        "verification_status": rng.choice([0, 1, 2], n),
        "emp_length_missing": rng.choice([0, 1], n, p=[0.85, 0.15]),
    })
    y = pd.Series(rng.choice([0, 1], n, p=[0.8, 0.2]), name="default")
    return X, y, feature_cols


def _train_dummy_model(X, y):
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


# --- compute_disparate_impact ---

class TestDisparateImpactRatio:
    def test_equal_rates_returns_one(self):
        rates = {"GroupA": 0.50, "GroupB": 0.50}
        result = compute_disparate_impact(rates, "GroupA")
        assert result["ratios"]["GroupB"] == pytest.approx(1.0)
        assert result["violations"] == []

    def test_below_threshold_flagged(self):
        rates = {"Privileged": 0.60, "Unprivileged": 0.40}
        result = compute_disparate_impact(rates, "Privileged")
        # 0.40 / 0.60 = 0.6667 < 0.80
        assert result["ratios"]["Unprivileged"] < DIR_THRESHOLD
        assert "Unprivileged" in result["violations"]

    def test_above_threshold_not_flagged(self):
        rates = {"Privileged": 0.50, "Unprivileged": 0.45}
        result = compute_disparate_impact(rates, "Privileged")
        # 0.45 / 0.50 = 0.90 > 0.80
        assert result["ratios"]["Unprivileged"] >= DIR_THRESHOLD
        assert result["violations"] == []

    def test_zero_privileged_rate(self):
        rates = {"Privileged": 0.0, "Other": 0.50}
        result = compute_disparate_impact(rates, "Privileged")
        assert result["ratios"] == {}
        assert result["privileged_rate"] == 0

    def test_multiple_groups(self):
        rates = {"P": 0.60, "A": 0.55, "B": 0.30, "C": 0.50}
        result = compute_disparate_impact(rates, "P")
        assert len(result["ratios"]) == 4
        assert "B" in result["violations"]  # 0.30/0.60 = 0.50 < 0.80


# --- compute_equal_opportunity_diff ---

class TestEqualOpportunityDiff:
    def test_equal_tpr_returns_zero(self):
        tprs = {"GroupA": 0.80, "GroupB": 0.80}
        result = compute_equal_opportunity_diff(tprs, "GroupA")
        assert result["GroupB"] == pytest.approx(0.0)
        assert result["GroupA"] == pytest.approx(0.0)

    def test_higher_unprivileged_tpr(self):
        tprs = {"Privileged": 0.70, "Unprivileged": 0.85}
        result = compute_equal_opportunity_diff(tprs, "Privileged")
        assert result["Unprivileged"] == pytest.approx(0.15)

    def test_lower_unprivileged_tpr(self):
        tprs = {"Privileged": 0.80, "Unprivileged": 0.60}
        result = compute_equal_opportunity_diff(tprs, "Privileged")
        assert result["Unprivileged"] == pytest.approx(-0.20)


# --- compute_statistical_parity_diff ---

class TestStatisticalParityDiff:
    def test_equal_approval_rates(self):
        rates = {"A": 0.50, "B": 0.50}
        result = compute_statistical_parity_diff(rates, "A")
        assert result["B"] == pytest.approx(0.0)

    def test_unequal_rates(self):
        rates = {"Privileged": 0.60, "Unprivileged": 0.40}
        result = compute_statistical_parity_diff(rates, "Privileged")
        assert result["Unprivileged"] == pytest.approx(-0.20)
        assert result["Privileged"] == pytest.approx(0.0)


# --- compute_group_metrics ---

class TestGroupMetrics:
    def test_correct_number_of_groups(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 300, p=[0.8, 0.2])
        y_score = rng.rand(300)
        groups = rng.choice([0, 1, 2], 300)
        names = {0: "A", 1: "B", 2: "C"}
        result = compute_group_metrics(y_true, y_score, groups, names)
        assert len(result) == 3

    def test_auc_in_range(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 500, p=[0.7, 0.3])
        y_score = rng.rand(500)
        groups = rng.choice([0, 1], 500)
        names = {0: "A", 1: "B"}
        result = compute_group_metrics(y_true, y_score, groups, names)
        for m in result:
            if m["auc"] is not None:
                assert 0 <= m["auc"] <= 1

    def test_rates_between_zero_and_one(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 300, p=[0.8, 0.2])
        y_score = rng.rand(300)
        groups = rng.choice([0, 1], 300)
        names = {0: "X", 1: "Y"}
        result = compute_group_metrics(y_true, y_score, groups, names)
        for m in result:
            assert 0 <= m["default_rate"] <= 1
            assert 0 <= m["approval_rate"] <= 1
            assert 0 <= m["decline_rate"] <= 1

    def test_skips_small_groups(self):
        y_true = np.array([0, 1, 0, 0, 0] + [0] * 100)
        y_score = np.random.RandomState(42).rand(105)
        groups = np.array([99, 99, 99, 99, 99] + [0] * 100)  # group 99 has only 5 members
        names = {0: "Big", 99: "Tiny"}
        result = compute_group_metrics(y_true, y_score, groups, names)
        group_names = [m["group"] for m in result]
        assert "Big" in group_names
        assert "Tiny" not in group_names  # < 10 members


# --- analyze_attribute ---

class TestAnalyzeAttribute:
    def test_returns_expected_keys(self):
        rng = np.random.RandomState(42)
        n = 500
        y_true = rng.choice([0, 1], n, p=[0.8, 0.2])
        y_score = rng.rand(n)
        attr_values = rng.choice([0, 1], n)
        config = {
            "reverse_map": {0: "Group0", 1: "Group1"},
            "privileged_code": 0,
            "description": "Test Attribute",
        }
        result = analyze_attribute(y_true, y_score, attr_values, config, "test_attr")
        assert "attribute" in result
        assert "group_metrics" in result
        assert "disparate_impact" in result
        assert "equal_opportunity_diff" in result
        assert "statistical_parity_diff" in result
        assert "has_dir_violation" in result

    def test_flags_dir_violation(self):
        # Create data where group 1 has much lower approval rate
        n = 1000
        y_true = np.zeros(n)
        y_score = np.concatenate([
            np.full(500, 0.05),  # group 0: all approved
            np.full(500, 0.50),  # group 1: all declined
        ])
        attr_values = np.concatenate([np.zeros(500), np.ones(500)])
        config = {
            "reverse_map": {0: "Privileged", 1: "Unprivileged"},
            "privileged_code": 0,
            "description": "Test",
        }
        result = analyze_attribute(y_true, y_score, attr_values, config, "test")
        assert result["has_dir_violation"] is True


# --- End-to-end ---

class TestFairnessRun:
    def test_run_with_synthetic_data(self):
        X, y, feature_cols = _make_fairness_data()
        model = _train_dummy_model(X, y)
        report = run_fairness(model=model, X_test=X, y_test=y)
        assert "n_observations" in report
        assert "attributes" in report
        assert "overall_approval_rate" in report

    def test_report_contains_expected_attributes(self):
        X, y, feature_cols = _make_fairness_data()
        model = _train_dummy_model(X, y)
        report = run_fairness(model=model, X_test=X, y_test=y)
        # Should analyze attributes that exist in the data
        for attr in ["home_ownership", "verification_status", "emp_length_missing"]:
            assert attr in report["attributes"]

    def test_format_report_produces_string(self):
        X, y, feature_cols = _make_fairness_data()
        model = _train_dummy_model(X, y)
        report = run_fairness(model=model, X_test=X, y_test=y)
        formatted = format_report(report)
        assert isinstance(formatted, str)
        assert "FAIRNESS ANALYSIS REPORT" in formatted
        assert "Disparate Impact" in formatted
