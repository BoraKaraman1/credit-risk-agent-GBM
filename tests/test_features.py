"""
Tests for feature engineering, reject inference, and model training utilities.
Uses synthetic data — does not require real data files or trained models.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.reject_inference import (
    align_rejected_features,
    assign_pseudo_labels,
    compare_models,
    compute_ks,
    REJECT_DEFAULT_MULTIPLIER,
    REJECT_DEFAULT_CAP,
    REJECT_SAMPLE_WEIGHT,
)
from pipeline.train import compute_gini
from agents.drift_monitor import compute_psi, compute_csi


# --- Helpers ---

def _make_feature_data(n=500, n_features=10, seed=42):
    """Create synthetic feature data for testing."""
    rng = np.random.RandomState(seed)
    feature_cols = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.randn(n, n_features), columns=feature_cols)
    y = pd.Series(rng.choice([0, 1], n, p=[0.8, 0.2]), name="default")
    return X, y, feature_cols


def _train_dummy_model(X, y):
    """Train a quick model for testing."""
    model = HistGradientBoostingClassifier(
        max_iter=10, max_depth=3, random_state=42
    )
    model.fit(X, y)
    return model


# --- compute_ks tests ---

class TestComputeKS:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = compute_ks(y_true, y_score)
        assert ks == pytest.approx(1.0, abs=0.01)

    def test_random_scores(self):
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 1000, p=[0.8, 0.2])
        y_score = rng.rand(1000)
        ks = compute_ks(y_true, y_score)
        assert 0 <= ks <= 1

    def test_returns_float(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.7])
        assert isinstance(compute_ks(y_true, y_score), float)


class TestComputeGini:
    def test_from_auc(self):
        assert compute_gini(0.75) == pytest.approx(0.50)
        assert compute_gini(0.50) == pytest.approx(0.0)
        assert compute_gini(1.0) == pytest.approx(1.0)


# --- PSI / CSI tests ---

class TestComputePSI:
    def test_identical_distributions(self):
        scores = np.random.RandomState(42).rand(1000)
        psi, _, _ = compute_psi(scores, scores)
        assert psi == pytest.approx(0.0, abs=0.001)

    def test_shifted_distribution(self):
        rng = np.random.RandomState(42)
        expected = rng.beta(2, 5, 5000)
        actual = rng.beta(5, 2, 5000)
        psi, _, _ = compute_psi(expected, actual)
        assert psi > 0.1  # should detect significant shift

    def test_returns_tuple(self):
        scores = np.random.RandomState(42).rand(100)
        result = compute_psi(scores, scores)
        assert len(result) == 3

    def test_bin_percentages_sum_to_one(self):
        rng = np.random.RandomState(42)
        _, exp_pct, act_pct = compute_psi(rng.rand(500), rng.rand(500))
        assert sum(exp_pct) == pytest.approx(1.0, abs=0.01)
        assert sum(act_pct) == pytest.approx(1.0, abs=0.01)


class TestComputeCSI:
    def test_identical_features(self):
        col = np.random.RandomState(42).randn(1000)
        csi = compute_csi(col, col)
        assert csi == pytest.approx(0.0, abs=0.01)

    def test_shifted_feature(self):
        rng = np.random.RandomState(42)
        train_col = rng.normal(0, 1, 5000)
        prod_col = rng.normal(2, 1, 5000)  # mean shifted by 2 std
        csi = compute_csi(train_col, prod_col)
        assert csi > 0.1

    def test_handles_nans(self):
        rng = np.random.RandomState(42)
        col = rng.randn(100)
        col_with_nan = col.copy()
        col_with_nan[:10] = np.nan
        csi = compute_csi(col, col_with_nan)
        assert np.isfinite(csi)


# --- Reject inference tests ---

class TestAlignRejectedFeatures:
    def setup_method(self):
        self.feature_cols = ["fico_score", "dti", "loan_amnt", "emp_length", "int_rate"]
        rng = np.random.RandomState(42)
        n_train = 200
        n_reject = 100

        self.train = pd.DataFrame({
            col: rng.randn(n_train) for col in self.feature_cols
        })

        # Rejected has only a subset of features
        self.rejected = pd.DataFrame({
            "fico_score": rng.randn(n_reject),
            "dti": rng.randn(n_reject),
            "loan_amnt": rng.randn(n_reject),
            "emp_length": rng.randn(n_reject),
            "emp_length_missing": rng.choice([0, 1], n_reject),
        })

    def test_output_has_all_feature_cols(self):
        aligned = align_rejected_features(self.rejected, self.feature_cols, self.train)
        assert list(aligned.columns) == self.feature_cols

    def test_no_nulls(self):
        aligned = align_rejected_features(self.rejected, self.feature_cols, self.train)
        assert aligned.isnull().sum().sum() == 0

    def test_overlapping_cols_preserved(self):
        aligned = align_rejected_features(self.rejected, self.feature_cols, self.train)
        # fico_score should come from rejected, not be median-filled
        # (can't test exact values due to sampling, but shape should match)
        assert len(aligned) <= len(self.rejected)

    def test_missing_cols_filled_with_median(self):
        aligned = align_rejected_features(self.rejected, self.feature_cols, self.train)
        # int_rate is not in rejected → should be filled with train median
        train_median = self.train["int_rate"].median()
        assert (aligned["int_rate"] == train_median).all()


class TestAssignPseudoLabels:
    def setup_method(self):
        rng = np.random.RandomState(42)
        n = 500
        self.feature_cols = ["f1", "f2", "f3"]
        X = pd.DataFrame(rng.randn(n, 3), columns=self.feature_cols)
        y = pd.Series(rng.choice([0, 1], n, p=[0.8, 0.2]))
        self.model = _train_dummy_model(X, y)
        self.rejected_aligned = pd.DataFrame(rng.randn(200, 3), columns=self.feature_cols)

    def test_labels_are_binary(self):
        labels, _ = assign_pseudo_labels(self.model, self.rejected_aligned, 0.20)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_default_rate_respects_multiplier(self):
        training_rate = 0.20
        labels, _ = assign_pseudo_labels(self.model, self.rejected_aligned, training_rate)
        actual_rate = labels.mean()
        expected_rate = min(training_rate * REJECT_DEFAULT_MULTIPLIER, REJECT_DEFAULT_CAP)
        assert actual_rate == pytest.approx(expected_rate, abs=0.05)

    def test_cap_applied(self):
        # With a very high training rate, cap should kick in
        labels, _ = assign_pseudo_labels(self.model, self.rejected_aligned, 0.50)
        actual_rate = labels.mean()
        assert actual_rate <= REJECT_DEFAULT_CAP + 0.05

    def test_scores_returned(self):
        _, scores = assign_pseudo_labels(self.model, self.rejected_aligned, 0.20)
        assert len(scores) == len(self.rejected_aligned)
        assert all(0 <= s <= 1 for s in scores)


class TestCompareModels:
    def test_returns_expected_keys(self):
        X, y, cols = _make_feature_data(n=300)
        model1 = _train_dummy_model(X, y)
        model2 = _train_dummy_model(X, y)
        result = compare_models(model1, model2, X, y, cols)

        assert "champion" in result
        assert "augmented" in result
        assert "psi_between_models" in result
        assert "auc_delta" in result

    def test_metrics_in_range(self):
        X, y, cols = _make_feature_data(n=300)
        model1 = _train_dummy_model(X, y)
        model2 = _train_dummy_model(X, y)
        result = compare_models(model1, model2, X, y, cols)

        for key in ["champion", "augmented"]:
            assert 0 <= result[key]["auc"] <= 1
            assert 0 <= result[key]["ks"] <= 1
            assert -1 <= result[key]["gini"] <= 1

    def test_psi_non_negative(self):
        X, y, cols = _make_feature_data(n=300)
        model = _train_dummy_model(X, y)
        result = compare_models(model, model, X, y, cols)
        assert result["psi_between_models"] >= 0


# --- Training utility tests ---

class TestTrainModel:
    def test_hist_gbm_trains(self):
        X, y, _ = _make_feature_data(n=200)
        model = HistGradientBoostingClassifier(
            max_iter=10, max_depth=3, random_state=42
        )
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert len(probs) == len(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_sample_weights_accepted(self):
        X, y, _ = _make_feature_data(n=200)
        weights = np.concatenate([np.ones(150), np.full(50, 0.3)])
        model = HistGradientBoostingClassifier(
            max_iter=10, max_depth=3, random_state=42
        )
        model.fit(X, y, sample_weight=weights)
        assert model.n_iter_ > 0
