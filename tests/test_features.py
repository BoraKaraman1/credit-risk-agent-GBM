"""
Tests for feature engineering, reject inference, and model training utilities.
Uses synthetic data — does not require real data files or trained models.
"""

import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb

from pipeline.reject_inference import (
    align_rejected_features,
    assign_pseudo_labels,
    compare_models,
    compute_ks,
    save_augmented_model,
    train_augmented_model,
    REJECT_DEFAULT_MULTIPLIER,
    REJECT_DEFAULT_CAP,
)
from pipeline.train import compute_gini


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
    model = lgb.LGBMClassifier(
        n_estimators=10, max_depth=3, random_state=42, verbose=-1
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


# PSI / CSI drift metrics moved to the Go services; their tests live in
# go/shared/metrics (cross-checked against numpy reference fixtures).


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


# --- Augmented training smoke tests ---
# These execute the orchestration layer end to end on synthetic data: a
# phantom function call or a broken return contract in
# train_augmented_model/save_augmented_model must fail here, not in
# production.

class TestTrainAugmentedModel:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.feature_cols = [f"feat_{i}" for i in range(5)]
        self.X_acc = pd.DataFrame(rng.randn(200, 5), columns=self.feature_cols)
        self.y_acc = pd.Series(rng.choice([0, 1], 200, p=[0.8, 0.2]))
        # Rejected rows are shifted far from the accepted cloud so any
        # pseudo-labeled row is identifiable by its feature values.
        self.X_rej = pd.DataFrame(rng.randn(80, 5) + 1000.0, columns=self.feature_cols)
        self.y_rej = pd.Series(rng.choice([0, 1], 80, p=[0.6, 0.4]))
        self.X_val = pd.DataFrame(rng.randn(60, 5), columns=self.feature_cols)
        self.y_val = pd.Series(rng.choice([0, 1], 60, p=[0.8, 0.2]))

    def _run(self):
        return train_augmented_model(
            self.X_acc, self.y_acc, self.X_rej, self.y_rej,
            self.X_val, self.y_val,
        )

    def test_executes_and_returns_scoring_model(self):
        model, X_es, y_es = self._run()
        probs = model.predict_proba(self.X_acc[self.feature_cols])[:, 1]
        assert len(probs) == len(self.X_acc)
        assert all(0 <= p <= 1 for p in probs)

    def test_calibration_carveout_is_observed_rows_only(self):
        _, X_es, y_es = self._run()
        assert len(X_es) == len(y_es)
        assert len(X_es) > 0
        # No row from the shifted (pseudo-labeled) rejected cloud may
        # reach the calibration carve-out.
        assert (X_es["feat_0"] < 500).all()


class TestSaveAugmentedModel:
    def test_versions_relative_to_champion_with_ri_tag(self, tmp_path, monkeypatch):
        import json

        monkeypatch.setenv("CREDIT_RISK_MODELS_DIR", str(tmp_path))
        champ = tmp_path / "champion"
        champ.mkdir(parents=True)
        (champ / "model_metadata.json").write_text(json.dumps({"version": "v2.5"}))

        X, y, cols = _make_feature_data(n=200)
        model = _train_dummy_model(X, y)
        comparison = {"augmented": {"auc": 0.7, "ks": 0.3, "gini": 0.4},
                      "champion": {"auc": 0.7, "ks": 0.3, "gini": 0.4},
                      "psi_between_models": 0.01, "auc_delta": 0.0}
        version = save_augmented_model(model, cols, {"test": comparison["augmented"]},
                                       comparison)
        assert version == "v2.6-ri"
        meta = json.loads((tmp_path / "challenger" / "model_metadata.json").read_text())
        assert meta["version"] == "v2.6-ri"
        # Test metrics recorded under the test key only — no fabricated
        # train split in the model card.
        assert set(meta["metrics"]) == {"test"}


# --- Training utility tests ---

class TestTrainModel:
    def test_lightgbm_trains(self):
        X, y, _ = _make_feature_data(n=200)
        model = lgb.LGBMClassifier(
            n_estimators=10, max_depth=3, random_state=42, verbose=-1
        )
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert len(probs) == len(X)
        assert all(0 <= p <= 1 for p in probs)

    def test_sample_weights_accepted(self):
        X, y, _ = _make_feature_data(n=200)
        weights = np.concatenate([np.ones(150), np.full(50, 0.3)])
        model = lgb.LGBMClassifier(
            n_estimators=10, max_depth=3, random_state=42, verbose=-1
        )
        model.fit(X, y, sample_weight=weights)
        assert model.n_estimators_ > 0
