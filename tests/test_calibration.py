"""
Tests for probability calibration and scorecard scaling.
Uses synthetic data — does not require real data files or trained models.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb

from pipeline.calibrate import (
    BASE_ODDS,
    BASE_SCORE,
    FACTOR,
    OFFSET,
    PDO,
    calibrate_model,
    fit_calibrator,
    pd_to_score,
    reliability_table,
    scorecard_params,
)


# --- Helpers ---

def _make_scored_data(n=5000, seed=42):
    """Synthetic binary outcomes whose true PD is a known function of x."""
    rng = np.random.RandomState(seed)
    true_pd = rng.uniform(0.01, 0.6, n)
    y = (rng.uniform(size=n) < true_pd).astype(int)
    # Distorted scores: rank-order preserved, calibration broken
    raw = true_pd ** 2
    return raw, y, true_pd


# --- Scorecard scaling ---

class TestScorecard:
    def test_base_anchor(self):
        """PD at exactly 30:1 good:bad odds maps to the base score 600."""
        pd_at_base = 1 / (BASE_ODDS + 1)
        assert pd_to_score(pd_at_base) == BASE_SCORE

    def test_points_to_double_odds(self):
        """Doubling the odds adds exactly PDO points."""
        pd_at_base = 1 / (BASE_ODDS + 1)
        pd_at_double = 1 / (2 * BASE_ODDS + 1)
        assert pd_to_score(pd_at_double) - pd_to_score(pd_at_base) == PDO

    def test_monotonic_decreasing_in_pd(self):
        pds = np.linspace(0.001, 0.999, 200)
        scores = pd_to_score(pds)
        assert (np.diff(scores) <= 0).all()

    def test_extreme_pds_stay_finite(self):
        assert np.isfinite(pd_to_score(0.0))
        assert np.isfinite(pd_to_score(1.0))
        assert pd_to_score(0.0) > pd_to_score(1.0)

    def test_returns_integers(self):
        scores = pd_to_score(np.array([0.05, 0.2, 0.5]))
        assert scores.dtype.kind == "i"

    def test_params_consistent(self):
        params = scorecard_params()
        assert params["factor"] == float(FACTOR)
        assert params["offset"] == float(OFFSET)
        assert params["base_score"] == BASE_SCORE
        # offset + factor*ln(base_odds) recovers the base score
        recovered = params["offset"] + params["factor"] * np.log(params["base_odds"])
        assert abs(recovered - BASE_SCORE) < 1e-9


# --- Isotonic calibration ---

class TestCalibration:
    def test_calibrator_is_monotonic(self):
        raw, y, _ = _make_scored_data()
        iso = fit_calibrator(raw, y)
        grid = np.linspace(raw.min(), raw.max(), 100)
        preds = iso.predict(grid)
        assert (np.diff(preds) >= 0).all()

    def test_calibration_recovers_true_pd(self):
        """Isotonic on distorted scores should approximate the true PD."""
        raw, y, true_pd = _make_scored_data(n=20000)
        iso = fit_calibrator(raw, y)
        cal = iso.predict(raw)
        # Distorted scores are far from truth; calibrated ones close
        assert np.abs(cal - true_pd).mean() < np.abs(raw - true_pd).mean()
        assert np.abs(cal - true_pd).mean() < 0.05

    def test_out_of_range_scores_clip(self):
        raw, y, _ = _make_scored_data()
        iso = fit_calibrator(raw, y)
        lo, hi = iso.predict([raw.min()]), iso.predict([raw.max()])
        assert iso.predict([-1.0]) == lo
        assert iso.predict([2.0]) == hi

    def test_calibrate_model_improves_brier(self):
        rng = np.random.RandomState(0)
        n = 8000
        X = pd.DataFrame(rng.randn(n, 5), columns=[f"f{i}" for i in range(5)])
        logit = X["f0"] * 2 + X["f1"]
        y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
        model = lgb.LGBMClassifier(n_estimators=20, max_depth=3, random_state=42, verbose=-1)
        model.fit(X.iloc[: n // 2], y.iloc[: n // 2])

        X_cal, y_cal = X.iloc[n // 2 : 3 * n // 4], y.iloc[n // 2 : 3 * n // 4]
        X_test, y_test = X.iloc[3 * n // 4 :], y.iloc[3 * n // 4 :]
        calibrator, report = calibrate_model(model, X_cal, y_cal, X_test, y_test)

        assert report["method"] == "isotonic"
        assert report["n_calibration_rows"] == len(X_cal)
        assert report["brier_calibrated"] <= report["brier_raw"] + 0.005
        assert len(report["reliability_calibrated"]) > 1
        assert calibrator.X_thresholds_.shape == calibrator.y_thresholds_.shape


# --- Reliability table ---

class TestReliabilityTable:
    def test_bins_cover_all_rows(self):
        raw, y, _ = _make_scored_data()
        table = reliability_table(y, raw, n_bins=10)
        assert sum(row["n"] for row in table) == len(y)

    def test_perfectly_calibrated_scores_match_observed(self):
        rng = np.random.RandomState(1)
        pd_true = rng.uniform(0.05, 0.5, 50000)
        y = (rng.uniform(size=50000) < pd_true).astype(int)
        table = reliability_table(y, pd_true, n_bins=5)
        for row in table:
            assert abs(row["mean_predicted"] - row["observed_default_rate"]) < 0.02

    def test_constant_scores_collapse_to_one_bin(self):
        y = np.array([0, 1, 0, 1])
        proba = np.full(4, 0.5)
        table = reliability_table(y, proba, n_bins=10)
        assert len(table) == 1
        assert table[0]["n"] == 4
