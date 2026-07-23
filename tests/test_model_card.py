"""
Tests for the model card generator.
Uses synthetic metadata dicts — does not require trained models.
"""

from pipeline.model_card import render, _validation_status


def _meta(with_calibration=True, with_violation=False):
    meta = {
        "version": "v9.9",
        "trained_at": "2026-06-13T00:00:00+00:00",
        "n_features": 33,
        "metrics": {
            "train": {"auc": 0.75, "ks": 0.36, "gini": 0.49},
            "val": {"auc": 0.74, "ks": 0.35, "gini": 0.48},
            "test": {"auc": 0.72, "ks": 0.32, "gini": 0.43},
        },
        "hyperparameters": {"max_depth": 6, "learning_rate": 0.05},
        "scorecard": {"base_score": 600, "base_odds": 30, "pdo": 20,
                      "factor": 28.85, "offset": 501.86},
        "fairness": {
            "dir_threshold": 0.80,
            "attributes": {
                "home_ownership": {
                    "description": "Home Ownership Status",
                    "privileged_group": "MORTGAGE",
                    "groups": {
                        "MORTGAGE": {"dir": 1.0, "approval_rate": 0.5, "default_rate": 0.2},
                        "RENT": {"dir": 0.7 if with_violation else 0.9,
                                 "approval_rate": 0.35, "default_rate": 0.3},
                    },
                    "violations": ["RENT"] if with_violation else [],
                },
            },
        },
    }
    if with_calibration:
        meta["calibration"] = {
            "method": "isotonic", "n_calibration_rows": 100, "n_breakpoints": 50,
            "brier_raw": 0.17, "brier_calibrated": 0.169,
            "reliability_calibrated": [
                {"n": 100, "mean_predicted": 0.1, "observed_default_rate": 0.11},
            ],
        }
    return meta


_FEATURE_META = {
    "split_method": "time-aware",
    "splits": {
        "train": {"rows": 831039, "default_rate": 0.1862},
        "val": {"rows": 392090, "default_rate": 0.2482},
        "test": {"rows": 145841, "default_rate": 0.2642},
    },
}


class TestValidationStatus:
    def test_approved_when_clean(self):
        status, _ = _validation_status(_meta())
        assert status == "APPROVED"

    def test_review_on_fairness_violation(self):
        status, rationale = _validation_status(_meta(with_violation=True))
        assert status == "REVIEW REQUIRED"
        assert "home_ownership/RENT" in rationale

    def test_review_when_uncalibrated(self):
        status, rationale = _validation_status(_meta(with_calibration=False))
        assert status == "REVIEW REQUIRED"
        assert "not calibrated" in rationale

    def test_missing_fairness_fails_closed(self):
        meta = _meta()
        del meta["fairness"]
        status, rationale = _validation_status(meta)
        assert status == "REVIEW REQUIRED"
        assert "No fairness analysis" in rationale


def _champion_fairness(rent_dir):
    """Champion fairness summary with RENT at the given DIR."""
    return {
        "dir_threshold": 0.80,
        "attributes": {
            "home_ownership": {
                "groups": {
                    "MORTGAGE": {"dir": 1.0},
                    "RENT": {"dir": rent_dir},
                },
                "violations": ["RENT"] if rent_dir < 0.80 else [],
            },
        },
    }


class TestChampionRelativeStatus:
    """Champion-relative verdicts must mirror the Go retrain gate
    (go/monitoring/retrain.go fairnessGate): block only new-or-worsened
    violations, with the contract's dir_worsen_tolerance."""

    def test_inherited_violation_not_worsened_approves(self):
        # Challenger RENT 0.70 vs champion 0.69: inherited, no worse.
        status, rationale = _validation_status(
            _meta(with_violation=True), _champion_fairness(0.69))
        assert status == "APPROVED"
        assert "champion-relative" in rationale

    def test_new_violation_still_blocks(self):
        # Champion clean on RENT (0.90): the challenger's violation is new.
        status, rationale = _validation_status(
            _meta(with_violation=True), _champion_fairness(0.90))
        assert status == "REVIEW REQUIRED"
        assert "home_ownership/RENT" in rationale

    def test_worsened_beyond_tolerance_blocks(self):
        # Champion violates at 0.75; challenger 0.70 is worse by > 0.01.
        status, rationale = _validation_status(
            _meta(with_violation=True), _champion_fairness(0.75))
        assert status == "REVIEW REQUIRED"
        assert "worsened" in rationale

    def test_worsening_within_tolerance_passes(self):
        # Champion 0.705 -> challenger 0.70: within the noise tolerance.
        status, _ = _validation_status(
            _meta(with_violation=True), _champion_fairness(0.705))
        assert status == "APPROVED"


class TestChampionFairnessFor:
    def test_resolves_only_for_challenger_dir(self, tmp_path, monkeypatch):
        import json

        from pipeline.model_card import champion_fairness_for

        monkeypatch.setenv("CREDIT_RISK_MODELS_DIR", str(tmp_path))
        champ = tmp_path / "champion"
        champ.mkdir(parents=True)
        (champ / "model_metadata.json").write_text(
            json.dumps({"fairness": {"dir_threshold": 0.8, "attributes": {}}}))

        assert champion_fairness_for(tmp_path / "challenger") == {
            "dir_threshold": 0.8, "attributes": {}}
        # The champion must never be scored against its own summary,
        # and arbitrary dirs (versions/, tmp) use the absolute rule.
        assert champion_fairness_for(champ) is None
        assert champion_fairness_for(tmp_path / "versions" / "v1.0") is None

    def test_none_without_champion(self, tmp_path, monkeypatch):
        from pipeline.model_card import champion_fairness_for

        monkeypatch.setenv("CREDIT_RISK_MODELS_DIR", str(tmp_path))
        assert champion_fairness_for(tmp_path / "challenger") is None


class TestRender:
    def test_contains_all_sections(self):
        card = render(_meta(), _FEATURE_META)
        for heading in ("# Model Card", "## Validation Status", "## Data Window",
                        "## Discrimination Metrics", "## Calibration",
                        "## Scorecard Scaling", "## Fairness", "## Hyperparameters"):
            assert heading in card

    def test_no_em_dashes(self):
        # Documentation convention for this repo: no em dashes.
        assert "—" not in render(_meta(), _FEATURE_META)

    def test_violation_surfaced_in_table(self):
        card = render(_meta(with_violation=True), _FEATURE_META)
        assert "VIOLATION" in card

    def test_missing_hyperparameters_handled(self):
        meta = _meta()
        del meta["hyperparameters"]
        card = render(meta, _FEATURE_META)
        assert "Not recorded" in card
