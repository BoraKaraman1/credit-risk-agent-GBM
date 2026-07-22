"""
Tests for the Streamlit UI's pure helpers (ui/core.py). These shape
drift_log rows and API responses into frames; no streamlit, no network.
"""

from datetime import datetime

from ui.core import (
    adverse_actions_frame,
    csi_frame,
    decile_frame,
    decision_presentation,
    fairness_frame,
    metric_history_frame,
    psi_status,
)

FAIRNESS_DETAILS = {
    "dir_threshold": 0.8,
    "attributes": {
        "home_ownership": {
            "description": "Home Ownership Status",
            "privileged_group": "MORTGAGE",
            "groups": {
                "MORTGAGE": {"dir": 1.0, "eod": 0.0, "spd": 0.0,
                             "approval_rate": 0.44, "default_rate": 0.22},
                "RENT": {"dir": 0.634, "eod": -0.15, "spd": -0.16,
                         "approval_rate": 0.28, "default_rate": 0.32},
            },
            "violations": ["RENT"],
        },
    },
}


class TestDecisionPresentation:
    def test_known_bands(self):
        assert decision_presentation("approve")["status"] == "good"
        assert decision_presentation("review")["status"] == "warning"
        assert decision_presentation("decline")["status"] == "critical"

    def test_unknown_band_passes_through(self):
        pres = decision_presentation("weird")
        assert pres["label"] == "weird" and pres["status"] == "warning"


class TestPsiStatus:
    def test_thresholds(self):
        assert psi_status(0.05) == "OK"
        assert psi_status(0.15) == "WARNING"
        assert psi_status(0.30) == "CRITICAL"


class TestMetricHistoryFrame:
    def test_sorts_by_time(self):
        rows = [
            (datetime(2026, 6, 2), 0.2, "v1.2", {}),
            (datetime(2026, 6, 1), 0.1, "v1.2", {}),
        ]
        df = metric_history_frame(rows)
        assert list(df["metric_value"]) == [0.1, 0.2]

    def test_empty_input(self):
        assert metric_history_frame([]).empty


class TestFairnessFrame:
    def test_flattens_groups_with_flags(self):
        df = fairness_frame(FAIRNESS_DETAILS)
        assert len(df) == 2
        rent = df[df["group"] == "RENT"].iloc[0]
        assert rent["violation"] and not rent["privileged"]
        assert rent["dir"] == 0.634 and rent["eod"] == -0.15 and rent["spd"] == -0.16
        mortgage = df[df["group"] == "MORTGAGE"].iloc[0]
        assert mortgage["privileged"] and not mortgage["violation"]

    def test_empty_details(self):
        assert fairness_frame({}).empty
        assert fairness_frame(None).empty


class TestDetailFrames:
    def test_csi_sorted_descending(self):
        df = csi_frame({"csi": {"a": 0.01, "b": 0.30, "c": 0.10}})
        assert list(df["feature"]) == ["b", "c", "a"]

    def test_decile_frame_columns(self):
        df = decile_frame({"decile_analysis": [
            {"decile": 1, "count": 100, "default_rate": 0.05, "avg_score": 0.08},
        ]})
        assert list(df.columns) == ["decile", "count", "default_rate", "avg_score"]
        assert df.iloc[0]["default_rate"] == 0.05

    def test_missing_details_yield_empty(self):
        assert csi_frame({}).empty
        assert decile_frame(None).empty

    def test_adverse_actions(self):
        df = adverse_actions_frame([
            {"code": 2, "reason": "Excessive obligations", "feature_name": "DTI",
             "shap_value": 0.09, "feature_value": 38.2},
        ])
        assert df.iloc[0]["code"] == 2 and df.iloc[0]["feature"] == "DTI"
        assert adverse_actions_frame(None).empty
