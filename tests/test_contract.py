"""
Cross-language threshold contract.

go/shared/config/contract.json is the single source of truth for the
decision, monitoring, and fairness thresholds. The Go services embed it
at compile time; pipeline/config.py and ui/core.py load it at import.
These tests pin every Python consumer to the contract so a drifted copy
can never pass CI. (config_test.go covers the Go side.)
"""

import json
from pathlib import Path

from pipeline import config, fairness
from ui import core as ui_core

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = json.loads((ROOT / "go" / "shared" / "config" / "contract.json").read_text())


def test_contract_is_sane():
    d, m = CONTRACT["decision"], CONTRACT["monitoring"]
    assert 0 < d["approve_below"] < d["review_below"] < 1
    assert 0 < m["psi_warning"] < m["psi_critical"]
    assert m["csi_threshold"] > 0
    assert m["auc_drop_threshold"] > 0
    assert 0 < CONTRACT["fairness"]["dir_threshold"] <= 1
    assert 0 <= CONTRACT["fairness"]["dir_worsen_tolerance"] < CONTRACT["fairness"]["dir_threshold"]


def test_pipeline_config_matches_contract():
    assert config.APPROVE_THRESHOLD == CONTRACT["decision"]["approve_below"]
    assert config.REVIEW_THRESHOLD == CONTRACT["decision"]["review_below"]
    assert config.PSI_WARNING == CONTRACT["monitoring"]["psi_warning"]
    assert config.PSI_CRITICAL == CONTRACT["monitoring"]["psi_critical"]
    assert config.CSI_THRESHOLD == CONTRACT["monitoring"]["csi_threshold"]
    assert config.AUC_DROP_THRESHOLD == CONTRACT["monitoring"]["auc_drop_threshold"]
    assert config.DIR_THRESHOLD == CONTRACT["fairness"]["dir_threshold"]
    assert config.DIR_WORSEN_TOLERANCE == CONTRACT["fairness"]["dir_worsen_tolerance"]


def test_fairness_module_matches_contract():
    assert fairness.APPROVE_THRESHOLD == CONTRACT["decision"]["approve_below"]
    assert fairness.REVIEW_THRESHOLD == CONTRACT["decision"]["review_below"]
    assert fairness.DIR_THRESHOLD == CONTRACT["fairness"]["dir_threshold"]


def test_ui_matches_contract():
    assert ui_core.PSI_WARNING == CONTRACT["monitoring"]["psi_warning"]
    assert ui_core.PSI_CRITICAL == CONTRACT["monitoring"]["psi_critical"]
    assert ui_core.AUC_DROP_THRESHOLD == CONTRACT["monitoring"]["auc_drop_threshold"]
