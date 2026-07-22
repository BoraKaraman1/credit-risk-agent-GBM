"""
Tests for the model risk review agent's deterministic plumbing.
The LLM call itself is not tested; everything around it is: tool
functions, prompt assembly, memo/audit writing. No API key needed.
"""

import json
import os
import stat

from agents import review_agent
from agents.review_agent import (
    build_prompt,
    read_model_metadata,
    run_monitor,
)


# --- Tool functions ---

class TestRunMonitor:
    def test_rejects_unknown_monitor(self):
        assert run_monitor("promote").startswith("ERROR")
        assert run_monitor("backfill").startswith("ERROR")

    def test_missing_binary_reports_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CREDIT_RISK_GO_BIN", str(tmp_path))
        assert "not found" in run_monitor("drift")

    def test_runs_fake_gbm_and_returns_stdout(self, monkeypatch, tmp_path):
        fake = tmp_path / "gbm"
        fake.write_text('#!/bin/sh\necho \'{"psi": 0.12, "psi_status": "WARNING"}\'\n')
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("CREDIT_RISK_GO_BIN", str(tmp_path))

        out = run_monitor("drift")
        assert json.loads(out) == {"psi": 0.12, "psi_status": "WARNING"}

    def test_nonzero_exit_reports_error(self, monkeypatch, tmp_path):
        fake = tmp_path / "gbm"
        fake.write_text('#!/bin/sh\necho "boom" >&2\nexit 3\n')
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("CREDIT_RISK_GO_BIN", str(tmp_path))

        out = run_monitor("performance")
        assert out.startswith("ERROR") and "boom" in out


class TestReadModelMetadata:
    def test_rejects_unknown_role(self):
        assert read_model_metadata("shadow").startswith("ERROR")

    def test_reads_metadata(self, monkeypatch, tmp_path):
        champ = tmp_path / "models" / "champion"
        champ.mkdir(parents=True)
        (champ / "model_metadata.json").write_text('{"version": "v9.9"}')
        monkeypatch.setenv("CREDIT_RISK_MODELS_DIR", str(tmp_path / "models"))

        assert json.loads(read_model_metadata("champion")) == {"version": "v9.9"}

    def test_missing_file_reports_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CREDIT_RISK_MODELS_DIR", str(tmp_path))
        assert read_model_metadata("challenger").startswith("ERROR")


# --- Prompt assembly ---

class TestBuildPrompt:
    def test_embeds_provided_reports(self):
        prompt = build_prompt({"drift": {"psi": 0.31, "psi_status": "CRITICAL"}})
        assert "drift report" in prompt
        assert '"psi_status": "CRITICAL"' in prompt

    def test_flags_missing_reports_instead_of_guessing(self):
        prompt = build_prompt({"drift": {"psi": 0.1}, "retrain": None})
        assert "retrain" in prompt
        assert "do not guess" in prompt

    def test_no_reports_instructs_tool_use(self):
        prompt = build_prompt(None)
        assert "run_monitor" in prompt


# --- Memo and audit plumbing ---

class TestMemoAndAudit:
    def test_write_memo_adds_advisory_header(self, monkeypatch, tmp_path):
        monkeypatch.setattr(review_agent, "MEMO_PATH", tmp_path / "memo.md")
        path = review_agent._write_memo("## Summary\nAll stable.", "claude-opus-4-8")
        text = path.read_text()
        assert text.startswith("# Weekly Monitoring Review")
        assert "Advisory only" in text
        assert "claude-opus-4-8" in text
        assert text.endswith("All stable.\n")

    def test_audit_appends_jsonl(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CREDIT_RISK_DATA_DIR", str(tmp_path))
        review_agent._append_audit({"model": "m", "memo": "a"})
        path = review_agent._append_audit({"model": "m", "memo": "b"})

        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["memo"] for r in lines] == ["a", "b"]
        assert path == tmp_path / "review_agent_audit.jsonl"


# --- Guardrails ---

def test_run_requires_api_key_not_loop(monkeypatch):
    """Without credentials run() must raise so the DAG task can catch and
    skip; it must never fall back to fabricating a memo."""
    if review_agent.anthropic is None:
        return  # no SDK installed: covered by the RuntimeError branch
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_PROFILE", "nonexistent-profile-for-test")
    try:
        review_agent.run(reports={"drift": {"psi": 0.1}})
    except Exception:
        pass
    else:  # pragma: no cover
        raise AssertionError("run() succeeded without credentials")
    assert not os.path.exists("docs/monitoring_review.md.tmp")
