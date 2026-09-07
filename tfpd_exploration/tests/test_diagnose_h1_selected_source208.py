"""Synthetic contracts for the selected-EMA source208 descriptive diagnostic."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1 import diagnose_h1_selected_source208 as diagnostic


def _fixed():
    # The evaluator itself deliberately accepts a test-sized fixed set; the
    # production hard cardinality belongs to _require_ids and postcondition.
    return {"ses-a": np.arange(diagnostic.PER_SESSION, dtype=np.int64)}


def _cache(*, target_dtype=np.float32, bad_start=False):
    n = diagnostic.W + diagnostic.PER_SESSION
    target = np.repeat(np.arange(n, dtype=target_dtype)[:, None], diagnostic.OUTPUTS, axis=1)
    return {"train": {"ses-a": {"neural": np.ones((n, diagnostic.UNITS), np.float32),
                               "velocity": target, "query_starts": np.arange(diagnostic.PER_SESSION, dtype=np.int64), "bank": {}}}}


class _Model:
    def eval(self): pass
    def forward_last(self, x, bank): return torch.zeros((len(x), diagnostic.OUTPUTS), dtype=torch.float32, device=x.device)


def test_evaluate_exact_native_source_windows_metadata_and_float64_archive(monkeypatch):
    fixed = _fixed()
    # Patch just the final cardinality, leaving per-session 16 and native W700
    # semantics observable without allocating 13 full test sessions.
    monkeypatch.setattr(diagnostic, "COUNT", diagnostic.PER_SESSION)
    result, arrays = diagnostic.evaluate_source208(_Model(), _cache(), fixed, torch.device("cpu"), lambda row, dev: object())
    assert result["windows"] == diagnostic.PER_SESSION
    assert arrays["prediction"].shape == (diagnostic.PER_SESSION, 7)
    assert arrays["prediction"].dtype == np.float64 and arrays["target"].dtype == np.float64
    assert arrays["start"].dtype == np.int64 and arrays["session_id"].tolist() == ["ses-a"] * diagnostic.PER_SESSION
    assert result["per_session"]["ses-a"]["starts"] == list(range(diagnostic.PER_SESSION))
    assert result["pooled"]["r2_concat_float64"] == pytest.approx(-23489.04705882353)


@pytest.mark.parametrize("cache", [_cache(target_dtype=np.float64),
                                    {"train": {"ses-a": {"neural": np.ones((diagnostic.W - 1, diagnostic.UNITS), np.float32),
                                                        "velocity": np.zeros((diagnostic.W - 1, 7), np.float32), "query_starts": np.arange(16), "bank": {}}}}])
def test_evaluate_rejects_bad_source_geometry_and_target_dtype(monkeypatch, cache):
    monkeypatch.setattr(diagnostic, "COUNT", diagnostic.PER_SESSION)
    with pytest.raises(RuntimeError, match="source208 canonical batch"):
        diagnostic.evaluate_source208(_Model(), cache, _fixed(), torch.device("cpu"), lambda row, dev: object())


def test_preflight_gate_precedes_formal_audit_and_any_load(tmp_path, monkeypatch):
    calls = []
    monkeypatch.delenv("H1_SELECTED_SOURCE208_DIAGNOSTIC_GO", raising=False)
    monkeypatch.setattr(diagnostic, "preflight_audit", lambda formal: calls.append("audit") or {})
    monkeypatch.setattr(diagnostic, "code_source_audit", lambda formal, pre: calls.append("source") or {})
    with pytest.raises(RuntimeError, match="explicit source208 diagnostic gate"):
        diagnostic.run(tmp_path / "formal", tmp_path / "new", device="cpu", threads=1)
    assert calls == []


def test_nonoverwrite_and_postclosure_mutation_are_hard_failures(tmp_path, monkeypatch):
    out = tmp_path / "existing"; out.mkdir()
    monkeypatch.setenv("H1_SELECTED_SOURCE208_DIAGNOSTIC_GO", "1")
    with pytest.raises(FileExistsError): diagnostic.run(tmp_path / "formal", out, device="cpu", threads=1)

    monkeypatch.setattr(diagnostic, "preflight_audit", lambda formal: {"v": 2})
    monkeypatch.setattr(diagnostic, "code_source_audit", lambda formal, pre: {"same": True})
    with pytest.raises(RuntimeError, match="fresh post"):
        diagnostic.require_fresh_post(tmp_path / "formal", {"v": 1}, {"same": True})
    # The helper has no output path and therefore cannot create/overwrite one.
    assert not (tmp_path / "unexpected-output").exists()


def test_hard_frozen_id_and_gate_hashes_and_cli_parses_once(tmp_path, monkeypatch):
    assert len(diagnostic.IDS_SHA256) == 64 and len(diagnostic.GATE1040_SHA256) == 64
    called = []
    monkeypatch.setattr(diagnostic, "run", lambda *args, **kwargs: called.append((args, kwargs)) or {"ok": True})
    assert diagnostic.main(["--formal", str(tmp_path / "f"), "--out", str(tmp_path / "o"), "--device", "cpu", "--threads", "1"]) == {"ok": True}
    assert len(called) == 1 and called[0][1] == {"device": "cpu", "threads": 1}


def test_current_gate_and_manifest_metadata_have_the_expected_readonly_schema():
    gate = diagnostic.read(diagnostic.GATE1040)
    manifest = diagnostic.read(diagnostic.IDS)
    assert gate["status"] == "COMPLETE" and gate["updates_completed"] == 1040
    assert set(gate["after"]) == {"flat", "route"}
    assert set(manifest["ids"]) == set(manifest["source_authority"]["arrays"]["train"])
    assert sum(map(len, manifest["ids"].values())) == 208
