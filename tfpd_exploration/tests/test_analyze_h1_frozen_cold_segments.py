"""Synthetic tests for archive-only H1 cold/full-W700 analysis."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from tfpd_exploration.src.family_runtime_v1 import analyze_h1_frozen_cold_segments as analysis


def arrays(count=8, sessions=2):
    ends = np.array([697, 698, 699, 700, 697, 698, 699, 700], np.int64)[:count]
    labels = np.array(["s0", "s0", "s0", "s0", "s1", "s1", "s1", "s1"], dtype="U2")[:count]
    target = np.arange(count * 7, dtype=np.float64).reshape(count, 7)
    prediction = target + .25
    return {"prediction": prediction, "target": target, "end": ends, "session_id": labels}


def test_exact_699_boundary_and_group_sse_recombination(monkeypatch):
    monkeypatch.setattr(analysis, "COUNT", 8); monkeypatch.setattr(analysis, "SESSIONS", 2)
    result = analysis.analyze_arrays(arrays())
    assert result["groups"]["cold_history_lt_699"]["pooled"]["count"] == 4
    assert result["groups"]["full_w700_ge_699"]["pooled"]["count"] == 4
    all_group = result["groups"]["all"]["pooled"]
    split_sse = sum(result["groups"][name]["pooled"]["sse_float64"] for name in ("cold_history_lt_699", "full_w700_ge_699"))
    assert split_sse == all_group["sse_float64"]
    assert all_group["r2_concat_float64"] != pytest.approx(np.mean([result["groups"][name]["pooled"]["r2_concat_float64"] for name in ("cold_history_lt_699", "full_w700_ge_699")]))


def test_metadata_mismatch_finite_variance_and_paired_count_guards(monkeypatch):
    monkeypatch.setattr(analysis, "COUNT", 8); monkeypatch.setattr(analysis, "SESSIONS", 2)
    value = arrays(); changed = {key: item.copy() for key, item in value.items()}; changed["end"][0] = 99
    with pytest.raises(RuntimeError, match="identity"):
        analysis.compare.same_surface(changed, value)
    flat = arrays(); flat["target"].fill(1)
    with pytest.raises(RuntimeError, match="variance"):
        analysis.analyze_arrays(flat)
    left, right = analysis.analyze_arrays(arrays()), analysis.analyze_arrays(arrays())
    right["groups"]["all"]["pooled"]["count"] = 7
    with pytest.raises(RuntimeError, match="count"):
        analysis.paired_deltas(left, right)


def test_gate_precedes_formal_and_archive_work(monkeypatch, tmp_path):
    args = SimpleNamespace(output=tmp_path / "out", formal=tmp_path / "formal", original_receipt=tmp_path / "r", original_npz=tmp_path / "n")
    monkeypatch.delenv(analysis.GO, raising=False)
    monkeypatch.setattr(analysis, "artifact_audit", lambda _: pytest.fail("formal read before gate"))
    with pytest.raises(RuntimeError, match="GO"):
        analysis.run(args)
    assert not args.output.exists()


def test_post_mutation_helper_detects_changed_bound_file(tmp_path):
    path = tmp_path / "bound"; path.write_text("before")
    bound = {str(path): analysis.sha(path)}
    path.write_text("after")
    with pytest.raises(RuntimeError):
        analysis.require_same_files(bound)


def test_constants_bind_original_reference_and_fixed_split():
    assert analysis.ORIGINAL_RECEIPT_SHA == "539bfa832c650df51e22be14b5ee0dba0c1f49fbb595f9fb869328a601f30c48"
    assert analysis.ORIGINAL_NPZ_SHA == "f1bb6739415ea66bb86bf4285035f131333e350072ae364139b81488912ba81c"
    assert analysis.COLD_END_EXCLUSIVE == 699
