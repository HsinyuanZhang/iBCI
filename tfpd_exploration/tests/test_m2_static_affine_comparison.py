"""Synthetic-only guards for the gated selected M2 static-affine comparison."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from tfpd_exploration.src.family_runtime_v1 import m2_static_affine_comparison as comparison


def args(tmp_path, **overrides):
    value = dict(output=tmp_path / "out.json", calls=32, warmup=128, threads=1,
                 container_reference=True, image_digest=comparison.IMAGE_DIGEST)
    value.update(overrides)
    return SimpleNamespace(**value)


def test_public_contract_and_stats():
    comparison.assert_public("ok", np.zeros((7, 2), np.float32))
    for value in (np.zeros((7, 2), np.float64), np.zeros((1, 2), np.float32), np.full((7, 2), np.nan, np.float32)):
        with pytest.raises(RuntimeError, match="owning contiguous"):
            comparison.assert_public("bad", value)
    assert comparison.stats([1., 2., 3.])["p95_ms"] == pytest.approx(2.9)
    for values in ([], [0.], [np.nan], [[1.]]):
        with pytest.raises(RuntimeError, match="timing"):
            comparison.stats(values)
    assert comparison.stream_count(32, 128) == 161
    assert comparison.stream_count(2048, 128) == 2177
    with pytest.raises(RuntimeError, match="cardinality"):
        comparison.stream_count(64, 128)
    assert comparison.ratio_stats([.5, 1., 2.]) == {"mean": pytest.approx(7 / 6), "p50": 1., "p95": pytest.approx(1.9), "p99": pytest.approx(1.98), "max": 2.}
    with pytest.raises(RuntimeError, match="dimensionless"):
        comparison.ratio_stats([0.])
    paired = comparison.paired_ratio_summary([1., 2.], [2., 1.])
    assert paired["raw_sample_ratio_static_over_base"] == [.5, 2.]
    assert paired["dimensionless_ratio_stats"]["p50"] == 1.25
    with pytest.raises(RuntimeError, match="equal nonempty"):
        comparison.paired_ratio_summary([1.], [])


def test_gate_precedes_preimport_hashing_and_output(monkeypatch, tmp_path):
    monkeypatch.delenv(comparison.GO, raising=False)
    monkeypatch.setattr(comparison, "sha", lambda _: pytest.fail("gate must precede hashes"))
    with pytest.raises(RuntimeError, match="comparison gate"):
        comparison.run(args(tmp_path))
    assert not args(tmp_path).output.exists()


def test_gate_requires_exact_image_cpu_empty_and_fixed_options(monkeypatch, tmp_path):
    monkeypatch.setenv(comparison.GO, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for changed in (dict(container_reference=False), dict(image_digest="wrong"), dict(calls=33), dict(warmup=0), dict(threads=3)):
        with pytest.raises(RuntimeError):
            comparison._gate(args(tmp_path, **changed))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    with pytest.raises(RuntimeError, match="CPU-only"):
        comparison._gate(args(tmp_path))


def test_proof_requires_status_prepost_and_selected_source_consistency(monkeypatch, tmp_path):
    authority = {"selected": {"FLAT": 1}, "finalized_model_source": {"x": 2}, "verified_training_recipe": {"y": 3}}
    proof = {"status": "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY", "authority_pre": authority, "authority_post": authority}
    path = tmp_path / "proof.json"; path.write_text(__import__("json").dumps(proof))
    monkeypatch.setattr(comparison, "PROOF", path); monkeypatch.setattr(comparison, "PROOF_SHA256", comparison.sha(path))
    assert comparison._proof(authority)["status"].startswith("PASS")
    with pytest.raises(RuntimeError, match="disagreement"):
        comparison._proof({**authority, "selected": {"FLAT": 2}})


def test_static_metadata_is_explicitly_unpromoted_and_frozen_image_is_bound():
    assert comparison.STATUS.startswith("EXPERIMENTAL_NOT_PROMOTED")
    assert comparison.IMAGE_DIGEST.endswith("8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f")
    assert comparison.PROOF_SHA256 == "a76c6cc871a27e5a450998a02d5e299a09992335b1efec85d0b31891890f6a38"
    assert comparison.LOCAL_REFERENCE_FILES[comparison.PAYLOAD] == "4e4dae8f7239582a26d44cdb449e674710f28223523dd691dd4f8758b05220e0"
    assert comparison.IMAGE_SPINT_SHA256 == "855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519"
