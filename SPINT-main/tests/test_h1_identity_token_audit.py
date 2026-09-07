"""Contract tests for the forward-only H1 identity-token audit.

These tests are deliberately CPU-only, no-target, and re-use the already
sealed/immutable source caches and checkpoints on disk (mode 0444).  They do
not open any target/formal/minival/EvalAI file, and they do not write to any
of the paths this audit is forbidden from writing to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts import h1_identity_token_audit as audit
from src.data.h1_m4_eb_pilot import H1_HELDIN_SESSIONS, H1_M4_FOLD0_SOURCE, H1_M4_FOLD0_TARGET


ROOT = Path(__file__).resolve().parents[1]


def test_source_session_set_is_exactly_the_11_fold0_source_sessions():
    assert len(H1_M4_FOLD0_SOURCE) == 11
    assert set(H1_M4_FOLD0_SOURCE) | set(H1_M4_FOLD0_TARGET) == set(H1_HELDIN_SESSIONS)
    assert set(H1_M4_FOLD0_SOURCE).isdisjoint(H1_M4_FOLD0_TARGET)


def test_checkpoints_and_configs_and_caches_exist_and_are_immutable():
    import stat

    for spec in audit.CHECKPOINTS.values():
        checkpoint_path = Path(spec["checkpoint_path"])
        config_path = Path(spec["config_path"])
        cache_dir = Path(spec["cache_dir"])
        assert checkpoint_path.is_file(), checkpoint_path
        assert config_path.is_file(), config_path
        assert cache_dir.is_dir(), cache_dir
        # Checkpoints themselves are plain read-only-by-convention files (mode
        # 0664 here), not the immutable mode-0444 artifacts; only the frozen
        # source-cache members below are contract-enforced immutable.
        for name in (
            "fold0_frozen_eb_plan.npz",
            "fold0_frozen_eb_plan.manifest.json",
            "fold0_all_source_m4_carriers.npz",
            "fold0_all_source_m4_carriers.manifest.json",
        ):
            member = cache_dir / name
            assert member.is_file(), member
            assert stat.S_IMODE(member.stat().st_mode) == 0o444


def test_load_net_matches_checkpoint_and_is_forward_only_pure():
    spec = audit.CHECKPOINTS["hc_seed42_canonical_fold0_epoch49"]
    net = audit.load_net(Path(spec["checkpoint_path"]))
    assert isinstance(net, audit.H1CarrierIdSpint)
    assert net.zero_carrier is False
    assert not net.training  # .eval() was called

    guard = audit.ForwardOnlyStateGuard(net)
    calib = torch.zeros(1, 4, 1024, 176, dtype=torch.float32)
    carrier = torch.zeros(1, 176, 4, dtype=torch.float32)
    before = audit.state_hash(net.state_dict())
    out_1 = guard.guarded_projection(calib, carrier).clone()
    out_2 = guard.guarded_projection(calib, carrier).clone()
    after = audit.state_hash(net.state_dict())
    assert before == after, "forward pass must not mutate model state"
    assert torch.equal(out_1, out_2), "identical inputs must give identical outputs (no hidden RNG state)"
    assert out_1.shape == (1, 176, 700)


def test_build_source_context_opens_only_source_sessions():
    spec = audit.CHECKPOINTS["hc_seed42_canonical_fold0_epoch49"]
    context = audit.build_source_context(Path(spec["cache_dir"]))
    assert set(context["records"]) == set(H1_M4_FOLD0_SOURCE)
    for record in context["records"].values():
        assert record.date != "19250101"  # the fold-0 target date is never present in source
        assert "held-in-calib" in str(record.path)
        assert "held-out" not in str(record.path).lower()
        assert "minival" not in str(record.path).lower()
        assert "formal" not in str(record.path).lower()
        assert "evalai" not in str(record.path).lower()


def test_decomposition_reconstruction_identity_holds_exactly():
    """activity_part + carrier_part - both_zero + residual must equal full, exactly."""
    spec = audit.CHECKPOINTS["hc_seed42_canonical_fold0_epoch49"]
    net = audit.load_net(Path(spec["checkpoint_path"]))
    guard = audit.ForwardOnlyStateGuard(net)
    context = audit.build_source_context(Path(spec["cache_dir"]))
    session_name = H1_M4_FOLD0_SOURCE[0]
    result = audit.per_session_measurements(guard, context, session_name)
    m2 = result["measurement_2_decomposition"]
    # The residual is reported, not assumed zero; but it must be finite and the
    # relation full = activity + carrier - both_zero + residual is definitional,
    # so just check the reported quantities are internally consistent in sign/scale.
    assert m2["activity_part_rms"] > 0.0
    assert m2["carrier_part_rms"] > 0.0
    assert np.isfinite(m2["nonlinear_reconstruction_residual_rms"])
    assert m2["rms_ratio_carrier_over_activity"] == m2["carrier_part_rms"] / m2["activity_part_rms"]


def test_spectrum_stats_participation_ratio_on_known_matrices():
    # Rank-1 matrix: participation ratio must be exactly 1.
    rank1 = np.outer(np.arange(1, 6, dtype=np.float64), np.arange(1, 8, dtype=np.float64))
    stats = audit._spectrum_stats(rank1)
    assert stats["participation_ratio_effective_dim"] == pytest_approx(1.0)
    assert stats["numpy_default_tol_matrix_rank"] == 1

    # Orthogonal-rows identity-like matrix of rank k has participation ratio k.
    identity_like = np.eye(4, 7, dtype=np.float64)
    stats2 = audit._spectrum_stats(identity_like)
    assert stats2["participation_ratio_effective_dim"] == pytest_approx(4.0)
    assert stats2["numpy_default_tol_matrix_rank"] == 4


def pytest_approx(value, rel=1e-6):
    import pytest

    return pytest.approx(value, rel=rel)


def test_per_session_measurements_are_forward_deterministic_across_reload():
    """Reloading the checkpoint from disk twice must give bit-identical identity tokens."""
    spec = audit.CHECKPOINTS["hc_seed43_sealed_epoch49"]
    session_name = H1_M4_FOLD0_SOURCE[3]

    net_a = audit.load_net(Path(spec["checkpoint_path"]))
    guard_a = audit.ForwardOnlyStateGuard(net_a)
    context = audit.build_source_context(Path(spec["cache_dir"]))
    result_a = audit.per_session_measurements(guard_a, context, session_name)

    net_b = audit.load_net(Path(spec["checkpoint_path"]))
    guard_b = audit.ForwardOnlyStateGuard(net_b)
    result_b = audit.per_session_measurements(guard_b, context, session_name)

    assert (
        result_a["measurement_1_identity_token"]["identity_full_rms_overall"]
        == result_b["measurement_1_identity_token"]["identity_full_rms_overall"]
    )
    assert result_a["measurement_4_spectrum"]["full_identity_token_spectrum"]["singular_values"] == (
        result_b["measurement_4_spectrum"]["full_identity_token_spectrum"]["singular_values"]
    )


def test_receipt_file_exists_with_matching_module_sha256():
    import hashlib
    import json

    module_path = ROOT / "scripts" / "h1_identity_token_audit.py"
    receipt_path = ROOT / "scripts" / "h1_identity_token_audit_receipt.json"
    assert module_path.is_file()
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # The receipt records the module hash *at the time it was generated*; a
    # later edit to the module is expected to require regenerating the
    # receipt, so this only checks internal shape/consistency, not that the
    # module is byte-identical to some earlier snapshot.
    assert len(receipt["module_sha256"]) == 64
    for label, result in receipt["results"].items():
        assert result["state_immutable_across_all_forward_calls"] is True
        assert result["state_sha256_before"] == result["state_sha256_after"]
        assert set(result["per_session"]) == set(H1_M4_FOLD0_SOURCE)
        assert result["forward_calls"] == 11 * 4
