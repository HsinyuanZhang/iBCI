"""In-memory contracts for the isolated RT AFC4 LS null-strength audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_rt_afc4_ls_null_strength.py"
SPEC = importlib.util.spec_from_file_location("rt_afc4_ls_null_strength_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _diverse_reach_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Six equal-length reaches with distinct velocity directions and speeds."""

    groups = np.repeat(np.arange(6, dtype=np.int64), 8)
    values = []
    for reach in range(6):
        theta = 2.0 * np.pi * reach / 6.0
        for block in range(8):
            jitter = (block - 3.5) * 0.025
            values.append((1.0 + 0.04 * block) * np.asarray([np.cos(theta + jitter), np.sin(theta + jitter)]))
    return groups, np.asarray(values, dtype=np.float64)


def test_current_ls_is_within_reach_and_preserves_each_reach_direction() -> None:
    groups, velocity = _diverse_reach_fixture()
    from src.data.falcon_k4_features import deterministic_k4_block_label_permutation

    permutation = deterministic_k4_block_label_permutation(groups, session_name="synthetic", seed=42)
    report = AUDIT.permutation_diagnostics(groups, velocity, permutation)
    assert report["labels_changed_blocks"] == groups.size
    assert report["same_reach_assigned_fraction"] == 1.0
    assert report["velocity_marginal"]["preserved_exactly_by_index_permutation"] is True
    assert report["block_length_statistics"]["per_reach_neural_block_counts_unchanged"] is True
    assert report["reach_direction_transfer"]["mean_cosine"] == pytest.approx(1.0, abs=1.0e-12)


def test_cross_reach_null_is_bijective_nonidentity_and_breaks_coarse_association() -> None:
    groups, velocity = _diverse_reach_fixture()
    permutation = AUDIT.deterministic_cross_reach_derangement(
        groups, velocity, session_name="synthetic", seed=42
    )
    report = AUDIT.permutation_diagnostics(groups, velocity, permutation)
    np.testing.assert_array_equal(np.sort(permutation), np.arange(groups.size))
    assert np.all(permutation != np.arange(groups.size))
    assert np.all(groups[permutation] != groups)
    assert report["same_reach_assigned_blocks"] == 0
    assert report["reach_direction_transfer"]["mean_cosine"] <= AUDIT.MAX_STRONG_NULL_GROUP_DIRECTION_COSINE
    assert report["reach_direction_transfer"]["median_cosine"] <= AUDIT.MAX_STRONG_NULL_GROUP_DIRECTION_COSINE
    assert report["velocity_marginal"]["global_mean_delta"] == pytest.approx([0.0, 0.0], abs=1.0e-12)
    assert report["velocity_marginal"]["global_speed_mean_delta"] == pytest.approx(0.0, abs=1.0e-12)


def test_cross_reach_null_fails_closed_when_coarse_direction_cannot_be_broken() -> None:
    groups = np.repeat(np.arange(3, dtype=np.int64), 6)
    # Cross-reach source IDs exist, but every reach has exactly the same
    # direction.  It would be misleading to call that a direction-mismatch
    # null, so the predeclared strength gate must reject it.
    velocity = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (groups.size, 1))
    with pytest.raises(AUDIT.StrongNullError, match="coarse reach-direction association"):
        AUDIT.deterministic_cross_reach_derangement(groups, velocity, session_name="collinear", seed=42)


def test_descriptor_comparison_records_signed_w_geometry_and_rate_components() -> None:
    aligned = np.asarray([[1.0, 0.0, 1.0, 3.0], [0.0, 2.0, 2.0, 4.0]], dtype=np.float32)
    null = np.asarray([[0.0, 1.0, 1.0, 5.0], [2.0, 0.0, 2.0, 1.0]], dtype=np.float32)
    comparison = AUDIT.descriptor_comparison(aligned, null)
    rows = comparison["per_channel"]
    assert rows[0]["w_cosine"] == pytest.approx(0.0)
    assert rows[0]["signed_w_angle_deg"] == pytest.approx(90.0)
    assert rows[1]["signed_w_angle_deg"] == pytest.approx(-90.0)
    assert rows[0]["delta_b"] == pytest.approx(2.0)
    assert rows[1]["delta_b"] == pytest.approx(-3.0)
    assert comparison["w_norm_difference"]["mean_absolute_delta"] == pytest.approx(0.0)
    assert comparison["baseline_b_difference"]["mean_absolute_delta"] == pytest.approx(2.5)


def test_source_is_isolated_from_full_session_loader_gpu_and_active_rt_producers() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "from src.data.rt_k4_loader import load_rt_session" not in source
    assert "from src.data.rt_datamodule import RtDataModule" not in source
    assert "torch" not in source
    assert "velocity_series.data[:]" not in source
    assert "velocity_series.timestamps[:]" not in source
    assert '"--execute-support-audit"' in source
    assert "CUDA_VISIBLE_DEVICES" in source
