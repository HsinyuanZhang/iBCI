"""Contracts for the rejected-v1 / non-transferable v2 RT LS audit."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_rt_afc4_ls_null_strength_v2.py"
SPEC = importlib.util.spec_from_file_location("rt_afc4_ls_null_strength_audit_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def _diverse_reaches(*, session_shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    groups = np.repeat(np.arange(8, dtype=np.int64), 7)
    rows = []
    for reach in range(8):
        angle = session_shift + 2.0 * np.pi * reach / 8.0
        for block in range(7):
            rows.append((0.9 + block * 0.07) * np.asarray([
                np.cos(angle + (block - 3) * 0.02), np.sin(angle + (block - 3) * 0.02)
            ]))
    return groups, np.asarray(rows, dtype=np.float64)


def test_v2_is_cross_reach_bijective_and_nonextreme_without_rates_or_w() -> None:
    groups, velocity = _diverse_reaches()
    permutation, selection = AUDIT.deterministic_random_cross_reach_derangement(
        groups, velocity, session_name="random-session", seed=42
    )
    diagnostics = selection["permutation_diagnostics"]
    np.testing.assert_array_equal(np.sort(permutation), np.arange(groups.size))
    assert np.all(permutation != np.arange(groups.size))
    assert np.all(groups[permutation] != groups)
    assert diagnostics["same_reach_assigned_blocks"] == 0
    transfer = diagnostics["reach_direction_transfer"]
    assert abs(transfer["mean_cosine"]) <= AUDIT.MAX_ABS_SESSION_REACH_DIRECTION_COSINE
    assert abs(transfer["median_cosine"]) <= AUDIT.MAX_ABS_SESSION_REACH_DIRECTION_COSINE
    assert tuple(inspect.signature(AUDIT.deterministic_random_cross_reach_derangement).parameters) == (
        "groups", "velocity", "session_name", "seed"
    )
    names = AUDIT.deterministic_random_cross_reach_derangement.__code__.co_names
    assert "fit_descriptor" not in names and "descriptor_comparison" not in names


def test_v2_rejects_collinear_labels_instead_of_hiding_a_common_transfer() -> None:
    groups = np.repeat(np.arange(4, dtype=np.int64), 6)
    velocity = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (groups.size, 1))
    with pytest.raises(AUDIT.V2NullError, match="no session-namespaced random cross-reach"):
        AUDIT.deterministic_random_cross_reach_derangement(groups, velocity, session_name="collinear", seed=42)


def _fixed_map_records(matrix: np.ndarray) -> list[dict]:
    records = []
    for session in range(4):
        rng = np.random.RandomState(100 + session)
        null = rng.normal(size=(11, 2))
        null += np.asarray([0.5 * session, -0.2 * session])
        records.append({
            "session_name": f"session-{session}", "null_reach_means": null,
            "correct_reach_means": null @ matrix,
        })
    return records


@pytest.mark.parametrize("matrix", [np.asarray([[-1.0, 0.0], [0.0, -1.0]]), np.asarray([[0.0, -1.0], [1.0, 0.0]])])
def test_transfer_gate_fails_fixed_negation_and_fixed_rotation(matrix: np.ndarray) -> None:
    audit = AUDIT.leave_one_session_out_transfer_audit(_fixed_map_records(matrix))
    assert audit["linear"]["mean"] == pytest.approx(1.0, abs=1.0e-12)
    assert audit["orthogonal"]["mean"] == pytest.approx(1.0, abs=1.0e-12)
    assert audit["predeclared_gate"]["pass"] is False


def test_transfer_gate_passes_session_namespaced_noncommon_cross_reach_maps() -> None:
    records = []
    for session in range(6):
        groups, velocity = _diverse_reaches(session_shift=session * 0.19)
        permutation, _selection = AUDIT.deterministic_random_cross_reach_derangement(
            groups, velocity, session_name=f"session-{session}", seed=42
        )
        null, correct = AUDIT.reach_summary_pairs(groups, velocity, permutation)
        records.append({"session_name": f"session-{session}", "null_reach_means": null, "correct_reach_means": correct})
    audit = AUDIT.leave_one_session_out_transfer_audit(records)
    assert audit["predeclared_gate"]["pass"] is True


def test_source_records_rejected_v1_reason_and_never_imports_active_rt_producer() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "systematic_antialignment_learnable_by_source_trained_consumer" in source
    assert "load_rt_session" not in source
    assert "RtDataModule" not in source
    assert "torch" not in source
    assert '"--execute-support-audit"' in source
