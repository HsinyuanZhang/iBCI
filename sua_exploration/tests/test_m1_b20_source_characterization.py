"""Synthetic/no-NWB contracts for the blocked M1 B20 characterization."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.m1_b20_source_characterization import (  # noqa: E402
    ROW_ATTACHMENT_NULL_REPLICATES,
    ROW_ATTACHMENT_RANDOM_SCHEDULES,
    A2_B20_R2_BY_SESSION,
    WIDTH,
    attachment_shuffle_carrier,
    attachment_shuffled_outer_carriers,
    b20_carrier,
    b20_split_reliability,
    build_outer_loso_carriers,
    complementary_binomial_thinning,
    deterministic_unit_row_attachment_permutation,
    monte_carlo_upper_rank,
    odd_even_bin_split,
    outer_loso_proxy,
    a2_b20_reproduction_receipt,
    family_distinguishable,
    terminal_characterization_decision,
    q10_carrier,
    r10_carrier,
    rate_only_carrier,
    row_attachment_schedule_receipt,
)

SCRIPT = ROOT / "sua_exploration" / "scripts" / "audit_m1_b20_source_characterization.py"


def _trials(offset: int, *, units: int = 4) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for index in range(10):
        bins = 4 + index % 3
        base = np.arange(bins * units, dtype=np.int64).reshape(bins, units)
        result.append((base + offset + 3 * index) % 9)
    return tuple(result)


def _support() -> dict[str, tuple[np.ndarray, ...]]:
    return {name: _trials(index + 1) for index, name in enumerate(("s0", "s1", "s2", "s3"))}


def _targets() -> dict[str, np.ndarray]:
    return {
        name: np.add.outer(np.arange(4, dtype=np.float64) + index * 0.1, np.arange(4, dtype=np.float64) * 0.01)
        for index, name in enumerate(("s0", "s1", "s2", "s3"))
    }


def _script_module():
    spec = importlib.util.spec_from_file_location("m1_b20_source_characterization_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_r10_q10_b20_are_width20_exact_and_order_invariant():
    trials = _trials(2)
    r10 = r10_carrier(trials)
    q10 = q10_carrier(trials)
    b20 = b20_carrier(trials)
    assert r10.shape == q10.shape == b20.shape == (4, WIDTH)
    assert np.all(r10[:, 10:] == 0.0)
    assert np.all(q10[:, :10] == 0.0)
    assert np.allclose(b20, r10 + q10)
    permuted_bins = tuple(trial[::-1, :] for trial in reversed(trials))
    assert np.allclose(r10, r10_carrier(permuted_bins))
    assert np.allclose(q10, q10_carrier(permuted_bins))
    assert np.allclose(b20, b20_carrier(permuted_bins))
    assert rate_only_carrier(trials).shape == (4, WIDTH)


def test_attachment_shuffle_moves_complete_rows_only_and_keeps_per_column_marginals():
    carrier = b20_carrier(_trials(3))
    identity = attachment_shuffle_carrier(carrier, session_name="s0", replicate=0)
    shuffled = attachment_shuffle_carrier(carrier, session_name="s0", replicate=7)
    assert np.array_equal(identity, carrier)
    assert np.array_equal(np.sort(shuffled, axis=0), np.sort(carrier, axis=0))
    assert {tuple(row) for row in shuffled} == {tuple(row) for row in carrier}
    assert np.array_equal(
        deterministic_unit_row_attachment_permutation(session_name="s0", replicate=7, units=4),
        deterministic_unit_row_attachment_permutation(session_name="s0", replicate=7, units=4),
    )
    receipt = row_attachment_schedule_receipt(("s0", "s1", "s2", "s3"), units=4)
    assert receipt["replicates_including_observed_identity"] == ROW_ATTACHMENT_NULL_REPLICATES
    assert receipt["random_replicates_after_identity"] == ROW_ATTACHMENT_RANDOM_SCHEDULES
    assert receipt["identity_schedule_indices"][0] == 0
    assert len(receipt["schedule_sha256"]) == ROW_ATTACHMENT_NULL_REPLICATES
    order = deterministic_unit_row_attachment_permutation(session_name="s0", replicate=7, units=4)
    r_only = attachment_shuffle_carrier(carrier, session_name="s0", replicate=7, family="A-R")
    q_only = attachment_shuffle_carrier(carrier, session_name="s0", replicate=7, family="A-Q")
    assert np.array_equal(r_only[:, :10], carrier[order, :10]) and np.array_equal(r_only[:, 10:], carrier[:, 10:])
    assert np.array_equal(q_only[:, :10], carrier[:, :10]) and np.array_equal(q_only[:, 10:], carrier[order, 10:])


def test_outer_loso_builder_is_target_free_and_shuffled_arm_remains_width20():
    base = build_outer_loso_carriers(_support())
    assert "target" not in build_outer_loso_carriers.__code__.co_varnames
    assert set(base.public) == {"s0", "s1", "s2", "s3"}
    for fold in base.public.values():
        assert set(fold) == {"R10", "Q10", "B20", "rate_only"}
        assert all(value.shape[1] == WIDTH for arm in fold.values() for value in arm.values())
    shuffled = attachment_shuffled_outer_carriers(base, replicate=3, family="A-R")
    assert all(set(fold) == {"A-R"} for fold in shuffled.values())
    assert all(value.shape[1] == WIDTH for fold in shuffled.values() for value in fold["A-R"].values())
    proxy = outer_loso_proxy(base.public, _targets())
    assert set(proxy["arms"]) == {"R10", "Q10", "B20", "rate_only"}
    assert all(0.0 < row["bounded_u"] <= 1.0 for rows in proxy["arms"].values() for row in rows)


def test_bounded_monte_carlo_rank_is_robust_and_requires_frozen_schedule_count():
    null = np.full(ROW_ATTACHMENT_NULL_REPLICATES, 0.2)
    null[0] = 0.9
    rank = monte_carlo_upper_rank(0.9, null)
    assert rank["conservative_upper_tail_p_value"] == pytest.approx(1.0 / ROW_ATTACHMENT_NULL_REPLICATES)
    assert rank["one_sided_97p5_binomial_upper_bound"] < 0.025
    with pytest.raises(ValueError, match="4,096"):
        monte_carlo_upper_rank(0.9, null[:-1])


def test_split_reliability_uses_complementary_thinning_x2_and_odd_even_diagnostic():
    trials = _trials(5)
    left, right = complementary_binomial_thinning(trials, session_name="s0", repeat=1)
    for original, a, b in zip(trials, left, right):
        assert np.array_equal((a + b) // 2, original)
        assert np.all(a % 2 == 0) and np.all(b % 2 == 0)
    odd, even = odd_even_bin_split(trials)
    assert all(a.shape[0] + b.shape[0] == original.shape[0] for original, a, b in zip(trials, odd, even))
    receipt = b20_split_reliability(trials, session_name="s0", repeats=5)
    assert receipt["thinning"]["method"].endswith("rescaled_x2")
    assert set(receipt["thinning"]["median_row_cosine_quantiles_2p5_50_97p5"]) == {"R10", "Q10", "B20"}
    assert all(len(values) == 3 for values in receipt["thinning"]["median_row_cosine_quantiles_2p5_50_97p5"].values())
    assert all(len(values) == WIDTH for values in receipt["odd_even_bin_diagnostic"]["per_coordinate_pearson_including_undefined"].values())
    assert receipt["independent_biological_samples_added"] == 0


def test_protocol_reproduction_and_terminal_decision_helpers_are_exact():
    proxy = {"arms": {"B20": [{"left_out_session": session, "r2": value} for session, value in A2_B20_R2_BY_SESSION.items()]}}
    assert a2_b20_reproduction_receipt(proxy)["pass"] is True
    observed = {name: 0.9 for name in ("s0", "s1", "s2", "s3")}
    rank = {"conservative_upper_tail_p_value": 1.0 / 4096, "one_sided_97p5_binomial_upper_bound": 0.001}
    family = family_distinguishable(
        observed_u_by_session=observed,
        null_u_by_session={name: [0.2] * ROW_ATTACHMENT_RANDOM_SCHEDULES for name in observed}, rank=rank,
    )
    assert family["distinguishable"] is True
    # Protocol-precedence ambiguity is resolved by the reviewed lower-state
    # prior: if both components qualify for simplification, retain R10 only.
    decision = terminal_characterization_decision(
        valid=True, a2_reproduction_pass=True, all_family=family,
        r_family={"distinguishable": False}, q_family={"distinguishable": False},
        d_r_by_session=[0.01] * 4, d_q_by_session=[0.01] * 4,
    )
    assert "retain_R10_lower_state_candidate" in decision["decision"]


def test_prelaunch_is_fresh_no_nwb_and_execute_has_sha_bound_guard(tmp_path, monkeypatch):
    module = _script_module()
    monkeypatch.setattr(module, "conservative_no_nwb_runtime_benchmark", lambda: {"within_24h_fixed_worker_projection": True, "cuda_visible_devices": "", "kind": "synthetic-test"})
    receipt = module.build_prelaunch_receipt()
    assert receipt["hard_exclusions"]["nwb_opened_for_this_receipt"] is False
    assert receipt["status"] == "pending_root_review_no_nwb_opened"
    path = module.write_prelaunch(tmp_path / "prelaunch")
    loaded = json.loads(path.read_text())
    assert loaded["contracts"]["attachment_null"]["replicates"] == ROW_ATTACHMENT_NULL_REPLICATES
    assert (path.parent / "prelaunch_receipt.sha256").is_file()
    with pytest.raises(FileExistsError):
        module.write_prelaunch(path.parent)
    source = SCRIPT.read_text()
    assert "FalconDataModule" not in source
    assert "load_raw_source_sessions" in source
    assert "--execute-source-audit" in source
    assert 'CUDA_VISIBLE_DEVICES"] = ""' in source


def test_execution_review_binds_prelaunch_protocol_and_manifest_hashes(tmp_path):
    module = _script_module()
    prelaunch = tmp_path / "prelaunch.json"
    prelaunch.write_text(json.dumps({
        "schema_version": module.SCHEMA_PRELAUNCH,
        "manifest": {"sha256": module.sha256(module.MANIFEST)},
        "protocol": {"sha256": module.sha256(module.PROTOCOL)},
        "code": {
            "runner": {"sha256": module.sha256(Path(module.__file__))},
            "pure_contracts": {"sha256": module.sha256(module.SUA / "mc_maze" / "m1_b20_source_characterization.py")},
            "a2_exact_raw_loader": {"sha256": module.sha256(module.SUA / "scripts" / "audit_m1_fixed_k_prototypes_gate_a.py")},
        },
    }))
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "authorization": module.REVIEW_AUTHORIZATION,
        "prelaunch_receipt_sha256": module.sha256(prelaunch),
        "protocol_sha256": module.sha256(module.PROTOCOL),
        "manifest_sha256": module.sha256(module.MANIFEST),
    }))
    module._check_execution_review(review, prelaunch)
    review.write_text(json.dumps({"authorization": module.REVIEW_AUTHORIZATION, "prelaunch_receipt_sha256": "bad"}))
    with pytest.raises(PermissionError, match="prelaunch"):
        module._check_execution_review(review, prelaunch)
