"""No-NWB contracts for the A2 fixed-K temporal-prototype controls."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))

from mc_maze.fixed_k_temporal_prototypes import FIXED_K  # noqa: E402
from mc_maze.m1_fixed_k_prototype_gate_a2 import (  # noqa: E402
    a2_cpu_gate,
    MARGINAL_CONTROL_COLUMNS,
    SLOT_NULL_REPLICATES,
    WIDTH,
    build_a2_base_carriers,
    canonical_slot_identity_rank_diagnostic,
    deterministic_trial_time_permutation,
    exact_group_upper_rank,
    exact_slot_chunk_ranges,
    exact_slot_null_plans,
    generic_outer_loso_proxy,
    marginal_distribution_carrier,
    mean_arm_metric,
    paired_b20_content_summary,
    slot_null_carriers,
    slot_null_plan_receipt,
    scheduled_time_upper_rank,
    time_null_receipt,
    time_order_null_carriers,
    within_trial_time_order_null,
)

SCRIPT = ROOT / "sua_exploration" / "scripts" / "audit_m1_fixed_k_prototypes_gate_a2.py"


def _trials(offset: int, *, units: int = 3) -> tuple[np.ndarray, ...]:
    values: list[np.ndarray] = []
    for trial in range(10):
        bins = 3 + (trial % 3)
        base = np.arange(bins * units, dtype=np.int64).reshape(bins, units)
        values.append((base + offset + trial) % 7)
    return tuple(values)


def _support() -> dict[str, tuple[np.ndarray, ...]]:
    return {name: _trials(index + 1) for index, name in enumerate(("s0", "s1", "s2", "s3"))}


def _d4() -> dict[str, np.ndarray]:
    return {name: np.arange(12, dtype=np.float64).reshape(3, 4) + index for index, name in enumerate(("s0", "s1", "s2", "s3"))}


def _targets() -> dict[str, np.ndarray]:
    return {
        name: np.asarray([[0.1 + index, 0.2, 0.3, 0.4], [0.4, 0.2 + index, 0.1, 0.5], [0.3, 0.6, 0.2 + index, 0.1]], dtype=np.float64)
        for index, name in enumerate(("s0", "s1", "s2", "s3"))
    }


def _script_module():
    spec = importlib.util.spec_from_file_location("m1_fixed_k_prototype_a2_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_marginal_carrier_is_exactly_width20_and_invariant_to_time_and_trial_order():
    trials = _trials(2)
    carrier = marginal_distribution_carrier(trials)
    shuffled = within_trial_time_order_null(trials, session_name="s0", replicate=9)
    reversed_trials = tuple(reversed(trials))
    assert carrier.shape == (3, WIDTH) == (3, 20)
    assert len(MARGINAL_CONTROL_COLUMNS) == 20
    assert np.allclose(carrier, marginal_distribution_carrier(shuffled))
    assert np.allclose(carrier, marginal_distribution_carrier(reversed_trials))
    with pytest.raises(ValueError, match="exactly ten"):
        marginal_distribution_carrier(trials[:9])
    # First coordinate is the sorted trial log-rate with the mandated +0.5
    # pseudocount; coordinate 10 begins the mandated pooled log1p quantiles.
    expected_rate = sorted(
        np.log((trial[:, 0].sum() + 0.5) / (0.020 * trial.shape[0])) for trial in trials
    )
    assert np.allclose(carrier[0, :10], expected_rate)
    assert carrier[0, 10] == pytest.approx(np.quantile(np.log1p(np.concatenate(trials, axis=0)[:, 0]), 0.05, method="linear"))


def test_time_null_is_joint_across_units_nonidentity_and_preserves_trial_unit_marginals():
    trials = _trials(3)
    null = within_trial_time_order_null(trials, session_name="s1", replicate=17)
    for index, (before, after) in enumerate(zip(trials, null)):
        order = deterministic_trial_time_permutation(session_name="s1", replicate=17, trial_index=index, bins=before.shape[0])
        assert np.array_equal(after, before[order, :])
        assert np.array_equal(after.sum(axis=0), before.sum(axis=0))
        assert np.array_equal(np.sort(after, axis=0), np.sort(before, axis=0))
    assert np.array_equal(
        deterministic_trial_time_permutation(session_name="s1", replicate=0, trial_index=0, bins=trials[0].shape[0]),
        np.arange(trials[0].shape[0]),
    )
    receipt = time_null_receipt({"s0": trials, "s1": trials, "s2": trials, "s3": trials})
    assert receipt["within_trial_only"] is True
    assert receipt["same_bin_permutation_for_all_units_within_trial"] is True
    assert len(receipt["plan_sha256"]) == 64
    assert receipt["identity_schedule_indices"] == [0]
    assert len(receipt["schedule_sha256"]) == 4096


def test_exact_slot_null_enumerates_quotient_once_and_uses_reference_identity():
    plans = exact_slot_null_plans(("s0", "s1", "s2", "s3"))
    assert len(plans) == SLOT_NULL_REPLICATES == 24 ** 3
    identity = np.arange(FIXED_K)
    assert all(np.array_equal(plan["s0"], identity) for plan in plans)
    canonical = {
        tuple(tuple(plan[name].tolist()) for name in ("s0", "s1", "s2", "s3"))
        for plan in plans
    }
    assert len(canonical) == 24 ** 3
    receipt = slot_null_plan_receipt(plans)
    assert receipt["permutation_group"] == "(S4)^4 / global_S4_exact_quotient"
    assert receipt["enumerated_classes"] == 24 ** 3
    assert receipt["reference_session"] == "s0"
    chunks = exact_slot_chunk_ranges(workers=8)
    assert chunks[0][0] == 0 and chunks[-1][1] == 24 ** 3
    assert all(left[1] == right[0] for left, right in zip(chunks, chunks[1:]))


def test_exact_rank_is_bounded_u_group_rank_and_rejects_incomplete_group():
    values = np.linspace(0.1, 0.9, SLOT_NULL_REPLICATES)
    result = exact_group_upper_rank(0.9, values)
    assert result["upper_rank"] == 1
    assert result["exact_upper_tail_p_value"] == pytest.approx(1.0 / SLOT_NULL_REPLICATES)
    assert result["r2_is_secondary_only"] is True
    with pytest.raises(ValueError, match=r"24\*\*3"):
        exact_group_upper_rank(0.5, [0.1, 0.2])
    time_values = np.full(4096, 0.2)
    time_values[0] = 0.5
    time_rank = scheduled_time_upper_rank(0.5, time_values)
    assert time_rank["scheduled_upper_tail_p_value"] == pytest.approx(1.0 / 4096)
    assert time_rank["one_sided_97p5_binomial_upper_bound"] < 0.025
    # U=1/(2-R2) smoothly maps arbitrarily negative finite R2 toward zero;
    # no hard clipping/outlier deletion is involved.
    assert 1.0 / (2.0 - (-200.0)) < 1.0 / (2.0 - (-2.0))


def test_base_and_null_carriers_remain_fold_specific_width20_and_no_target_argument_to_builder():
    base = build_a2_base_carriers(_support(), _d4())
    assert set(base.public) == {"s0", "s1", "s2", "s3"}
    plans = exact_slot_null_plans(("s0", "s1", "s2", "s3"))
    carrier = slot_null_carriers(base, plan_for_replicate=plans[0])
    for left_out in base.public:
        assert left_out not in base.anchor_receipts[left_out]["source_sessions"]
        assert set(carrier[left_out]) == {"D4", "rate_only", "marginal_distribution", "prototype", "slot_null"}
        assert all(value.shape[1] == 20 for arm in carrier[left_out].values() for value in arm.values())
    # The source-only builder accepts support and the visible D4 reference;
    # later-neural targets cannot enter its API.
    assert "target" not in build_a2_base_carriers.__code__.co_varnames


def test_generic_proxy_reports_bounded_u_and_time_null_has_no_d4_or_target_input():
    base = build_a2_base_carriers(_support(), _d4())
    proxy = generic_outer_loso_proxy(base.public, _targets())
    for rows in proxy["arms"].values():
        assert len(rows) == 4
        assert all(0.0 <= row["bounded_u"] <= 1.0 and "r2" in row for row in rows)
    assert 0.0 <= mean_arm_metric(proxy, "prototype") <= 1.0
    time = time_order_null_carriers(_support(), replicate=2)
    assert all(set(fold) == {"time_order_null"} for fold in time.values())
    assert all(value.shape[1] == 20 for fold in time.values() for value in fold["time_order_null"].values())


def test_protocol_gate_helpers_expose_all_required_cpu_conditions_on_synthetic_scores():
    plans = exact_slot_null_plans(("s0", "s1", "s2", "s3"))
    group_h = np.full(SLOT_NULL_REPLICATES, 0.2)
    identity = next(index for index, plan in enumerate(plans) if all(np.array_equal(order, np.arange(4)) for order in plan.values()))
    group_h[identity] = 0.9
    conditional = canonical_slot_identity_rank_diagnostic(plans, group_h)
    assert conditional["all_sessions_strictly_above_conditional_median"] is True
    content = {
        "all_four_positive": True, "paired_ci95": [0.031, 0.05], "mde80": 0.029,
    }
    slot = {"exact_upper_tail_p_value": 1.0 / SLOT_NULL_REPLICATES}
    time = {"scheduled_upper_tail_p_value": 1.0 / 4096, "one_sided_97p5_binomial_upper_bound": 0.001}
    time_h = np.full(4096, 0.2); time_h[0] = 0.9
    decision = a2_cpu_gate(
        content=content, slot_rank=slot, time_rank=time,
        observed_h=0.9, slot_h=group_h, time_h=time_h,
        observed_u_by_session={name: 0.9 for name in ("s0", "s1", "s2", "s3")},
        slot_u_by_session={name: np.full(SLOT_NULL_REPLICATES, 0.2) for name in ("s0", "s1", "s2", "s3")},
        time_u_by_session={name: np.full(4096, 0.2) for name in ("s0", "s1", "s2", "s3")},
        validity_contract_pass=True, repeatability_contract_pass=True,
    )
    assert decision["decision"] == "gate_a2_pass_cpu_only_prepare_separate_decoder_gpu_protocol"
    assert all(decision["slot_session_canonical_U_above_full_null_median"].values())
    assert decision["gpu_authorized"] is False and decision["decoder_authorized"] is False
    marginal_stop = a2_cpu_gate(
        content={"all_four_positive": False, "paired_ci95": [0.1, 0.2], "mde80": 0.01}, slot_rank=slot, time_rank=time,
        observed_h=0.9, slot_h=group_h, time_h=time_h,
        observed_u_by_session={name: 0.9 for name in ("s0", "s1", "s2", "s3")},
        slot_u_by_session={name: np.full(SLOT_NULL_REPLICATES, 0.2) for name in ("s0", "s1", "s2", "s3")},
        time_u_by_session={name: np.full(4096, 0.2) for name in ("s0", "s1", "s2", "s3")},
        validity_contract_pass=True, repeatability_contract_pass=True,
    )
    precision_stop = a2_cpu_gate(
        content={"all_four_positive": True, "paired_ci95": [0.01, 0.2], "mde80": 0.01}, slot_rank=slot, time_rank=time,
        observed_h=0.9, slot_h=group_h, time_h=time_h,
        observed_u_by_session={name: 0.9 for name in ("s0", "s1", "s2", "s3")},
        slot_u_by_session={name: np.full(SLOT_NULL_REPLICATES, 0.2) for name in ("s0", "s1", "s2", "s3")},
        time_u_by_session={name: np.full(4096, 0.2) for name in ("s0", "s1", "s2", "s3")},
        validity_contract_pass=True, repeatability_contract_pass=True,
    )
    assert marginal_stop["decision"] == "marginal_baseline_not_beaten_stop"
    assert precision_stop["decision"] == "precision_insufficient_stop"


def test_a2_prelaunch_is_no_nwb_and_static_runner_has_no_discovery_or_cuda_route(tmp_path, monkeypatch):
    module = _script_module()
    protocol = tmp_path / "A2_PROTOCOL.md"
    protocol.write_text("frozen synthetic protocol for readiness test\n")
    monkeypatch.setattr(module, "PROTOCOL", protocol)
    monkeypatch.setattr(module, "_load_sessions_after_review", lambda *_: (_ for _ in ()).throw(AssertionError("must not load in prelaunch")))
    monkeypatch.setattr(module, "conservative_no_nwb_runtime_benchmark", lambda: {
        "within_24h_serial_projection": True, "kind": "synthetic-test", "cuda_visible_devices": "",
    })
    receipt = module.build_prelaunch_receipt()
    assert receipt["status"] == "pending_root_review_no_nwb_opened"
    assert receipt["hard_exclusions"]["nwb_opened_for_this_receipt"] is False
    assert receipt["a2_controls"]["slot_null"]["replicates"] == 24 ** 3
    path = module.write_prelaunch(tmp_path / "prelaunch")
    loaded = json.loads(path.read_text())
    assert loaded["endpoint"]["primary_slot_statistic"].startswith("mean four-session bounded U")
    assert (path.parent / "prelaunch_receipt.sha256").is_file()
    with pytest.raises(FileExistsError):
        module.write_prelaunch(path.parent)
    source = SCRIPT.read_text()
    assert "FalconDataModule" not in source
    assert ".setup(" not in source
    assert "rglob(" not in source
    assert 'CUDA_VISIBLE_DEVICES"] = ""' in source
