from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest

from mc_maze import carrier_value_mask_core as core


ROOT = Path(__file__).resolve().parents[2]


def _script(name: str):
    path = ROOT / "sua_exploration" / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(value: float) -> dict:
    return {"per_session_mean_r2": {"s1": value, "s2": value}}


def test_source_gate_constants_are_frozen_and_bounded() -> None:
    assert core.SEED == 42
    assert core.SOURCE_EPOCH == 5
    assert core.WINDOW_CAP_PER_SESSION == 128
    assert core.SOURCE_DIRECTION_AUDIT_TRIALS == 50
    assert core.MASK_FRACTION == 0.25
    assert core.MIN_MEDIAN_VALUE_WEIGHTED_SHARE == 0.10
    assert core.session_seed("session-a") == core.session_seed("session-a")
    assert core.session_seed("session-a") != core.session_seed("session-b")


def _trials(count: int = 55, *, missing: tuple[int, ...] = ()) -> tuple[list[dict], list[dict]]:
    record = []
    production = []
    for index in range(count):
        base = {
            "trial_index": 100 + index,
            "start": index * 60,
            "stop": index * 60 + 55,
            "target_dir": None if index in missing else float(index % 8),
        }
        record.append(dict(base))
        production.append(dict(base))
    return record, production


def test_source_semantics_accept_production_missing_direction_without_weakening_query() -> None:
    record, production = _trials(missing=(27, 41))
    receipt = core.source_trial_semantics(record, production, session="source-s")
    assert receipt["query_usable_trial_indices_start"] == 30
    assert receipt["query_usable_trial_count"] == 25
    assert receipt["post30_query_window_count"] == 25 * 6
    assert receipt["activity_support_usable_indices"] == list(range(30))
    assert receipt["first30_direction_coverage"]["finite_target_dir_count"] == 29
    assert receipt["first30_direction_coverage"]["missing_target_dir_usable_indices"] == [27]
    assert receipt["first30_direction_coverage"]["target_dir_all_finite"] is False
    assert receipt["first50_direction_coverage"]["finite_target_dir_count"] == 48
    assert receipt["first50_direction_coverage"]["missing_target_dir_usable_indices"] == [27, 41]
    assert receipt["source_target_all_label_gate_applied"] is False
    semantics = receipt["production_t4_missing_direction_semantics"]
    assert semantics["missing_direction_index"] == -1
    assert semantics["missing_trial_retained_in_chronological_pool_and_rate_matrix"] is True
    assert semantics["missing_trial_excluded_from_present_directions_and_direction_conditioned_means"] is True
    assert semantics["missing_trial_imputed_or_label_added"] is False
    receipt["available_post30_query_windows"] = receipt["post30_query_window_count"]
    receipt["query_window_cap"] = core.WINDOW_CAP_PER_SESSION
    core.validate_source_session_receipt(receipt, session="source-s")


def test_source_semantics_fail_closed_on_chronology_or_unsanitized_nonfinite() -> None:
    record, production = _trials()
    poisoned = [dict(row) for row in production]
    poisoned[27]["trial_index"] += 1
    with pytest.raises(core.CarrierValueMaskContractError, match="trial_index drift"):
        core.source_trial_semantics(record, poisoned, session="source-s")
    poisoned = [dict(row) for row in production]
    poisoned[27]["target_dir"] = float("nan")
    with pytest.raises(core.CarrierValueMaskContractError, match="did not normalize non-finite"):
        core.source_trial_semantics(record, poisoned, session="source-s")


def test_source_semantics_require_post30_query_and_do_not_apply_target_label_gate() -> None:
    record, production = _trials(count=30, missing=(27,))
    with pytest.raises(Exception, match="more than 30"):
        core.source_trial_semantics(record, production, session="source-s")


def test_source_session_receipt_rejects_false_all_finite_or_skip_semantics() -> None:
    record, production = _trials(missing=(27,))
    receipt = core.source_trial_semantics(record, production, session="source-s")
    receipt["available_post30_query_windows"] = receipt["post30_query_window_count"]
    receipt["query_window_cap"] = core.WINDOW_CAP_PER_SESSION
    receipt["first30_direction_coverage"]["target_dir_all_finite"] = True
    with pytest.raises(core.CarrierValueMaskContractError, match="all-finite ledger drift"):
        core.validate_source_session_receipt(receipt, session="source-s")
    receipt["first30_direction_coverage"]["target_dir_all_finite"] = False
    receipt["production_t4_missing_direction_semantics"]["missing_trial_imputed_or_label_added"] = True
    with pytest.raises(core.CarrierValueMaskContractError, match="semantics drift"):
        core.validate_source_session_receipt(receipt, session="source-s")


def test_source_gate_binding_surface_is_additive_and_contains_runtime() -> None:
    bindings = core.current_bindings()
    assert "math" in bindings
    assert "source_probe" in bindings
    assert "scorer" in bindings
    assert "aggregator" in bindings
    assert "queue" in bindings
    assert "streaming_spint" in bindings
    assert "a2_scorer" in bindings
    assert all(len(row["sha256"]) == 64 for row in bindings.values())


def test_mask_lattice_and_domain_contrasts_are_complete() -> None:
    aggregate = _script("aggregate_carrier_value_mask.py")
    observed = aggregate.summarize_domain(
        {
            "low_t4": _row(0.60),
            "random_t4": _row(0.56),
            "high_t4": _row(0.50),
            "low_z4": _row(0.32),
            "random_z4": _row(0.31),
        },
        {"source_t4": _row(0.55), "source_z4": _row(0.30)},
    )
    assert observed["deltas_from_matched_parent"]["low_t4"] == pytest.approx(0.05)
    assert observed["deltas_from_matched_parent"]["low_z4"] == pytest.approx(0.02)
    assert observed["low_mask_carrier_interaction"] == pytest.approx(0.03)
    assert observed["low_t4_minus_random_t4"] == pytest.approx(0.04)
    assert observed["low_t4_minus_high_t4"] == pytest.approx(0.10)


def test_score_refuses_body_or_sidecar_collision(tmp_path: Path) -> None:
    scorer = _script("score_carrier_value_mask.py")
    body = tmp_path / "score.json"
    scorer._fresh(body)
    body.write_text("occupied", encoding="utf-8")
    with pytest.raises(scorer.CarrierValueMaskScoreError, match="output exists"):
        scorer._fresh(body)
    body.unlink()
    Path(str(body) + ".sha256").write_text("occupied", encoding="utf-8")
    with pytest.raises(scorer.CarrierValueMaskScoreError, match="sidecar exists"):
        scorer._fresh(body)


def test_only_canonical_source_gate_score_and_terminal_outputs_are_allowed(tmp_path: Path) -> None:
    probe = _script("probe_carrier_value_mask_source.py")
    scorer = _script("score_carrier_value_mask.py")
    aggregate = _script("aggregate_carrier_value_mask.py")
    with pytest.raises(core.CarrierValueMaskContractError, match="canonical path"):
        probe._require_canonical_output(tmp_path / "other_gate.json")
    with pytest.raises(scorer.CarrierValueMaskScoreError, match="canonical path"):
        scorer._require_canonical_output("low_t4", "within_subject", tmp_path / "other_score.json")
    with pytest.raises(aggregate.CarrierValueMaskAggregateError, match="canonical path"):
        aggregate._require_canonical_output(tmp_path / "other_terminal.json")


def test_queue_receipt_shell_smoke_is_data_and_gpu_free() -> None:
    queue = ROOT / "sua_exploration" / "scripts" / "watch_and_run_post_setkv_mask.sh"
    result = subprocess.run([str(queue), "--shell-smoke"], check=True, capture_output=True, text=True)
    assert "SHELL_SMOKE_PASS__NO_DATA_NO_GPU" in result.stdout


def test_engineering_receipt_requires_cached_bool_state_and_explicit_latency() -> None:
    aggregate = _script("aggregate_carrier_value_mask.py")
    row = {
        "per_session_mean_r2": {"s1": 0.4},
        "mask_runtime_shape_by_session": {
            "s1": {
                "unit_count": 8,
                "masked_unit_count": 2,
                "remaining_unit_count": 6,
                "persistent_boolean_mask_state_bytes": 8,
                "persistent_boolean_mask_element_bytes": 1,
            }
        },
        "persistent_mask_state_bytes_by_session": {"s1": 8},
        "persistent_mask_state_definition": "one cached torch.bool [N] mask per target session",
        "analytic_dense_mha_mac_delta_current_key_padding_mask_path": 0,
        "masked_tokens_are_not_physically_compacted_in_current_pytorch_path": True,
        "model_parameter_count": 10,
        "parameter_delta": 0,
        "measured_forward_wall_seconds": 1.0,
        "measured_mask_setup_wall_seconds": 0.1,
        "measured_forward_samples_across_epochs": 4,
        "measured_forward_seconds_per_sample": 0.25,
    }
    aggregate._validate_engineering_receipt(row)
    row["persistent_mask_state_bytes_by_session"] = {"s1": 7}
    with pytest.raises(aggregate.CarrierValueMaskAggregateError, match="state receipt drift"):
        aggregate._validate_engineering_receipt(row)
