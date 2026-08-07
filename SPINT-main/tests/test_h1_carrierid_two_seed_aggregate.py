"""Synthetic/no-NWB contracts for the H1 CarrierID strict two-seed aggregator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

import pytest

from scripts import h1_carrierid_two_seed_aggregate as aggregate


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(seed: int, *, status: str | None = None, schedule: str | None = None) -> dict[str, object]:
    sessions = ["ses-A", "ses-B"]
    session_samples = {"ses-A": 7, "ses-B": 3}
    target = {
        "sessions": sessions, "files": {"ses-A": "a" * 64, "ses-B": "b" * 64},
        "strict_query_window_indices_sha256": "c" * 64,
        "support_and_carrier_hashes": {
            "ses-A": {"support_sha256": "7" * 64, "carrier_sha256": {"full": "8" * 64}},
            "ses-B": {"support_sha256": "9" * 64, "carrier_sha256": {"full": "a" * 64}},
        },
        "all_query_histories_start_at_or_after_fifth_trial": True,
        "pooled_recordings_before_variance_weighted_r2": True, "remainder_preserved": True,
    }
    def metric(score: float) -> dict[str, object]:
        return {
            "pooled_r2": score, "r2_accumulator_dtype": "float64", "samples": 10,
            "session_samples": session_samples,
            "per_session": {"ses-A": {"samples": 7, "r2": score}, "ses-B": {"samples": 3, "r2": score - 0.1}},
            "state_immutable": True, "state_sha256_before": "d" * 64, "state_sha256_after": "d" * 64,
            "query_window_indices_sha256": "c" * 64,
        }
    hs_meta = {"arm": "base", "config_sha256": "1" * 64, "source_manifest_sha256": "6" * 64, "checkpoint_epoch_zero_based": 49,
               "epochs_completed": 50, "selected_by": "fixed_terminal_epoch_no_selection",
               "base_residual_literal_zero": True, "residual_trainable": False}
    hc_meta = {"arm": "full", "config_sha256": "2" * 64, "source_manifest_sha256": "6" * 64, "checkpoint_epoch_zero_based": 49,
               "epochs_completed": 50, "selected_by": "fixed_terminal_epoch_no_selection", "carrier_mode": "full",
               "deployment_target_optimizer_steps": 0, "deployment_target_backward_steps": 0,
               "carrierid_parameters": 10, "spint_identity_parameters": 100, "whole_model_parameters": 1000,
               "spint_whole_model_parameters": 2000}
    hc0_meta = {**hc_meta, "arm": "zero", "config_sha256": "3" * 64,
                "carrier_mode": "literal_zero_at_model_boundary"}
    checkpoints = {
        "h_s_matched_spint": {"sha256": "e" * 64, "config_sha256": "1" * 64, "metadata": hs_meta},
        "h_c_full": {"sha256": "f" * 64, "config_sha256": "2" * 64, "metadata": hc_meta},
        "h_c0_separate_literal_zero": {"sha256": "0" * 64, "config_sha256": "3" * 64, "metadata": hc0_meta},
    }
    common = {
        "schema": aggregate.SEED42_SCHEMA if seed == 42 else aggregate.SEED43_SCHEMA,
        "status": status or (aggregate.SEED42_STATUS if seed == 42 else "PASS_H1_CARRIERID_H32_SEED43_TERMINAL_EVALUATED"),
        "fold_date": "19250101", "checkpoint_binding_completed_before_target_open": True,
        "metric_contract": {"prediction_dtype": "float32", "r2_sse_tss_accumulator_dtype": "float64"},
        "data_scope": {"minival_opened": False, "heldout_opened": False, "formal_heldout_opened": False, "evalai_opened": False},
        "target": target, "checkpoints": checkpoints,
        "metrics": {"h_s": metric(0.4), "h_c_interventions": {"full": metric(0.5)}, "h_c0": metric(0.45)},
        "parameter_accounting": {
            "static_session_identity_encoder_parameters": {"h_s_spint": 100, "h_c_h32": 10},
            "whole_model_parameters": {"h_s_spint": 2000, "h_c_h32": 1000},
            "target_session_updated_parameters": {"h_s": 0, "h_c": 0, "h_c0": 0},
            "target_session_optimizer_steps": {"h_s": 0, "h_c": 0, "h_c0": 0},
            "target_session_backward_steps": {"h_s": 0, "h_c": 0, "h_c0": 0},
        },
    }
    if seed == 42:
        common["source_manifest_sha256"] = "6" * 64
        common["source_manifest"] = {"calibration_schedule_sha256": schedule or "4" * 64}
    else:
        common["seed"] = 43
        common["no_target_preflight"] = {
            "path": "/result-only/no-target-preflight.json", "sha256": "b" * 64,
            "status": aggregate.SEED43_NO_TARGET_PREFLIGHT_STATUS,
        }
        common["source"] = {
            "manifest_sha256": "6" * 64, "calibration_schedule_sha256": schedule or "5" * 64,
            "seed42_schedule_sha256": "4" * 64, "schedule_is_new_for_seed43": True,
        }
    return common


def _roots(tmp_path: Path, *, include43: bool = True) -> tuple[Path, Path]:
    seed42, seed43 = tmp_path / "seed42", tmp_path / "seed43"
    seed43.mkdir(parents=True, exist_ok=True)
    _write(seed42 / "float64.json", _receipt(42))
    # Same legacy schema but without the Float64 metric contract: discovery
    # must reject it as an ambiguous predecessor, not silently choose it.
    legacy = _receipt(42)
    legacy["metric_contract"] = {}
    _write(seed42 / "legacy.json", legacy)
    if include43:
        _write(seed43 / "future_seed43_terminal.json", _receipt(43))
    return seed42, seed43


def test_auto_discovery_selects_unique_seed42_float64_result_and_seed43_schema(tmp_path):
    root42, root43 = _roots(tmp_path)
    found42 = aggregate.discover_terminal_result(seed=42, root=root42)
    found43 = aggregate.discover_terminal_result(seed=43, root=root43)
    assert found42.path.name == "float64.json" and found43.path.name == "future_seed43_terminal.json"
    assert found42.sha256 == _sha(found42.path)


def test_discovery_ignores_nonobject_json_and_fails_on_two_valid_candidates(tmp_path):
    root42, root43 = _roots(tmp_path)
    _write(root42 / "unrelated_array.json", ["not", "a", "result"])
    assert aggregate.discover_terminal_result(seed=42, root=root42).path.name == "float64.json"
    _write(root43 / "second_valid_terminal.json", _receipt(43))
    with pytest.raises(aggregate.TwoSeedAggregateError, match="found 2"):
        aggregate.discover_terminal_result(seed=43, root=root43)


def test_missing_seed43_fails_closed_before_formal_output(tmp_path):
    root42, root43 = _roots(tmp_path, include43=False)
    output = tmp_path / "formal.json"
    with pytest.raises(aggregate.TwoSeedAggregateError, match="seed43 needs exactly one"):
        aggregate.aggregate(seed42_root=root42, seed43_root=root43, output=output)
    assert not output.exists()


def test_terminal_validation_rejects_target_update_or_parameter_drift(tmp_path):
    root42, root43 = _roots(tmp_path)
    bad = _receipt(43)
    bad["parameter_accounting"]["target_session_backward_steps"]["h_c"] = 1
    _write(root43 / "future_seed43_terminal.json", bad)
    found = aggregate.discover_terminal_result(seed=43, root=root43)
    with pytest.raises(aggregate.TwoSeedAggregateError, match="nonzero target update"):
        aggregate.validate_terminal_result(found)
    bad = _receipt(43)
    bad["parameter_accounting"]["static_session_identity_encoder_parameters"]["h_c_h32"] = 11
    _write(root43 / "future_seed43_terminal.json", bad)
    found = aggregate.discover_terminal_result(seed=43, root=root43)
    with pytest.raises(aggregate.TwoSeedAggregateError, match="parameter accounting drift"):
        aggregate.validate_terminal_result(found)
    bad = _receipt(43)
    bad["no_target_preflight"]["status"] = "WRONG"
    _write(root43 / "future_seed43_terminal.json", bad)
    found = aggregate.discover_terminal_result(seed=43, root=root43)
    with pytest.raises(aggregate.TwoSeedAggregateError, match="no-target preflight status"):
        aggregate.validate_terminal_result(found)


def test_two_seed_statistics_are_dispersion_only_not_a_confidence_interval():
    row = aggregate._sample_dispersion([0.1, 0.3])
    assert row["mean"] == pytest.approx(0.2)
    assert row["sample_standard_deviation_ddof1"] == pytest.approx(0.1414213562)
    assert row["range"] == pytest.approx(0.2) and row["positive_seed_count"] == 2
    assert "not a confidence interval" in row["interpretation"]


def test_cross_seed_same_schedule_or_support_carrier_drift_fails_before_output(tmp_path):
    root42, root43 = _roots(tmp_path)
    _write(root43 / "future_seed43_terminal.json", _receipt(43, schedule="4" * 64))
    output = tmp_path / "same_schedule.json"
    with pytest.raises(aggregate.TwoSeedAggregateError, match="not independent"):
        aggregate.aggregate(seed42_root=root42, seed43_root=root43, output=output)
    assert not output.exists()
    bad = _receipt(43)
    bad["target"]["support_and_carrier_hashes"]["ses-A"]["carrier_sha256"]["full"] = "f" * 64
    _write(root43 / "future_seed43_terminal.json", bad)
    with pytest.raises(aggregate.TwoSeedAggregateError, match="support_and_carrier_hashes"):
        aggregate.aggregate(seed42_root=root42, seed43_root=root43, output=output)
    assert not output.exists()


def test_formal_output_is_o_excl_0444_and_seed43_stop_has_nonpass_status(tmp_path):
    root42, root43 = _roots(tmp_path)
    output = tmp_path / "pass.json"
    result = aggregate.aggregate(seed42_root=root42, seed43_root=root43, output=output)
    assert result["status"] == aggregate.AGGREGATE_STATUS_PASS
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(aggregate.TwoSeedAggregateError, match="refusing to overwrite"):
        aggregate.aggregate(seed42_root=root42, seed43_root=root43, output=output)
    stopped = _receipt(43, status="STOP_H1_CARRIERID_H32_SEED43_TERMINAL_GATE_FAILED")
    _write(root43 / "future_seed43_terminal.json", stopped)
    stop_output = tmp_path / "stop.json"
    stopped_result = aggregate.aggregate(seed42_root=root42, seed43_root=root43, output=stop_output)
    assert stopped_result["status"] == aggregate.AGGREGATE_STATUS_SEED43_GATE_FAILED
    payload = json.loads(stop_output.read_text(encoding="utf-8"))
    assert payload["all_seed_terminal_gates_pass"] is False and payload["status"] == aggregate.AGGREGATE_STATUS_SEED43_GATE_FAILED


def test_aggregator_source_is_result_json_only_without_nwb_checkpoint_or_gpu_imports():
    source = Path(aggregate.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source and "src.data" not in source
    assert "load_target_records" not in source and "torch.load" not in source
    assert '"checkpoint_opened_by_aggregator": False' in source
