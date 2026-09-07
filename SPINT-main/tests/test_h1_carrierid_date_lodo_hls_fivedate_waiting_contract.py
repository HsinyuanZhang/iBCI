"""Pure-JSON/no-NWB contracts for the deferred five-date H1 H-LS expansion."""
from __future__ import annotations

import json
from pathlib import Path
import stat
import ast

import pytest

from scripts import h1_carrierid_date_lodo_hls_fivedate_contract as contract
from scripts import h1_carrierid_date_lodo_hls_fivedate_launch_receipt as launcher
from scripts import h1_carrierid_date_lodo_hls_fivedate_target_evaluator_preflight as evaluator
from scripts import h1_carrierid_date_lodo_hls_fivedate_terminal_checker as checker
from scripts import h1_carrierid_date_lodo_hls_fivedate_waiting_preflight as waiting


def _write_immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def _aggregate() -> dict[str, object]:
    return {
        "schema": contract.UPSTREAM_AGGREGATE_SCHEMA,
        "status": contract.UPSTREAM_AGGREGATE_STATUS,
        "required_outer_dates": list(contract.DATES),
        "all_five_date_receipts_present_and_validated": True,
        "route_prerequisite": {
            "status": "source/date screen complete", "automatic_route_selection": "FORBIDDEN",
        },
    }


def _source_preflight(date: str) -> dict[str, object]:
    return {
        "schema": contract.SOURCE_PREFLIGHT_SCHEMA, "status": contract.SOURCE_PREFLIGHT_STATUS,
        "outer_date": date, "source_binding_sha256": (date * 8)[:64],
        "source_controls": {
            "carrier_intervention": "temporal_velocity_label_rotation", "same_h_c_source_windows": True,
            "same_h_c_source_schedule": True, "same_h_c_normalizer": True, "seed": 42,
            "epochs": 50, "fixed_terminal_epoch_zero_based": 49,
        },
        "scope": {"target_recordings_opened": 0, "target_bytes_read": 0,
                  "cuda_constructed_or_launched": False, "trainer_constructed_or_launched": False},
    }


def _terminal(date: str, binding_sha: str) -> dict[str, object]:
    return {
        "schema": contract.SOURCE_TERMINAL_SCHEMA, "status": contract.SOURCE_TERMINAL_STATUS,
        "outer_date": date, "arm": "H-LS", "source_binding_sha256": binding_sha,
        "source_controls": {"carrier_intervention": "temporal_velocity_label_rotation", "seed": 42,
                            "epochs": 50, "fixed_terminal_epoch_zero_based": 49, "warm_start": False},
        "deployment_updates": {"target_optimizer_steps": 0, "target_backward_steps": 0,
                               "target_model_state_updated": False},
    }


def test_hls_waiting_plan_binds_the_actual_sealed_null_strength_audit_and_no_compute(tmp_path: Path):
    missing = tmp_path / "not_yet_complete_aggregate.json"
    output = tmp_path / "waiting.json"
    result = waiting.build(output=output, upstream_aggregate=missing)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == contract.WAITING_STATUS
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    assert body["current_gate"]["observed_at_plan_creation"]["reason"] == "MISSING"
    assert body["control_semantics"]["not_a_pure_label_deletion_control"] is True
    assert body["control_semantics"]["strong_null_audit"]["sha256"] == contract.NULL_AUDIT_SHA256
    assert body["current_scope"] == {
        "nwb_opened": 0, "target_recordings_opened": 0, "target_bytes_read": 0,
        "trainer_constructed_or_launched": False, "cuda_constructed_or_launched": False,
        "tmux_started": False, "checkpoint_created_or_loaded": False,
    }


def test_waiting_plan_never_selects_partial_date_and_launcher_waits_for_complete_aggregate(tmp_path: Path):
    waiting_path = tmp_path / "waiting.json"
    missing = tmp_path / "missing.json"
    waiting.build(output=waiting_path, upstream_aggregate=missing)
    with pytest.raises(contract.HlsFiveDateContractError, match="waits for H-S/H-C aggregate"):
        launcher.inspect_waiting_plan(waiting_plan=waiting_path, upstream_aggregate=missing)

    invalid = _aggregate()
    invalid["all_five_date_receipts_present_and_validated"] = False
    aggregate = _write_immutable(tmp_path / "invalid.json", invalid)
    with pytest.raises(contract.HlsFiveDateContractError, match="waits for H-S/H-C aggregate"):
        launcher.inspect_waiting_plan(waiting_plan=waiting_path, upstream_aggregate=aggregate)


def test_no_gpu_launcher_requires_all_five_hls_source_only_preflights(tmp_path: Path):
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate())
    waiting_path = tmp_path / "waiting.json"
    waiting.build(output=waiting_path, upstream_aggregate=aggregate)
    preflights = {
        date: _write_immutable(tmp_path / "preflights" / f"{date}.json", _source_preflight(date))
        for date in contract.DATES
    }
    output = tmp_path / "launch.json"
    result = launcher.prepare(waiting_plan=waiting_path, upstream_aggregate=aggregate,
                              source_preflights=preflights, explicit_route="H1-HLS-FIVEDATE", output=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == contract.LAUNCH_STATUS
    assert body["not_a_gpu_launcher"] is True and body["launch_authorized"] is False
    assert tuple(body["fixed_grid"]) == contract.DATES
    assert stat.S_IMODE(output.stat().st_mode) == 0o444

    with pytest.raises(launcher.HlsFiveDateLaunchError, match="exactly all five"):
        launcher.prepare(waiting_plan=waiting_path, upstream_aggregate=aggregate,
                         source_preflights=dict(list(preflights.items())[:-1]),
                         explicit_route="H1-HLS-FIVEDATE", output=tmp_path / "launch_bad.json")


def test_hls_source_terminal_checker_and_evaluator_preflight_stay_target_closed(tmp_path: Path):
    aggregate = _write_immutable(tmp_path / "aggregate.json", _aggregate())
    waiting_path = tmp_path / "waiting.json"
    waiting.build(output=waiting_path, upstream_aggregate=aggregate)
    preflights = {
        date: _write_immutable(tmp_path / "preflights" / f"{date}.json", _source_preflight(date))
        for date in contract.DATES
    }
    launch_path = tmp_path / "launch.json"
    launcher.prepare(waiting_plan=waiting_path, upstream_aggregate=aggregate,
                     source_preflights=preflights, explicit_route="H1-HLS-FIVEDATE", output=launch_path)
    terminals = {
        date: _write_immutable(tmp_path / "terminals" / f"{date}.json",
                               _terminal(date, _source_preflight(date)["source_binding_sha256"]))
        for date in contract.DATES
    }
    checked = tmp_path / "checked.json"
    result = checker.check(launch_receipt=launch_path, upstream_aggregate=aggregate,
                           terminal_receipts=terminals, output=checked)
    assert result["status"] == checker.CHECKER_STATUS
    checked_body = json.loads(checked.read_text(encoding="utf-8"))
    assert checked_body["scope"]["nwb_opened"] == 0
    assert checked_body["scope"]["target_evaluator"] == "NOT_RUN"
    evaluator_output = tmp_path / "evaluator_ready.json"
    ready = evaluator.prepare(source_terminal_aggregate=checked, output=evaluator_output)
    ready_body = json.loads(evaluator_output.read_text(encoding="utf-8"))
    assert ready["status"] == contract.EVALUATOR_PREFLIGHT_STATUS
    assert ready_body["not_a_target_evaluator"] is True
    assert ready_body["scope"]["target_optimizer_steps"] == 0

    terminals[contract.DATES[0]].chmod(0o644)
    broken = _terminal(contract.DATES[0], _source_preflight(contract.DATES[0])["source_binding_sha256"])
    broken["deployment_updates"]["target_backward_steps"] = 1  # type: ignore[index]
    _write_immutable(terminals[contract.DATES[0]], broken)
    with pytest.raises(checker.HlsFiveDateTerminalError, match="target-update"):
        checker.check(launch_receipt=launch_path, upstream_aggregate=aggregate,
                      terminal_receipts=terminals, output=tmp_path / "checked_bad.json")


def test_hls_prep_scripts_are_json_only_and_cannot_construct_runtime_paths():
    files = [
        Path("scripts/h1_carrierid_date_lodo_hls_fivedate_contract.py"),
        Path("scripts/h1_carrierid_date_lodo_hls_fivedate_waiting_preflight.py"),
        Path("scripts/h1_carrierid_date_lodo_hls_fivedate_launch_receipt.py"),
        Path("scripts/h1_carrierid_date_lodo_hls_fivedate_terminal_checker.py"),
        Path("scripts/h1_carrierid_date_lodo_hls_fivedate_target_evaluator_preflight.py"),
    ]
    imported: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
    for forbidden in ("torch", "lightning", "hydra", "pynwb", "h5py", "subprocess"):
        assert forbidden not in imported
    merged = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
    assert "not_a_gpu_launcher" in merged and "not_a_target_evaluator" in merged
