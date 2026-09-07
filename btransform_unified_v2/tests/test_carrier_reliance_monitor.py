from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "carrier_reliance_v1" / "monitor_carrier_reliance.py"
SPEC = importlib.util.spec_from_file_location("carrier_reliance_monitor", SCRIPT)
assert SPEC and SPEC.loader
monitor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = monitor
SPEC.loader.exec_module(monitor)


def test_h1_terminal_requires_matching_nonempty_payload_hash() -> None:
    launch = {"payload_sha256_before": "sealed"}
    good = {"schema": "h1_bt_eort_direct_carrier_reliance_v1", "status": "COMPLETE", "payload": {"sha256_before_after": "sealed"}}
    legacy_good = {"schema": "h1_bt_eort_direct_carrier_reliance_v1", "status": "COMPLETE", "payload": {"sha256_before": "sealed", "sha256_after": "sealed"}}
    bad = {"schema": good["schema"], "status": "COMPLETE", "payload": {"sha256_before_after": "wrong"}}
    assert monitor.completed_h1_report(good, launch)
    assert monitor.completed_h1_report(legacy_good, launch)
    assert not monitor.completed_h1_report(bad, launch)
    assert not monitor.completed_h1_report({"schema": good["schema"], "status": "COMPLETE"}, launch)


def test_m2_terminal_requires_schema_complete_and_launch_hash() -> None:
    launch = {"payload_sha256_before": "sealed"}
    good = {"schema": "m2_move_t4_concat_carrier_reliance_v2", "status": "COMPLETE", "payload_sha256_before_after": "sealed", "protocol": {"temporal_coordinate_contract": "end=start+49"}}
    assert monitor.completed_m2_report(good, launch)
    assert not monitor.completed_m2_report(good, launch, invalidated=True)
    assert not monitor.completed_m2_report({**good, "payload_sha256_before_after": "wrong"}, launch)
    assert not monitor.completed_m2_report({**good, "status": "RUNNING"}, launch)


def test_m2_state_uses_launcher_dest_not_newest_glob(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results"; results.mkdir()
    dest = results / "declared"; dest.mkdir()
    (dest / "launch_receipt.json").write_text(json.dumps({"payload_sha256_before": "sealed"}))
    (dest / "report.json").write_text(json.dumps({"schema": "m2_move_t4_concat_carrier_reliance_v2", "status": "COMPLETE", "payload_sha256_before_after": "sealed", "protocol": {"temporal_coordinate_contract": "end=start+49"}}))
    state = results / "launcher.json"; state.write_text(json.dumps({"m2_dest": str(dest)}))
    monkeypatch.setattr(monitor, "M2_LAUNCHER", state)
    monkeypatch.setattr(monitor, "RESULTS", results)
    monkeypatch.setattr(monitor, "process", lambda pid, expected: {"pid": pid, "alive": False, "expected_script": expected})
    value = monitor.m2_status()
    assert value["m2_dest"] == str(dest)
    assert value["semantic_complete"]
    (dest / "INVALID_ENDPOINT_ALIGNMENT.json").write_text("{}")
    assert not monitor.m2_status()["semantic_complete"]
