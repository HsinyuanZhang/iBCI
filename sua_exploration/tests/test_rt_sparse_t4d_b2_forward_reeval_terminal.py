"""Contracts for the uniform, no-launch RT B2 forward-only terminal workflow."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/rt_sparse_t4d_b2_forward_reeval_terminal.py"


def _module():
    spec = importlib.util.spec_from_file_location("rt_b2_forward_terminal_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_fold0_imported_evidence_is_sha_bound_and_read_only():
    mod = _module()
    receipt = mod._check_fold0_import()
    assert receipt["status"] == "PASS_SHA_VERIFIED_NONDESTRUCTIVE_IMPORT"
    assert receipt["files"]["checkpoint"]["sha256"] == "078ac6dca35809b602ae09d86d9ba6d8af41f5e9615d79de6223fc5003cb2396"


def test_plan_contract_pins_legacy_config_working_directories_and_has_explicit_execute_mode():
    mod = _module()
    assert mod.EVALUATOR_CWD == ROOT / "streaming_calibration_exp"
    assert mod.REMOTE_EVALUATOR_CWD.endswith("rt_sparse_endpoint_stage2_5070_v1_20260810/streaming_calibration_exp")
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"execute"' in source
    assert "PASS_EXACT_ONCE_FORWARD_ONLY" in source
    assert "pre_execution_actual_query_digests" in source


def test_summary_requires_15fold_consumer_to_construct_data_elsewhere():
    mod = _module()
    result = mod._summary([0.1, -0.1, 0.0], ["a", "b", "c"])
    assert result["leave_largest_absolute_out_mean"] == -0.05
    assert result["removed"] == {"index": 0, "label": "a", "delta": 0.1}
    assert result["two_sided_exact_sign_test_p"] == 1.0


def test_launch_binding_rejects_tampered_source_command_exit_and_same_count_different_query(tmp_path):
    mod = _module()
    output = tmp_path / "new.json"
    output.write_text("{}", encoding="utf-8")
    digest = "a" * 64
    stage = (digest, "b" * 64, "c" * 64)
    manifest = {"surfaces": {name: {"sha256": digest} for name in ("outer_evaluator", "datamodule", "falcon_dataset")}}
    plan = {"command": ["python", "eval.py"], "cwd": "/fixed"}
    launch = {"status": "PASS_EXACT_ONCE_FORWARD_ONLY", "fold": 1, "command": plan["command"], "cwd": plan["cwd"], "output": {"path": str(output), "sha256": mod._sha(output)}, "pre_execution_actual_query_digests": list(stage), "stage2_actual_query_digests": list(stage), "execution": {"exit_code": 0, "source_sha256": {name: digest for name in manifest["surfaces"]}}}
    mod._validate_launch_binding(launch, plan=plan, manifest=manifest, fold=1, output=output, stage_actual=stage)
    bad_source = {**launch, "execution": {**launch["execution"], "source_sha256": {**launch["execution"]["source_sha256"], "datamodule": "0" * 64}}}
    with pytest.raises(mod.TerminalError, match="SHA differs"):
        mod._validate_launch_binding(bad_source, plan=plan, manifest=manifest, fold=1, output=output, stage_actual=stage)
    bad_command = {**launch, "command": ["tampered"]}
    with pytest.raises(mod.TerminalError, match="command"):
        mod._validate_launch_binding(bad_command, plan=plan, manifest=manifest, fold=1, output=output, stage_actual=stage)
    bad_exit = {**launch, "execution": {**launch["execution"], "exit_code": 1}}
    with pytest.raises(mod.TerminalError, match="exit code"):
        mod._validate_launch_binding(bad_exit, plan=plan, manifest=manifest, fold=1, output=output, stage_actual=stage)
    same_count_wrong_query = {**launch, "pre_execution_actual_query_digests": [digest, "d" * 64, "e" * 64]}
    with pytest.raises(mod.TerminalError, match="query identity"):
        mod._validate_launch_binding(same_count_wrong_query, plan=plan, manifest=manifest, fold=1, output=output, stage_actual=stage)


def test_remote_receipt_data_dir_is_overridden_by_sha_bound_local_allowlist():
    mod = _module()
    root = ROOT / "sua_exploration/data/dandi_000688/sub-C"
    local = next(root.glob("*.nwb"))
    digest = mod._sha(local)
    remote_outer = {"data_dir": "/definitely/nonexistent/remote/stage/data", "outer_target_path": "/remote/elsewhere/" + local.name}
    manifest = {"nwb_allowlist": [{"path": "/remote/allowlist/" + local.name, "sha256": digest}]}
    bound_root, bound_sha = mod._bound_local_nwb_sha(manifest, remote_outer)
    assert bound_root == root
    assert bound_sha == digest
