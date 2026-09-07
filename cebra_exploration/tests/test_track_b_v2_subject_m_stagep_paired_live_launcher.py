"""No-target tests for the additive paired live launcher."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_subject_m_stagep_paired_live_launcher as launcher  # noqa: E402
import track_b_v2_subject_m_stagep_paired_real_producer as producer  # noqa: E402
import track_b_v2_subject_m_stagep_runtime as runtime  # noqa: E402


CLI = ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_paired_live_launcher.py"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _cap(view: str) -> producer.ViewExecutionCapability:
    return producer.ViewExecutionCapability(
        view=view, cell=runtime.StagePCell.from_view(view).as_dict(),
        admission_sha256=_sha(f"admission-{view}"),
        official_preflight_body_sha256=_sha(f"official-{view}"),
        implementation_closure_sha256=_sha("sealed"),
        source_authority_set_sha256=_sha(f"source-{view}"),
        v9_preflight_sha256=producer.V9_PREFLIGHT_SHA256)


def _fake_prerequisites(view: str = "sua") -> launcher.ExecutionPrerequisites:
    cap = _cap(view)
    closure = launcher.implementation_closure()
    preflight_payload = {
        "launcher_preflight_payload_sha256": _sha("preflight-payload"),
        "launcher_implementation_closure": closure,
        "cost_bound_cuda_identity": {},
        "target_authority_bindings_by_view": {},
    }
    return launcher.ExecutionPrerequisites(
        admissions={}, capabilities={view: cap},
        producer_addendum={"body_sha256": _sha("addendum")},
        launcher_preflight={"body_sha256": _sha("preflight"), "payload": preflight_payload},
        cost_identity={}, target_authority_bindings={})


def _valid_semantic_chain(view: str = "sua") -> tuple:
    cap = _cap(view)
    start_payload = {"start_payload_sha256": _sha("start")}
    source_payload = {
        "status": "STRICT27_SOURCE_MATERIALIZED_AND_EXACT_AUTHORITY_MATCH",
        "cell": cap.cell, "admission_sha256": cap.admission_sha256,
        "source_authority_set_sha256": cap.source_authority_set_sha256,
        "source_session_count": 27, "ordered_source_session_ids": [f"s{i}" for i in range(27)],
        "source_authority_exact_rebuild_match": True,
        "historical_selector_plan_executed_or_selected": False,
        "target_opened": False, "query_opened": False}
    source_payload["source_payload_sha256"] = launcher._sha_json(source_payload)
    target_payload = {
        "status": "TARGET_M50_AND_SPARSE_V9_QUERY_MATERIALIZED",
        "cell": cap.cell, "start_payload_sha256": start_payload["start_payload_sha256"],
        "query": {"semantics": "SPARSE_EXACT_V9_ENDPOINT_GATHER_FROM_CONTINUOUS_SUFFIX_TRANSFORM",
                  "query_row_count": producer.V9_QUERY_COUNT,
                  "valid_starts_int64_sha256": producer.V9_VALID_STARTS_SHA256,
                  "ordered_target_behavior_float32_sha256": producer.V9_TARGET_SHA256,
                  "sealed_v9_preflight_sha256": producer.V9_PREFLIGHT_SHA256,
                  "ordered_prediction_endpoint_int64_sha256": _sha("endpoints"),
                  "ordered_offset10_RF_int64_sha256": _sha("rf"),
                  "not_contiguous_5_to_minus5_crop": True,
                  "each_RF_is_range_endpoint_minus5_to_endpoint_plus5_exclusive": True,
                  "every_RF_wholly_inside_held_suffix": True,
                  "every_RF_support_disjoint": True,
                  "query_neural_or_auxiliary_enters_any_fit": False},
        "private_snapshot": {"parser_consumed_continuously_held_fd": True,
                             "pathname_reopen_permitted": False},
        "source_only_behavior_normalizer_used": True,
        "target_query_neural_or_auxiliary_entered_fit": False}
    target_payload["target_payload_sha256"] = launcher._sha_json(target_payload)
    source_loaded = {"payload": source_payload, "body_sha256": _sha("source-body")}
    target_loaded = {"payload": target_payload, "body_sha256": _sha("target-body")}
    encoder_payload = {
        "status": "JOINT_ENCODER_PERSISTED_AND_RELOADED_EXACT",
        "cell": cap.cell, "start_payload_sha256": start_payload["start_payload_sha256"],
        "source_payload_sha256": source_payload["source_payload_sha256"],
        "source_receipt_body_sha256": source_loaded["body_sha256"],
        "target_payload_sha256": target_payload["target_payload_sha256"],
        "fit_proof": {"fit_stream_count": 28, "fit_call_count": 1,
                      "source_session_count": 27, "target_support_session_count": 1,
                      "target_query_enters_fit": False, "model_contract": producer.MODEL_CONTRACT},
        "same_encoder_services_all_six_readouts": True, "cross_view_encoder_reuse": False}
    encoder_payload["encoder_payload_sha256"] = launcher._sha_json(encoder_payload)
    scores = {}
    for route in producer.ROUTES:
        for decoder in producer.DECODERS:
            payload = {
                "status": "ROUTE_DECODER_SCORE_COMPLETE",
                "cell": cap.cell, "readout_route": route, "decoder": decoder,
                "target_payload_sha256": target_payload["target_payload_sha256"],
                "encoder_payload_sha256": encoder_payload["encoder_payload_sha256"],
                "target_float32_sha256": producer.V9_TARGET_SHA256,
                "query_row_count": producer.V9_QUERY_COUNT,
                "readout_proof": {"query_enters_fit": False,
                                  "sparse_query_semantics": "exact_V9_endpoint_gather__not_contiguous_crop",
                                  "query_block_receipt_sha256": launcher._sha_json(target_payload["query"])},
                "metric": {"implementation": "torchmetrics.regression.R2Score", "version": "1.5.1",
                           "dtype": "float32", "device": "cpu", "multioutput": "variance_weighted",
                           "update_scope": "one_complete_ordered_external_target_session_query_then_compute_once",
                           "update_call_count": 1, "compute_call_count": 1,
                           "target_float32_bytes_sha256": producer.V9_TARGET_SHA256,
                           "custom_numpy_float64_pooled_r2_used": False}}
            payload["score_payload_sha256"] = launcher._sha_json(payload)
            scores[f"{route}__{decoder}"] = {"payload": payload, "body_sha256": _sha(route + decoder)}
    return (cap, {"payload": start_payload}, source_loaded, target_loaded,
            {"payload": encoder_payload, "body_sha256": _sha("encoder-body")}, scores)


def _temp_topologies(tmp_path: Path) -> dict[str, dict]:
    result = {}
    for view in launcher.PAIR_ORDER:
        root = tmp_path / "cells" / view
        scores = {f"{route}__{decoder}": str(root / "scores" / f"{route}__{decoder}.json")
                  for route in producer.ROUTES for decoder in producer.DECODERS}
        result[view] = {
            "cell_root": str(root), "start": str(root / "start.json"),
            "source": str(root / "source_materialization.json"),
            "target": str(root / "target_materialization.json"),
            "private_snapshot": str(root / "private_snapshot/held_target.nwb"),
            "encoder": str(root / "encoder.json"), "checkpoint": str(root / "model.pt"),
            "embeddings": str(root / "embeddings.npz"), "scores": scores,
            "completion": str(root / "completion.json"), "terminal": str(root / "terminal.json"),
            "caller_path_override_permitted": False,
        }
    return result


def _patch_topology(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    topologies = _temp_topologies(tmp_path)
    monkeypatch.setattr(producer, "_topology", lambda view: copy.deepcopy(topologies[view]))
    monkeypatch.setattr(launcher, "PAIRED_COMPLETION", tmp_path / "aggregate/paired.json")
    return topologies


def test_default_cli_is_no_write_and_fresh_import_avoids_ml_runtime(tmp_path: Path) -> None:
    probe = f"""
import json,sys
before={{n:(n in sys.modules) for n in ('torch','cebra','pynwb')}}
import track_b_v2_subject_m_stagep_paired_live_launcher as x
p=x.build_dry_plan()
after={{n:(n in sys.modules) for n in before}}
assert before == after, (before,after)
assert p['target_opened'] is p['torch_imported'] is p['gpu_queried'] is p['write_performed'] is False
print(json.dumps({{'status':p['status'],'count':p['topology']['prospective_pair_count_per_view']}}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    done = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, env=env,
                          text=True, capture_output=True, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"status": launcher.STATUS_DRY, "count": 15}
    assert not launcher.LAUNCHER_PREFLIGHT.exists()


def test_cli_single_flags_and_missing_root_env_refuse_before_authorities() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env.pop(launcher.ROOT_ENV, None)
    for flag in ("--execute", "--i-have-authorization"):
        done = subprocess.run([sys.executable, str(CLI), flag], cwd=ROOT, env=env,
                              text=True, capture_output=True, check=False)
        assert done.returncode != 0 and "requires both" in done.stderr
    both = subprocess.run([sys.executable, str(CLI), "--execute", "--i-have-authorization"],
                          cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert both.returncode != 0 and f"{launcher.ROOT_ENV}=1" in both.stderr


def test_dual_flags_with_root_env_still_fail_closed_on_missing_immutable_preflights() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    env[launcher.ROOT_ENV] = "1"
    done = subprocess.run([sys.executable, str(CLI), "--execute", "--i-have-authorization"],
                          cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert done.returncode != 0
    assert not launcher.LAUNCHER_PREFLIGHT.exists()
    assert not launcher.PAIRED_COMPLETION.exists()


def test_internal_child_cannot_be_called_without_parent_token() -> None:
    env = dict(os.environ)
    env.update(launcher._child_env_contract("sua"))
    env.pop(launcher.INTERNAL_TOKEN_ENV, None)
    done = subprocess.run([sys.executable, str(CLI), "--execute", "--i-have-authorization",
                           "--internal-child", "--view", "sua"], cwd=ROOT, env=env,
                          text=True, capture_output=True, check=False)
    assert done.returncode != 0 and "token" in done.stderr


def test_arbitrary_internal_token_cannot_consume_parent_launch_contract(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_topology(tmp_path, monkeypatch)
    prerequisites = _fake_prerequisites()
    payload = launcher._launch_contract_payload(
        view="sua", token="parent-secret", prerequisites=prerequisites,
        parent_pid=os.getppid())
    launcher._publish(launcher._launch_contract_path("sua"), payload)
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="token/parent"):
        launcher._load_child_launch_contract(view="sua", token="caller-chosen")


def test_direct_pmua_child_is_blocked_by_immutable_sua_predecessor_before_gpu_or_target(
        monkeypatch: pytest.MonkeyPatch) -> None:
    prerequisites = _fake_prerequisites("pseudo_mua")
    contract = {"payload": launcher._launch_contract_payload(
                    view="pseudo_mua", token="secret", prerequisites=prerequisites,
                    parent_pid=os.getppid()),
                "path": str(launcher._launch_contract_path("pseudo_mua")),
                "body_sha256": _sha("contract")}
    monkeypatch.setattr(launcher, "_load_child_launch_contract", lambda **_kwargs: contract)
    monkeypatch.setattr(launcher, "_verify_child_environment", lambda **_kwargs: None)
    monkeypatch.setattr(launcher, "load_launcher_preflight", lambda **_kwargs: {
        "body_sha256": contract["payload"]["launcher_preflight_body_sha256"]})
    monkeypatch.setattr(launcher, "validate_pretarget_prerequisites",
                        lambda **_kwargs: touched.append("target-authority-resolution"))
    monkeypatch.setattr(launcher, "_launch_contract_payload", lambda **_kwargs: contract["payload"])
    monkeypatch.setattr(launcher, "_require_sua_terminal_before_pmua",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            launcher.StagePPairedLiveLauncherError("missing SUA predecessor")))
    touched = []
    monkeypatch.setattr(launcher, "_verify_gpu_against_cost", lambda *_args: touched.append("gpu"))
    monkeypatch.setattr(producer, "materialize_target_from_private_snapshot",
                        lambda *_args: touched.append("target"))
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="SUA predecessor"):
        launcher.execute_internal_child(view="pseudo_mua", token="secret")
    assert touched == []


def test_topology_has_exact_fifteen_per_view_including_launch_contract_and_source() -> None:
    topology = launcher.canonical_topology()
    assert topology["prospective_pair_count_per_view"] == 15
    assert topology["total_prospective_pair_count"] == 31
    assert topology["source_receipt_is_mandatory_and_independently_published"] is True
    for view in launcher.PAIR_ORDER:
        assert Path(topology["per_view"][view]["source"]) in launcher._prospective_paths(view)
        assert launcher._launch_contract_path(view).resolve() in launcher._prospective_paths(view)


def test_freshness_gate_precedes_a2_target_path_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    order = []
    monkeypatch.setattr(launcher, "_fresh_admissions", lambda: {})
    monkeypatch.setattr(producer, "build_no_target_review_plan", lambda: {})
    monkeypatch.setattr(producer, "bind_execution_capabilities", lambda **_kwargs: {})
    monkeypatch.setattr(producer, "load_execution_addendum", lambda **_kwargs: {"body_sha256": _sha("a")})
    monkeypatch.setattr(launcher, "load_launcher_preflight", lambda **_kwargs: {
        "payload": {"producer_addendum_body_sha256": _sha("a"),
                    "cost_bound_cuda_identity": {"cost": True},
                    "target_authority_bindings_by_view": {}}})
    monkeypatch.setattr(launcher, "_cost_identity", lambda: {"cost": True})
    def fail_freshness(**_kwargs):
        order.append("freshness")
        raise launcher.StagePPairedLiveLauncherError("poisoned source output")
    monkeypatch.setattr(launcher, "require_all_outputs_fresh", fail_freshness)
    monkeypatch.setattr(launcher.target_materializer,
                        "build_development_target_materializer_dry_plan",
                        lambda **_kwargs: order.append("target_path_resolution"))
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="poisoned"):
        launcher.validate_pretarget_prerequisites()
    assert order == ["freshness"]


def test_a2_target_authority_must_exact_match_preflight_after_freshness(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _cap("sua")
    capabilities = {view: cap for view in launcher.PAIR_ORDER}
    monkeypatch.setattr(launcher, "_fresh_admissions", lambda: {})
    monkeypatch.setattr(producer, "build_no_target_review_plan", lambda: {})
    monkeypatch.setattr(producer, "bind_execution_capabilities", lambda **_kwargs: capabilities)
    monkeypatch.setattr(producer, "load_execution_addendum",
                        lambda **_kwargs: {"body_sha256": _sha("addendum")})
    monkeypatch.setattr(launcher, "load_launcher_preflight", lambda **_kwargs: {
        "payload": {"producer_addendum_body_sha256": _sha("addendum"),
                    "cost_bound_cuda_identity": {},
                    "target_authority_bindings_by_view": {"frozen": "wrong"}}})
    monkeypatch.setattr(launcher, "_cost_identity", lambda: {})
    order = []
    monkeypatch.setattr(launcher, "require_all_outputs_fresh",
                        lambda **_kwargs: order.append("freshness"))
    monkeypatch.setattr(launcher, "_target_authority_bindings",
                        lambda **_kwargs: order.append("target-authority") or {"live": "binding"})
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="A2 target authority"):
        launcher.validate_pretarget_prerequisites()
    assert order == ["freshness", "target-authority"]


@pytest.mark.parametrize("poison", ("source_body", "source_sidecar", "parent_symlink"))
def test_freshness_rejects_source_pair_and_parent_alias(
        poison: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topologies = _patch_topology(tmp_path, monkeypatch)
    if poison == "source_body":
        path = Path(topologies["sua"]["source"]); path.parent.mkdir(parents=True); path.write_text("old")
    elif poison == "source_sidecar":
        path = Path(f"{topologies['sua']['source']}.sha256"); path.parent.mkdir(parents=True); path.write_text("old")
    else:
        real = tmp_path / "real"; real.mkdir()
        alias = Path(topologies["sua"]["cell_root"]); alias.parent.mkdir(parents=True)
        alias.symlink_to(real, target_is_directory=True)
    if poison == "parent_symlink":
        with pytest.raises(launcher.StagePPairedLiveLauncherError, match="real directory"):
            launcher.require_all_outputs_fresh()
    else:
        with pytest.raises(launcher.StagePPairedLiveLauncherError, match="not fresh"):
            launcher.require_all_outputs_fresh()


def test_child_environment_and_gpu_cost_tamper_fail_closed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "CONDA_PYTHON", Path(sys.executable))
    monkeypatch.setattr(launcher, "REPO_ROOT", Path.cwd())
    token = "token"
    for key, value in launcher._child_env_contract("sua").items():
        monkeypatch.setenv(key, value)
    for key in launcher.FORBIDDEN_CHILD_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(launcher.INTERNAL_TOKEN_ENV, token)
    contract = {"payload": {
        "actual_argv_required": launcher._child_argv("sua"),
        "actual_environment_required": launcher._child_env_contract("sua"),
        "actual_cwd_required": str(Path.cwd()), "actual_python_required": sys.executable}}
    monkeypatch.setattr(launcher.sys, "argv", launcher._child_argv("sua")[1:])
    launcher._verify_child_environment(view="sua", token=token, launch_contract=contract)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="environment drift"):
        launcher._verify_child_environment(view="sua", token=token, launch_contract=contract)
    monkeypatch.setattr(producer, "verify_isolated_cuda_identity", lambda: {
        "physical_uuid": "GPU-wrong", "physical_pci_bus_id": "x", "driver_version": "x",
        "name": "x", "torch_version": "x", "torch_cuda_runtime": "x",
        "torch_module_path": str(tmp_path / "torch.py")})
    monkeypatch.setattr(producer, "_read_same_fd", lambda *_args, **_kwargs:
                        SimpleNamespace(sha256=_sha("wrong-torch")))
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="differs"):
        launcher._verify_gpu_against_cost({
            "physical_uuid": "GPU-right", "physical_pci_bus_id": "bdf", "driver_version": "driver",
            "device_name": "gpu", "torch_version": "torch", "torch_cuda_runtime": "cuda",
            "torch_path": str(tmp_path / "expected.py"), "torch_sha256": _sha("torch"),
            "cost_body_sha256": _sha("cost")})


def test_sua_failure_blocks_pmua_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launcher.ROOT_ENV, "1")
    prerequisites = _fake_prerequisites()
    closure = launcher.implementation_closure()
    prerequisites.launcher_preflight["payload"]["launcher_implementation_closure"] = closure
    monkeypatch.setattr(launcher, "validate_pretarget_prerequisites", lambda: prerequisites)
    calls = []
    def fail_sua(*, view, token, runner, prerequisites):
        calls.append(view)
        raise launcher.StagePPairedLiveLauncherError("sua failed")
    monkeypatch.setattr(launcher, "_run_one_child", fail_sua)
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="sua failed"):
        launcher._orchestrate_authorized_pair(runner=lambda *_args: (0, b"", b""))
    assert calls == ["sua"]


def test_unverifiable_sua_terminal_blocks_pmua_even_after_zero_child_exit(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launcher.ROOT_ENV, "1")
    prerequisites = _fake_prerequisites()
    closure = launcher.implementation_closure()
    prerequisites.launcher_preflight["payload"]["launcher_implementation_closure"] = closure
    monkeypatch.setattr(launcher, "validate_pretarget_prerequisites", lambda: prerequisites)
    calls = []
    def returned_terminal(*, view, token, runner, prerequisites):
        calls.append(view)
        return {"schema": producer.SCHEMA_TERMINAL,
                "status": "TERMINAL_SUCCESS__DEVELOPMENT_PILOT_CELL",
                "cell": runtime.StagePCell.from_view(view).as_dict(),
                "terminal_payload_sha256": _sha("terminal")}
    monkeypatch.setattr(launcher, "_run_one_child", returned_terminal)
    monkeypatch.setattr(launcher, "_load_output", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        launcher.StagePPairedLiveLauncherError("terminal sidecar poison")))
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="sidecar poison"):
        launcher._orchestrate_authorized_pair(runner=lambda *_args: (0, b"", b""))
    assert calls == ["sua"]


def test_nonzero_subprocess_publishes_exact_failure_completion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topologies = _patch_topology(tmp_path, monkeypatch)
    argv_seen = []
    def runner(view, argv, env):
        argv_seen.extend(argv)
        snapshot = Path(topologies["sua"]["private_snapshot"])
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(b"synthetic-held-snapshot")
        return 17, b"stdout", b"stderr"
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="exit code 17"):
        launcher._run_one_child(view="sua", token="token", runner=runner,
                                prerequisites=_fake_prerequisites())
    loaded = launcher._read_pair(Path(topologies["sua"]["completion"]), label="failure")
    payload = loaded["payload"]
    assert payload["schema"] == launcher.SCHEMA_FAILURE_COMPLETION
    assert payload["actual_exit_code"] == 17 and payload["actual_started"] is True
    assert payload["start_receipt_published"] is False
    assert payload["pMUA_start_permitted"] is False
    assert payload["actual_argv"] == argv_seen
    contract = launcher._read_pair(launcher._launch_contract_path("sua"), label="launch contract")
    assert payload["parent_launch_contract"]["body_sha256"] == contract["body_sha256"]
    assert payload["target_opened_or_may_have_opened"] is True


def test_zero_exit_with_missing_outputs_publishes_postvalidation_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topologies = _patch_topology(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "_finalize_success",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            launcher.StagePPairedLiveLauncherError("missing score")))
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="exited zero"):
        launcher._run_one_child(view="sua", token="token",
                                runner=lambda *_args: (0, b"out", b"err"),
                                prerequisites=_fake_prerequisites())
    payload = launcher._read_pair(
        Path(topologies["sua"]["completion"]), label="postvalidation failure")["payload"]
    assert payload["actual_exit_code"] == 0
    assert payload["failure_stage"] == "post_exit_immutable_validation"
    assert payload["status"] == "TERMINAL_CHILD_FAILURE__CELL_NONREUSABLE"


def test_synthetic_internal_child_exact_producer_order_and_source_receipt_binding(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topologies = _patch_topology(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "CONDA_PYTHON", Path(sys.executable))
    monkeypatch.setattr(launcher, "REPO_ROOT", Path.cwd())
    token = "child-token"
    for key, value in launcher._child_env_contract("sua").items():
        monkeypatch.setenv(key, value)
    for key in launcher.FORBIDDEN_CHILD_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(launcher.INTERNAL_TOKEN_ENV, token)
    monkeypatch.setattr(launcher.sys, "argv", launcher._child_argv("sua")[1:])
    cap = _cap("sua")
    launcher_closure = launcher.implementation_closure()
    cost = {
        "cost_body_sha256": _sha("cost"), "physical_uuid": "GPU-test",
        "physical_pci_bus_id": "bdf", "driver_version": "driver", "device_name": "gpu",
        "torch_version": "torch", "torch_cuda_runtime": "cuda", "torch_sha256": _sha("torch")}
    preflight_payload = {
        "launcher_preflight_payload_sha256": _sha("preflight-payload"),
        "launcher_implementation_closure": launcher_closure,
        "producer_closure_sha256": producer.implementation_closure()["closure_sha256"],
        "producer_addendum_body_sha256": _sha("addendum"),
        "cost_bound_cuda_identity": cost, "target_authority_bindings_by_view": {}}
    prereq = launcher.ExecutionPrerequisites(
        admissions={}, capabilities={"sua": cap},
        producer_addendum={"body_sha256": _sha("addendum")},
        launcher_preflight={"body_sha256": _sha("preflight"), "payload": preflight_payload},
        cost_identity=cost, target_authority_bindings={})
    monkeypatch.setattr(launcher, "validate_pretarget_prerequisites", lambda **_kwargs: prereq)
    launch_payload = launcher._launch_contract_payload(
        view="sua", token=token, prerequisites=prereq, parent_pid=os.getppid())
    launcher._publish(launcher._launch_contract_path("sua"), launch_payload)
    monkeypatch.setattr(launcher, "_verify_gpu_against_cost", lambda _cost: {
        "physical_uuid": "GPU-test", "physical_pci_bus_id": "bdf", "driver_version": "driver",
        "name": "gpu", "torch_version": "torch", "torch_cuda_runtime": "cuda",
        "torch_module_path": "/conda/torch.py", "torch_module_sha256": _sha("torch"),
            "cost_receipt_body_sha256": _sha("cost")})
    order = []
    source_payload = {
        "schema": producer.SCHEMA_SOURCE,
        "status": "STRICT27_SOURCE_MATERIALIZED_AND_EXACT_AUTHORITY_MATCH",
        "cell": cap.cell, "admission_sha256": cap.admission_sha256,
        "source_authority_set_sha256": cap.source_authority_set_sha256,
        "ordered_source_session_ids": [f"source-{i}" for i in range(27)],
        "source_session_count": 27, "source_authority_exact_rebuild_match": True,
        "historical_selector_plan_executed_or_selected": False,
        "target_opened": False, "query_opened": False}
    source_payload["source_payload_sha256"] = launcher._sha_json(source_payload)
    source = producer.SourceProducerOutput(cap, {}, (), source_payload)
    target = SimpleNamespace(capability=cap)
    class FakeEstimator:
        @classmethod
        def load(cls, *_args, **_kwargs): return cls()
    encoder = producer.JointEncoderOutput(cap, FakeEstimator(), (), {}, {}, {}, {}, {
        "fit_stream_count": 28, "fit_call_count": 1, "source_session_count": 27,
        "target_support_session_count": 1, "target_query_enters_fit": False,
        "model_contract": producer.MODEL_CONTRACT})
    monkeypatch.setattr(producer, "materialize_and_verify_strict27_source",
                        lambda _cap: order.append("source") or source)
    monkeypatch.setattr(producer, "materialize_target_from_private_snapshot",
                        lambda _cap, _source: order.append("target") or target)
    monkeypatch.setattr(producer, "fit_one_joint_encoder",
                        lambda **_kwargs: order.append("fit") or encoder)
    def fake_persist(**_kwargs):
        order.append("persist")
        checkpoint = producer.publish_immutable_raw_pair(
            Path(topologies["sua"]["checkpoint"]), b"checkpoint")
        embeddings = producer.publish_immutable_raw_pair(
            Path(topologies["sua"]["embeddings"]), b"embeddings")
        return {
            "checkpoint": checkpoint | {"same_fd_reload_exact": True},
            "embeddings": embeddings | {"same_fd_reload_exact": True}}
    monkeypatch.setattr(producer, "persist_sklearn_checkpoint_and_embeddings", fake_persist)
    target_payload = {
        "schema": producer.SCHEMA_TARGET,
        "status": "TARGET_M50_AND_SPARSE_V9_QUERY_MATERIALIZED", "cell": cap.cell,
        "start_payload_sha256": "filled-below",
        "private_snapshot": {"parser_consumed_continuously_held_fd": True,
                             "pathname_reopen_permitted": False},
        "query": {"semantics": "SPARSE_EXACT_V9_ENDPOINT_GATHER_FROM_CONTINUOUS_SUFFIX_TRANSFORM",
                  "query_row_count": producer.V9_QUERY_COUNT,
                  "valid_starts_int64_sha256": producer.V9_VALID_STARTS_SHA256,
                  "ordered_target_behavior_float32_sha256": producer.V9_TARGET_SHA256,
                  "sealed_v9_preflight_sha256": producer.V9_PREFLIGHT_SHA256,
                  "ordered_prediction_endpoint_int64_sha256": _sha("endpoints"),
                  "ordered_offset10_RF_int64_sha256": _sha("rf"),
                  "not_contiguous_5_to_minus5_crop": True,
                  "each_RF_is_range_endpoint_minus5_to_endpoint_plus5_exclusive": True,
                  "every_RF_wholly_inside_held_suffix": True,
                  "every_RF_support_disjoint": True,
                  "query_neural_or_auxiliary_enters_any_fit": False},
        "source_only_behavior_normalizer_used": True,
        "target_query_neural_or_auxiliary_entered_fit": False}
    monkeypatch.setattr(producer, "finalize_target_payload",
                        lambda **kwargs: order.append("finalize_target") or (
                            target_payload.update({"start_payload_sha256": kwargs["start_sha256"]}) or
                            target_payload.update({"target_payload_sha256": launcher._sha_json(target_payload)}) or
                            target_payload))
    def fake_scores(**kwargs):
        order.append("scores")
        outputs = {}
        for route in producer.ROUTES:
            for decoder in producer.DECODERS:
                payload = {
                    "schema": producer.SCHEMA_SCORE, "status": "ROUTE_DECODER_SCORE_COMPLETE",
                    "cell": cap.cell,
                    "readout_route": route, "decoder": decoder,
                    "target_payload_sha256": kwargs["target_payload"]["target_payload_sha256"],
                    "encoder_payload_sha256": kwargs["encoder_payload"]["encoder_payload_sha256"],
                    "prediction_float32_sha256": _sha(f"pred-{route}-{decoder}"),
                    "target_float32_sha256": producer.V9_TARGET_SHA256,
                    "query_row_count": producer.V9_QUERY_COUNT,
                    "readout_proof": {"query_enters_fit": False,
                                      "sparse_query_semantics": "exact_V9_endpoint_gather__not_contiguous_crop",
                                      "query_block_receipt_sha256": launcher._sha_json(
                                          kwargs["target_payload"]["query"])},
                    "metric": {"implementation": "torchmetrics.regression.R2Score", "version": "1.5.1",
                               "dtype": "float32", "device": "cpu", "multioutput": "variance_weighted",
                               "update_scope": "one_complete_ordered_external_target_session_query_then_compute_once",
                               "update_call_count": 1, "compute_call_count": 1,
                               "target_float32_bytes_sha256": producer.V9_TARGET_SHA256,
                               "custom_numpy_float64_pooled_r2_used": False}}
                payload["score_payload_sha256"] = launcher._sha_json(payload)
                outputs[f"{route}__{decoder}"] = payload
        return outputs
    monkeypatch.setattr(producer, "score_all_six_readouts", fake_scores)
    monkeypatch.setattr(launcher, "_cleanup_private_snapshot_after_success",
                        lambda **_kwargs: order.append("cleanup"))
    rc = launcher.execute_internal_child(view="sua", token=token)
    assert rc == 0
    assert order == ["source", "target", "fit", "persist", "finalize_target", "scores", "cleanup"]
    source_loaded = launcher._read_pair(Path(topologies["sua"]["source"]), label="source")
    encoder_loaded = launcher._read_pair(Path(topologies["sua"]["encoder"]), label="encoder")
    assert encoder_loaded["payload"]["source_receipt_body_sha256"] == source_loaded["body_sha256"]
    assert all(Path(path).exists() for path in topologies["sua"]["scores"].values())
    monkeypatch.setattr(launcher, "load_launcher_preflight", lambda **_kwargs: prereq.launcher_preflight)
    monkeypatch.setattr(launcher, "_fresh_admissions", lambda: {})
    monkeypatch.setattr(producer, "build_no_target_review_plan", lambda: {
        "implementation_closure": producer.implementation_closure()})
    monkeypatch.setattr(producer, "bind_execution_capabilities", lambda **_kwargs: {"sua": cap})
    monkeypatch.setattr(producer, "load_execution_addendum", lambda **_kwargs: prereq.producer_addendum)
    terminal = launcher._finalize_success(
        view="sua", returncode=0, argv=launcher._child_argv("sua"))
    completion = launcher._read_pair(Path(topologies["sua"]["completion"]), label="completion")
    assert completion["payload"]["actual_exit_code"] == 0
    assert completion["payload"]["immutable_output_bindings"]["source"]["body_sha256"] == \
        source_loaded["body_sha256"]
    assert terminal["source_receipt_body_sha256"] == source_loaded["body_sha256"]
    assert terminal["launcher_launch_final_live_exact_equal"] is True
    assert all(path.exists() and Path(f"{path}.sha256").exists()
               for path in launcher._prospective_paths("sua"))


def test_start_payload_rejects_actual_child_argv_substitution(monkeypatch: pytest.MonkeyPatch) -> None:
    prerequisites = _fake_prerequisites()
    launch_contract = {"path": str(launcher._launch_contract_path("sua")),
                       "body_sha256": _sha("contract"),
                       "payload": {"launch_contract_payload_sha256": _sha("contract-payload")}}
    monkeypatch.setattr(launcher.sys, "argv", ["unexpected.py"])
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="actual argv"):
        launcher._start_payload(
            capability=prerequisites.capabilities["sua"], prerequisites=prerequisites,
            gpu_identity={}, launch_contract=launch_contract)


@pytest.mark.parametrize("poison", ("source_cell", "target_upstream", "encoder_upstream", "score_role"))
def test_semantic_chain_rejects_same_schema_wrong_cell_upstream_or_role(poison: str) -> None:
    cap, start, source, target, encoder, scores = copy.deepcopy(_valid_semantic_chain())
    if poison == "source_cell":
        source["payload"]["cell"] = _cap("pseudo_mua").cell
        source["payload"]["source_payload_sha256"] = launcher._sha_json({
            key: value for key, value in source["payload"].items()
            if key != "source_payload_sha256"})
    elif poison == "target_upstream":
        target["payload"]["start_payload_sha256"] = _sha("wrong-start")
        target["payload"]["target_payload_sha256"] = launcher._sha_json({
            key: value for key, value in target["payload"].items()
            if key != "target_payload_sha256"})
    elif poison == "encoder_upstream":
        encoder["payload"]["source_payload_sha256"] = _sha("wrong-source")
        encoder["payload"]["encoder_payload_sha256"] = launcher._sha_json({
            key: value for key, value in encoder["payload"].items()
            if key != "encoder_payload_sha256"})
    else:
        role = next(iter(scores))
        scores[role]["payload"]["readout_route"] = "wrong_role"
        scores[role]["payload"]["score_payload_sha256"] = launcher._sha_json({
            key: value for key, value in scores[role]["payload"].items()
            if key != "score_payload_sha256"})
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="drift"):
        launcher._validate_scientific_output_chain(
            view="sua", capability=cap, start=start, source=source, target=target,
            encoder=encoder, scores=scores)


def test_finalize_reloads_preflight_and_rejects_midrun_launcher_closure_drift_before_outputs(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _cap("sua")
    producer_closure = producer.implementation_closure()
    launch_closure = {"files": {}, "closure_sha256": _sha("launch")}
    final_closure = {"files": {}, "closure_sha256": _sha("final")}
    monkeypatch.setattr(launcher, "load_launcher_preflight", lambda **_kwargs: {
        "body_sha256": _sha("pf"), "payload": {
            "launcher_implementation_closure": launch_closure,
            "producer_closure_sha256": producer_closure["closure_sha256"],
            "producer_addendum_body_sha256": _sha("addendum")}})
    monkeypatch.setattr(launcher, "_fresh_admissions", lambda: {})
    monkeypatch.setattr(producer, "build_no_target_review_plan", lambda: {})
    monkeypatch.setattr(producer, "bind_execution_capabilities", lambda **_kwargs: {"sua": cap})
    monkeypatch.setattr(producer, "load_execution_addendum",
                        lambda **_kwargs: {"body_sha256": _sha("addendum")})
    monkeypatch.setattr(launcher, "implementation_closure", lambda: final_closure)
    monkeypatch.setattr(launcher, "_load_output", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("decisive outputs must not be loaded after closure drift")))
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="launch/final/live"):
        launcher._finalize_success(view="sua", returncode=0, argv=launcher._child_argv("sua"))


def test_launcher_preflight_publication_rolls_back_on_final_closure_drift(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "launcher_preflight.json"
    monkeypatch.setattr(launcher, "LAUNCHER_PREFLIGHT", path)
    calls = 0
    def candidate():
        nonlocal calls
        calls += 1
        return {"schema": launcher.SCHEMA_PREFLIGHT, "canonical_path": str(path), "generation": calls}
    monkeypatch.setattr(launcher, "build_launcher_preflight_candidate", candidate)
    with pytest.raises(launcher.StagePPairedLiveLauncherError, match="launch/final"):
        launcher.publish_launcher_preflight(i_have_independent_root_review=True)
    assert not path.exists() and not Path(f"{path}.sha256").exists()


def test_parent_success_order_is_sua_then_pmua_then_paired_completion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_topology(tmp_path, monkeypatch)
    monkeypatch.setenv(launcher.ROOT_ENV, "1")
    prerequisites = _fake_prerequisites()
    closure = launcher.implementation_closure()
    prerequisites.launcher_preflight["payload"]["launcher_implementation_closure"] = closure
    monkeypatch.setattr(launcher, "validate_pretarget_prerequisites", lambda **_kwargs: prerequisites)
    calls = []
    def success(*, view, token, runner, prerequisites):
        calls.append(view)
        payload = {
            "schema": producer.SCHEMA_TERMINAL,
            "status": "TERMINAL_SUCCESS__DEVELOPMENT_PILOT_CELL",
            "cell": runtime.StagePCell.from_view(view).as_dict(),
            "terminal_payload_sha256": _sha(f"terminal-{view}"),
            "launcher_preflight_body_sha256": prerequisites.launcher_preflight["body_sha256"],
            "launcher_closure_sha256_at_launch": closure["closure_sha256"],
            "launcher_closure_sha256_at_final": closure["closure_sha256"],
            "launcher_launch_final_live_exact_equal": True,
            "producer_launch_final_live_exact_equal": True}
        return payload
    monkeypatch.setattr(launcher, "_run_one_child", success)
    monkeypatch.setattr(launcher, "load_launcher_preflight",
                        lambda **_kwargs: prerequisites.launcher_preflight)
    monkeypatch.setattr(launcher, "_load_output", lambda path, **_kwargs: {
        "payload": {
            "schema": producer.SCHEMA_TERMINAL,
            "status": "TERMINAL_SUCCESS__DEVELOPMENT_PILOT_CELL",
            "cell": runtime.StagePCell.from_view("sua" if "/sua/" in str(path) else "pseudo_mua").as_dict(),
            "terminal_payload_sha256": _sha(
                "terminal-sua" if "/sua/" in str(path) else "terminal-pseudo_mua"),
            "launcher_preflight_body_sha256": prerequisites.launcher_preflight["body_sha256"],
            "launcher_closure_sha256_at_launch": closure["closure_sha256"],
            "launcher_closure_sha256_at_final": closure["closure_sha256"],
            "launcher_launch_final_live_exact_equal": True,
            "producer_launch_final_live_exact_equal": True},
        "body_sha256": _sha("terminal-body")})
    result = launcher._orchestrate_authorized_pair(runner=lambda *_args: (0, b"", b""))
    assert calls == ["sua", "pseudo_mua"]
    assert result["status"] == "PAIRED_STAGEP_PILOT_COMPLETE__NO_POPULATION_INFERENCE"
    assert Path(launcher.PAIRED_COMPLETION).exists()
