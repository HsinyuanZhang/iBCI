"""No-target tests for the paired live launcher v2 successor."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import stat
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_subject_m_stagep_paired_live_launcher_v2 as launcher  # noqa: E402
import track_b_v2_subject_m_stagep_paired_real_producer_v2 as producer  # noqa: E402
import track_b_v2_subject_m_stagep_paired_live_launcher as launcher_v1  # noqa: E402
import track_b_v2_subject_m_stagep_paired_real_producer as producer_v1  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _temporary_topology(tmp_path: Path, view: str) -> dict:
    root = tmp_path / view
    scores = {f"{route}__{decoder}": str(root / "scores" / f"{route}__{decoder}.json")
              for route in producer.ROUTES for decoder in producer.DECODERS}
    return {"cell_root": str(root), "start": str(root / "start.json"),
            "source": str(root / "source.json"), "target_access_attempt": str(root / "attempt.json"),
            "target_lineage": str(root / "lineage.json"), "target": str(root / "target.json"),
            "private_snapshot": str(root / "private/target.nwb"), "encoder": str(root / "encoder.json"),
            "checkpoint": str(root / "model.pt"), "embeddings": str(root / "embeddings.npz"),
            "scores": scores, "child_stdout": str(root / "stdout.raw"),
            "child_stderr": str(root / "stderr.raw"), "completion": str(root / "completion.json"),
            "terminal": str(root / "terminal.json"), "caller_path_override_permitted": False,
            "successor_version": "v2"}


def _cap(view: str = "sua"):
    return producer.ViewExecutionCapability(
        view=view, cell=launcher_v1.sealed_runtime.StagePCell.from_view(view).as_dict(),
        admission_sha256=_sha("admission"), official_preflight_body_sha256=_sha("official"),
        implementation_closure_sha256=_sha("sealed"),
        source_authority_set_sha256=_sha("source-authority"),
        v9_preflight_sha256=producer.V9_PREFLIGHT_SHA256)


def test_dry_plan_is_no_write_and_v1_is_rejected() -> None:
    plan = launcher.build_dry_plan()
    assert plan["status"] == launcher.STATUS_DRY
    assert plan["topology"]["prospective_pair_count_per_view"] == 22
    assert plan["topology"]["total_prospective_pair_count"] == 45
    assert plan["v1_failure_provenance"]["v1_output_reuse_permitted"] is False
    assert plan["target_opened"] is plan["gpu_queried"] is plan["write_performed"] is False


def test_topology_includes_attempt_lineage_and_both_raw_streams() -> None:
    for view in launcher.PAIR_ORDER:
        paths = launcher._prospective_paths(view)
        topology = producer._topology(view)
        assert len(paths) == 22
        for role in ("target_access_attempt", "target_lineage", "child_stdout", "child_stderr"):
            assert Path(topology[role]).resolve() in paths
        assert launcher._snapshot_cleanup_path(view).resolve() in paths
        assert all(path.resolve() in paths for path in launcher._forensic_stream_paths(view))


def test_failure_persists_exact_empty_and_nonempty_stream_pairs_and_attempt_honesty(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topologies = {view: _temporary_topology(tmp_path, view) for view in launcher.PAIR_ORDER}
    monkeypatch.setattr(producer, "_topology", lambda view: copy.deepcopy(topologies[view]))
    monkeypatch.setattr(launcher, "_launch_contract_path",
                        lambda view: Path(topologies[view]["cell_root"]) / "launch.json")
    monkeypatch.setattr(launcher, "_publish_launch_contract", lambda **_kwargs:
                        launcher._publish(Path(topologies[_kwargs["view"]]["cell_root"]) / "launch.json",
                                          {"schema": launcher.SCHEMA_LAUNCH_CONTRACT}))
    class Prereq: pass
    def runner(view, argv, env):
        launcher._publish(Path(topologies[view]["target_access_attempt"]),
                          {"schema": producer.SCHEMA_TARGET_ACCESS_ATTEMPT})
        return 7, b"", b"full traceback\nline2\n"
    with pytest.raises(launcher.StagePPairedLiveLauncherV2Error, match="raw stderr"):
        launcher._run_one_child(view="sua", token="t", runner=runner, prerequisites=Prereq())
    completion = launcher._read_pair(Path(topologies["sua"]["completion"]), label="failure")["payload"]
    assert completion["stdout_byte_count"] == 0
    assert completion["stderr_byte_count"] == len(b"full traceback\nline2\n")
    assert completion["target_access_attempt_receipt_present"] is True
    assert completion["target_opened_or_may_have_opened"] is True
    terminal = launcher._read_pair(Path(topologies["sua"]["terminal"]), label="failure terminal")["payload"]
    assert terminal["schema"] == launcher.SCHEMA_FAILURE_TERMINAL
    assert terminal["target_opened_or_may_have_opened"] is True
    stderr = producer._read_same_fd(Path(topologies["sua"]["child_stderr"]),
                                    label="stderr", required_mode=0o444)
    assert stderr.raw == b"full traceback\nline2\n"


def test_failure_before_attempt_is_honestly_not_known_open(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topologies = {view: _temporary_topology(tmp_path, view) for view in launcher.PAIR_ORDER}
    monkeypatch.setattr(producer, "_topology", lambda view: copy.deepcopy(topologies[view]))
    monkeypatch.setattr(launcher, "_launch_contract_path",
                        lambda view: Path(topologies[view]["cell_root"]) / "launch.json")
    monkeypatch.setattr(launcher, "_publish_launch_contract", lambda **_kwargs:
                        launcher._publish(Path(topologies[_kwargs["view"]]["cell_root"]) / "launch.json",
                                          {"schema": launcher.SCHEMA_LAUNCH_CONTRACT}))
    with pytest.raises(launcher.StagePPairedLiveLauncherV2Error):
        launcher._run_one_child(view="sua", token="t", runner=lambda *_args: (2, b"", b"early"),
                                prerequisites=object())
    payload = launcher._read_pair(Path(topologies["sua"]["completion"]), label="failure")["payload"]
    assert payload["target_access_attempt_receipt_present"] is False
    assert payload["target_opened_or_may_have_opened"] is False
    terminal = launcher._read_pair(Path(topologies["sua"]["terminal"]), label="failure terminal")["payload"]
    assert terminal["pMUA_start_permitted"] is False


def test_single_flag_and_missing_root_env_refuse() -> None:
    cli = launcher.CLI
    import subprocess
    env = dict(os.environ); env.pop(launcher.ROOT_ENV, None)
    for args in (["--execute"], ["--i-have-authorization"]):
        done = subprocess.run([sys.executable, str(cli), *args], cwd=ROOT, env=env,
                              text=True, capture_output=True, check=False)
        assert done.returncode != 0
    done = subprocess.run([sys.executable, str(cli), "--execute", "--i-have-authorization"],
                          cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    assert done.returncode != 0 and launcher.ROOT_ENV in done.stderr


def test_success_finalizer_binds_attempt_lineage_streams_and_launch_final_closure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topology = _temporary_topology(tmp_path, "sua")
    monkeypatch.setattr(producer, "_topology", lambda _view: copy.deepcopy(topology))
    monkeypatch.setattr(launcher, "_launch_contract_path", lambda _view: tmp_path / "sua/launch.json")
    cap = _cap(); closure = launcher.implementation_closure(); pclosure = producer.implementation_closure()
    target_authority = {"canonical": "target-authority"}
    preflight = {"body_sha256": _sha("preflight-body"), "payload": {
        "launcher_implementation_closure": closure,
        "producer_closure_sha256": pclosure["closure_sha256"],
        "target_authority_bindings_by_view": {"sua": target_authority}}}
    monkeypatch.setattr(launcher, "load_launcher_preflight", lambda **_kwargs: preflight)
    monkeypatch.setattr(launcher, "_fresh_admissions", lambda: {})
    monkeypatch.setattr(producer, "build_no_target_review_plan", lambda: {})
    monkeypatch.setattr(producer, "bind_execution_capabilities", lambda **_kwargs: {"sua": cap})
    monkeypatch.setattr(producer, "load_execution_addendum",
                        lambda **_kwargs: {"body_sha256": _sha("addendum")})
    monkeypatch.setattr(launcher_v1, "_validate_start_chain", lambda **_kwargs: None)
    monkeypatch.setattr(launcher_v1, "_validate_scientific_output_chain", lambda **_kwargs: None)

    source_payload = {"schema": producer.SCHEMA_SOURCE,
                      "source_payload_sha256": _sha("source-payload")}
    source_binding = launcher._publish(Path(topology["source"]), source_payload)
    attempt_payload = {
        "schema": producer.SCHEMA_TARGET_ACCESS_ATTEMPT, "cell": cap.cell,
        "source_receipt_body_sha256": source_binding["body_sha256"],
        "source_payload_sha256": source_payload["source_payload_sha256"],
        "launcher_preflight_body_sha256": preflight["body_sha256"],
        "target_authority": target_authority, "target_opened_before_publication": False,
        "next_operation_may_open_target": True}
    attempt_payload["target_access_attempt_payload_sha256"] = launcher._sha_json(attempt_payload)
    attempt_binding = launcher._publish(Path(topology["target_access_attempt"]), attempt_payload)
    lineage_payload = {
        "schema": producer.SCHEMA_TARGET_LINEAGE, "cell": cap.cell,
        "source_receipt_body_sha256": source_binding["body_sha256"],
        "target_access_attempt_body_sha256": attempt_binding["body_sha256"],
        "valid_starts_int64_sha256": producer.V9_VALID_STARTS_SHA256,
        "ordered_target_behavior_float32_sha256": producer.V9_TARGET_SHA256,
        "private_snapshot": {
            "private_snapshot_path": topology["private_snapshot"],
            "private_snapshot_sha256": _sha("snapshot"),
            "private_snapshot_identity_sha256": _sha("snapshot-identity")},
        "query_row_count": producer.V9_QUERY_COUNT, "cebra_fit_started": False,
        "gpu_fit_started": False, "formal_data_opened": False}
    lineage_payload["target_lineage_payload_sha256"] = launcher._sha_json(lineage_payload)
    lineage_binding = launcher._publish(Path(topology["target_lineage"]), lineage_payload)
    cleanup_payload = {
        "schema": launcher.SCHEMA_SNAPSHOT_CLEANUP, "cell": cap.cell,
        "private_snapshot_path": topology["private_snapshot"],
        "private_snapshot_sha256": _sha("snapshot"),
        "private_snapshot_identity_sha256": _sha("snapshot-identity"),
        "link_count_after_unlink": 0, "held_fd_identity_unchanged_after_unlink": True,
        "canonical_path_absent_after_unlink": True}
    cleanup_payload["snapshot_cleanup_payload_sha256"] = launcher._sha_json(cleanup_payload)
    cleanup_binding = launcher._publish(
        launcher._snapshot_cleanup_path("sua"), cleanup_payload)
    launch_binding = launcher._publish(tmp_path / "sua/launch.json",
                                       {"schema": launcher.SCHEMA_LAUNCH_CONTRACT})
    start_payload = {"schema": producer.SCHEMA_START, "start_payload_sha256": _sha("start"),
                     "launcher_closure_sha256": closure["closure_sha256"]}
    launcher._publish(Path(topology["start"]), start_payload)
    target_payload = {
        "schema": producer.SCHEMA_TARGET, "target_payload_sha256": _sha("target"),
        "target_access_attempt_receipt_body_sha256": attempt_binding["body_sha256"],
        "prefit_target_lineage_receipt_body_sha256": lineage_binding["body_sha256"]}
    launcher._publish(Path(topology["target"]), target_payload)
    Path(topology["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = producer.publish_immutable_raw_pair(Path(topology["checkpoint"]), b"checkpoint")
    embeddings = producer.publish_immutable_raw_pair(Path(topology["embeddings"]), b"embeddings")
    encoder_payload = {
        "schema": producer.SCHEMA_ENCODER, "encoder_payload_sha256": _sha("encoder"),
        "checkpoint": {"body_sha256": checkpoint["body_sha256"]},
        "embedding_bundle": {"body_sha256": embeddings["body_sha256"]},
        "prefit_target_lineage_receipt_body_sha256": lineage_binding["body_sha256"]}
    launcher._publish(Path(topology["encoder"]), encoder_payload)
    for route in producer.ROUTES:
        for decoder in producer.DECODERS:
            role = f"{route}__{decoder}"
            launcher._publish(Path(topology["scores"][role]), {
                "schema": producer.SCHEMA_SCORE, "score_payload_sha256": _sha(role)})
    stdout = producer.publish_immutable_raw_pair(Path(topology["child_stdout"]), b"") | {"byte_count": 0}
    stderr = producer.publish_immutable_raw_pair(Path(topology["child_stderr"]), b"") | {"byte_count": 0}
    terminal = launcher._finalize_success(
        view="sua", returncode=0, argv=launcher._child_argv("sua"),
        stdout_binding=stdout, stderr_binding=stderr)
    completion = launcher._read_pair(Path(topology["completion"]), label="completion")["payload"]
    assert completion["target_access_attempt_receipt_body_sha256"] == attempt_binding["body_sha256"]
    assert completion["prefit_target_lineage_receipt_body_sha256"] == lineage_binding["body_sha256"]
    assert completion["snapshot_cleanup_receipt_body_sha256"] == cleanup_binding["body_sha256"]
    assert completion["child_stdout_raw_pair"]["byte_count"] == 0
    assert terminal["launcher_launch_final_live_exact_equal"] is True
    assert launch_binding["body_sha256"] == completion["parent_launch_contract_body_sha256"]


def test_finalizer_rejects_midrun_closure_drift_before_outputs(
        monkeypatch: pytest.MonkeyPatch) -> None:
    launch = {"files": {}, "closure_sha256": _sha("launch")}
    final = {"files": {}, "closure_sha256": _sha("final")}
    pclosure = producer.implementation_closure()
    monkeypatch.setattr(launcher, "load_launcher_preflight", lambda **_kwargs: {
        "payload": {"launcher_implementation_closure": launch,
                    "producer_closure_sha256": pclosure["closure_sha256"]}})
    monkeypatch.setattr(launcher, "_fresh_admissions", lambda: {})
    monkeypatch.setattr(producer, "build_no_target_review_plan", lambda: {})
    monkeypatch.setattr(producer, "bind_execution_capabilities", lambda **_kwargs: {"sua": _cap()})
    monkeypatch.setattr(producer, "load_execution_addendum", lambda **_kwargs: {})
    monkeypatch.setattr(launcher, "implementation_closure", lambda: final)
    monkeypatch.setattr(launcher, "_load_output", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("outputs must not load after closure drift")))
    with pytest.raises(launcher.StagePPairedLiveLauncherV2Error, match="closure drift"):
        launcher._finalize_success(
            view="sua", returncode=0, argv=launcher._child_argv("sua"),
            stdout_binding={}, stderr_binding={})


def test_sua_failure_blocks_pmua_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launcher.ROOT_ENV, "1")
    monkeypatch.setattr(launcher, "validate_pretarget_prerequisites", lambda **_kwargs: object())
    calls = []
    def fail(*, view, **_kwargs):
        calls.append(view)
        raise launcher.StagePPairedLiveLauncherV2Error("SUA failed")
    monkeypatch.setattr(launcher, "_run_one_child", fail)
    with pytest.raises(launcher.StagePPairedLiveLauncherV2Error, match="SUA failed"):
        launcher._orchestrate_authorized_pair(runner=lambda *_args: (0, b"", b""))
    assert calls == ["sua"]


def test_direct_pmua_requires_sua_terminal_before_target_authority_or_gpu(
        monkeypatch: pytest.MonkeyPatch) -> None:
    contract = {"payload": {"launcher_preflight_body_sha256": _sha("preflight")}}
    monkeypatch.setattr(launcher, "_load_child_launch_contract", lambda **_kwargs: contract)
    monkeypatch.setattr(launcher, "_verify_child_environment", lambda **_kwargs: None)
    monkeypatch.setattr(launcher, "load_launcher_preflight",
                        lambda **_kwargs: {"body_sha256": _sha("preflight")})
    touched = []
    monkeypatch.setattr(launcher, "_require_sua_terminal_before_pmua",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            launcher.StagePPairedLiveLauncherV2Error("SUA terminal absent")))
    monkeypatch.setattr(launcher, "validate_pretarget_prerequisites",
                        lambda **_kwargs: touched.append("target-authority"))
    monkeypatch.setattr(launcher, "_verify_gpu_against_cost", lambda *_args: touched.append("gpu"))
    with pytest.raises(launcher.StagePPairedLiveLauncherV2Error, match="SUA terminal absent"):
        launcher.execute_internal_child(view="pseudo_mua", token="untrusted")
    assert touched == []


def test_preattempt_gate_never_calls_target_authority_or_materializer_helper(
        monkeypatch: pytest.MonkeyPatch) -> None:
    cap = _cap()
    frozen = {view: {"frozen": view} for view in launcher.PAIR_ORDER}
    monkeypatch.setattr(launcher, "_fresh_admissions", lambda: {})
    monkeypatch.setattr(producer, "build_no_target_review_plan", lambda: {})
    monkeypatch.setattr(producer, "bind_execution_capabilities",
                        lambda **_kwargs: {view: cap for view in launcher.PAIR_ORDER})
    monkeypatch.setattr(producer, "load_execution_addendum",
                        lambda **_kwargs: {"body_sha256": _sha("addendum")})
    monkeypatch.setattr(launcher, "load_launcher_preflight", lambda **_kwargs: {
        "body_sha256": _sha("preflight"), "payload": {
            "producer_addendum_body_sha256": _sha("addendum"),
            "cost_bound_cuda_identity": {"gpu": "frozen"},
            "target_authority_bindings_by_view": frozen}})
    monkeypatch.setattr(launcher, "_cost_identity", lambda: {"gpu": "frozen"})
    monkeypatch.setattr(launcher, "require_all_outputs_fresh", lambda **_kwargs: None)
    monkeypatch.setattr(launcher, "_target_authority_bindings", lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("pre-attempt target authority helper forbidden")))
    monkeypatch.setattr(launcher.l1.target_materializer,
                        "build_development_target_materializer_dry_plan",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            AssertionError("pre-attempt target materializer forbidden")))
    prerequisites = launcher.validate_pretarget_prerequisites(only_view="sua")
    assert prerequisites.target_authority_bindings == frozen


def test_attempt_pair_is_durable_before_exact_one_view_live_target_helper(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topology = _temporary_topology(tmp_path, "sua")
    monkeypatch.setattr(producer, "_topology", lambda _view: copy.deepcopy(topology))
    cap = _cap()
    frozen = {
        "status": "CANONICAL_DEVELOPMENT_AUTHORITY_AND_SUBM_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS",
        "dataset": "subject_m", "view": "sua", "target_session_id": producer.TARGET_SESSION_ID,
        "target_data_opened": False}
    prerequisites = launcher.ExecutionPrerequisites(
        {}, {"sua": cap}, {}, {"body_sha256": _sha("preflight")}, {}, {"sua": frozen})
    source = producer.SourceProducerOutput(
        capability=cap, request={}, rows=(),
        payload={"source_payload_sha256": _sha("source-payload")})
    source_binding = launcher._publish(Path(topology["source"]), {"schema": producer.SCHEMA_SOURCE})
    calls = []
    def target_helper(**kwargs):
        calls.append(dict(kwargs))
        attempt = Path(topology["target_access_attempt"])
        assert attempt.exists() and Path(f"{attempt}.sha256").exists()
        loaded = launcher._read_pair(attempt, label="durable-before-helper")
        assert loaded["payload"]["target_authority"] == frozen
        return copy.deepcopy(frozen)
    monkeypatch.setattr(launcher.l1.target_materializer,
                        "build_development_target_materializer_dry_plan", target_helper)
    binding, live = launcher._publish_attempt_then_resolve_live_target_authority(
        view="sua", prerequisites=prerequisites, source=source, source_binding=source_binding)
    assert len(calls) == 1 and calls[0]["view"] == "sua"
    assert live == frozen and launcher._read_pair(
        Path(topology["target_access_attempt"]), label="attempt")["body_sha256"] == binding["body_sha256"]


def test_stream_group_sidecar_race_rolls_back_canonical_and_preserves_forensic_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topologies = {view: _temporary_topology(tmp_path, view) for view in launcher.PAIR_ORDER}
    monkeypatch.setattr(producer, "_topology", lambda view: copy.deepcopy(topologies[view]))
    monkeypatch.setattr(launcher, "_launch_contract_path",
                        lambda view: Path(topologies[view]["cell_root"]) / "launch.json")
    monkeypatch.setattr(launcher, "_publish_launch_contract", lambda **_kwargs:
                        launcher._publish(Path(topologies[_kwargs["view"]]["cell_root"]) / "launch.json",
                                          {"schema": launcher.SCHEMA_LAUNCH_CONTRACT}))
    original_link = launcher.os.link
    injected = {"done": False}
    stderr_side_name = Path(f"{topologies['sua']['child_stderr']}.sha256").name
    def racing_link(src, dst, *args, **kwargs):
        if dst == stderr_side_name and not injected["done"]:
            injected["done"] = True
            fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444,
                         dir_fd=kwargs["dst_dir_fd"])
            os.write(fd, b"foreign-race\n"); os.fsync(fd); os.close(fd)
        return original_link(src, dst, *args, **kwargs)
    monkeypatch.setattr(launcher.os, "link", racing_link)
    stdout_raw = b"complete stdout bytes\n"; stderr_raw = b"complete stderr bytes\n"
    def runner(view, _argv, _env):
        launcher._publish(Path(topologies[view]["target_access_attempt"]),
                          {"schema": producer.SCHEMA_TARGET_ACCESS_ATTEMPT})
        return 9, stdout_raw, stderr_raw
    with pytest.raises(launcher.StagePPairedLiveLauncherV2Error, match="forensic fallback"):
        launcher._run_one_child(view="sua", token="t", runner=runner, prerequisites=object())
    assert not Path(topologies["sua"]["child_stdout"]).exists()
    assert not Path(f"{topologies['sua']['child_stdout']}.sha256").exists()
    assert not Path(topologies["sua"]["child_stderr"]).exists()
    forensic_stdout, forensic_stderr = launcher._forensic_stream_paths("sua")
    assert producer._read_same_fd(forensic_stdout, label="forensic stdout", required_mode=0o444).raw == stdout_raw
    assert producer._read_same_fd(forensic_stderr, label="forensic stderr", required_mode=0o444).raw == stderr_raw
    failure = launcher._read_pair(Path(topologies["sua"]["completion"]), label="failure")["payload"]
    terminal = launcher._read_pair(Path(topologies["sua"]["terminal"]), label="terminal")["payload"]
    assert failure["child_stream_storage_role"] == "forensic_fallback_after_canonical_group_failure"
    assert failure["target_opened_or_may_have_opened"] is True
    assert terminal["failure_stage"] == "raw_stream_group_publication_failure"


def test_zero_exit_finalization_failure_still_terminalizes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topologies = {view: _temporary_topology(tmp_path, view) for view in launcher.PAIR_ORDER}
    monkeypatch.setattr(producer, "_topology", lambda view: copy.deepcopy(topologies[view]))
    monkeypatch.setattr(launcher, "_launch_contract_path",
                        lambda view: Path(topologies[view]["cell_root"]) / "launch.json")
    monkeypatch.setattr(launcher, "_publish_launch_contract", lambda **_kwargs:
                        launcher._publish(Path(topologies[_kwargs["view"]]["cell_root"]) / "launch.json",
                                          {"schema": launcher.SCHEMA_LAUNCH_CONTRACT}))
    monkeypatch.setattr(launcher, "_finalize_success", lambda **_kwargs: (_ for _ in ()).throw(
        launcher.StagePPairedLiveLauncherV2Error("terminal publication drift")))
    with pytest.raises(launcher.StagePPairedLiveLauncherV2Error, match="finalization failed"):
        launcher._run_one_child(view="sua", token="t", runner=lambda *_args: (0, b"out", b"err"),
                                prerequisites=object())
    failure = launcher._read_pair(Path(topologies["sua"]["completion"]), label="failure")["payload"]
    terminal = launcher._read_pair(Path(topologies["sua"]["terminal"]), label="terminal")["payload"]
    assert failure["failure_stage"] == "post_exit_immutable_validation"
    assert terminal["schema"] == launcher.SCHEMA_FAILURE_TERMINAL


def _target_with_snapshot(cap, path: Path):
    raw = b"held target snapshot bytes"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw); path.chmod(0o444)
    info = path.stat()
    identity = (info.st_dev, info.st_ino, info.st_size, info.st_mode,
                info.st_mtime_ns, info.st_ctime_ns)
    return producer.TargetProducerOutput(
        capability=cap, neural_support=None, behavior_support=None, neural_suffix=None,
        behavior_suffix=None, valid_starts=None, payload_inputs={"private_snapshot": {
            "private_snapshot_path": str(path), "private_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
            "private_snapshot_byte_count": len(raw),
            "private_snapshot_identity_sha256": launcher._sha_json(identity)}})


def test_snapshot_cleanup_holds_original_fd_through_unlink_and_publishes_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topology = _temporary_topology(tmp_path, "sua")
    monkeypatch.setattr(producer, "_topology", lambda _view: copy.deepcopy(topology))
    target = _target_with_snapshot(_cap(), Path(topology["private_snapshot"]))
    binding = launcher._cleanup_private_snapshot_after_success(view="sua", target=target)
    payload = launcher._read_pair(launcher._snapshot_cleanup_path("sua"), label="cleanup")["payload"]
    assert binding["body_sha256"]
    assert payload["link_count_after_unlink"] == 0
    assert payload["held_fd_identity_unchanged_after_unlink"] is True
    assert not Path(topology["private_snapshot"]).exists()


def test_snapshot_cleanup_rename_swap_fails_without_cleanup_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    topology = _temporary_topology(tmp_path, "sua")
    monkeypatch.setattr(producer, "_topology", lambda _view: copy.deepcopy(topology))
    snapshot = Path(topology["private_snapshot"])
    target = _target_with_snapshot(_cap(), snapshot)
    original_unlink = launcher.os.unlink
    injected = {"done": False}
    def swapping_unlink(name, *args, **kwargs):
        if name == snapshot.name and not injected["done"]:
            injected["done"] = True
            os.rename(name, "moved-original.nwb", src_dir_fd=kwargs["dir_fd"],
                      dst_dir_fd=kwargs["dir_fd"])
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444,
                         dir_fd=kwargs["dir_fd"])
            os.write(fd, b"foreign"); os.close(fd)
        return original_unlink(name, *args, **kwargs)
    monkeypatch.setattr(launcher.os, "unlink", swapping_unlink)
    with pytest.raises(launcher.StagePPairedLiveLauncherV2Error, match="canonical held inode"):
        launcher._cleanup_private_snapshot_after_success(view="sua", target=target)
    assert not launcher._snapshot_cleanup_path("sua").exists()
