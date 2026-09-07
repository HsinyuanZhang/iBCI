from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

import pytest

from scripts import h1_carrierid_date_lodo_hls_early_v2_remote_transfer as transfer
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_LAUNCH_SCHEMA,
    EARLY_LAUNCH_STATUS,
    EARLY_PREFLIGHT_SCHEMA,
    EARLY_PREFLIGHT_STATUS,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
)


def _immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path.resolve()


def _file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.resolve()


def _row(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": transfer.sha256_file(path)}


def _closure(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Path], Path]:
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    _file(repo / "code.py", "CODE_CLOSURE = True\n")
    phase1 = _immutable(repo / "producer/phase1.json", {"kind": "phase1"})
    null = _immutable(repo / "producer/null.json", {"kind": "null"})
    waiting = _immutable(repo / "producer/waiting.json", {"kind": "waiting"})
    preflights: dict[str, Path] = {}
    terminals: dict[str, Path] = {}
    launch_rows: dict[str, Any] = {}
    for date in DATES:
        pair = _immutable(repo / f"producer/{date}/pair.json", {"date": date})
        plan_manifest = _immutable(repo / f"producer/{date}/frozen_m4_plan.manifest.json",
                                   {"date": date, "kind": "plan"})
        _immutable(repo / f"producer/{date}/frozen_m4_plan.npz",
                   {"date": date, "synthetic": "arrays"})
        normalizer = _immutable(repo / f"producer/{date}/source_rms_normalizer.manifest.json",
                                {"date": date, "kind": "normalizer"})
        manifest = _immutable(repo / f"producer/{date}/source_manifest.json", {
            "frozen_plan": {"manifest_path": str(plan_manifest),
                            "manifest_sha256": transfer.sha256_file(plan_manifest)},
            "normalizer": {"manifest_path": str(normalizer),
                           "manifest_file_sha256": transfer.sha256_file(normalizer)},
        })
        preflight = _immutable(repo / f"early/preflights/{date}.json", {
            "schema": EARLY_PREFLIGHT_SCHEMA, "status": EARLY_PREFLIGHT_STATUS,
            "outer_date": date,
            "matched_h_c_pair_preflight": _row(pair),
            "phase1_preflight": _row(phase1), "null_strength_audit": _row(null),
            "waiting_plan": _row(waiting),
            "source_binding": {"source_manifest_path": str(manifest),
                               "source_manifest_sha256": transfer.sha256_file(manifest)},
        })
        preflights[date] = preflight
        launch_rows[date] = {**_row(preflight), "source_binding_sha256": f"source-{date}"}
        claim = _immutable(repo / f"early/claims/{date}.json", {
            "schema": transfer.WRITER_CLAIM_SCHEMA, "status": transfer.WRITER_CLAIM_STATUS,
            "outer_date": date,
        })
        checkpoint = _file(repo / f"early/runs/{date}/checkpoints/fixed_epoch50/epoch_049.ckpt",
                           f"checkpoint-{date}\n")
        config = _file(repo / f"early/runs/{date}/.hydra/config.yaml", f"date: {date}\n")
        terminal = _immutable(repo / f"early/terminals/{date}.json", {
            "schema": EARLY_TERMINAL_SCHEMA, "status": EARLY_TERMINAL_STATUS,
            "outer_date": date, "arm": "H-LS", "source_preflight": _row(preflight),
            "writer_claim": _row(claim),
            "checkpoint": {**_row(checkpoint), "config_path": str(config),
                           "config_sha256": transfer.sha256_file(config)},
            "scope": {"target_recordings_opened": 0, "target_bytes_read": 0},
            "deployment_updates": {"target_optimizer_steps": 0, "target_backward_steps": 0,
                                   "target_model_state_updated": False},
        })
        terminals[date] = terminal
    launch = _immutable(repo / "early/launch.json", {
        "schema": EARLY_LAUNCH_SCHEMA, "status": EARLY_LAUNCH_STATUS,
        "fixed_grid": list(DATES), "source_preflights": launch_rows,
    })
    output = repo / "early/import/remote_transfer.json"
    plan = transfer.build_plan(
        launch_receipt=launch, source_terminals=terminals, output=output,
        upstream_aggregate=repo / "producer/upstream.json", repo_root=repo,
        code_files=("code.py",),
    )
    return plan, repo, terminals, output


def _state(row: dict[str, Any]) -> dict[str, Any]:
    return {"state": "FILE", "sha256": row["sha256"], "size": row["size"],
            "mode": row.get("mode", 0o444)}


def _inspection(plan: dict[str, Any], requested: list[str], *, transfer_missing: bool,
                gate_ready: bool = True, mismatch: str | None = None) -> dict[str, Any]:
    transfer_rows = {row["path"]: row for row in plan["transfer_files"]}
    prerequisite_rows = {row["path"]: row for row in plan["remote_existing_prerequisites"]}
    files: dict[str, Any] = {}
    for path in requested:
        if path == plan["transfer_receipt_output"]:
            local = Path(path)
            files[path] = ({"state": "FILE", "sha256": transfer.sha256_file(local),
                            "size": local.stat().st_size, "mode": 0o444}
                           if local.exists() else {"state": "MISSING"})
        elif path in prerequisite_rows:
            files[path] = _state(prerequisite_rows[path])
        elif path in transfer_rows:
            files[path] = {"state": "MISSING"} if transfer_missing else _state(transfer_rows[path])
        else:
            files[path] = {"state": "MISSING"}
    if mismatch is not None:
        files[mismatch] = {"state": "FILE", "sha256": "0" * 64,
                           "size": transfer_rows.get(mismatch, prerequisite_rows.get(mismatch))["size"],
                           "mode": 0o444}
    return {
        "repo_root": plan["remote_repo_root"], "files": files,
        "active_main_chain_producers": [],
        "upstream_gate": ({"ready": True, "sha256": "a" * 64}
                          if gate_ready else {"ready": False, "reason": "UPSTREAM_INCOMPLETE"}),
    }


def _completed(command, returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def test_dry_run_requires_five_passed_terminals_and_builds_minimal_data_closure(tmp_path: Path) -> None:
    plan, _repo, _terminals, output = _closure(tmp_path)
    assert plan["status"] == "DRY_RUN_LOCAL_CLOSURE_READY_NOT_TRANSFERRED"
    labels = [row["label"] for row in plan["transfer_files"]]
    assert len([label for label in labels if "source terminal" in label]) == 5
    assert len([label for label in labels if "source preflight" in label]) == 5
    assert len([label for label in labels if "writer claim" in label]) == 5
    assert len([label for label in labels if "e49 checkpoint" in label]) == 5
    assert len([label for label in labels if "resolved config" in label]) == 5
    assert "early-v2 launch receipt" in labels and "code closure:code.py" in labels
    assert not output.exists()
    assert plan["policy"]["binder_called"] is False
    assert plan["policy"]["target_opened"] == 0


def test_nonpass_source_terminal_blocks_plan_before_any_remote_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, repo, terminals, output = _closure(tmp_path)
    terminal = terminals[DATES[0]]
    body = json.loads(terminal.read_text())
    body["status"] = "FAILED"
    terminal.chmod(0o644)
    terminal.write_text(json.dumps(body), encoding="utf-8")
    terminal.chmod(0o444)
    monkeypatch.setattr(transfer.subprocess, "run",
                        lambda *_args, **_kwargs: pytest.fail("dry-run validation must not use SSH"))
    with pytest.raises(Exception, match="schema/status drift"):
        transfer.build_plan(
            launch_receipt=Path(plan["launch_receipt"]["path"]), source_terminals=terminals,
            output=output, upstream_aggregate=repo / "producer/upstream.json",
            repo_root=repo, code_files=("code.py",),
        )


def test_execute_fails_closed_when_remote_main_chain_is_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _terminals, output = _closure(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        rendered = [str(value) for value in command]
        calls.append(rendered)
        assert rendered[0] == "ssh"
        request = json.loads(kwargs["input"])
        return _completed(rendered, stdout=json.dumps(
            _inspection(plan, request["paths"], transfer_missing=True, gate_ready=False)))

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    with pytest.raises(transfer.RemoteTransferError, match="main chain is not complete"):
        transfer.execute(plan)
    assert len(calls) == 1 and calls[0][0] == "ssh"
    assert not output.exists()


def test_existing_remote_mismatch_is_refused_before_rsync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _terminals, _output = _closure(tmp_path)
    mismatch = plan["transfer_files"][0]["path"]
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        rendered = [str(value) for value in command]
        calls.append(rendered)
        assert rendered[0] == "ssh"
        request = json.loads(kwargs["input"])
        return _completed(rendered, stdout=json.dumps(
            _inspection(plan, request["paths"], transfer_missing=True, mismatch=mismatch)))

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    with pytest.raises(transfer.RemoteTransferError, match="refusing overwrite"):
        transfer.execute(plan)
    assert all(command[0] != "rsync" for command in calls)


def test_active_remote_hs_hc_producer_blocks_all_writes_even_with_complete_aggregate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _terminals, _output = _closure(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        rendered = [str(value) for value in command]
        calls.append(rendered)
        request = json.loads(kwargs["input"])
        body = _inspection(plan, request["paths"], transfer_missing=True)
        body["active_main_chain_producers"] = [
            {"pid": 1234, "command": "python src/train.py experiment=h1_carrierid_date_lodo_hc_19250120"}
        ]
        return _completed(rendered, stdout=json.dumps(body))

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    with pytest.raises(transfer.RemoteTransferError, match="producer/finalizer is still active"):
        transfer.execute(plan)
    assert len(calls) == 1 and calls[0][0] == "ssh"


def test_remote_producer_prerequisite_is_never_transferred_and_must_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _terminals, _output = _closure(tmp_path)
    mismatch = plan["remote_existing_prerequisites"][0]["path"]
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        rendered = [str(value) for value in command]
        calls.append(rendered)
        request = json.loads(kwargs["input"])
        return _completed(rendered, stdout=json.dumps(
            _inspection(plan, request["paths"], transfer_missing=True, mismatch=mismatch)))

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    with pytest.raises(transfer.RemoteTransferError, match="producer/base prerequisite"):
        transfer.execute(plan)
    assert all(command[0] != "rsync" for command in calls)


def test_success_transfers_only_missing_then_verifies_every_sha_and_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _terminals, output = _closure(tmp_path)
    calls: list[tuple[list[str], dict[str, Any]]] = []
    ssh_count = 0

    def fake_run(command, **kwargs):
        nonlocal ssh_count
        rendered = [str(value) for value in command]
        calls.append((rendered, kwargs))
        if rendered[0] == "ssh":
            ssh_count += 1
            request = json.loads(kwargs["input"])
            return _completed(rendered, stdout=json.dumps(_inspection(
                plan, request["paths"], transfer_missing=ssh_count == 1,
            )))
        assert rendered[0] == "rsync"
        return _completed(rendered)

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    result = transfer.execute(plan)
    assert result["status"] == transfer.TRANSFER_STATUS
    assert result["transferred_missing"] == len(plan["transfer_files"])
    assert output.is_file() and output.stat().st_mode & 0o777 == 0o444
    body = json.loads(output.read_text())
    assert body["scope"]["binder_called"] is False and body["scope"]["target_opened"] == 0
    rsync_calls = [command for command, _kwargs in calls if command[0] == "rsync"]
    assert len(rsync_calls) == 2
    assert all("--ignore-existing" in command and "--delete" not in command for command in rsync_calls)
    producer_paths = {row["path"] for row in plan["remote_existing_prerequisites"]}
    assert producer_paths.isdisjoint(rsync_calls[0])
    assert output.as_posix() in rsync_calls[1]
    assert all("--execute-target-evaluation" not in command for command, _kwargs in calls)


def test_remote_repo_root_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan, _repo, _terminals, _output = _closure(tmp_path)

    def fake_run(command, **kwargs):
        request = json.loads(kwargs["input"])
        body = _inspection(plan, request["paths"], transfer_missing=True)
        body["repo_root"] = "/different/repo"
        return _completed(command, stdout=json.dumps(body))

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    with pytest.raises(transfer.RemoteTransferError, match="identical absolute path"):
        transfer.execute(plan)


def test_ssh_remote_multiline_python_is_one_shell_quoted_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _terminals, _output = _closure(tmp_path)
    captured: list[list[str]] = []

    def fake_run(command, **kwargs):
        rendered = [str(value) for value in command]
        captured.append(rendered)
        request = json.loads(kwargs["input"])
        return _completed(rendered, stdout=json.dumps(
            _inspection(plan, request["paths"], transfer_missing=True)))

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    transfer._remote_inspect(remote=transfer.REMOTE, plan=plan, paths=[])
    assert len(captured) == 1
    command = captured[0]
    assert command[-2] == transfer.REMOTE
    assert len(command[command.index(transfer.REMOTE) + 1:]) == 1
    remote_command = command[-1]
    assert "\n" in remote_command and " " in remote_command
    assert shlex.split(remote_command) == ["python3", "-c", transfer.REMOTE_INSPECT_CODE]


def test_read_only_probe_uses_only_one_ssh_and_never_rsync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    plan, _repo, _terminals, _output = _closure(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        rendered = [str(value) for value in command]
        calls.append(rendered)
        assert rendered[0] == "ssh"
        request = json.loads(kwargs["input"])
        return _completed(rendered, stdout=json.dumps(
            _inspection(plan, request["paths"], transfer_missing=True)))

    monkeypatch.setattr(transfer.subprocess, "run", fake_run)
    result = transfer.probe_remote_read_only(plan)
    assert result["status"] == "PASS_REMOTE_READ_ONLY_PROBE_NO_TRANSFER"
    assert result["remote_writes"] == 0 and result["rsync_invoked"] is False
    assert len(calls) == 1 and calls[0][0] == "ssh"


@pytest.mark.parametrize("label", ["checkpoint", "resolved config"])
def test_local_checkpoint_and_config_terminal_symlinks_are_rejected_before_resolve(
    label: str, tmp_path: Path,
) -> None:
    repo = (tmp_path / "repo").resolve()
    target = _file(repo / "real/file.bin", "payload")
    link = repo / "links/file.bin"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    with pytest.raises(transfer.RemoteTransferError, match="cannot be a symlink"):
        transfer._local_file(link, repo_root=repo, label=label,
                             expected_sha256=transfer.sha256_file(target))


def test_receipt_dependency_terminal_symlink_is_rejected_before_resolve(tmp_path: Path) -> None:
    repo = (tmp_path / "repo").resolve()
    target = _immutable(repo / "real/dependency.json", {"ok": True})
    link = repo / "links/dependency.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    with pytest.raises(transfer.RemoteTransferError, match="cannot be a symlink"):
        transfer._receipt_dependency(
            {"path": str(link), "sha256": transfer.sha256_file(target)},
            repo_root=repo, label="matched pair dependency",
        )


def test_local_rsync_relative_absolute_source_lands_at_identical_absolute_suffix(tmp_path: Path) -> None:
    if shutil.which("rsync") is None:
        pytest.skip("local rsync is unavailable")
    source = _file(tmp_path / "source/tree/payload.txt", "payload\n")
    destination = tmp_path / "destination"
    destination.mkdir()
    result = subprocess.run(
        ["rsync", "-r", "-R", str(source), f"{destination}/"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    expected = destination / str(source).lstrip("/")
    assert expected.read_bytes() == source.read_bytes()


def test_code_closure_contains_remote_post_import_and_actual_evaluator_dependencies() -> None:
    required = {
        "scripts/h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder.py",
        "scripts/h1_carrierid_date_lodo_hls_early_v2_terminal_checker.py",
        "scripts/h1_carrierid_date_lodo_hls_fivedate_target_evaluator_preflight.py",
        "scripts/h1_carrierid_date_lodo_hls_target_evaluator_closure.py",
        "scripts/h1_carrierid_date_lodo_hls_terminal_evaluate.py",
        "scripts/h1_carrierid_date_lodo_hls_fivedate_aggregate.py",
        "scripts/h1_carrierid_date_lodo_hls_early_v2_remote5070_replay.py",
        "src/data/h1_carrierid_date_lodo_hls_target.py",
        "src/models/h1_carrierid_date_lodo_hls_module.py",
    }
    assert required.issubset(set(transfer.CODE_FILES))
