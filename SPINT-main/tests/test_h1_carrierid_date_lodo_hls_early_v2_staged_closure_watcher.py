from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import h1_carrierid_date_lodo_hls_early_v2_staged_closure_watcher as watcher
from scripts.h1_carrierid_date_lodo_hls_early_v2_contract import (
    DATES,
    EARLY_TERMINAL_SCHEMA,
    EARLY_TERMINAL_STATUS,
)


def _immutable(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path.resolve()


def _terminals(root: Path) -> dict[str, Path]:
    rows = watcher.source_terminals(root)
    for date, path in rows.items():
        _immutable(path, {
            "schema": EARLY_TERMINAL_SCHEMA,
            "status": EARLY_TERMINAL_STATUS,
            "outer_date": date,
            "arm": "H-LS",
            "scope": {"target_recordings_opened": 0, "target_bytes_read": 0},
            "deployment_updates": {
                "target_optimizer_steps": 0,
                "target_backward_steps": 0,
                "target_model_state_updated": False,
            },
        })
    return rows


def test_missing_local_terminal_waits_without_ssh_or_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "terminals"
    rows = _terminals(root)
    rows[DATES[-1]].chmod(0o644)
    rows[DATES[-1]].unlink()
    monkeypatch.setattr(
        watcher.transfer, "build_plan",
        lambda **_kwargs: pytest.fail("incomplete local grid must not build a transfer plan"),
    )
    monkeypatch.setattr(
        watcher, "_pretransfer_remote_state",
        lambda _plan: pytest.fail("incomplete local grid must not use SSH"),
    )
    state = watcher.poll_once(
        terminal_root=root,
        transfer_receipt=tmp_path / "transfer.json",
        execute=False,
    )
    assert state["status"] == "WAITING_LOCAL_SOURCE_TERMINALS"
    assert state["missing_dates"] == [DATES[-1]]
    assert state["remote_inspected"] is False
    assert state["writes"] == 0 and state["gpu_started"] is False


def test_existing_bad_terminal_fails_closed_instead_of_becoming_waiting(tmp_path: Path) -> None:
    rows = _terminals(tmp_path / "terminals")
    bad = rows[DATES[0]]
    body = json.loads(bad.read_text())
    body["deployment_updates"]["target_backward_steps"] = 1
    bad.chmod(0o644)
    bad.write_text(json.dumps(body), encoding="utf-8")
    bad.chmod(0o444)
    with pytest.raises(watcher.StagedClosureWatcherError, match="not target-closed"):
        watcher.inspect_local_terminals(rows)


def test_complete_local_grid_waits_for_remote_upstream_without_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "terminals"
    _terminals(root)
    plan = {"synthetic": True}
    monkeypatch.setattr(watcher.transfer, "build_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(
        watcher, "_pretransfer_remote_state",
        lambda observed: ({"status": "WAITING_REMOTE_UPSTREAM_PRODUCER_EXIT", "active": [7]}
                          if observed is plan else pytest.fail("unexpected plan")),
    )
    monkeypatch.setattr(
        watcher.transfer, "execute",
        lambda *_args, **_kwargs: pytest.fail("waiting upstream must not transfer"),
    )
    state = watcher.poll_once(
        terminal_root=root,
        transfer_receipt=tmp_path / "transfer.json",
        execute=True,
    )
    assert state["status"] == "WAITING_REMOTE_UPSTREAM_PRODUCER_EXIT"
    assert state["writes"] == 0 and state["gpu_started"] is False


def test_ready_dry_run_neither_transfers_nor_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "terminals"
    _terminals(root)
    monkeypatch.setattr(watcher.transfer, "build_plan", lambda **_kwargs: {"plan": 1})
    monkeypatch.setattr(
        watcher, "_pretransfer_remote_state",
        lambda _plan: {"status": "READY_FOR_VERIFIED_NO_OVERWRITE_TRANSFER"},
    )
    monkeypatch.setattr(
        watcher.transfer, "execute",
        lambda *_args, **_kwargs: pytest.fail("dry run must not transfer"),
    )
    monkeypatch.setattr(
        watcher, "launch_remote_replay",
        lambda *_args, **_kwargs: pytest.fail("dry run must not launch tmux"),
    )
    state = watcher.poll_once(
        terminal_root=root,
        transfer_receipt=tmp_path / "transfer.json",
        execute=False,
    )
    assert state["status"] == "READY_FOR_VERIFIED_NO_OVERWRITE_TRANSFER"
    assert state["writes"] == 0 and state["gpu_started"] is False


def test_existing_transfer_and_tmux_are_observed_without_relaunch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "terminals"
    _terminals(root)
    receipt = tmp_path / "transfer.json"
    receipt.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        watcher, "inspect_remote_replay",
        lambda *_args, **_kwargs: {
            "status": "OBSERVING_EXISTING_REMOTE5070_TMUX_OWNER",
            "tmux": {"exists": True},
        },
    )
    monkeypatch.setattr(
        watcher, "launch_remote_replay",
        lambda *_args, **_kwargs: pytest.fail("existing owner must never be relaunched"),
    )
    state = watcher.poll_once(
        terminal_root=root, transfer_receipt=receipt, execute=True,
    )
    assert state["status"] == "OBSERVING_EXISTING_REMOTE5070_TMUX_OWNER"
    assert state["writes"] == 0 and state["gpu_started"] is False


def test_ready_existing_transfer_launches_exactly_one_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "terminals"
    _terminals(root)
    receipt = tmp_path / "transfer.json"
    receipt.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        watcher, "inspect_remote_replay",
        lambda *_args, **_kwargs: {
            "status": "READY_TO_START_UNIQUE_REMOTE5070_TMUX_OWNER",
        },
    )
    launches: list[tuple[Path, str]] = []

    def launch(path: Path, *, owner: str) -> dict[str, Any]:
        launches.append((Path(path), owner))
        return {"status": "LAUNCHED_UNIQUE_REMOTE5070_TMUX_OWNER", "owner": owner}

    monkeypatch.setattr(watcher, "launch_remote_replay", launch)
    state = watcher.poll_once(
        terminal_root=root, transfer_receipt=receipt, execute=True,
    )
    assert state["status"] == "LAUNCHED_UNIQUE_REMOTE5070_TMUX_OWNER"
    assert launches == [(receipt.resolve(), watcher.TMUX_OWNER)]
    assert state["writes"] == 1 and state["gpu_started"] is True


def _transfer_receipt(path: Path) -> Path:
    return _immutable(path, {
        "schema": watcher.transfer.TRANSFER_SCHEMA,
        "status": watcher.transfer.TRANSFER_STATUS,
        "remote": watcher.transfer.REMOTE,
        "identical_absolute_repo_root": str(watcher.ROOT),
        "remote_upstream_aggregate": {
            "path": str(watcher.ROOT / "synthetic_upstream.json"),
            "sha256": "a" * 64,
            "five_date_complete": True,
        },
    })


def _gate_inspection(receipt: Path, requested_paths: list[str]) -> dict[str, Any]:
    files = {path: {"state": "MISSING"} for path in requested_paths}
    files[str(receipt)] = {
        "state": "FILE",
        "sha256": watcher.sha256_file(receipt),
        "size": receipt.stat().st_size,
        "mode": 0o444,
    }
    return {
        "repo_root": str(watcher.ROOT),
        "active_main_chain_producers": [],
        "upstream_gate": {"ready": True, "sha256": "a" * 64},
        "files": files,
    }


def test_remote_tmux_state_is_idempotent_observation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    receipt = _transfer_receipt(tmp_path / "transfer.json")

    def inspect(*, paths, **_kwargs):
        return _gate_inspection(receipt, list(paths))

    monkeypatch.setattr(watcher.transfer, "_remote_inspect", inspect)
    monkeypatch.setattr(watcher, "_remote_owner_inspect", lambda **kwargs: {
        "tmux": {"exists": True, "panes": ["123 0 python"]},
        "active_replay_processes": [{"pid": 123}],
        "files": {path: {"state": "MISSING"} for path in kwargs["paths"]},
    })
    state = watcher.inspect_remote_replay(receipt)
    assert state["status"] == "OBSERVING_EXISTING_REMOTE5070_TMUX_OWNER"


def test_orphaned_replay_claim_without_tmux_or_terminal_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    receipt = _transfer_receipt(tmp_path / "transfer.json")
    monkeypatch.setattr(
        watcher.transfer, "_remote_inspect",
        lambda *, paths, **_kwargs: _gate_inspection(receipt, list(paths)),
    )

    def owner_state(**kwargs):
        files = {path: {"state": "MISSING"} for path in kwargs["paths"]}
        claim = next(path for path in kwargs["paths"] if path.endswith(".claim.json"))
        files[claim] = {"state": "FILE", "mode": 0o444}
        return {"tmux": {"exists": False, "panes": []},
                "active_replay_processes": [], "files": files}

    monkeypatch.setattr(watcher, "_remote_owner_inspect", owner_state)
    with pytest.raises(watcher.StagedClosureWatcherError, match="orphaned one-shot"):
        watcher.inspect_remote_replay(receipt)
