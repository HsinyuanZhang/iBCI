from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/orchestrate_paired_view_c1_shared_zero4_postcompletion.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("zero4_postcompletion_orchestrator_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


orch = _load_module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    adapter = tmp_path / "adapter.json"
    adapter.write_text("{}\n", encoding="utf-8")
    adapter.chmod(0o444)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"fixed":true}\n', encoding="utf-8")
    python = tmp_path / "python"
    python.write_bytes(b"fixture executable\n")
    python.chmod(0o755)
    run_dirs = {}
    rows = {}
    for seed in orch.SEEDS:
        run_dir = tmp_path / orch.RUN_NAME.format(seed=seed)
        (run_dir / "epoch_ckpts").mkdir(parents=True)
        metadata = run_dir / "run_metadata.json"
        terminal = run_dir / "epoch_ckpts/epoch_011.ckpt"
        metadata.write_text("{}\n", encoding="utf-8")
        terminal.write_bytes(b"checkpoint\n")
        metadata.chmod(0o444)
        terminal.chmod(0o444)
        run_dirs[seed] = run_dir
        rows[str(seed)] = {
            "run_metadata": {"canonical_path": str(metadata.resolve())},
            "terminal_checkpoint": {"canonical_path": str(terminal.resolve())},
        }
    verified = {
        "kind": "v3_external_v7_adapter",
        "schema": "fixture-v3-adapter",
        "status": "fixture-completed",
        "rows": rows,
    }
    monkeypatch.setattr(orch, "_verify_completion_receipt", lambda _path: verified)
    monkeypatch.setattr(orch, "EXPECTED_MANIFEST_SHA256", _sha(manifest))
    monkeypatch.setattr(orch, "_completion_run_dir", lambda _verified, seed: run_dirs[seed])
    args = argparse.Namespace(
        matrix_completion_receipt=adapter,
        train_val_manifest=manifest,
        zero4_result_root=tmp_path / "zero4",
        t4_ts4_receipt=tmp_path / "t4_ts4_receipt.json",
        t4_ts4_result_root=tmp_path / "t4_ts4",
        aggregate_out=tmp_path / "three_arm.json",
        python=python,
        device="cuda",
    )
    return args, run_dirs


def _successful_runner(calls: list[list[str]], *, fail_cell: tuple[int, str] | None = None):
    def runner(command):
        command = list(command)
        calls.append(command)
        if Path(command[1]) == orch.EVALUATOR:
            seed = int(Path(command[command.index("--run-dir") + 1]).name.rsplit("_s", 1)[1])
            view = command[command.index("--signal-view") + 1]
            if fail_cell == (seed, view):
                return 19
            artifact = Path(command[command.index("--out-path") + 1])
            artifact.parent.mkdir(parents=True, exist_ok=True)
            # Deliberately contains a conspicuous score.  The orchestrator must
            # treat these bytes as opaque and never print or JSON-parse them.
            artifact.write_text(
                json.dumps({"variant_score": 12345.6789, "seed": seed, "view": view}) + "\n",
                encoding="utf-8",
            )
            return 0
        assert Path(command[1]) == orch.AGGREGATOR
        output = Path(command[command.index("--out") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"status":"fixture"}\n', encoding="utf-8")
        output.chmod(0o444)
        return 0

    return runner


def test_invalid_or_missing_adapter_prevents_any_evaluator_or_aggregator_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    missing = tmp_path / "missing_adapter.json"
    args = argparse.Namespace(matrix_completion_receipt=missing)

    def reject(_path):
        raise orch.completion_bridge.TerminalAdapterError("adapter missing")

    monkeypatch.setattr(orch, "_verify_completion_receipt", reject)
    monkeypatch.setattr(orch, "_run_quiet", lambda command: calls.append(list(command)) or 0)
    sys.modules.pop("eval_paired_view_c1_shared_zero4_terminal", None)
    with pytest.raises(orch.completion_bridge.TerminalAdapterError, match="adapter missing"):
        orch.run(args)
    assert calls == []
    assert "eval_paired_view_c1_shared_zero4_terminal" not in sys.modules


def test_partial_continuation_uses_commits_and_aggregates_only_after_six_of_six(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    first_calls: list[list[str]] = []
    monkeypatch.setattr(
        orch,
        "_run_quiet",
        _successful_runner(first_calls, fail_cell=(43, "sua")),
    )
    with pytest.raises(orch.OrchestrationError, match=r"seed43/sua returncode=19"):
        orch.run(args)
    assert sum(Path(call[1]) == orch.EVALUATOR for call in first_calls) == 3
    assert all(Path(call[1]) != orch.AGGREGATOR for call in first_calls)
    assert len(list((args.zero4_result_root / "commits").glob("*.commit.json"))) == 2

    second_calls: list[list[str]] = []
    monkeypatch.setattr(orch, "_run_quiet", _successful_runner(second_calls))
    report = orch.run(args)
    assert report["completed_cell_count"] == 6
    assert report["parent_parsed_cell_score_artifacts"] is False
    evaluator_calls = [call for call in second_calls if Path(call[1]) == orch.EVALUATOR]
    assert len(evaluator_calls) == 4
    assert Path(second_calls[-1][1]) == orch.AGGREGATOR
    commits = list((args.zero4_result_root / "commits").glob("*.commit.json"))
    assert len(commits) == 6
    # JSON objects are not hashable; normalize to canonical strings to prove
    # all six cells bind one exact path/SHA/schema/status/kind identity.
    identities = {
        json.dumps(
            json.loads(path.read_text(encoding="utf-8"))["completion_identity"],
            sort_keys=True,
        )
        for path in commits
    }
    assert len(identities) == 1


def test_existing_committed_artifact_tamper_fails_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        orch,
        "_run_quiet",
        _successful_runner(calls, fail_cell=(42, "pseudo_mua")),
    )
    with pytest.raises(orch.OrchestrationError):
        orch.run(args)
    artifact = orch._artifact_path(args.zero4_result_root, seed=42, view="sua")
    artifact.chmod(0o644)
    artifact.write_text('{"variant_score":-999}\n', encoding="utf-8")
    artifact.chmod(0o444)
    calls.clear()
    with pytest.raises(orch.OrchestrationError, match="commit binding drift"):
        orch.run(args)
    assert calls == []


@pytest.mark.parametrize("orphan_kind", ["artifact", "commit"])
def test_orphan_artifact_or_commit_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, orphan_kind: str
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    artifact = orch._artifact_path(args.zero4_result_root, seed=42, view="sua")
    commit = orch._commit_path(args.zero4_result_root, seed=42, view="sua")
    target = artifact if orphan_kind == "artifact" else commit
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}\n", encoding="utf-8")
    target.chmod(0o444)
    monkeypatch.setattr(
        orch,
        "_run_quiet",
        lambda _command: pytest.fail("orphan must fail before subprocess"),
    )
    with pytest.raises(orch.OrchestrationError, match="orphan artifact/commit"):
        orch.run(args)


def test_evaluator_failure_never_aggregates_and_reports_no_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        orch,
        "_run_quiet",
        _successful_runner(calls, fail_cell=(42, "sua")),
    )
    with pytest.raises(orch.OrchestrationError) as caught:
        orch.run(args)
    assert str(caught.value) == "terminal evaluator failed for seed42/sua returncode=19"
    assert all(Path(call[1]) != orch.AGGREGATOR for call in calls)
    captured = capsys.readouterr()
    assert "12345.6789" not in captured.out + captured.err


def test_subprocess_stdio_is_devnull(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(orch.subprocess, "run", fake_run)
    assert orch._run_quiet(["fixture", "command"]) == 7
    assert observed == {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "check": False,
    }


def test_controlled_source_drift_during_cells_stops_without_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    baseline = orch._controlled_sources()
    source_checks = 0

    def changing_sources():
        nonlocal source_checks
        source_checks += 1
        if source_checks <= 2:  # initial pin + first-cell immediate recheck
            return baseline
        drifted = json.loads(json.dumps(baseline))
        drifted["orchestrator"]["sha256"] = "0" * 64
        return drifted

    calls: list[list[str]] = []
    monkeypatch.setattr(orch, "_controlled_sources", changing_sources)
    monkeypatch.setattr(orch, "_run_quiet", _successful_runner(calls))
    with pytest.raises(
        orch.OrchestrationError,
        match=r"controlled source drift before terminal evaluator seed42/pseudo_mua",
    ):
        orch.run(args)
    assert [Path(call[1]) for call in calls] == [orch.EVALUATOR]
    assert len(list((args.zero4_result_root / "commits").glob("*.commit.json"))) == 1
    assert all(Path(call[1]) != orch.AGGREGATOR for call in calls)


def test_controlled_source_drift_at_aggregate_boundary_refuses_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    baseline = orch._controlled_sources()
    source_checks = 0

    def changing_sources():
        nonlocal source_checks
        source_checks += 1
        # One initial pin plus one recheck before each of six evaluator cells.
        if source_checks <= 7:
            return baseline
        drifted = json.loads(json.dumps(baseline))
        drifted["three_arm_aggregator"]["size_bytes"] += 1
        return drifted

    calls: list[list[str]] = []
    monkeypatch.setattr(orch, "_controlled_sources", changing_sources)
    monkeypatch.setattr(orch, "_run_quiet", _successful_runner(calls))
    with pytest.raises(
        orch.OrchestrationError, match="controlled source drift before three-arm aggregation"
    ):
        orch.run(args)
    assert sum(Path(call[1]) == orch.EVALUATOR for call in calls) == 6
    assert all(Path(call[1]) != orch.AGGREGATOR for call in calls)
    assert len(list((args.zero4_result_root / "commits").glob("*.commit.json"))) == 6


def test_write_once_aggregate_refuses_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    args.aggregate_out.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(
        orch,
        "_run_quiet",
        lambda _command: pytest.fail("no subprocess may run when aggregate exists"),
    )
    with pytest.raises(FileExistsError, match="write-once three-arm aggregate exists"):
        orch.run(args)


def test_aggregator_failure_preserves_six_commits_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _fixture(tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def runner(command):
        command = list(command)
        if Path(command[1]) == orch.AGGREGATOR:
            calls.append(command)
            return 23
        return _successful_runner(calls)(command)

    monkeypatch.setattr(orch, "_run_quiet", runner)
    with pytest.raises(orch.OrchestrationError, match="aggregator failed returncode=23"):
        orch.run(args)
    commits = sorted((args.zero4_result_root / "commits").glob("*.commit.json"))
    assert len(commits) == 6
    hashes = {path: _sha(path) for path in commits}

    # A continuation must schedule no evaluator cells.  The existing six
    # commits remain byte-identical while aggregation is retried once.
    retry_calls: list[list[str]] = []
    monkeypatch.setattr(orch, "_run_quiet", _successful_runner(retry_calls))
    orch.run(args)
    assert [Path(call[1]) for call in retry_calls] == [orch.AGGREGATOR]
    assert hashes == {path: _sha(path) for path in commits}
