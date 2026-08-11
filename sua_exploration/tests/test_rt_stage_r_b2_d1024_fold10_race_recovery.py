"""Synthetic/no-NWB contracts for the isolated RT fold-10 race recovery."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import rt_stage_r_b2_d1024_fold10_race_recovery_checker as checker
from scripts import run_rt_stage_r_b2_d1024_fold10_race_recovery as recovery


def _write(path: Path, value: object, *, immutable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        path.chmod(0o444)
    return path


def _old_failure(tmp_path: Path) -> Path:
    return _write(tmp_path / "old/cell_terminal.json", {
        "schema": checker.OLD_TERMINAL_SCHEMA,
        "status": checker.OLD_TERMINAL_STATUS,
        "fold": 10,
        "seed": 42,
        "arm": "zero4",
        "formal_heldout_opened": False,
    }, immutable=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _successful_recovery(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    old = _old_failure(tmp_path)
    root = tmp_path / "fold10_race_recovery_v1"
    paths = checker.recovery_paths(root)
    _write(paths["preflight"], {"status": "READY_NOT_LAUNCHED"})
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    paths["config"].write_text("calibration_n_trials: 24\n", encoding="utf-8")
    _write(paths["split"], {"outer_loso_fold": 10})
    checkpoint = paths["fit"] / "checkpoints/best_ckpt/epoch_005.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fresh, isolated checkpoint")
    _write(paths["selection"], {
        "config_sha256": _sha(paths["config"]),
        "split_manifest_sha256": _sha(paths["split"]),
        "best_model_path": str(checkpoint.resolve()),
        "best_model_sha256": _sha(checkpoint),
    })
    _write(paths["outer"], {"checkpoint_sha256": _sha(checkpoint)})
    old_binding = checker.validate_old_failure(terminal_path=old)
    files = {
        name: {"path": str(paths[name].resolve()), "sha256": _sha(paths[name])}
        for name in ("preflight", "selection", "config", "split", "outer")
    }
    files["selected_checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint)}
    _write(paths["terminal"], {
        "schema": checker.RECOVERY_TERMINAL_SCHEMA,
        "status": checker.RECOVERY_TERMINAL_STATUS,
        "fold": 10,
        "seed": 42,
        "arm": "zero4",
        "formal_heldout_opened": False,
        "preserved_old_failure": old_binding,
        "freshness": {
            "recovery_root": str(root.resolve()),
            "cell": str(paths["cell"]),
            "old_cell_reused": False,
            "warmstart_forbidden": True,
            "configured_ckpt_path": None,
            "recovery_fit_attempts": 1,
        },
        "files": files,
    }, immutable=True)
    return root, old, paths


def test_plan_is_nonexecuting_preserves_old_failure_and_forbids_warmstart(monkeypatch, tmp_path):
    old = _old_failure(tmp_path)
    root = tmp_path / "fold10_race_recovery_v1"
    monkeypatch.setattr(recovery.supervisor, "_require_program", lambda python: None)

    plan = recovery.plan(recovery_root=root, python=Path("/synthetic/python"), gpu=0, old_terminal_path=old)

    assert plan["mode"] == "plan_only_no_execution"
    assert plan["old_failure_preserved"]["path"] == str(old.resolve())
    assert plan["freshness_contract"]["old_cell_reused"] is False
    assert plan["freshness_contract"]["warmstart_forbidden"] is True
    assert plan["freshness_contract"]["configured_ckpt_path"] is None
    assert "ckpt_path=null" in plan["commands"]["fresh_source_fit_gpu_only_if_execute"]
    assert root.exists() is False
    assert old.stat().st_mode & 0o777 == 0o444


def test_plan_fails_closed_if_recovery_root_exists_or_old_failure_is_mutable(monkeypatch, tmp_path):
    old = _old_failure(tmp_path)
    monkeypatch.setattr(recovery.supervisor, "_require_program", lambda python: None)
    root = tmp_path / "fold10_race_recovery_v1"
    root.mkdir()
    with pytest.raises(FileExistsError, match="refusing resume/reuse"):
        recovery.plan(recovery_root=root, python=Path("/synthetic/python"), gpu=0, old_terminal_path=old)

    root.rmdir()
    old.chmod(0o644)
    with pytest.raises(checker.RecoveryError, match="immutable mode 0444"):
        recovery.plan(recovery_root=root, python=Path("/synthetic/python"), gpu=0, old_terminal_path=old)


def test_recovery_checker_requires_fresh_terminal_source_config_and_checkpoint_hashes(tmp_path):
    root, old, paths = _successful_recovery(tmp_path)
    audited = checker.validate_recovery_terminal(recovery_root=root, old_terminal_path=old)
    assert audited["terminal"]["status"] == checker.RECOVERY_TERMINAL_STATUS
    assert audited["files"]["selected_checkpoint"]["sha256"] == _sha(
        Path(json.loads(paths["selection"].read_text())["best_model_path"])
    )

    # The checker must not accept an old hash binding after any source config
    # mutation, even though the immutable historical failure still exists.
    paths["config"].write_text("calibration_n_trials: 25\n", encoding="utf-8")
    with pytest.raises(checker.RecoveryError, match="config SHA mismatch"):
        checker.validate_recovery_terminal(recovery_root=root, old_terminal_path=old)


def test_recovery_checker_rejects_warmstart_claim_and_old_terminal_reuse(tmp_path):
    root, old, paths = _successful_recovery(tmp_path)
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    paths["terminal"].chmod(0o644)
    terminal["freshness"]["warmstart_forbidden"] = False
    _write(paths["terminal"], terminal, immutable=True)
    with pytest.raises(checker.RecoveryError, match="warmstart_forbidden"):
        checker.validate_recovery_terminal(recovery_root=root, old_terminal_path=old)

    # A successful recovery terminal can never be placed over the historical
    # old-cell path: its unique recovery-cell location is part of the contract.
    with pytest.raises(checker.RecoveryError, match="unique fresh-cell path"):
        checker.validate_recovery_terminal(
            recovery_root=root, terminal_path=old, old_terminal_path=old,
        )


def test_saved_fit_lineage_requires_null_checkpoint_and_a_checkpoint_inside_new_fit(tmp_path):
    paths = checker.recovery_paths(tmp_path / "fresh")
    paths["config"].parent.mkdir(parents=True, exist_ok=True)
    paths["config"].write_text("ckpt_path: null\n", encoding="utf-8")
    checkpoint = paths["fit"] / "checkpoints/best/epoch_001.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"new")
    _write(paths["selection"], {"best_model_path": str(checkpoint)})
    recovery._validate_fresh_fit_lineage(paths)

    paths["config"].write_text("ckpt_path: /old/partial.ckpt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ckpt_path: null"):
        recovery._validate_fresh_fit_lineage(paths)


def test_runner_requires_explicit_execute_flag_and_has_no_implicit_gpu_execution():
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    assert "--execute-recovery" in source
    assert "if not args.execute_recovery" in source
    assert "ckpt_path=null" in source
    assert "warmstart_forbidden" in source
    assert "_assert_launch_isolation(gpu)" in source


def test_execute_checks_launch_isolation_before_creating_recovery_root(monkeypatch, tmp_path):
    old = _old_failure(tmp_path)
    root = tmp_path / "fold10_race_recovery_v1"
    monkeypatch.setattr(recovery.supervisor, "_require_program", lambda python: None)
    monkeypatch.setattr(
        recovery, "_assert_launch_isolation",
        lambda gpu: (_ for _ in ()).throw(RuntimeError("occupied exact fold")),
    )

    with pytest.raises(RuntimeError, match="occupied exact fold"):
        recovery.execute(
            recovery_root=root, python=Path("/synthetic/python"), gpu=0,
            old_terminal_path=old,
        )
    assert root.exists() is False
    assert old.stat().st_mode & 0o777 == 0o444
