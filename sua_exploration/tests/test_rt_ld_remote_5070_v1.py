"""Adversarial/static tests for the isolated RT L-D 5070 Ti transport."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/rt_ld_remote_5070_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("rt_ld_remote_5070_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_scientific_order_and_commands_do_not_change_seed_fold_or_arms(tmp_path: Path):
    m = _module()
    assert m.ARM_ORDER == (
        "rt_ld_a0_full_m24_fold0_seed42",
        "rt_ld_g_full_m24_fold0_seed42",
        "rt_ld_g_xls_m24_fold0_seed42",
    )
    command = m._train_command(
        Path("/opt/spint/bin/python"), m.ARM_ORDER[2], tmp_path / "03_g_xls",
        Path("/data/rt"), Path("/data/teacher.ckpt"),
    )
    assert command[2:5] == [
        "experiment=rt_ld_g_xls_m24_fold0_seed42", "seed=42",
        f"hydra.run.dir={(tmp_path / '03_g_xls').resolve()}",
    ]
    assert "data.data_dir=/data/rt" in command
    assert "paths.teacher_ckpt_path=/data/teacher.ckpt" in command
    assert not any("gate" in token.lower() or "r2" in token.lower() for token in command)


def test_run_root_is_fresh_direct_isolated_child_and_reuse_fails(tmp_path: Path):
    m = _module()
    stage = tmp_path / "stage"
    stage.mkdir()
    base = stage / m.RESULT_BASE_RELATIVE / "runs"
    candidate = base / "fold0_seed42_20260810"
    root, observed_base = m._validate_run_root(stage, candidate)
    assert root == candidate.resolve() and observed_base == base.resolve()
    with pytest.raises(ValueError, match="direct child"):
        m._validate_run_root(stage, base / "nested" / "escape")
    candidate.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        m._validate_run_root(stage, candidate)


def test_binding_validator_rejects_missing_symlink_and_byte_drift(tmp_path: Path):
    m = _module()
    root = tmp_path / "stage"
    root.mkdir()
    file = root / "sealed.txt"
    file.write_bytes(b"correct")
    digest = m.hashlib.sha256(b"correct").hexdigest()
    assert m._validate_bindings(root, {"sealed.txt": digest}) == {"sealed.txt": digest}
    file.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash drift"):
        m._validate_bindings(root, {"sealed.txt": digest})
    file.unlink()
    target = root / "target"
    target.write_bytes(b"correct")
    file.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        m._validate_bindings(root, {"sealed.txt": digest})


def test_optimizer_drift_strictly_extends_v2_and_rejects_replacement_drift() -> None:
    m = _module()
    receipt = m.V8.V4._validate_v2_anchor()
    base = m._json(m.V8.V4.DRIFT)
    supplement = m._json(m.OPTIMIZER_DRIFT)
    m._validate_optimizer_drift(receipt, base, supplement)

    mutated = json.loads(json.dumps(supplement))
    mutated["artifact_drift"][
        "streaming_calibration_exp/src/models/rt_ld_streaming_module.py"
    ]["replacement_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="optimizer replacement mismatch"):
        m._validate_optimizer_drift(receipt, base, mutated)


def test_dataset_requires_exactly_15_nonempty_unique_rt_sessions(tmp_path: Path):
    m = _module()
    for index in range(15):
        (tmp_path / f"sub-C_ses-RT-20{index:06d}_behavior+ecephys.nwb").write_bytes(b"nwb")
    result = m._validate_data_dir(tmp_path)
    assert result["session_count"] == 15 and result["total_bytes"] == 45
    (tmp_path / "sub-C_ses-RT-20999999_behavior+ecephys.nwb").write_bytes(b"nwb")
    with pytest.raises(ValueError, match="exactly 15"):
        m._validate_data_dir(tmp_path)
    (tmp_path / "sub-C_ses-RT-20999999_behavior+ecephys.nwb").unlink()
    next(tmp_path.glob("*.nwb")).write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        m._validate_data_dir(tmp_path)


def test_gpu_inventory_is_exactly_one_5070ti_and_compute_parser_fails_closed():
    m = _module()
    rows = m._parse_gpu_inventory("0, GPU-abc, NVIDIA GeForce RTX 5070 Ti Laptop GPU\n")
    assert rows == [{"index": "0", "uuid": "GPU-abc", "name": "NVIDIA GeForce RTX 5070 Ti Laptop GPU"}]
    with pytest.raises(ValueError, match="inventory"):
        m._parse_gpu_inventory("0, missing-name\n")
    assert m._parse_compute_pids("No running processes found\n") == []
    assert m._parse_compute_pids("123\n456\n") == [123, 456]
    with pytest.raises(ValueError, match="compute PID"):
        m._parse_compute_pids("not-a-pid\n")


def test_existing_ld_process_detection_ignores_only_wrapper_ancestry():
    m = _module()
    rows = [
        (10, "python rt_ld_remote_5070_v1.py --mode run"),
        (11, "python src/train.py experiment=rt_ld_g_full_m24_fold0_seed42"),
        (12, "python src/rt_clean_nested_loso_eval.py --config x"),
        (13, "python unrelated.py"),
    ]
    found = m._ld_processes_from_rows(rows, {10})
    assert [row["pid"] for row in found] == [11, 12]
    assert m._ld_processes_from_rows([(10, rows[0][1])], {10}) == []


def test_runtime_release_requires_both_empty_samples_and_no_competing_process(monkeypatch):
    m = _module()
    monkeypatch.setattr(m, "_gpu_inventory", lambda: [{"index": "0", "uuid": "x", "name": "5070 Ti"}])
    probes = iter([[], []])
    monkeypatch.setattr(m, "_compute_pids", lambda: next(probes))
    monkeypatch.setattr(m, "_existing_ld_processes", lambda: [])
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    assert m._runtime_release(0)["compute_pids_second"] == []

    probes = iter([[], [999]])
    monkeypatch.setattr(m, "_compute_pids", lambda: next(probes))
    with pytest.raises(RuntimeError, match="not empty"):
        m._runtime_release(0)


def test_readiness_writer_is_append_only_immutable_and_rejects_nan(tmp_path: Path):
    m = _module()
    path = tmp_path / "receipt.json"
    m._write_immutable_json(path, {"schema": "x", "score_present": False})
    assert path.stat().st_mode & 0o777 == 0o444
    assert json.loads(path.read_text()) == {"schema": "x", "score_present": False}
    with pytest.raises(FileExistsError, match="already exists"):
        m._write_immutable_json(path, {"schema": "replacement"})
    bad = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        m._write_immutable_json(bad, {"bad": float("nan")})
    assert not bad.exists()


def test_cli_run_requires_all_explicit_transport_paths():
    m = _module()
    parser = m._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "run"])
