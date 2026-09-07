"""Tests for unlaunched GPU experiment contracts, preflights, runners, and aggregators."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUA = REPO / "sua_exploration"
PY = "/home/xinyuan/miniconda3/envs/spint/bin/python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_shell(cmd: list[str], *, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


@pytest.mark.parametrize(
    ("runner", "contract", "extra_env"),
    [
        (
            SUA / "scripts/run_a2_matched_correspondence_one_cell.sh",
            SUA / "docs/A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md",
            {"CELL": "within_t4", "SEED": "42", "GPU": "0"},
        ),
        (
            SUA / "scripts/run_a10_no_backprop_cost_one_cell.sh",
            SUA / "docs/A10_NO_BACKPROP_COST_CONTRACT_20260812.md",
            {"ARM": "readout_probe", "SEED": "42", "GPU": "0"},
        ),
        (
            SUA / "scripts/run_a14_b3t_t4_efficiency_completion_one_cell.sh",
            SUA / "docs/A14_B3T_T4_EFFICIENCY_COMPLETION_CONTRACT_20260812.md",
            {"GPU": "0"},
        ),
    ],
)
def test_runner_refuses_without_authorization(
    runner: Path,
    contract: Path,
    extra_env: dict[str, str],
) -> None:
    env = {**extra_env, "PYTHON_BIN": PY}
    proc = run_shell(["bash", str(runner), "--launch"], env=env)
    assert proc.returncode == 3
    assert "Refusing" in proc.stderr or "Refusing" in proc.stdout

    dry = run_shell(["bash", str(runner), "--dry-run"], env=env)
    assert dry.returncode == 0
    assert sha256_file(contract) in dry.stdout


def test_a2_runner_contract_sha_matches_document() -> None:
    contract = SUA / "docs/A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md"
    proc = run_shell(
        ["bash", str(SUA / "scripts/run_a2_matched_correspondence_one_cell.sh"), "--dry-run"],
        env={"CELL": "within_z4", "SEED": "42", "GPU": "0", "PYTHON_BIN": PY},
    )
    assert proc.returncode == 0
    assert sha256_file(contract) in proc.stdout


def test_a10_preflight_fail_closed_on_nonempty_root(tmp_path: Path) -> None:
    result_root = tmp_path / "a10"
    result_root.mkdir()
    (result_root / "gradient_free_s42.json").write_text("{}")
    proc = subprocess.run(
        [
            PY,
            str(SUA / "scripts/a10_no_backprop_cost_preflight.py"),
            "--result-root",
            str(result_root),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert any(
        b.get("code") == "NONEMPTY_OUTPUT_ROOT"
        for b in payload.get("implementation_blockers", [])
    )


def test_a2_preflight_fail_closed_on_nonempty_root(tmp_path: Path) -> None:
    result_root = tmp_path / "a2"
    result_root.mkdir()
    (result_root / "within_z4_s42.json").write_text("{}")
    proc = subprocess.run(
        [
            PY,
            str(SUA / "scripts/a2_matched_correspondence_preflight.py"),
            "--result-root",
            str(result_root),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert any(
        b.get("code") == "NONEMPTY_OUTPUT_ROOT"
        for b in payload.get("implementation_blockers", [])
    )


def test_a14_preflight_reports_missing_cell() -> None:
    proc = subprocess.run(
        [PY, str(SUA / "scripts/a14_b3t_t4_efficiency_completion_preflight.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    payload = json.loads(proc.stdout)
    assert payload["missing_cell"] == "b3t_ts4_s42"
    assert payload["expected_new_gpu_cells"] == 1
    if proc.returncode == 0:
        assert payload["status"] == "READY_FOR_SEPARATE_GPU_REVIEW"
    else:
        assert payload["status"] == "STOP_A14_PREFLIGHT_BLOCKERS"


def _minimal_a2_epoch_artifact(
    path: Path,
    *,
    seed: int,
    side: str,
    sessions: list[str],
) -> None:
    per_epoch = {}
    for epoch in range(5, 13):
        per_epoch[str(epoch)] = {
            "per_session_r2": {s: 0.3 + 0.01 * epoch for s in sessions},
            "mean_r2": 0.3 + 0.01 * epoch,
        }
    metadata = {
        "schema_version": 1,
        "variant": "B3S",
        "seed": seed,
        "side_features": {"group": side},
        "training": {"loss_mode": "task_only"},
    }
    meta_path = path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata))
    artifact = {
        "schema_version": 1,
        "seed": seed,
        "variant": "B3S",
        "no_test_files_evaluated": True,
        "run_metadata_path": str(meta_path),
        "protocol": {
            "calibration_n": 30,
            "pool_size": 30,
            "total_epochs": 12,
            "epoch_window": list(range(5, 13)),
        },
        "per_epoch": per_epoch,
    }
    path.write_text(json.dumps(artifact))


def test_a2_aggregator_rejects_incomplete_matrix(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    proc = subprocess.run(
        [
            PY,
            str(SUA / "scripts/aggregate_a2_matched_correspondence.py"),
            "--result-dir",
            str(result_dir),
            "--contract",
            str(SUA / "docs/A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md"),
            "--out",
            str(tmp_path / "aggregate.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode != 0


def test_sealed_session_guard_unit() -> None:
    import sys

    sys.path.insert(0, str(SUA))
    from mc_maze.gpu_contract_common import assert_sessions_not_sealed

    with pytest.raises(ValueError, match="sealed"):
        assert_sessions_not_sealed(["sub-C_ses-CO-20151113"])


def test_b1_aggregator_rejects_incomplete_matrix(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    proc = subprocess.run(
        [
            PY,
            str(SUA / "scripts/aggregate_b1_m2_t4_loss_mode.py"),
            "--result-dir",
            str(result_dir),
            "--contract",
            str(SUA / "docs/B1_M2_T4_LOSS_MODE_CONTRACT_20260812.md"),
            "--out",
            str(tmp_path / "aggregate.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(SUA),
    )
    assert proc.returncode != 0


def test_eval_cell_scripts_refuse_launch_without_auth() -> None:
    proc = subprocess.run(
        [
            PY,
            str(SUA / "scripts/a10_no_backprop_cost_eval_cell.py"),
            "--arm",
            "full_finetune",
            "--seed",
            "42",
            "--source-run-dir",
            str(SUA / "checkpoints"),
            "--out-path",
            "/tmp/a10_out.json",
            "--launch",
        ],
        capture_output=True,
        text=True,
        cwd=str(SUA),
    )
    assert proc.returncode == 3

    proc2 = subprocess.run(
        [
            PY,
            str(SUA / "scripts/a2_matched_correspondence_score_external.py"),
            "--run-dir",
            str(SUA / "checkpoints"),
            "--cell",
            "cross_t4",
            "--seed",
            "42",
            "--out-path",
            "/tmp/a2_out.json",
            "--launch",
        ],
        capture_output=True,
        text=True,
        cwd=str(SUA),
    )
    assert proc2.returncode == 3


def test_contract_documents_exist() -> None:
    for name in (
        "A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md",
        "A10_NO_BACKPROP_COST_CONTRACT_20260812.md",
        "A14_B3T_T4_EFFICIENCY_COMPLETION_CONTRACT_20260812.md",
        "B1_M2_T4_LOSS_MODE_CONTRACT_20260812.md",
    ):
        path = SUA / "docs" / name
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "Authorizes no GPU" in text or "authorizes nothing" in text.lower()
