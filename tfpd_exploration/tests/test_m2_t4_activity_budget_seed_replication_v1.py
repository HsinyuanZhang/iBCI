from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tfpd_exploration.src.m2_t4_activity_budget_seed_replication_v1 import core, plan, physical


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_seed_grid_and_checkpoint_literals_are_complete() -> None:
    assert plan.SEEDS == (42, 43, 44)
    assert set(plan.CHECKPOINTS) == set(plan.SEEDS)
    assert len({value[1] for value in plan.CHECKPOINTS.values()}) == 3
    assert all(len(value[1]) == 64 for value in plan.CHECKPOINTS.values())


def test_frozen_checkpoint_manifests_and_normalizers_match() -> None:
    for seed in plan.SEEDS:
        path, digest = physical._checkpoint(REPO_ROOT, seed)
        assert path.name == "best.ckpt"
        assert digest == plan.CHECKPOINTS[seed][1]


def test_parent_has_exact_seed42_grid() -> None:
    rows, digest = physical._load_parent(REPO_ROOT)
    assert digest == plan.PARENT_SCORE_SHA256
    assert len(rows) == 65
    assert {key[2] for key in rows} == {
        "ridge_static_m30",
        "ridge_static_m10",
        "ridge_activity30_m10",
        "ridge_static_m4",
        "ridge_activity30_m4",
    }


def test_aggregation_is_equal_key_and_paired() -> None:
    candidate = {"b": 0.7, "a": 0.2, "c": 0.1}
    reference = {"a": 0.1, "b": 0.8, "c": -0.2}
    result = core.paired(candidate, reference)
    assert result["positive"] == 2
    assert result["mean_delta"] == pytest.approx((0.1 - 0.1 + 0.3) / 3)
    summary = core.summarize(candidate)
    assert summary["mean"] == pytest.approx(1.0 / 3.0)
    with pytest.raises(core.ReplicationError):
        core.paired(candidate, {"a": 0.1})


def test_atomic_publisher_is_terminal_pair_only(tmp_path: Path) -> None:
    root = tmp_path / "replication"
    physical._publish(root, {"schema": plan.SCHEMA, "status": "TERMINAL"})
    assert sorted(path.name for path in root.iterdir()) == ["score.json", "score.json.sha256"]
    assert root.stat().st_mode & 0o777 == 0o555
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in root.iterdir())
    with pytest.raises(core.ReplicationError):
        physical._publish(root, {})


def test_dry_cli_is_inert_and_torch_free() -> None:
    script = REPO_ROOT / "tfpd_exploration/scripts/run_m2_t4_activity_budget_seed_replication_v1.py"
    command = [sys.executable, str(script)]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload == {
        "new_forward_rows": 130,
        "reused_rows": 65,
        "schema": "m2_t4_activity_budget_seed_replication_v1_dry",
        "seeds": [42, 43, 44],
        "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
    }
    probe = (
        "import runpy,sys; "
        f"runpy.run_path({str(script)!r}, run_name='__main__'); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
