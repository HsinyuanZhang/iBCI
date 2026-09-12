from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

WORKSPACE = Path(__file__).resolve().parents[4]
ROOT = WORKSPACE / "btransform_unified_v2"
sys.path[:0] = [str(ROOT), str(ROOT / "learnable_recency_v1" / "src"),
                str(WORKSPACE / "btransform_unified_v1" / "src")]
import btransform_unified_v2
btransform_unified_v2.__path__ = [str(ROOT), str(ROOT / "src" / "btransform_unified_v2")]

from btransform_unified_v2.dandi688_bench_v2.flat_extension import run_campaign as campaign


def _plan(tmp_path: Path, *, seed42_only: bool = False) -> tuple[Path, dict]:
    root = tmp_path / "flat"
    return root, campaign.create_plan(root, seed42_only=seed42_only)


def test_plan_freezes_registered_default_ten_cell_scope(tmp_path: Path) -> None:
    root, plan = _plan(tmp_path)
    assert plan["status"] == campaign.STATUS
    assert len(plan["main_cells"]) == 6
    assert len(plan["supplemental_cells"]) == 4
    assert plan["flat_final_cells"] == plan["planned_cells"]
    assert [task["cell"] for task in plan["training_tasks"][:5]] == [
        "full_flat_sua", "activity_flat_sua", "raw_set_flat_sua", "full_flat_sua_s43", "full_flat_sua_s44",
    ]
    assert plan["budget"] == {"segments": 24, "updates_per_segment": 3165, "batch": 32, "total_steps": 75960}
    assert campaign._load_plan(root)["sha256"] == plan["sha256"]


def test_seed42_only_plan_is_exactly_six_cells(tmp_path: Path) -> None:
    _root, plan = _plan(tmp_path, seed42_only=True)
    assert len(plan["planned_cells"]) == 6
    assert plan["supplemental_cells"] == []
    assert {task["seed"] for task in plan["training_tasks"]} == {42}


def test_worker_argv_is_module_based_and_representation_scoped(tmp_path: Path) -> None:
    root, _plan_value = _plan(tmp_path, seed42_only=True)
    argv = campaign._worker_command(root, "pmua", 7)
    assert argv[:3] == [campaign.sys.executable, "-m", "btransform_unified_v2.dandi688_bench_v2.flat_extension.run_campaign"]
    assert argv[-4:] == ["--representation", "pmua", "--gpu", "7"]


def test_plan_self_hash_rejects_tampering(tmp_path: Path) -> None:
    root, _plan_value = _plan(tmp_path)
    path = root / "plan.json"
    changed = json.loads(path.read_text())
    changed["main_cells"].append("unregistered")
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        campaign._load_plan(root)


def test_rehashed_scientific_plan_tampering_is_still_rejected(tmp_path: Path) -> None:
    root, _plan_value = _plan(tmp_path)
    path = root / "plan.json"
    changed = json.loads(path.read_text())
    changed["budget"]["batch"] = 16
    changed["sha256"] = campaign._digest({key: value for key, value in changed.items() if key != "sha256"})
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="scientific scope"):
        campaign._load_plan(root)


def test_worker_stops_after_first_failed_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _plan_value = _plan(tmp_path, seed42_only=True)
    calls: list[list[str]] = []

    def failed(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(pid=999, wait=lambda: 9)

    monkeypatch.setattr(campaign.subprocess, "Popen", failed)
    result = campaign.worker(root, representation="sua", gpu=0)
    assert result["status"] == "FAILED"
    assert len(calls) == 1
    assert result["tasks"][0]["status"] == "FAILED"
    assert result["tasks"][0]["child_pid"] == 999
    assert "exception" in result


def test_worker_source_drift_is_recorded_as_failed_before_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _plan_value = _plan(tmp_path, seed42_only=True)
    monkeypatch.setattr(campaign, "_verify_plan_dependencies", lambda _plan: (_ for _ in ()).throw(ValueError("source drift")))
    result = campaign.worker(root, representation="sua", gpu=0)
    assert result["status"] == "FAILED"
    assert result["tasks"] == []
    assert result["phase"] == "training"


def test_start_uses_detached_representation_workers_and_exclusive_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _plan_value = _plan(tmp_path, seed42_only=True)
    monkeypatch.setattr(campaign, "_cuda_available", lambda: 4)
    pids = iter((101, 102))

    def spawned(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(pid=next(pids))

    monkeypatch.setattr(campaign.subprocess, "Popen", spawned)
    result = campaign.start(root, gpus=(2, 3))
    assert result["status"] == "RUNNING"
    assert result["children"]["sua"]["gpu"] == 2
    assert result["children"]["pmua"]["gpu"] == 3
    with pytest.raises(PermissionError, match="claim"):
        campaign.start(root, gpus=(2, 3))
