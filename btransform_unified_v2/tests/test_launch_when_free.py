from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "rift_v1" / "launch_pair_when_free.py"
SPEC = importlib.util.spec_from_file_location("rift_launch_when_free", SCRIPT)
assert SPEC and SPEC.loader
launch = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launch
SPEC.loader.exec_module(launch)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_ready(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    results = project / "results" / "rift_v1"
    run_root = results / "pair"
    results.mkdir(parents=True)
    source = project / "source.py"
    source.write_text("stable source\n")
    ready = results / "ready.json"
    ready.write_text(json.dumps({
        "status": "CPU_PREFLIGHT_PASS",
        "sources": [{"path": "source.py", "sha256": digest(source)}],
        "variants": {
            "recency": {"smoke_argv": ["smoke-recency", "{smoke_dest}"], "train_argv": ["train-recency", "{train_dest}"]},
            "flat": {"smoke_argv": ["smoke-flat", "{smoke_dest}"], "train_argv": ["train-flat", "{train_dest}"]},
        },
    }))
    return project, results, ready


def gpu(*, busy: bool = False, index: str = "2") -> launch.GPU:
    return launch.GPU(index, f"GPU-test-{index}", 100 if not busy else 50, 0, (9123,) if busy else ())


class Process:
    next_pid = 4000

    def __init__(self, code: int = 0) -> None:
        self.pid = Process.next_pid
        Process.next_pid += 1
        self.code = code

    def wait(self) -> int:
        return self.code

    def poll(self) -> int:
        return self.code


def queue(project: Path, results: Path, ready: Path, queries, calls: list[list[str]], codes=None) -> launch.Queue:
    launch.PROJECT_ROOT = project
    query_iter = iter(queries)
    last_query = queries[-1] if queries else []
    code_iter = iter(codes or [])

    def fake_query():
        return next(query_iter, last_query)

    def fake_popen(argv, **kwargs):
        assert kwargs["shell"] is False
        calls.append(argv)
        return Process(next(code_iter, 0))

    return launch.Queue(ready, results / "pair", results_root=results, sample_seconds=10,
                        query=fake_query, popen=fake_popen, clock=lambda: 1_700_000_000.0, sleep=lambda _: None)


def test_missing_ready_and_source_change_are_refused(tmp_path: Path) -> None:
    project, results, ready = setup_ready(tmp_path)
    launch.PROJECT_ROOT = project
    with pytest.raises(ValueError, match="does not exist"):
        launch.load_ready(results / "missing.json", results / "pair", results)
    (project / "source.py").write_text("changed\n")
    with pytest.raises(ValueError, match="SHA"):
        launch.load_ready(ready, results / "pair", results)


def test_busy_gpu_is_refused_after_two_samples(tmp_path: Path) -> None:
    project, results, ready = setup_ready(tmp_path)
    calls: list[list[str]] = []
    # Two busy samples do not launch; it remains polling until the later idle pair.
    result = queue(project, results, ready, [[gpu(busy=True)], [gpu(busy=True)], [gpu()]] * 4, calls).execute()
    assert calls[0][0] == "smoke-recency"
    assert result["runs"]["recency"]["status"] == "completed"


def test_two_idle_samples_launch_recency_before_flat_and_record(tmp_path: Path) -> None:
    project, results, ready = setup_ready(tmp_path)
    calls: list[list[str]] = []
    # Per variant: two samples to qualify, one check after smoke.
    result = queue(project, results, ready, [[gpu()]] * 6, calls).execute()
    assert [call[0] for call in calls] == ["smoke-recency", "train-recency", "smoke-flat", "train-flat"]
    assert result["runs"]["recency"]["status"] == "completed"
    assert result["runs"]["flat"]["status"] == "completed"
    assert result["runs"]["recency"]["gpu_uuid"] == "GPU-test-2"
    state = json.loads((results / "pair" / "launch_state.json").read_text())
    assert state["runs"]["flat"]["train_argv"][0] == "train-flat"


def test_completed_runs_are_not_repeated(tmp_path: Path) -> None:
    project, results, ready = setup_ready(tmp_path)
    calls: list[list[str]] = []
    first = queue(project, results, ready, [[gpu()]] * 6, calls)
    first.execute()
    second_calls: list[list[str]] = []
    # No GPU query should be needed when both variants have a durable record.
    queue(project, results, ready, [], second_calls).execute()
    assert len(calls) == 4
    assert second_calls == []


def test_smoke_failure_never_starts_its_formal_training(tmp_path: Path) -> None:
    project, results, ready = setup_ready(tmp_path)
    calls: list[list[str]] = []
    # Recency smoke exits 7.  Flat can still be independently considered.
    result = queue(project, results, ready, [[gpu()]] * 6, calls, codes=[7, 0, 0]).execute()
    assert calls[0][0] == "smoke-recency"
    assert "train-recency" not in [call[0] for call in calls]
    assert result["runs"]["recency"]["status"] == "failed"
    assert result["runs"]["flat"]["status"] == "completed"


def test_dry_run_validates_without_spawning(tmp_path: Path) -> None:
    project, results, ready = setup_ready(tmp_path)
    calls: list[list[str]] = []
    result = queue(project, results, ready, [[gpu()]] * 4, calls).execute(dry_run=True)
    assert calls == []
    assert result["runs"] == {}
    assert result["last_dry_run"]["candidates"] == ["GPU-test-2"]


def test_second_idle_gpu_launches_flat_while_recency_is_running(tmp_path: Path) -> None:
    project, results, ready = setup_ready(tmp_path)
    calls: list[list[str]] = []
    pair = [gpu(index="0"), gpu(index="1")]
    result = queue(project, results, ready, [pair] * 4, calls).execute()
    assert [call[0] for call in calls] == ["smoke-recency", "smoke-flat", "train-recency", "train-flat"]
    assert result["runs"]["recency"]["gpu_index"] == "0"
    assert result["runs"]["flat"]["gpu_index"] == "1"
