from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/paper_program_v1/guarded_sequence.py"
    spec = importlib.util.spec_from_file_location("guarded_sequence_test", source)
    module = importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module)
    program = tmp_path / "program"; specs = program / "specs"; specs.mkdir(parents=True)
    import guarded_cell
    monkeypatch.setattr(guarded_cell, "PROGRAM", program)
    bound = tmp_path / "bound.py"; bound.write_text("v=1\n")
    source_hash = hashlib.sha256(bound.read_bytes()).hexdigest()
    predecessor = tmp_path / "predecessor.json"
    predecessor.write_text(json.dumps({"status": "COMPLETE", "cell": "prior", "spec_sha256": "prior-sha"}))
    launches = []
    def write_spec(name):
        cell = f"cell_{name}"; path = specs / f"{name}.json"
        path.write_text(json.dumps({"status": "READY", "cell": cell, "source_hashes": {str(bound): source_hash}}))
        return {"name": name, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cell": cell}
    rows = [write_spec("one"), write_spec("two")]
    sequence = tmp_path / "sequence.json"; state = tmp_path / "sequence_state.json"
    def write_sequence(spec_rows=rows):
        sequence.write_text(json.dumps({"schema": module.SCHEMA, "status": "READY", "state_path": str(state),
            "predecessor": {"state_path": str(predecessor), "cell": "prior", "spec_sha256": "prior-sha"}, "specs": spec_rows}))
        return sequence
    def popen(argv, **kwargs):
        launches.append(argv); spec_path = Path(argv[-1]); item = json.loads(spec_path.read_text()); cell_state = program / item["cell"] / "state.json"
        cell_state.parent.mkdir(parents=True); cell_state.write_text(json.dumps({"status": "COMPLETE", "cell": item["cell"], "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest()}))
        return SimpleNamespace(pid=100 + len(launches), wait=lambda: 0)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    return SimpleNamespace(module=module, program=program, bound=bound, predecessor=predecessor, rows=rows,
                           sequence=write_sequence, state=state, launches=launches)


def test_normal_two_cell_sequence_is_strictly_serial(queue):
    assert queue.module.run_sequence(queue.sequence()) == 0
    assert [Path(call[-1]).name for call in queue.launches] == ["one.json", "two.json"]
    state = json.loads(queue.state.read_text())
    assert state["status"] == "COMPLETE" and state["current_index"] == 2 and state["child_pid"] is None


def test_failure_stops_without_launching_next_cell(queue, monkeypatch):
    def fail(argv, **kwargs): return SimpleNamespace(pid=1, wait=lambda: 7)
    monkeypatch.setattr(queue.module.subprocess, "Popen", fail)
    assert queue.module.run_sequence(queue.sequence()) == 1
    assert json.loads(queue.state.read_text())["status"] == "FAILED_CELL"


def test_mismatched_spec_source_and_predecessor_failure_never_launch(queue):
    queue.rows[0]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="hash drift"):
        queue.module.run_sequence(queue.sequence())
    queue.rows[0]["sha256"] = hashlib.sha256(Path(queue.rows[0]["path"]).read_bytes()).hexdigest()
    queue.predecessor.write_text(json.dumps({"status": "FAILED_STAGE", "cell": "prior", "spec_sha256": "prior-sha"}))
    assert queue.module.run_sequence(queue.sequence()) == 1
    assert queue.launches == []


def test_running_or_completed_sequence_never_double_launches(queue):
    queue.state.write_text(json.dumps({"status": "RUNNING"}))
    with pytest.raises(RuntimeError, match="completed or running"):
        queue.module.run_sequence(queue.sequence())
    assert queue.launches == []


@pytest.mark.parametrize("field,value", [("cell", "wrong_cell"), ("spec_sha256", "wrong_hash")])
def test_stale_complete_identity_never_advances(queue, monkeypatch, field, value):
    def stale(argv, **kwargs):
        spec_path = Path(argv[-1]); item = json.loads(spec_path.read_text()); cell_state = queue.program / item["cell"] / "state.json"
        cell_state.parent.mkdir(parents=True); cell_state.write_text(json.dumps({"status": "COMPLETE", "cell": item["cell"], "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(), field: value}))
        return SimpleNamespace(pid=1, wait=lambda: 0)
    monkeypatch.setattr(queue.module.subprocess, "Popen", stale)
    assert queue.module.run_sequence(queue.sequence()) == 1
    assert json.loads(queue.state.read_text())["status"] == "FAILED_CELL"


def test_duplicate_cell_and_invalid_sidecar_stop(queue):
    duplicate = dict(queue.rows[1]); duplicate["cell"] = queue.rows[0]["cell"]
    with pytest.raises(ValueError, match="duplicate"):
        queue.module.run_sequence(queue.sequence([queue.rows[0], duplicate]))
    (queue.predecessor.parent / "INVALIDATED.json").write_text("{}")
    assert queue.module.run_sequence(queue.sequence()) == 1
    assert json.loads(queue.state.read_text())["status"] == "FAILED_PREDECESSOR"
