"""Resource and evidence gates for the paper experiment queue."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "scripts/paper_program_v1/guarded_cell.py"
    spec = importlib.util.spec_from_file_location("guarded_paper_cell_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROGRAM", tmp_path / "program")
    monkeypatch.setattr(module, "WORKSPACE", tmp_path)
    monkeypatch.setattr(module, "available_ram_bytes", lambda: 16 * 2**30)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    gpu = SimpleNamespace(index="1", uuid="GPU-test", memory_mib=0, utilization=0, compute_pids=())
    monkeypatch.setattr(module, "query_gpus", lambda: [gpu])
    source_file = tmp_path / "bound_source.py"
    source_file.write_text("version = 1\n")
    receipt = tmp_path / "result" / "receipt.json"
    specification = {
        "status": "READY", "cell": "test_cell",
        "source_hashes": {str(source_file): hashlib.sha256(source_file.read_bytes()).hexdigest()},
        "new_destinations": [str(receipt.parent)],
        "stages": [{"name": "train", "argv": ["test-only-command"],
                    "new_destinations": [str(receipt.parent)],
                    "receipts": [{"path": str(receipt), "fields": {"status": "COMPLETE", "steps": 24}}]}],
    }
    spec_path = tmp_path / "spec.json"

    def write_spec():
        spec_path.write_text(json.dumps(specification))
        return spec_path

    launched = []

    def launch(argv, **kwargs):
        launched.append((argv, kwargs))
        return SimpleNamespace(pid=123456, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    return SimpleNamespace(module=module, spec=specification, path=write_spec, receipt=receipt,
                           source=source_file, launched=launched, gpu=gpu)


class StopWaiting(Exception):
    pass


def stop_waiting(_):
    raise StopWaiting


@pytest.mark.parametrize("resource", ["disk", "ram", "occupied_gpu"])
def test_no_launch_until_resource_available(queue, monkeypatch, resource):
    if resource == "disk":
        monkeypatch.setattr(queue.module.shutil, "disk_usage", lambda _: SimpleNamespace(free=60 * 2**30))
    elif resource == "ram":
        monkeypatch.setattr(queue.module, "available_ram_bytes", lambda: 4 * 2**30)
    else:
        queue.gpu.compute_pids = (9876,)
    monkeypatch.setattr(queue.module.time, "sleep", stop_waiting)
    with pytest.raises(StopWaiting):
        queue.module.run_cell(queue.path())
    assert queue.launched == []


def test_exit_zero_without_complete_receipt_stops_cell(queue):
    queue.spec["stages"].append({"name": "score", "argv": ["must-not-launch"],
                                 "receipts": [{"path": str(queue.receipt), "fields": {"status": "COMPLETE"}}]})
    assert queue.module.run_cell(queue.path()) == 1
    assert len(queue.launched) == 1
    state = json.loads((queue.module.PROGRAM / "test_cell/state.json").read_text())
    assert state["status"] == "FAILED_STAGE" and not state["receipts_valid"]


def test_existing_artifact_and_source_drift_prevent_launch(queue):
    queue.receipt.parent.mkdir()
    queue.receipt.write_text('{"status":"COMPLETE","steps":24}')
    with pytest.raises(RuntimeError, match="already contains"):
        queue.module.run_cell(queue.path())
    queue.receipt.unlink()
    queue.source.write_text("version = 2\n")
    with pytest.raises(RuntimeError, match="frozen source changed"):
        queue.module.run_cell(queue.path())
    assert queue.launched == []


def test_complete_receipt_binds_command_to_chosen_gpu(queue, monkeypatch):
    def launch(argv, **kwargs):
        queue.launched.append((argv, kwargs))
        queue.receipt.parent.mkdir()
        queue.receipt.write_text('{"status":"COMPLETE","steps":24}')
        return SimpleNamespace(pid=123456, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(queue.module.subprocess, "Popen", launch)
    queue.spec["environment"] = {"CUDA_VISIBLE_DEVICES": "0"}
    assert queue.module.run_cell(queue.path()) == 0
    assert queue.launched[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    state = json.loads((queue.module.PROGRAM / "test_cell/state.json").read_text())
    assert state["status"] == "COMPLETE" and state["stages"][0]["receipts_valid"]
    with pytest.raises(RuntimeError, match="completed cell"):
        queue.module.run_cell(queue.path())


def test_empty_completion_criteria_rejected(queue):
    queue.spec["stages"][0]["receipts"][0]["fields"] = {}
    with pytest.raises(ValueError, match="nonempty exact fields"):
        queue.module.run_cell(queue.path())
    assert queue.launched == []
