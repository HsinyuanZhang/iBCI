from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rift_v1/monitor_m2_train.py"


def _monitor():
    spec = importlib.util.spec_from_file_location("m2_rift_monitor", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_snapshot_tolerates_startup_before_heartbeat(tmp_path: Path) -> None:
    monitor = _monitor()
    monitor._alive = lambda _pid, _dest: True
    row = monitor.snapshot(tmp_path, 99999999, None, now=100.0)
    assert row["state"] == "RUNNING"
    assert row["global_step"] is None


def test_snapshot_tracks_progress_and_requires_score_receipt_for_completion(tmp_path: Path) -> None:
    monitor = _monitor()
    monitor._alive = lambda _pid, _dest: False
    (tmp_path / "heartbeat.json").write_text(json.dumps({"global_step": 7, "epoch": 1, "status": "TRAINING"}))
    prior = {"global_step": 6, "last_progress_unix": 1.0}
    row = monitor.snapshot(tmp_path, 99999999, prior, now=100.0)
    assert row["global_step"] == 7 and row["last_progress_unix"] == 100.0
    (tmp_path / "score_receipt.json").write_text("{}")
    completed = monitor.snapshot(tmp_path, 99999999, row, now=101.0)
    assert completed["state"] == "FAILED_NO_SCORE_RECEIPT"


def test_completed_requires_well_formed_full_ext4_receipt(tmp_path: Path) -> None:
    monitor = _monitor(); monitor._alive = lambda _pid, _dest: False
    (tmp_path / "run_meta.json").write_text(json.dumps({"status": "FORMAL", "cell": "M2-RIFT-R50-D4-P16-RECENCY-V1", "epochs": 24}))
    (tmp_path / "score_receipt.json").write_text("not-json")
    assert monitor.snapshot(tmp_path, 7, None, now=1.0)["state"] == "FAILED_NO_SCORE_RECEIPT"
    (tmp_path / "score_receipt.json").write_text(json.dumps({"schema": "m2_rift_ext4_epoch_scan_v1", "status": "COMPLETED", "cell": "M2-RIFT-R50-D4-P16-RECENCY-V1", "ema_by_epoch": {str(i): {} for i in range(1, 25)}}))
    assert monitor.snapshot(tmp_path, 7, None, now=2.0)["state"] == "COMPLETED"


def test_scoring_heartbeat_advances_without_global_step(tmp_path: Path) -> None:
    monitor = _monitor(); monitor._alive = lambda _pid, _dest: True
    (tmp_path / "heartbeat.json").write_text(json.dumps({"status": "SCORING", "event": "ext4_score", "epoch": 1, "completed_epochs": 1}))
    first = monitor.snapshot(tmp_path, 7, None, now=10.0)
    (tmp_path / "heartbeat.json").write_text(json.dumps({"status": "SCORING", "event": "ext4_score", "epoch": 2, "completed_epochs": 2}))
    second = monitor.snapshot(tmp_path, 7, first, now=400.0)
    assert second["state"] == "RUNNING" and second["last_progress_unix"] == 400.0
