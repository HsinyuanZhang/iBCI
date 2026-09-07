"""Synthetic, no-GPU v5 terminal-receipt release contracts."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/rt_ld_device_handoff_v5.py"


def _module():
    spec = importlib.util.spec_from_file_location("rt_ld_handoff_v5", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipt(path: Path, date: str, *, status: str | None = None, mode: int = 0o444) -> None:
    path.write_text(json.dumps({"schema": "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1", "status": status or f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{date}_HS_HC_EVALUATED", "outer_date": date}))
    os.chmod(path, mode)


def _paths(tmp_path: Path, module, partition: str) -> dict[str, Path]:
    return {date: tmp_path / f"{date}.json" for date in module.TERMINALS[partition]}


def test_missing_wrong_mode_wrong_date_or_status_terminal_receipt_blocks_release(tmp_path: Path) -> None:
    module = _module(); name = "gpu0_h1_static_ci64"; paths = _paths(tmp_path, module, name)
    assert not module._terminal_receipts_ready(name, paths)
    for date, path in paths.items(): _receipt(path, date)
    assert module._terminal_receipts_ready(name, paths)
    bad = next(iter(paths)); os.chmod(paths[bad], 0o644)
    assert not module._terminal_receipts_ready(name, paths)
    os.chmod(paths[bad], 0o644); _receipt(paths[bad], "19250113")
    assert not module._terminal_receipts_ready(name, paths)
    os.chmod(paths[bad], 0o644); _receipt(paths[bad], bad, status="WRONG")
    assert not module._terminal_receipts_ready(name, paths)


def test_this_partition_complete_and_two_clean_samples_is_eligible(tmp_path: Path) -> None:
    module = _module(); name = "gpu1_h1_static_ci64"; paths = _paths(tmp_path, module, name)
    for date, path in paths.items(): _receipt(path, date)
    assert module._terminal_receipts_ready(name, paths)
    clean = ("exited", False, [])
    assert module._eligible_from_probes(True, clean, clean)
    assert not module._eligible_from_probes(False, clean, clean)


def test_other_partition_missing_receipt_does_not_block_this_partition(tmp_path: Path) -> None:
    module = _module(); name = "gpu0_h1_static_ci64"; paths = _paths(tmp_path, module, name)
    for date, path in paths.items(): _receipt(path, date)
    assert module._terminal_receipts_ready(name, paths)
    assert not module._terminal_receipts_ready("gpu1_h1_static_ci64", {date: tmp_path / f"missing-{date}.json" for date in module.TERMINALS["gpu1_h1_static_ci64"]})
    assert module._eligible_from_probes(True, ("exited", False, []), ("exited", False, []))
