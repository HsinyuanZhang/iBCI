"""Regression checks for the score-free r6b controlled-stop incident writer."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRITER = ROOT / "sua_exploration/scripts/write_m2_native_post33_phase_c_v4_r6b_abort_incident.py"


def _module():
    spec = importlib.util.spec_from_file_location("r6b_abort_incident_writer", WRITER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_snapshot_excludes_writer_and_matches_only_exact_r6b_workers() -> None:
    module = _module()
    root = str(module.R6B_ROOT)
    snapshot = "\n".join(
        (
            f"123 {module.__file__}",
            f"124 /bin/bash -c python {module.__file__}",
            f"125 python run_m2_native_post33_phase_c_v4_matrix.py --cell-root {root}",
            "126 python run_m2_native_post33_phase_c_v4_matrix.py --cell-root /tmp/other-root",
            f"127 python unrelated_r6b_tool.py --cell-root {root}",
        )
    )
    assert module._surviving_r6b_processes(snapshot, own_pid=123) == [
        f"125 python run_m2_native_post33_phase_c_v4_matrix.py --cell-root {root}"
    ]
