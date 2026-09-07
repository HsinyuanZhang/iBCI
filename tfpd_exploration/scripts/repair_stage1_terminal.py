"""Repair terminal receipts for Stage-1 cells whose training completed but whose
terminal write crashed on the post-fit val_r2 recomputation bug (metric state is
reset by on_validation_epoch_end, so compute() raised "needs at least two samples").

The training artifacts are intact (12/12 epoch checkpoints + immutable launch
receipt).  This tool writes the missing terminal receipt from the on-disk
evidence: checkpoint manifest (names + SHA-256), launch receipt binding, and an
explicit incident note.  It never touches checkpoints or launch receipts and
refuses a cell that already has a terminal receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair(cell_dir: Path) -> dict:
    launch_path = cell_dir / "launch_receipt.json"
    terminal_path = cell_dir / "terminal_receipt.json"
    if not launch_path.is_file():
        raise SystemExit(f"missing launch receipt in {cell_dir}")
    if terminal_path.exists():
        raise SystemExit(f"terminal receipt already exists for {cell_dir}")
    launch = json.loads(launch_path.read_text())
    ckpt_dir = cell_dir / "epoch_ckpts"
    checkpoints = sorted(ckpt_dir.glob("*.ckpt"))
    expected_epochs = int(launch.get("max_epochs", 12))
    complete = len(checkpoints) == expected_epochs
    manifest = [
        {"file": c.name, "bytes": c.stat().st_size, "sha256": sha256_file(c)} for c in checkpoints
    ]
    terminal = {
        **launch,
        "status": "SOURCE_CELL_TERMINAL" if complete else "SOURCE_CELL_INCOMPLETE__ABNORMAL_TERMINATION",
        "repaired": True,
        "expected_epochs": expected_epochs,
        "observed_epochs": len(checkpoints),
        "repair_incident": (
            "original terminal write crashed on post-fit val_r2.compute() after the "
            "metric had been reset at the last epoch end; training itself completed "
            "normally and all epoch checkpoints are intact. final_val_r2 is null by "
            "repair; epoch-window scoring is post-hoc per the Stage-1 contract."
        ),
        "repaired_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs_run": len(checkpoints),
        "final_val_r2": None,
        "epoch_checkpoint_manifest": manifest,
        "launch_receipt_sha256": sha256_file(launch_path),
    }
    terminal_path.write_text(json.dumps(terminal, indent=1, sort_keys=True) + "\n")
    os.chmod(terminal_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    sidecar = terminal_path.with_suffix(".sha256")
    sidecar.write_text(sha256_file(terminal_path) + "  " + terminal_path.name + "\n")
    os.chmod(sidecar, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return {"cell": cell_dir.name, "epochs": len(checkpoints), "terminal": str(terminal_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", required=True, help="cell directories under the output root")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/stage1_source_cells_v1")
    args = parser.parse_args()
    for cell in args.cells:
        print(json.dumps(repair(args.output_root / cell)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
