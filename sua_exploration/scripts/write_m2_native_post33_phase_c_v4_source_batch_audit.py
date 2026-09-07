#!/usr/bin/env python3
"""Build the seven-fold source-only batch-cardinality audit for both arms."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import file_metadata


WORKERS = {
    "spint": ROOT / "SPINT-main/src/audit_post33_source_batches_phase_c_v4.py",
    "t4": ROOT / "streaming_calibration_exp/src/audit_post33_source_batches_phase_c_v4.py",
}
WORKDIRS = {
    "spint": ROOT / "SPINT-main",
    "t4": ROOT / "streaming_calibration_exp",
}


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="m2-post33-source-audit-") as directory:
        temp = Path(directory)
        arm_payloads = {}
        for arm in ("spint", "t4"):
            output = temp / f"{arm}.json"
            completed = subprocess.run(
                [sys.executable, str(WORKERS[arm]), "--data-root", str(data_root), "--output", str(output)],
                cwd=WORKDIRS[arm], check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"{arm} source-only audit worker failed with {completed.returncode}")
            payload = json.loads(output.read_text(encoding="utf-8"))
            if (
                payload.get("schema") != "m2_post33_phase_c_source_batch_arm_audit_v4"
                or payload.get("arm") != arm
                or payload.get("fold_outer_role_included") is not False
                or payload.get("scorer_imported") is not False
            ):
                raise ValueError(f"{arm} source-only audit contract failed")
            arm_payloads[arm] = payload
    input_rows = {}
    arms = {}
    for arm, payload in arm_payloads.items():
        arms[arm] = {}
        input_rows[arm] = {}
        for fold, row in payload["folds"].items():
            copied = dict(row)
            input_rows[arm][fold] = copied.pop("source_input_files")
            arms[arm][fold] = copied
    result = {
        "schema": "m2_post33_phase_c_source_batch_cardinality_audit_v4",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "score_data_accessed": False,
        "fold_outer_role_included": False,
        "scorer_imported": False,
        "formal_data_accessed": False,
        "batch_policy": "session-local full batches of 32; incomplete tail dropped",
        "arms": arms,
        "source_input_files": input_rows,
        "source_bindings": {
            "spint_session_batch_sampler_source": file_metadata(
                ROOT / "SPINT-main/src/data/falcon_datamodule.py"
            ),
            "t4_session_batch_sampler_source": file_metadata(
                ROOT / "streaming_calibration_exp/src/data/falcon_datamodule.py"
            ),
            "spint_post33_split_source": file_metadata(
                ROOT / "SPINT-main/src/data/falcon_post33_confirm_v1_datamodule.py"
            ),
            "t4_post33_split_source": file_metadata(
                ROOT / "streaming_calibration_exp/src/data/falcon_post33_confirm_v3_datamodule.py"
            ),
            "spint_phase_c_v4_datamodule_source": file_metadata(
                ROOT / "SPINT-main/src/data/falcon_post33_confirm_v4_datamodule.py"
            ),
            "t4_phase_c_v4_datamodule_source": file_metadata(
                ROOT / "streaming_calibration_exp/src/data/falcon_post33_confirm_v4_datamodule.py"
            ),
            "spint_source_audit_worker": file_metadata(WORKERS["spint"]),
            "t4_source_audit_worker": file_metadata(WORKERS["t4"]),
            "source_audit_writer": file_metadata(Path(__file__)),
            "phase_a_scorefree_data_audit": file_metadata(
                ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_scorefree_audit_20260804/audit.json"
            ),
            "phase_a_live_split_audit": file_metadata(
                ROOT / "sua_exploration/results/m2_native_t4_spint_post33_confirm_v1_live_plumbing_20260804/live_plumbing.json"
            ),
        },
    }
    _write_exclusive(args.output.resolve(), result)


if __name__ == "__main__":
    main()
