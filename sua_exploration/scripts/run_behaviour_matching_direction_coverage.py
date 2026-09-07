#!/usr/bin/env python3
"""Supplement to the 2026-08-14 behaviour-matching measurement: per-session direction coverage.

The main run found centre-out pairs whose discrete direction-bin match rate falls to 0.40.
This supplement records, per session, which of the 8 canonical target directions actually
survive the rewarded + duration filter, because a direction-keyed pairing mechanism can only
pair classes that both sessions hold.

Behaviour and trial metadata only. CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "streaming_calibration_exp"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sua_exploration.behaviour_matching import core, loaders

DEFAULT_OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/behaviour_matching_feasibility_20260814"
RECEIPT_NAME = "direction_coverage_supplement_receipt.json"
N_DIRECTIONS = 8


def write_immutable(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, 0o444)
    return hashlib.sha256(payload).hexdigest()


def coverage(cohort: str, paths: list[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        session = loaders.load_centre_out_session(path, cohort=cohort)
        labels = session.labels["dir8"]
        counts = np.bincount(labels[labels >= 0], minlength=N_DIRECTIONS)
        reference_pool, query_pool = core.chronological_halves(session.n_samples)
        rows.append(
            {
                "session": session.session,
                "samples": session.n_samples,
                "samples_per_direction": [int(v) for v in counts.tolist()],
                "directions_present": int((counts > 0).sum()),
                "missing_directions": [int(v) for v in np.flatnonzero(counts == 0).tolist()],
                "directions_present_first_half": int(
                    np.unique(labels[reference_pool][labels[reference_pool] >= 0]).size
                ),
                "directions_present_second_half": int(
                    np.unique(labels[query_pool][labels[query_pool] >= 0]).size
                ),
                "unique_target_dir_values_in_trials_table": session.notes["unique_target_dir_values_rad"],
                "samples_without_direction_label": session.notes["samples_without_direction_label"],
            }
        )
    incomplete = [row for row in rows if row["directions_present"] < N_DIRECTIONS]
    shared = set(range(N_DIRECTIONS))
    for row in rows:
        shared &= set(range(N_DIRECTIONS)) - set(row["missing_directions"])
    return {
        "sessions": len(rows),
        "sessions_with_all_8_directions": len(rows) - len(incomplete),
        "sessions_missing_at_least_one_direction": len(incomplete),
        "directions_shared_by_every_session": sorted(shared),
        "per_session": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.time()
    payload = {
        "schema": "behaviour_matching_direction_coverage_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-14",
        "supplements": core.SCHEMA,
        "protocol": {"document": core.PROTOCOL_DOCUMENT},
        "note": (
            "A session's trials table can expose all 8 canonical target_dir values while fewer "
            "survive the rewarded + 50-bin duration filter that defines a usable sample; only "
            "the surviving set is pairable by a direction-keyed mechanism."
        ),
        "cohorts": {
            "co_subc_source": coverage("co_subc_source", loaders.discover_centre_out_subc(REPO_ROOT)),
            "co_subm_external": coverage("co_subm_external", loaders.discover_centre_out_subm(REPO_ROOT)),
        },
        "implementation_sha256": {
            "sua_exploration/behaviour_matching/loaders.py": hashlib.sha256(
                Path(loaders.__file__).read_bytes()
            ).hexdigest(),
            "sua_exploration/scripts/run_behaviour_matching_direction_coverage.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nice": os.nice(0),
            "elapsed_seconds": time.time() - started,
        },
    }
    receipt_path = args.output_dir.resolve() / RECEIPT_NAME
    core.require(not receipt_path.exists(), f"refusing to overwrite {receipt_path}")
    sha = write_immutable(receipt_path, core.canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha,
                "co_subc_source": {
                    key: payload["cohorts"]["co_subc_source"][key]
                    for key in (
                        "sessions",
                        "sessions_missing_at_least_one_direction",
                        "directions_shared_by_every_session",
                    )
                },
                "co_subm_external": {
                    key: payload["cohorts"]["co_subm_external"][key]
                    for key in (
                        "sessions",
                        "sessions_missing_at_least_one_direction",
                        "directions_shared_by_every_session",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
