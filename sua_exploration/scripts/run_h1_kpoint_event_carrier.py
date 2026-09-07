#!/usr/bin/env python3
"""Run the frozen CPU-only H1 K-point event carrier screen."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import h1_kpoint_event_carrier as screen
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    need(not output.exists(), f"refusing to overwrite K-point event carrier receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    need(stat.S_IMODE(output.stat().st_mode) == 0o444, "K-point event carrier receipt mode drift")
    return output, hashlib.sha256(encoded).hexdigest()


def compact_summary(body: Mapping[str, Any]) -> dict[str, Any]:
    by_arm: dict[str, Any] = {}
    for arm, aggregate in body["aggregates_by_arm"].items():
        intercept = aggregate["median_delta_intercept"]
        retained = aggregate["retained_variance"]
        by_arm[arm] = {
            "defined_sessions": aggregate["defined_sessions"],
            "median_delta_intercept_mean": intercept.get("mean"),
            "median_delta_intercept_median": intercept.get("median"),
            "median_delta_intercept_positive": intercept.get("positive"),
            "retained_variance_mean": retained.get("mean"),
        }
    contrast_table: dict[str, Any] = {}
    for family, contrasts in body["paired_contrasts"].items():
        contrast_table[family] = {}
        for label, summary in contrasts.items():
            contrast_table[family][label] = {
                "mean": summary.get("mean"),
                "median": summary.get("median"),
                "positive": summary.get("positive"),
                "leave_largest_absolute_out_mean": summary.get("leave_largest_absolute_out_mean"),
                "gate": body["gate_evaluation"][family][label],
            }
    supervision: dict[str, Any] = {}
    for k, block in body["supervision_accounting"].items():
        supervision[k] = {
            "fold0_ratio_vs_dense_velocity_coordinates": block["fold0_ratio_vs_dense_velocity_coordinates"],
            "fold0_acquisition_position_coordinates": block["fold0_acquisition_position_coordinates"],
        }
    return {
        "schema": body["schema"],
        "sealed_reproduction_passed": body["integrity_checks"]["sealed_hse5_reproduction"]["passed"],
        "runtime_seconds": body["runtime_seconds"],
        "by_arm": by_arm,
        "paired_contrasts": contrast_table,
        "supervision_accounting": supervision,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    paths = v1.index_heldin_calib(args.data_root.resolve())
    sessions = {name: v1.load_event_session(paths[name]) for name in v1.H1_HELDIN_SESSIONS}
    body = screen.run_screen(sessions)
    module_path = Path(screen.__file__).resolve()
    runner_path = Path(__file__).resolve()
    body.update({
        "runtime_seconds": time.monotonic() - started,
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS),
            "dates": list(v1.H1_DATES),
            "files": [
                {
                    "session": name,
                    "path": str(sessions[name].path),
                    "sha256": sessions[name].input_sha256,
                }
                for name in v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "module_path": str(module_path),
            "module_sha256": file_sha(module_path),
            "runner_path": str(runner_path),
            "runner_sha256": file_sha(runner_path),
        },
    })
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sua_exploration/results/h1_kpoint_event_carrier/source_screen.json",
    )
    args = parser.parse_args()
    result = run(args)
    output, digest = write_immutable(args.output, result)
    print(json.dumps({
        **compact_summary(result),
        "receipt": str(output),
        "sha256": digest,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
