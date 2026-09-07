#!/usr/bin/env python3
"""Run the frozen CPU-only H1 event-conditioning sweep screen."""
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

from sua_exploration.mc_maze import h1_event_conditioning_sweep as screen
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
    need(not output.exists(), f"refusing to overwrite event conditioning sweep receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    need(stat.S_IMODE(output.stat().st_mode) == 0o444, "event conditioning sweep receipt mode drift")
    return output, hashlib.sha256(encoded).hexdigest()


def compact_summary(body: Mapping[str, Any]) -> dict[str, Any]:
    by_k: dict[str, Any] = {}
    for key in map(str, body["subset_sizes"]):
        correlations = body["correlation_by_k"][key]
        intercept_corr = correlations.get("log10_condition_number_vs_median_delta_intercept") or {}
        fidelity_corr = correlations.get("log10_condition_number_vs_carrier_fidelity") or {}
        rule_rows: dict[str, Any] = {}
        for rule, aggregate in body["rule_aggregates_by_k"][key].items():
            rule_rows[rule] = {
                "median_delta_intercept_mean": aggregate["median_delta_intercept"].get("mean"),
                "carrier_fidelity_mean": aggregate["carrier_fidelity"].get("mean"),
            }
        contrast = body["paired_maxspread_minus_minspread_by_k"][key]
        by_k[key] = {
            "log10_condition_number_vs_median_delta_intercept": {
                "spearman": intercept_corr.get("spearman"),
                "pearson": intercept_corr.get("pearson"),
            },
            "log10_condition_number_vs_carrier_fidelity": {
                "spearman": fidelity_corr.get("spearman"),
                "pearson": fidelity_corr.get("pearson"),
            },
            "rule_means": rule_rows,
            "maxspread_minus_minspread": {
                "median_delta_intercept": contrast["median_delta_intercept"],
                "carrier_fidelity": contrast["carrier_fidelity"],
            },
        }
    return {
        "schema": body["schema"],
        "full_support_reproduction_passed": body["full_support_reproduction"]["passed"],
        "runtime_seconds": body["runtime_seconds"],
        "by_k": by_k,
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
                    "input_sha256": sessions[name].input_sha256,
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
        default=ROOT / "sua_exploration/results/h1_event_conditioning_sweep/source_screen.json",
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
