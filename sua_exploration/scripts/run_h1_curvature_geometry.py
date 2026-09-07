#!/usr/bin/env python3
"""Run the CPU-only H1 within-event position trajectory geometry diagnostic."""
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

from sua_exploration.mc_maze import h1_curvature_geometry as diagnostic
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
    need(not output.exists(), f"refusing to overwrite curvature geometry receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    need(stat.S_IMODE(output.stat().st_mode) == 0o444, "curvature geometry receipt mode drift")
    return output, hashlib.sha256(encoded).hexdigest()


def _format_percentile_block(label: str, block: Mapping[str, Any]) -> str:
    lines = [f"{label} (n={block['count']}):"]
    for field in ("median", "p90", "max"):
        value = block.get(field)
        lines.append(f"  {field}: {value}")
    return "\n".join(lines)


def print_summary(body: Mapping[str, Any]) -> None:
    overall = body["overall"]
    print("=== Overall geometry ===")
    print(f"events: {overall['event_count']}")
    print(f"degenerate_chord: {overall['degenerate_chord_count']}")
    print(_format_percentile_block("max_deviation_ratio", overall["max_deviation_ratio"]))
    print(_format_percentile_block("arc_chord_ratio", overall["arc_chord_ratio"]))
    print(_format_percentile_block("speed_cv", overall["speed_cv"]))

    print("\n=== max_deviation_ratio exceedance ===")
    for threshold, fraction in sorted(
        body["overall"]["max_deviation_ratio_exceedance"].items(),
        key=lambda item: float(item[0]),
    ):
        print(f"  > {threshold}: {fraction:.6f}")

    print("\n=== By movement tag ===")
    print(f"{'tag':<10} {'count':>8} {'med_max_dev':>12} {'med_arc':>12} {'med_speed_cv':>14}")
    for tag in v1.MOVEMENT_TAGS:
        row = body["by_tag"][tag]
        print(
            f"{tag:<10} {row['event_count']:>8} "
            f"{row['median_max_deviation_ratio']!s:>12} "
            f"{row['median_arc_chord_ratio']!s:>12} "
            f"{row['median_speed_cv']!s:>14}"
        )

    print("\n=== By degree of freedom ===")
    print(f"{'dof':<6} {'med_curv_ratio':>16} {'degenerate_delta':>18}")
    for name in diagnostic.DOF_NAMES:
        row = body["by_dimension"][name]
        print(
            f"{name:<6} {row['median_curvature_ratio']!s:>16} "
            f"{row['degenerate_delta_count']:>18}"
        )

    split = body["by_trial_split"]
    print("\n=== Support events (trial_index < 4) ===")
    support = split["support_events_trial_index_lt_4"]
    print(f"events: {support['event_count']}; degenerate_chord: {support['degenerate_chord_count']}")
    print(_format_percentile_block("max_deviation_ratio", support["max_deviation_ratio"]))
    print(_format_percentile_block("arc_chord_ratio", support["arc_chord_ratio"]))
    print(_format_percentile_block("speed_cv", support["speed_cv"]))

    print("\n=== Later events (trial_index >= 4) ===")
    later = split["later_events_trial_index_ge_4"]
    print(f"events: {later['event_count']}; degenerate_chord: {later['degenerate_chord_count']}")
    print(_format_percentile_block("max_deviation_ratio", later["max_deviation_ratio"]))
    print(_format_percentile_block("arc_chord_ratio", later["arc_chord_ratio"]))
    print(_format_percentile_block("speed_cv", later["speed_cv"]))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    paths = v1.index_heldin_calib(args.data_root.resolve())
    sessions = {name: v1.load_event_session(paths[name]) for name in v1.H1_HELDIN_SESSIONS}
    body = diagnostic.run_diagnostic(sessions)
    module_path = Path(diagnostic.__file__).resolve()
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
        default=ROOT / "sua_exploration/results/h1_curvature_geometry/diagnostic.json",
    )
    args = parser.parse_args()
    result = run(args)
    output, digest = write_immutable(args.output, result)
    print_summary(result)
    print(f"\nreceipt: {output}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
