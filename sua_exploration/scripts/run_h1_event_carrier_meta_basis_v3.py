#!/usr/bin/env python3
"""Run the CPU-only source meta-learned H1 event-carrier screen."""
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

from sua_exploration.mc_maze import h1_event_carrier_design_screen as v1screen
from sua_exploration.mc_maze import h1_event_carrier_meta_basis_v3 as meta
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_baseline(body: Mapping[str, Any], hse5: Mapping[str, Any]) -> dict[str, Any]:
    differences: list[float] = []
    for budget in (3, 4):
        for name in event_v1.H1_HELDIN_SESSIONS:
            observed = body["baseline_rows"][f"M{budget}"][name]
            expected = hse5["budgets"][f"M{budget}"]["sessions"][name]["forward"]
            for observed_key, expected_key in (
                ("median_r2_correct", "median_r2_correct"),
                ("median_delta_label_shuffle", "median_delta_shuffle"),
                ("median_delta_intercept", "median_delta_intercept"),
            ):
                differences.append(abs(float(observed[observed_key]) - float(expected[expected_key])))
    maximum = max(differences)
    need(maximum <= 1.0e-10, f"meta screen baseline reproduction failed: {maximum}")
    return {"passed": True, "comparisons": len(differences), "maximum_absolute_difference": maximum, "tolerance": 1.0e-10}


def write_immutable(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    need(not output.exists(), f"refusing to overwrite meta receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    need(stat.S_IMODE(output.stat().st_mode) == 0o444, "meta receipt mode drift")
    return output, hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument(
        "--hse5-receipt", type=Path,
        default=ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "sua_exploration/results/h1_event_carrier_meta_basis_v3/source_screen.json",
    )
    args = parser.parse_args()
    started = time.monotonic()
    data_root = args.data_root.resolve()
    hse5_path = args.hse5_receipt.resolve()
    hse5 = json.loads(hse5_path.read_text(encoding="utf-8"))
    sessions = v1screen.load_context_sessions(data_root)
    body = meta.run_screen(sessions)
    body["baseline_reproduction"] = verify_baseline(body, hse5)
    module_path = Path(meta.__file__).resolve()
    runner_path = Path(__file__).resolve()
    body.update({
        "runtime_seconds": time.monotonic() - started,
        "source_binding": {
            "sessions": list(event_v1.H1_HELDIN_SESSIONS),
            "dates": list(event_v1.H1_DATES),
            "files": [
                {
                    "session": name, "path": str(sessions[name].base.path),
                    "sha256": sessions[name].base.input_sha256,
                }
                for name in event_v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "module_path": str(module_path), "module_sha256": file_sha(module_path),
            "runner_path": str(runner_path), "runner_sha256": file_sha(runner_path),
            "v1_screen_module_path": str(Path(v1screen.__file__).resolve()),
            "v1_screen_module_sha256": file_sha(Path(v1screen.__file__).resolve()),
            "hse5_receipt_path": str(hse5_path), "hse5_receipt_sha256": file_sha(hse5_path),
        },
        "scope": {
            "public_held_in_calibration_nwbs_opened": 13,
            "minival_nwbs_opened": 0,
            "held_out_nwbs_opened": 0,
            "formal_test_labels_opened": 0,
            "dense_velocity_series_opened_by_carrier_screen": False,
            "decoder_constructed": False,
            "trainer_constructed": False,
            "cuda_used": False,
            "offline_source_basis_backward_steps_per_outer_arm": meta.OPTIMIZER_STEPS,
            "target_session_optimizer_steps": 0,
            "target_session_backward_steps": 0,
        },
    })
    output, digest = write_immutable(args.output, body)
    print(json.dumps({
        "status": body["status"],
        "passing_arms": body["passing_arms"],
        "selected_arm": body["selected_arm"],
        "receipt": str(output),
        "sha256": digest,
        "runtime_seconds": body["runtime_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
