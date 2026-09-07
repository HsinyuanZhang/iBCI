#!/usr/bin/env python3
"""Run the predeclared CPU-only H1 sparse-event estimator V5 screen."""
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

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_event_carrier_estimator_v5 as v5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


INVALIDATED_V5R1_RECEIPT = ROOT / "sua_exploration/results/h1_event_carrier_estimator_v5/source_screen.json"
INVALIDATED_V5R1_SHA256 = "a50ff3706c3b2bbc922d83bae09fc6b8cb4dee6d8eae74989a4a183af29eeb86"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hse5_reproduction(
    body: Mapping[str, Any], reference: Mapping[str, Any], *, tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    fields = {
        "median_r2_correct": "median_r2_correct",
        "median_delta_label_shuffle": "median_delta_shuffle",
        "median_delta_intercept": "median_delta_intercept",
    }
    differences: list[dict[str, Any]] = []
    for budget in (3, 4):
        observed = body["budgets"][f"M{budget}"]["candidates"]["hse5_pca_delta_q4"]["sessions"]
        expected = reference["budgets"][f"M{budget}"]["sessions"]
        for session in v1.H1_HELDIN_SESSIONS:
            for observed_key, expected_key in fields.items():
                delta = abs(float(observed[session][observed_key]) - float(expected[session]["forward"][expected_key]))
                differences.append({
                    "budget": budget,
                    "session": session,
                    "field": observed_key,
                    "absolute_difference": delta,
                })
    maximum = max(row["absolute_difference"] for row in differences)
    need(maximum <= tolerance, f"V5 failed exact H-SE5 reproduction: {maximum}")
    return {
        "passed": True,
        "comparisons": len(differences),
        "absolute_tolerance": tolerance,
        "maximum_absolute_difference": maximum,
    }


def write_immutable(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    need(not output.exists(), f"refusing to overwrite V5 receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)
    output.chmod(0o444)
    need(stat.S_IMODE(output.stat().st_mode) == 0o444, "V5 receipt mode drift")
    return output, hashlib.sha256(encoded).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    reference_path = args.hse5_v2_receipt.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    need(reference.get("schema") == v2.SCHEMA, "V5 H-SE5 reference schema mismatch")
    sessions = design.load_context_sessions(args.data_root.resolve())
    body = v5.run_screen(sessions)
    reproduction = verify_hse5_reproduction(body, reference)
    module_path = Path(v5.__file__).resolve()
    runner_path = Path(__file__).resolve()
    design_path = Path(design.__file__).resolve()
    parser_path = Path(v1.__file__).resolve()
    body.update({
        "runtime_seconds": time.monotonic() - started,
        "baseline_reproduction": {
            **reproduction,
            "reference_path": str(reference_path),
            "reference_sha256": file_sha(reference_path),
        },
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS),
            "dates": list(v1.H1_DATES),
            "files": [
                {"session": name, "path": str(sessions[name].base.path), "sha256": sessions[name].base.input_sha256}
                for name in v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "module_path": str(module_path), "module_sha256": file_sha(module_path),
            "runner_path": str(runner_path), "runner_sha256": file_sha(runner_path),
            "shared_design_path": str(design_path), "shared_design_sha256": file_sha(design_path),
            "event_parser_path": str(parser_path), "event_parser_sha256": file_sha(parser_path),
        },
        "scope": {
            **body["scope"],
            "public_held_in_calibration_nwbs_opened": 13,
            "minival_nwbs_opened": 0,
            "held_out_nwbs_opened": 0,
            "formal_test_labels_opened": 0,
            "decoder_constructed": False,
            "trainer_constructed": False,
            "cuda_used": False,
        },
        "supersedes": {
            "invalidated_receipt_path": str(INVALIDATED_V5R1_RECEIPT.resolve()),
            "invalidated_receipt_sha256": INVALIDATED_V5R1_SHA256,
            "invalidated_reason": (
                "V5r1 Poisson IRLS subtracted log(duration_seconds) from a working response "
                "whose eta was already log rate, double-applying the exposure offset"
            ),
            "old_receipt_preserved_unmodified": True,
            "same_predeclared_candidate_matrix": True,
        },
    })
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument("--hse5-v2-receipt", type=Path,
                        default=ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "sua_exploration/results/h1_event_carrier_estimator_v5/source_screen_v2.json")
    args = parser.parse_args()
    result = run(args)
    output, digest = write_immutable(args.output, result)
    print(json.dumps({
        "status": result["status"], "selected_candidate": result["selected_candidate"],
        "passing_candidates": result["passing_candidates"], "receipt": str(output),
        "sha256": digest, "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
