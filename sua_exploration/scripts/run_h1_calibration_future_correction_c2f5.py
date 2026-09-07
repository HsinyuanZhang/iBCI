#!/usr/bin/env python3
"""Run the frozen CPU-only H1 calibration-to-future correction (C2F5) screen."""
from __future__ import annotations

# Must precede NumPy import: this program intentionally leaves CPU capacity to
# the simultaneously running GPU training jobs.
import os
for _thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_key, "1")

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

from sua_exploration.mc_maze import h1_calibration_future_correction_c2f5 as c2f5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


PREDECLARATION_SCHEMA = "h1_c2f5_predeclaration_v1"


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def immutable_write(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    output = path.resolve()
    need(not output.exists(), f"refusing to overwrite immutable C2F5 artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(value)
    output.write_bytes(encoded)
    output.chmod(0o444)
    need(stat.S_IMODE(output.stat().st_mode) == 0o444, "C2F5 immutable mode drift")
    return output, hashlib.sha256(encoded).hexdigest()


def predeclaration() -> dict[str, Any]:
    """Literal design frozen before source NWB files are opened."""

    return {
        "schema": PREDECLARATION_SCHEMA,
        "protocol": c2f5.PROTOCOL,
        "candidate_name": "c2f5_shared_calibration_to_future_residual_operator",
        "candidate_formula": "B_corrected = B_support + F([B_support, analytic_quality])",
        "matched_quality_only_control": "B_support + F(analytic_quality), with B_support absent from F",
        "carrier_definition": "exact H-SE5 q4 [w1,w2,w3,w4,b]",
        "quality_features": list(c2f5.QUALITY_NAMES),
        "operator": {
            "family": "shared_affine_standardized_ridge",
            "has_channel_index_feature": False,
            "residual_teacher": "same-source-session later-event H-SE5 carrier",
            "ridge_grid": list(c2f5.C2F_RIDGE_GRID),
            "selection": "outer-date-excluded source-date LODO mean later-event raw-log-rate R2",
            "tie_break": "smallest lambda within 1e-12 of best score",
        },
        "controls": [
            "exact_raw_hse5",
            "endpoint_label_shuffled_support_refit_through_same_operator",
            "row_shuffled_corrected_carrier",
            "quality_only_no_B_support_operator",
            "intercept_only",
        ],
        "evaluation": {
            "cohort": "exact 13 public H1 held-in calibration sessions",
            "outer_split": "date LODO",
            "support_budgets": list(c2f5.SUPPORT_BUDGETS),
            "metric": "per-recording median-channel R2 on byte-identical H-SE5 later events, raw log-rate response",
            "raw_hse5_reference": "h1_sparse_event_endpoint_v2/source_audit.json",
        },
        "gate": {
            "both_M3_M4": True,
            "mean_delta_vs_hse5_at_least": c2f5.MATERIAL_MEAN_DELTA,
            "median_delta_vs_hse5_at_least": c2f5.MATERIAL_MEDIAN_DELTA,
            "positive_sessions_at_least": c2f5.MIN_POSITIVE_SESSIONS,
            "leave_largest_absolute_delta_out_mean_positive": True,
            "all_controls_positive_mean_median_10_of_13_and_leave_one_out": True,
        },
        "forbidden": {
            "dense_velocity": True,
            "minival": True,
            "held_out": True,
            "formal_test": True,
            "target_session_later_label_operator_fit": True,
            "target_session_optimizer": True,
            "target_session_backpropagation": True,
            "gpu": True,
        },
        "written_before_source_nwb_open": True,
    }


def ensure_predeclaration(path: Path) -> tuple[Path, str]:
    expected = predeclaration()
    resolved = path.resolve()
    if resolved.exists():
        need(stat.S_IMODE(resolved.stat().st_mode) == 0o444, "existing C2F5 predeclaration is not immutable")
        observed = json.loads(resolved.read_text(encoding="utf-8"))
        need(observed == expected, "existing C2F5 predeclaration differs from frozen literal")
        return resolved, file_sha(resolved)
    return immutable_write(resolved, expected)


def load_sessions(data_root: Path) -> dict[str, v1.EventSession]:
    indexed = v1.index_heldin_calib(data_root.resolve())
    return {name: v1.load_event_session(indexed[name]) for name in v1.H1_HELDIN_SESSIONS}


def run(args: argparse.Namespace) -> dict[str, Any]:
    predeclaration_path, predeclaration_sha = ensure_predeclaration(args.predeclaration_output)
    started = time.monotonic()
    reference_path = args.hse5_v2_receipt.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    need(reference.get("schema") == v2.SCHEMA, "C2F5 H-SE5 reference schema mismatch")
    sessions = load_sessions(args.data_root)
    body = c2f5.run_screen(sessions, hse5_reference=reference)
    module_path = Path(c2f5.__file__).resolve()
    runner_path = Path(__file__).resolve()
    parser_path = Path(v1.__file__).resolve()
    v2_path = Path(v2.__file__).resolve()
    body.update({
        "runtime_seconds": time.monotonic() - started,
        "predeclaration": {
            "path": str(predeclaration_path),
            "sha256": predeclaration_sha,
            "immutable_mode": "0444",
            "written_or_verified_before_source_nwb_open": True,
        },
        "baseline_reproduction": {
            "M3": body["budgets"]["M3"]["raw_hse5_v2_reproduction"],
            "M4": body["budgets"]["M4"]["raw_hse5_v2_reproduction"],
            "reference_path": str(reference_path),
            "reference_sha256": file_sha(reference_path),
        },
        "source_binding": {
            "sessions": list(v1.H1_HELDIN_SESSIONS),
            "dates": list(v1.H1_DATES),
            "files": [
                {"session": name, "path": str(sessions[name].path), "sha256": sessions[name].input_sha256}
                for name in v1.H1_HELDIN_SESSIONS
            ],
        },
        "implementation_binding": {
            "module_path": str(module_path), "module_sha256": file_sha(module_path),
            "runner_path": str(runner_path), "runner_sha256": file_sha(runner_path),
            "event_parser_path": str(parser_path), "event_parser_sha256": file_sha(parser_path),
            "hse5_v2_path": str(v2_path), "hse5_v2_sha256": file_sha(v2_path),
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
            "target_session_later_labels_used_only_for_outer_score": True,
        },
        "runtime_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "requested_nice": 19,
        },
    })
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument("--hse5-v2-receipt", type=Path,
                        default=ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json")
    parser.add_argument("--predeclaration-output", type=Path,
                        default=ROOT / "sua_exploration/results/h1_calibration_future_correction_c2f5/C2F5_PREDECLARATION_v1.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "sua_exploration/results/h1_calibration_future_correction_c2f5/source_screen.json")
    args = parser.parse_args()
    result = run(args)
    output, digest = immutable_write(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "passing_candidate": result["passing_candidate"],
        "receipt": str(output),
        "sha256": digest,
        "runtime_seconds": result["runtime_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
