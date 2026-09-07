#!/usr/bin/env python3
"""Run the immutable CPU-only QC2F5 source-date-LODO screen."""
from __future__ import annotations

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

from sua_exploration.mc_maze import h1_calibration_future_quadratic_c2f5 as qc2f5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_write(path: Path, value: Mapping[str, Any]) -> tuple[Path, str]:
    path = path.resolve()
    need(not path.exists(), f"refusing to overwrite immutable QC2F5 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    path.chmod(0o444)
    need(stat.S_IMODE(path.stat().st_mode) == 0o444, "QC2F5 immutable mode drift")
    return path, hashlib.sha256(encoded).hexdigest()


def predeclaration() -> dict[str, Any]:
    """Literal frozen before any source NWB is indexed or opened."""

    return {
        "schema": qc2f5.PREDECLARATION_SCHEMA, "protocol": qc2f5.PROTOCOL,
        "candidate_name": "qc2f5_fixed_degree2_source_support_to_future_residual_operator",
        "candidate_formula": "B_corrected = B_support + F2([B_support, analytic_quality])",
        "carrier_definition": "exact H-SE5 q4 [w1,w2,w3,w4,b]",
        "operator": {"family": "standardized_fixed_degree2_polynomial_ridge",
                     "terms": "linear + squares + pairwise crosses, ordered as declared; no channel index",
                     "ridge_grid": list(qc2f5.C2F_RIDGE_GRID),
                     "selection": "outer-date-excluded inner source-date LODO raw later-event log-rate R2",
                     "tie_break": "strongest ridge among lambdas within 1e-12 of best mean"},
        "controls": ["exact_raw_hse5", "independently_recomputed_linear_c2f5",
                     "endpoint_label_shuffled_support_refit_through_same_F2", "row_attachment_shuffle",
                     "quadratic_quality_only_no_B_in_F2", "source_future_teacher_pairing_shuffle_independently_trained_F2",
                     "intercept_only"],
        "evaluation": {"cohort": "exact 13 public H1 held-in calibration sessions", "outer_split": "date LODO",
                       "support_budgets": list(qc2f5.SUPPORT_BUDGETS),
                       "metric": "per-recording median-channel R2 on raw later-event log rates",
                       "hse5_receipt": "h1_sparse_event_endpoint_v2/source_audit.json",
                       "linear_c2f5_receipt": "h1_calibration_future_correction_c2f5/source_screen.json"},
        "gate": {"both_M3_M4": True, "mean_delta_vs_hse5_at_least": qc2f5.MATERIAL_MEAN_DELTA,
                 "median_delta_vs_hse5_at_least": qc2f5.MATERIAL_MEDIAN_DELTA,
                 "positive_sessions_at_least": qc2f5.MIN_POSITIVE_SESSIONS,
                 "required_positive_controls": ["linear_c2f5", "label_shuffled", "source_teacher_shuffled", "quality_only", "intercept_only"]},
        "forbidden": {"dense_velocity": True, "minival": True, "held_out": True, "formal_test": True,
                      "target_session_later_label_operator_fit": True, "target_session_optimizer": True,
                      "target_session_backpropagation": True, "gpu": True},
        "written_before_source_nwb_open": True,
    }


def ensure_predeclaration(path: Path) -> tuple[Path, str]:
    expected = predeclaration()
    path = path.resolve()
    if path.exists():
        need(stat.S_IMODE(path.stat().st_mode) == 0o444, "existing QC2F5 predeclaration not immutable")
        need(json.loads(path.read_text(encoding="utf-8")) == expected, "existing QC2F5 predeclaration changed")
        return path, file_sha(path)
    return immutable_write(path, expected)


def load_sessions(data_root: Path) -> dict[str, v1.EventSession]:
    indexed = v1.index_heldin_calib(data_root.resolve())
    return {name: v1.load_event_session(indexed[name]) for name in v1.H1_HELDIN_SESSIONS}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pre_path, pre_sha = ensure_predeclaration(args.predeclaration_output)
    started = time.monotonic()
    hse_path, linear_path = args.hse5_v2_receipt.resolve(), args.linear_c2f5_receipt.resolve()
    hse = json.loads(hse_path.read_text(encoding="utf-8"))
    linear = json.loads(linear_path.read_text(encoding="utf-8"))
    need(hse.get("schema") == v2.SCHEMA, "QC2F5 H-SE5 receipt schema mismatch")
    need(linear.get("schema") == "h1_calibration_future_correction_c2f5_source_screen_v1", "QC2F5 linear receipt schema mismatch")
    sessions = load_sessions(args.data_root)
    body = qc2f5.run_screen(sessions, hse5_reference=hse, linear_c2f5_reference=linear)
    body.update({
        "runtime_seconds": time.monotonic() - started,
        "predeclaration": {"path": str(pre_path), "sha256": pre_sha, "immutable_mode": "0444",
                           "written_or_verified_before_source_nwb_open": True},
        "baseline_reproduction": {"M3": body["budgets"]["M3"]["raw_hse5_v2_reproduction"],
                                  "M4": body["budgets"]["M4"]["raw_hse5_v2_reproduction"],
                                  "reference_path": str(hse_path), "reference_sha256": file_sha(hse_path)},
        "linear_c2f5_reference": {"path": str(linear_path), "sha256": file_sha(linear_path),
                                  "reproduction": body["linear_c2f5_exact_reproduction"]},
        "source_binding": {"sessions": list(v1.H1_HELDIN_SESSIONS), "dates": list(v1.H1_DATES),
                           "files": [{"session": name, "path": str(sessions[name].path), "sha256": sessions[name].input_sha256}
                                     for name in v1.H1_HELDIN_SESSIONS]},
        "implementation_binding": {"module_path": str(Path(qc2f5.__file__).resolve()), "module_sha256": file_sha(Path(qc2f5.__file__).resolve()),
                                   "runner_path": str(Path(__file__).resolve()), "runner_sha256": file_sha(Path(__file__).resolve()),
                                   "event_parser_path": str(Path(v1.__file__).resolve()), "event_parser_sha256": file_sha(Path(v1.__file__).resolve()),
                                   "hse5_v2_path": str(Path(v2.__file__).resolve()), "hse5_v2_sha256": file_sha(Path(v2.__file__).resolve())},
        "scope": {**body["scope"], "public_held_in_calibration_nwbs_opened": 13, "minival_nwbs_opened": 0,
                  "held_out_nwbs_opened": 0, "formal_test_labels_opened": 0, "decoder_constructed": False,
                  "trainer_constructed": False, "cuda_used": False, "target_session_later_labels_used_only_for_outer_score": True},
        "runtime_limits": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")} | {"requested_nice": 19},
    })
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument("--hse5-v2-receipt", type=Path, default=ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json")
    parser.add_argument("--linear-c2f5-receipt", type=Path, default=ROOT / "sua_exploration/results/h1_calibration_future_correction_c2f5/source_screen.json")
    parser.add_argument("--predeclaration-output", type=Path, default=ROOT / "sua_exploration/results/h1_calibration_future_quadratic_c2f5/QC2F5_PREDECLARATION_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "sua_exploration/results/h1_calibration_future_quadratic_c2f5/source_screen.json")
    args = parser.parse_args()
    try:
        os.nice(19)
    except OSError:
        pass
    result = run(args)
    output, digest = immutable_write(args.output, result)
    print(json.dumps({"status": result["status"], "passing_candidate": result["passing_candidate"], "receipt": str(output), "sha256": digest, "runtime_seconds": result["runtime_seconds"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
