#!/usr/bin/env python3
"""Guarded source-only CPU runner for Experiment B estimator selection.

Default operation is dry-run receipt generation.  Real data execution requires both an explicit
flag and ``T4_ESTIMATOR_B_V1_REVIEWED_SOURCE_ONLY=YES``.  It resolves only the 27 frozen source
names; development and formal names remain strings in the manifest receipt and are never paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
sys.path.insert(0, str(SUA))

from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials  # noqa: E402
from mc_maze.t4_cross_budget_audit import load_frozen_source_development_manifest, pool_trial_count_matrix_from_receipt  # noqa: E402
from mc_maze.t4_cross_budget_protocol import canonical_direction_indices  # noqa: E402
from mc_maze.t4_estimator_b_v1 import (  # noqa: E402
    CANDIDATE_NAMES, N_CALIBRATION_TRIALS, N_SOURCE_SESSIONS, POISSON_ETA_CLIP,
    POISSON_IRLS_MAX_ITERATIONS, POISSON_IRLS_TOLERANCE, SCORE_TRIAL_START, SCORE_TRIAL_STOP,
    SessionCounts, WITHIN_TRIAL_THINNING_SEEDS, run_all_candidates_parallel,
)


SOURCE_AUDIT = SUA / "results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json"
SOURCE_AUDIT_SHA256 = "42a48b978c4c3a27f35239adcc1c196dc8f3aaadfa7d4234955b96ee0b7c119d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def source_only_manifest_receipt(manifest_path: Path) -> dict:
    manifest = load_frozen_source_development_manifest(manifest_path)
    return {
        "manifest_path": manifest["manifest_path"], "manifest_sha256": manifest["manifest_sha256"],
        "source_session_names": list(manifest["nested_source_session_names"]),
        "sealed_development_names": list(manifest["development_validation_names"]),
        "sealed_formal_names": list(manifest["sealed_formal_test_names"]),
        "development_paths_resolved": False, "formal_paths_resolved": False,
    }


def build_dry_run_receipt(manifest_path: Path) -> dict:
    if sha256(SOURCE_AUDIT) != SOURCE_AUDIT_SHA256:
        raise ValueError("Step-2A source-only audit hash drift; refuse Experiment B binding")
    manifest = source_only_manifest_receipt(manifest_path)
    if len(manifest["source_session_names"]) != N_SOURCE_SESSIONS:
        raise ValueError("Experiment B requires exact 27 source session names")
    module = SUA / "mc_maze/t4_estimator_b_v1.py"
    return {
        "schema_version": "t4_estimator_b_v1_prelaunch_v1", "mode": "dry_run_only",
        "no_gpu_launch": True, "cuda_visible_devices": "", "no_dataset_opened": True,
        "no_development_session_opened": True, "no_formal_session_opened": True,
        "manifest": manifest,
        "reuse_binding_only": {"source_audit_path": str(SOURCE_AUDIT.resolve()), "source_audit_sha256": SOURCE_AUDIT_SHA256,
                                "explicitly_not_reused": "q_unit_plus_M_prediction_or_winner_evidence"},
        "chronology": {"rewarded_trial_order": "existing_datamodule_selector", "total_trials_loaded": 50,
                       "support": [0, N_CALIBRATION_TRIALS], "score": [SCORE_TRIAL_START, SCORE_TRIAL_STOP],
                       "support_score_disjoint": True, "signal_view": "sua"},
        "execution_plan": {"source_read": "each of the 27 source sessions is read once into caller-owned RAM",
                           "candidate_compute": "three pre-registered candidates run concurrently only after that shared read",
                           "outer_fold": "27-fold source LOSO; EB prior and all tuned hyperparameters exclude held fold",
                           "raw_counts_persisted": False},
        "candidates": {
            "eb_ridge": {"prior": "outer26_only_nonzero_mean_full_covariance_empirical_bayes_not_w3", "lambda_grid": [0.25, 1.0, 4.0], "selection": "inner_LOSO_only"},
            "second_harmonic": {"fit": "[1,cos,sin,cos2,sin2]", "export": "[a1,c1,hypot(a1,c1),b]", "score": "exported_first_harmonic_only"},
            "poisson_irls": {"max_iterations": POISSON_IRLS_MAX_ITERATIONS, "tolerance": POISSON_IRLS_TOLERANCE,
                              "eta_clip": list(POISSON_ETA_CLIP), "export_and_score": "quadrature_projected_rate_space_first_harmonic_only"},
        },
        "reliability": {"seeds": list(WITHIN_TRIAL_THINNING_SEEDS), "partition": "disjoint_two_way_multinomial_within_trial_rescaled_rate",
                        "unit_stat": "paired_common_valid_per_unit_[a,c]_cosine_median", "aggregate": "median_across_8_seeds",
                        "undefined_when_direction_norm_lte": 1.0e-8},
        "selection_gate": {"mean_deviance_ratio_lte": 0.98, "mean_reliability_delta_gte": 0.02,
                           "joint_nonworse_folds_gte": "20/27", "no_rank_increase": True, "no_nonconvergence_increase": True,
                           "no_invalid_descriptor_increase": True,
                           "irreversible_fail_fast": "stop candidate when joint-failure folds > 7"},
        "winner_tie_break": ["lower_mean_prospective_deviance_ratio", "lower_calibration_ops_proxy", "smaller_persistent_state_bytes", "remaining_tie_no_gpu_winner"],
        "implementation_hashes": {"module": sha256(module), "runner": sha256(Path(__file__))},
    }


def resolve_source_paths(manifest_receipt: dict, data_dir: Path) -> dict[str, Path]:
    """Resolve exactly source names; never resolve the sealed val/test names."""
    base = Path(data_dir).resolve() / "sub-C"
    if not base.is_dir():
        raise FileNotFoundError(f"missing sub-C data directory: {base}")
    paths: dict[str, Path] = {}
    for name in manifest_receipt["source_session_names"]:
        path = (base / f"{name}_behavior+ecephys.nwb").resolve()
        if path.parent != base or not path.is_file():
            raise FileNotFoundError(f"missing frozen source session: {path}")
        paths[name] = path
    return paths


def load_source_session(name: str, path: Path) -> tuple[SessionCounts, dict]:
    """Read the exact first 50 rewarded source trials once, retaining counts only in RAM."""
    trials = list_datamodule_rewarded_trials(path, bin_size_ms=20, window_size=50, trial_result_filter="R")
    if len(trials) < SCORE_TRIAL_STOP:
        raise ValueError(f"{name}: fewer than 50 rewarded trials")
    selected = trials[:SCORE_TRIAL_STOP]
    starts = np.asarray([float(row["start_time"]) for row in selected], dtype=np.float64)
    stops = np.asarray([float(row["stop_time"]) for row in selected], dtype=np.float64)
    if np.any(np.diff(starts) < 0.0) or np.any(stops <= starts):
        raise ValueError(f"{name}: chronology/boundary failure")
    raw_dirs = [row.get("target_dir") for row in selected]
    directions = canonical_direction_indices(raw_dirs)
    counts = pool_trial_count_matrix_from_receipt(path, trial_start_times=starts, trial_stop_times=stops, signal_view="sua")
    session = SessionCounts(name=name, counts=counts, durations_s=stops - starts, direction_indices=directions)
    return session, {"source_sha256": sha256(path), "unit_count": int(counts.shape[0]),
                     "chronology": {"rewarded_trial_count": 50, "ordinals": list(range(50)),
                                    "direction_indices": directions, "durations_s": stops - starts}}


def execute_source_only(manifest_path: Path, data_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    prelaunch = build_dry_run_receipt(manifest_path)
    paths = resolve_source_paths(prelaunch["manifest"], data_dir)
    sessions: list[SessionCounts] = []; source_receipts: dict[str, dict] = {}
    for name in prelaunch["manifest"]["source_session_names"]:
        session, receipt = load_source_session(name, paths[name]); sessions.append(session); source_receipts[name] = receipt
    result = run_all_candidates_parallel(sessions)
    output = {"schema_version": "t4_estimator_b_v1_source_only_cpu_audit_v1", "mode": "executed_source_only_cpu",
              "no_gpu_launch": True, "no_development_session_opened": True, "no_formal_session_opened": True,
              "prelaunch": prelaunch, "source_sessions": source_receipts, "result": result,
              "contains_raw_counts": False, "decoder_or_gpu_authorized": False}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "aggregate.json").write_text(json.dumps(strict_json(output), indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-source-only", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if int(args.dry_run) + int(args.execute_source_only) != 1:
        parser.error("choose exactly one of --dry-run or --execute-source-only")
    if args.execute_source_only:
        if os.environ.get("T4_ESTIMATOR_B_V1_REVIEWED_SOURCE_ONLY") != "YES":
            parser.error("real source audit requires T4_ESTIMATOR_B_V1_REVIEWED_SOURCE_ONLY=YES")
        if args.data_dir is None or args.output_dir is None:
            parser.error("--execute-source-only requires --data-dir and --output-dir")
        print(json.dumps(strict_json(execute_source_only(args.manifest, args.data_dir, args.output_dir)), indent=2, sort_keys=True))
        return
    receipt = build_dry_run_receipt(args.manifest)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(strict_json(receipt), indent=2, sort_keys=True) + "\n")
    print(json.dumps(strict_json(receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
