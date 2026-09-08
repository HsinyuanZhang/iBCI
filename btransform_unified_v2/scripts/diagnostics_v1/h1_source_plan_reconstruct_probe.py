#!/usr/bin/env python3
"""Read-only source-only replay of the frozen H1 582073 M3 carrier plan.

The probe never invokes candidate selection, target/query readers, scoring, or
training.  It uses the sealed q=12/lambda=10 selection and the current
historical source-only plan builder on exactly 13 public held-in calibrations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

WS = Path(__file__).resolve().parents[3]
ROOT = WS / "btransform_unified_v2"
HIST = Path("/home/xinyuan/Work_host/ibci_c3_film/SPINT-main")
AUTHORITY = Path("/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1/source_authority")
BINDING = ROOT / "docs/H1_FROZEN_CARRIER_METHOD_BINDING_20260908.json"
TAG = "S0_set_1"
SESSION = "ses-19250101T111740"
TRIALS = (3.0, 4.0, 5.0)
EXPECTED_T_SHA = "85d4801852554164421218d030404d07d8bcfea7795296e1ae28ffc82d17e2a2"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def import_historical_source() -> tuple[Any, Any]:
    """Import only the historical checkout's source-only carrier modules."""
    require(HIST.is_dir(), f"historical source checkout missing: {HIST}")
    text = str(HIST)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)
    from src.data import h1_m4_eb_pilot as pilot  # noqa: PLC0415
    from src import h1_hc_date_lodo_regen_v1 as regen  # noqa: PLC0415
    require(Path(pilot.__file__).resolve().is_relative_to(HIST), "pilot module escaped historical source checkout")
    require(Path(regen.__file__).resolve().is_relative_to(HIST), "plan module escaped historical source checkout")
    return pilot, regen


def plan_arrays(plan: Any) -> dict[str, np.ndarray]:
    return {
        "mean": np.asarray(plan.mean, dtype=np.float64),
        "scale": np.asarray(plan.scale, dtype=np.float64),
        "pcs": np.asarray(plan.pcs, dtype=np.float64),
        "U": np.asarray(plan.U, dtype=np.float64),
        "mu": np.asarray(plan.mu, dtype=np.float64),
        "tau2": np.asarray(plan.tau2, dtype=np.float64),
        "q": np.asarray(plan.q, dtype=np.int64),
        "lambda": np.asarray(plan.ridge_lambda, dtype=np.float64),
    }


def array_comparison(replayed: np.ndarray, frozen: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(replayed, np.float64) - np.asarray(frozen, np.float64)
    denom = float(np.linalg.norm(np.asarray(frozen, np.float64)))
    norm = float(np.linalg.norm(np.asarray(replayed, np.float64)))
    cosine = float(np.vdot(replayed.ravel().astype(np.float64), frozen.ravel().astype(np.float64)) / (norm * denom)) if norm > 0.0 and denom > 0.0 else None
    return {
        "byte_equal": bool(np.array_equal(replayed, frozen)),
        "max_abs_error": float(np.max(np.abs(delta))),
        "relative_frobenius_error": float(np.linalg.norm(delta) / denom) if denom > 0.0 else None,
        "cosine": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True, help="new result directory")
    args = parser.parse_args()
    require(not args.dest.exists(), f"destination must be fresh; refusing overwrite: {args.dest}")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""), "CPU-only probe requires CUDA_VISIBLE_DEVICES=''")
    require(sorted(os.sched_getaffinity(0)) == [14, 15], "launch with taskset -c 14,15")
    require(os.environ.get("OMP_NUM_THREADS") == "2", "launch with OMP_NUM_THREADS=2")
    require(os.environ.get("MKL_NUM_THREADS") == "2", "launch with MKL_NUM_THREADS=2")
    require(os.environ.get("OPENBLAS_NUM_THREADS") == "2", "launch with OPENBLAS_NUM_THREADS=2")
    import torch  # noqa: PLC0415
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)

    selection_path = AUTHORITY / "selection.json"
    plan_path = AUTHORITY / "plan.json"
    normalizer_path = AUTHORITY / "normalizer.json"
    for path in (selection_path, plan_path, normalizer_path, BINDING):
        require(path.is_file(), f"required sealed receipt missing: {path}")
    selection = json.loads(selection_path.read_text())
    plan_expected = json.loads(plan_path.read_text())
    normalizer = json.loads(normalizer_path.read_text())
    binding = json.loads(BINDING.read_text())
    selected = selection.get("selected")
    require(isinstance(selected, dict) and int(selected.get("q", -1)) == 12 and float(selected.get("lambda", -1)) == 10.0, "sealed selection is not q12/lambda10")
    require(int(plan_expected.get("q", -1)) == 12 and float(plan_expected.get("lambda", -1)) == 10.0, "sealed plan receipt is not q12/lambda10")
    require(float(normalizer.get("s_src", -1.0)) == float(binding["paper_binding"]["bound_parameters"]["s_src"]), "source RMS receipt drift")
    require(binding["array_lineage"]["s0_set_1"]["source_candidate_T_sha256"] == EXPECTED_T_SHA, "binding T anchor drift")

    pilot, regen = import_historical_source()
    data_root = HIST / "data/000954"
    indexed = pilot.index_heldin_calib(data_root)
    require(tuple(indexed) == tuple(pilot.H1_HELDIN_SESSIONS) and len(indexed) == 13, "expected exact 13-source roster")
    # No _date_selection call: q/lambda are frozen above.  All reads below are
    # public held-in calibration files; no heldout/minival/query path is used.
    records = {name: pilot.load_record(indexed[name]) for name in pilot.H1_HELDIN_SESSIONS}
    require(tuple(records) == tuple(pilot.H1_HELDIN_SESSIONS), "record order drift")
    raw_hashes = {name: sha_file(indexed[name]) for name in records}
    require(tuple(raw_hashes.values()) == tuple(plan_expected["source_input_sha256"]), "13 public source NWB hashes drift from sealed plan")

    selection_sha = sha_file(selection_path)
    t0 = time.perf_counter()
    plan = regen._make_final_plan(records, "h1_all_source_13", selected, selection_sha)
    source_plan_build_seconds = time.perf_counter() - t0
    arrays = plan_arrays(plan)
    expected_array_sha = dict(plan_expected["array_sha256"])
    observed_array_sha = {name: pilot.array_sha256(arrays[name]) for name in ("mean", "scale", "pcs", "U", "mu")}
    plan_array_matches = {name: observed_array_sha[name] == expected_array_sha[name] for name in observed_array_sha}
    plan_transform_match = str(plan.transform_sha256) == str(plan_expected["transform_sha256"])

    record = records[SESSION]
    require(tuple(float(value) for value in record.trial_values[:3]) == TRIALS, "S0 public M3 trials are not [3,4,5]")
    # Stop the target carrier timer before any identity/hash/metric work.
    t0 = time.perf_counter()
    raw = pilot.fit_deployment_carrier(record, plan, TRIALS)["carrier"]
    replayed_t = np.ascontiguousarray(raw / max(float(normalizer["s_src"]), 1.0e-12), dtype=np.float32)
    s0_m3_target_solve_seconds = time.perf_counter() - t0

    # The frozen T is loaded after the solve timer so artifact I/O cannot be
    # represented as solver cost.  It is used only for the comparison.
    source_payload = WS / "SPINT-main/local_data/h1_epfilm_evalai_v1/decoder.pt"
    payload = torch.load(source_payload, map_location="cpu", weights_only=False)
    frozen_t = np.ascontiguousarray(payload["sessions"][TAG]["carrier"], dtype=np.float32)
    require(sha_array(frozen_t) == EXPECTED_T_SHA, "frozen S0 carrier anchor drift")
    comparison = array_comparison(replayed_t, frozen_t)
    status = "PASS_SOURCE_PLAN_REPLAY_EXACT" if all(plan_array_matches.values()) and plan_transform_match and comparison["byte_equal"] else "SOURCE_PLAN_REPLAY_MISMATCH"

    args.dest.mkdir(parents=True)
    np.savez_compressed(args.dest / "replayed_plan_arrays.npz", **arrays)
    output_npz = args.dest / "replayed_plan_arrays.npz"
    report = {
        "schema": "h1_source_plan_reconstruct_probe_v1",
        "status": status,
        "utc": utc(),
        "scope": {"source_only": True, "heldin_calibration_nwbs_opened": 13, "date_selection_called": False, "target_or_query_opened": False, "scoring": False, "training": False, "optimizer_steps": 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "cpu": {"affinity": sorted(os.sched_getaffinity(0)), "torch_num_threads": torch.get_num_threads(), "omp_num_threads": os.environ["OMP_NUM_THREADS"], "mkl_num_threads": os.environ["MKL_NUM_THREADS"], "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"]},
        "sealed_inputs": {"selection": {"path": str(selection_path), "sha256": selection_sha, "selected": {"q": selected["q"], "lambda": selected["lambda"]}}, "plan_receipt": {"path": str(plan_path), "sha256": sha_file(plan_path), "expected_plan_npz_sha256": plan_expected["arrays_file_sha256"], "expected_transform_sha256": plan_expected["transform_sha256"], "expected_array_sha256": expected_array_sha}, "normalizer": {"path": str(normalizer_path), "sha256": sha_file(normalizer_path), "s_src": normalizer["s_src"]}, "binding": {"path": str(BINDING), "sha256": sha_file(BINDING), "submission_id": binding["submission"]["id"]}},
        "current_source_code": {str(Path(pilot.__file__).resolve()): sha_file(Path(pilot.__file__).resolve()), str(Path(regen.__file__).resolve()): sha_file(Path(regen.__file__).resolve())},
        "source_roster": [{"session": name, "path": str(indexed[name]), "sha256": raw_hashes[name]} for name in records],
        "source_plan_build": {"seconds": source_plan_build_seconds, "outer_domain": "h1_all_source_13", "plan_transform_sha256": plan.transform_sha256, "transform_matches_sealed": plan_transform_match, "array_sha256": observed_array_sha, "array_matches_sealed": plan_array_matches, "replayed_arrays_npz": {"path": str(output_npz), "container_sha256": sha_file(output_npz), "note": "container SHA may differ from the sealed NPZ due to ZIP metadata/compression; array schema and values above are the comparison authority."}},
        "s0_m3_target_solve": {"session": SESSION, "tag": TAG, "trials": list(TRIALS), "seconds": s0_m3_target_solve_seconds, "timer_scope": "fit_deployment_carrier + source-RMS normalize + contiguous float32 cast; elapsed captured before frozen-T I/O, hashes, and metrics", "replayed_T_sha256": sha_array(replayed_t), "frozen_T_sha256": EXPECTED_T_SHA, "comparison": comparison},
    }
    (args.dest / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.dest / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
