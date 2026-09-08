#!/usr/bin/env python3
"""Read-only 27-tag functional verification of the H1 source-plan replay.

Uses a prior replayed-plan NPZ as-is.  It never selects parameters, aligns
signs, changes a formula, scores query data, or mutates frozen artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

WS = Path(__file__).resolve().parents[3]
ROOT = WS / "btransform_unified_v2"
HIST = Path("/home/xinyuan/Work_host/ibci_c3_film/SPINT-main")
AUTHORITY = Path("/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1/source_authority")
SOURCE_PAYLOAD = WS / "SPINT-main/local_data/h1_epfilm_evalai_v1/decoder.pt"
RIFT_PAYLOAD = WS / "tfpd_exploration/submissions/evalai_h1_rift_r300_cached_v1/artifacts/h1_rift_r300_recency_e22.pkl"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def import_historical() -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(HIST))
    from src.data import h1_m4_eb_pilot as pilot  # noqa: PLC0415
    from src.data.h1_cal_aug_all_source_heldout_v1 import index_heldout_calib  # noqa: PLC0415
    require(Path(pilot.__file__).resolve().is_relative_to(HIST), "pilot import escaped historical checkout")
    return pilot, index_heldout_calib, None


def load_public_heldout_m3(path: Path, pilot: Any) -> Any:
    """The source payload packer's public-calibration-only held-out semantics.

    This is a local helper, intentionally separate from the historical M4
    query loader.  Its direct source is
    ``h1_m3_readout_calibration_v1/package.py::_load_heldout`` lines 121-168:
    exactly three public trials are valid and no fourth/fifth query trial is
    requested.
    """
    from falcon_challenge.config import FalconTask  # noqa: PLC0415
    from falcon_challenge.dataloaders import load_nwb  # noqa: PLC0415
    from pynwb import NWBHDF5IO  # noqa: PLC0415

    resolved = path.resolve()
    require("sub-HumanPitt-held-out-calib" in str(resolved), "held-out helper scope drift")
    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as handle:
        trial_num = np.asarray(handle.read().acquisition["TrialNum"].data[:], dtype=np.float64)
    spikes64 = np.asarray(neural, np.float64)
    targets64 = np.asarray(velocity, np.float64)
    spikes = spikes64.astype(np.float32)
    targets = targets64.astype(np.float32)
    changes = np.asarray(trial_change, bool).reshape(-1)
    mask = np.asarray(eval_mask, bool).reshape(-1)
    require(spikes.ndim == 2 and spikes.shape[1] == pilot.EXPECTED_NEURONS, "held-out neural shape drift")
    require(targets.shape == (len(spikes), 7), "held-out target shape drift")
    ordered = trial_num[mask & np.isfinite(trial_num)]
    require(ordered.size > 0 and not np.any(np.diff(ordered) < 0), "held-out TrialNum order drift")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    require(len(values) == 3, "official held-out calibration must contain exactly public M3")
    trials = tuple(pilot._trial_blocks(value, spikes64, targets64, mask, trial_num) for value in values)
    name = pilot.session_from_path(resolved)
    return pilot.H1PilotRecord(name, pilot.session_date(name), resolved, sha_file(resolved), spikes, targets,
                               changes, mask, trial_num, tuple(values), trials)


def compare(replayed: np.ndarray, frozen: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(replayed, np.float64) - np.asarray(frozen, np.float64)
    denom = float(np.linalg.norm(np.asarray(frozen, np.float64)))
    lhs = float(np.linalg.norm(np.asarray(replayed, np.float64)))
    return {
        "byte_equal": bool(np.array_equal(replayed, frozen)),
        "max_abs_error": float(np.max(np.abs(delta))),
        "relative_frobenius_error": None if denom == 0.0 else float(np.linalg.norm(delta) / denom),
        "cosine": None if lhs == 0.0 or denom == 0.0 else float(np.vdot(replayed.ravel().astype(np.float64), frozen.ravel().astype(np.float64)) / (lhs * denom)),
    }


def load_rift(path: Path) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):  # type: ignore[override]
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda value: torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
            return super().find_class(module, name)

    with path.open("rb") as f:
        value = CPUUnpickler(f).load()
    require(isinstance(value, dict) and "bank_by_dataset_tag" in value, "invalid frozen RIFT payload")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replayed-plan", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    args = ap.parse_args()
    require(not args.dest.exists(), f"destination must be fresh: {args.dest}")
    require(args.replayed_plan.is_file(), f"replayed plan missing: {args.replayed_plan}")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""), "CPU-only requires CUDA_VISIBLE_DEVICES=''")
    require(sorted(os.sched_getaffinity(0)) == [14, 15], "launch with taskset -c 14,15")
    require(all(os.environ.get(key) == "2" for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "launch with OMP/MKL/OPENBLAS thread counts all set to 2")
    import torch  # noqa: PLC0415
    torch.set_num_threads(2); torch.set_num_interop_threads(1)

    plan_receipt = json.loads((AUTHORITY / "plan.json").read_text())
    normalizer = json.loads((AUTHORITY / "normalizer.json").read_text())
    with np.load(args.replayed_plan, allow_pickle=False) as archive:
        needed = {"mean", "scale", "pcs", "U", "mu", "tau2", "q", "lambda"}
        require(set(archive.files) == needed, f"replayed plan schema drift: {archive.files}")
        arrays = {name: np.asarray(archive[name]) for name in needed}
    require(int(arrays["q"]) == 12 and float(arrays["lambda"]) == 10.0, "replayed plan is not q12/lambda10")
    plan = SimpleNamespace(mean=np.asarray(arrays["mean"], np.float64), scale=np.asarray(arrays["scale"], np.float64), pcs=np.asarray(arrays["pcs"], np.float64), U=np.asarray(arrays["U"], np.float64), mu=np.asarray(arrays["mu"], np.float64), tau2=float(arrays["tau2"]), q=int(arrays["q"]), ridge_lambda=float(arrays["lambda"]))

    pilot, index_heldout_calib, _ = import_historical()
    observed_plan_hashes = {name: pilot.array_sha256(getattr(plan, name)) for name in ("mean", "scale", "pcs", "U", "mu")}
    expected_plan_hashes = dict(plan_receipt["array_sha256"])
    plan_hash_difference = {name: {"replayed": observed_plan_hashes[name], "sealed_expected": expected_plan_hashes[name], "equal": observed_plan_hashes[name] == expected_plan_hashes[name]} for name in observed_plan_hashes}

    source = torch.load(SOURCE_PAYLOAD, map_location="cpu", weights_only=False)
    rift = load_rift(RIFT_PAYLOAD)
    source_rows = source.get("sessions", {})
    frozen_rows = rift["bank_by_dataset_tag"]
    require(set(source_rows) == set(frozen_rows) and len(source_rows) == 27, "source/frozen tag roster drift")
    heldin_paths = pilot.index_heldin_calib(HIST / "data/000954")
    heldout_paths = index_heldout_calib(HIST / "data/000954")
    rows: dict[str, Any] = {}
    started = time.perf_counter()
    for tag in sorted(source_rows):
        source_row = source_rows[tag]
        session = str(source_row["session"])
        trials = tuple(float(value) for value in source_row["calibration_trials"])
        require(len(trials) == 3 and len(set(trials)) == 3, f"{tag}: source payload is not exact public M3")
        if session in heldin_paths:
            path = heldin_paths[session]
            record = pilot.load_record(path)
            surface = "held-in-calib"
        else:
            require(session in heldout_paths, f"{tag}: no public calibration path for {session}")
            path = heldout_paths[session]
            record = load_public_heldout_m3(path, pilot)
            surface = "held-out-calib"
        require(tuple(float(value) for value in record.trial_values[:3]) == trials, f"{tag}: first public three trials differ from source payload")
        raw = pilot.fit_deployment_carrier(record, plan, trials)["carrier"]
        replayed_t = np.ascontiguousarray(raw / max(float(normalizer["s_src"]), 1.0e-12), dtype=np.float32)
        source_t = np.ascontiguousarray(source_row["carrier"], dtype=np.float32)
        frozen_t = np.ascontiguousarray(frozen_rows[tag]["T"], dtype=np.float32)
        rows[tag] = {"session": session, "surface": surface, "public_input": {"path": str(path), "sha256": sha_file(path)}, "trials": list(trials), "replayed_T_sha256": sha_array(replayed_t), "source_T_sha256": sha_array(source_t), "frozen_T_sha256": sha_array(frozen_t), "replayed_vs_source": compare(replayed_t, source_t), "replayed_vs_frozen": compare(replayed_t, frozen_t), "source_vs_frozen_byte_equal": bool(np.array_equal(source_t, frozen_t))}
    elapsed = time.perf_counter() - started
    all_source_equal = all(bool(row["replayed_vs_source"]["byte_equal"]) for row in rows.values())
    all_frozen_equal = all(bool(row["replayed_vs_frozen"]["byte_equal"]) for row in rows.values())
    source_frozen_equal = all(bool(row["source_vs_frozen_byte_equal"]) for row in rows.values())
    status = "PASS_ALL_27_FUNCTIONAL_REPLAY_T_BYTE_EQUAL" if all_source_equal and all_frozen_equal and source_frozen_equal else "SOURCE_PLAN_REPLAY_27_T_MISMATCH"
    packer_loader = WS / "tfpd_exploration/h1_series_20260830/src/h1_m3_readout_calibration_v1/package.py"
    report = {"schema": "h1_source_plan_replay_verify27_v1", "status": status, "utc": utc(), "scope": {"source_only_public_calibration": True, "tags": 27, "date_selection_called": False, "parameter_adjustment": False, "sign_or_basis_alignment": False, "query_or_target_opened": False, "scoring": False, "training": False, "optimizer_steps": 0}, "cpu": {"affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads()}, "source_functions": {"carrier_operator": str(Path(pilot.__file__).resolve()), "heldout_public_m3_loader_semantics": str(packer_loader) + "::_load_heldout", "heldout_loader_source_sha256": sha_file(packer_loader), "trial_mask": "eval_mask & finite TrialNum; chronological distinct values; exactly three trials; each trial feeds pilot._trial_blocks"}, "inputs": {"replayed_plan": {"path": str(args.replayed_plan), "container_sha256": sha_file(args.replayed_plan)}, "sealed_plan_receipt": {"path": str(AUTHORITY / "plan.json"), "sha256": sha_file(AUTHORITY / "plan.json")}, "source_payload": {"path": str(SOURCE_PAYLOAD), "sha256": sha_file(SOURCE_PAYLOAD)}, "frozen_rift_payload": {"path": str(RIFT_PAYLOAD), "sha256": sha_file(RIFT_PAYLOAD)}, "source_rms": normalizer["s_src"]}, "plan_array_hashes": {"note": "FP64 array hash differences are recorded as provenance differences and are not by themselves a functional T mismatch.", "per_array": plan_hash_difference}, "verification": {"elapsed_seconds": elapsed, "all_27_replayed_vs_source_byte_equal": all_source_equal, "all_27_replayed_vs_frozen_byte_equal": all_frozen_equal, "all_27_source_vs_frozen_byte_equal": source_frozen_equal}, "tags": rows}
    args.dest.mkdir(parents=True)
    (args.dest / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.dest / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
