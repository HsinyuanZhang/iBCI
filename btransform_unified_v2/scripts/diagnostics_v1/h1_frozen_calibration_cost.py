#!/usr/bin/env python3
"""Read-only CPU cost audit for frozen H1 submission 582073 calibration.

This audit is bound to the q=12/lambda=10 all-source M3 carrier used by the
frozen RIFT payload.  It deliberately distinguishes a measured cached carrier
from a fresh carrier solve: the immutable authority directory retained its JSON
receipts but not the sealed plan NPZ, so this checkout cannot honestly rerun
that numerical solve.  It does reconstruct the public M3 activity, materialize
C2 E0, and compare both frozen arrays against submission 582073.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT.parent
SPINT = WS / "SPINT-main"
TFPD = WS / "tfpd_exploration"
V1 = WS / "btransform_unified_v1"
BINDING = ROOT / "docs/H1_FROZEN_CARRIER_METHOD_BINDING_20260908.json"
SOURCE_PAYLOAD = SPINT / "local_data/h1_epfilm_evalai_v1/decoder.pt"
RIFT_PAYLOAD = TFPD / "submissions/evalai_h1_rift_r300_cached_v1/artifacts/h1_rift_r300_recency_e22.pkl"
C2_RECEIPT = TFPD / "submissions/evalai_h1_c2_cal1_b2_ort_v1/artifacts/payload.receipt.json"
AUTHORITY = Path("/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1/source_authority")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_path(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def require_new_dest(dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(f"destination must be fresh; refusing to overwrite: {dest}")


def load_payload(path: Path) -> dict[str, Any]:
    # These payloads are trusted frozen local artifacts. torch.load preserves
    # their historical numpy/pickle representation and maps tensor storage CPU.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise RuntimeError(f"payload is not a mapping: {path}")
    return value


def load_rift_payload(path: Path) -> dict[str, Any]:
    """Use submission 582073's own CPU unpickler for its protocol-4 payload."""
    for candidate in (WS, V1 / "src", ROOT / "src", path.parent.parent):
        add_path(candidate)
    module_path = path.parent.parent / "h1_rift_falcon_decoder.py"
    spec = importlib.util.spec_from_file_location("_frozen_h1_rift_decoder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen RIFT payload reader: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.load_payload(path)
    if not isinstance(value, dict):
        raise RuntimeError("RIFT payload is not a mapping")
    return value


def binding_for(tag: str) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads(BINDING.read_text())
    if receipt["submission"]["id"] != 582073:
        raise RuntimeError("binding receipt is not submission 582073")
    params = receipt["paper_binding"]["bound_parameters"]
    if params["q"] != 12 or float(params["ridge_lambda"]) != 10.0:
        raise RuntimeError("refusing non-q12/lambda10 binding")
    if tag != "S0_set_1":
        raise ValueError("this calibrated cost audit is pinned to S0_set_1")
    return receipt, params


def fresh_public_activity(session: str, trials: tuple[float, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    """Reopen only S0 held-in public calibration and rebuild its M3 identity."""
    # ``src`` is also used by the RIFT package, so load the audited source
    # file under a private module name instead of allowing package-order drift.
    spec = importlib.util.spec_from_file_location("_h1_m4_eb_pilot_audited", SPINT / "src/data/h1_m4_eb_pilot.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load audited H-C source operator")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    index_heldin_calib = pilot.index_heldin_calib
    interpolate_trial_identity = pilot.interpolate_trial_identity
    load_record = pilot.load_record

    paths = index_heldin_calib(SPINT / "data/000954")
    path = paths[session]
    t0 = time.perf_counter()
    record = load_record(path)
    raw_load_seconds = time.perf_counter() - t0
    got = tuple(float(value) for value in record.trial_values[:3])
    if got != trials:
        raise RuntimeError(f"{session}: expected public M3 {trials}, saw {got}")
    t0 = time.perf_counter()
    activity = np.ascontiguousarray(
        np.stack([interpolate_trial_identity(record, value) for value in trials]), dtype=np.float32
    )
    activity_build_seconds = time.perf_counter() - t0
    return activity, {
        "path": str(path), "sha256": sha_file(path), "raw_load_seconds": raw_load_seconds,
        "activity_build_seconds": activity_build_seconds, "session": record.session_name,
        "support_trials": list(trials), "activity": {"shape": list(activity.shape), "dtype": str(activity.dtype), "sha256": sha_array(activity), "nbytes": int(activity.nbytes)},
    }


def c2_materialize(activity: np.ndarray, carrier: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    add_path(WS)
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import load_frozen_c2_materializer

    t0 = time.perf_counter()
    materializer = load_frozen_c2_materializer().eval().to("cpu")
    checkpoint_load_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    e0, direct = materializer.materialize_bank(torch.from_numpy(activity), torch.from_numpy(carrier))
    embedding_build_seconds = time.perf_counter() - t0
    actual = np.ascontiguousarray(e0.numpy(), dtype=np.float32)
    copied_carrier = np.ascontiguousarray(direct.numpy(), dtype=np.float32)
    if not np.array_equal(copied_carrier, carrier):
        raise RuntimeError("C2 materializer altered direct frozen carrier")
    return actual, {
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "embedding_build_seconds": embedding_build_seconds,
        "c2_checkpoint_sha256": materializer.checkpoint_sha256,
        "e0": {"shape": list(actual.shape), "dtype": str(actual.dtype), "sha256": sha_array(actual), "nbytes": int(actual.nbytes)},
    }


def warm_carrier_e0_bank(activity: np.ndarray, frozen_t: np.ndarray, tag: str) -> dict[str, Any]:
    """One warm production bank construction; persist elapsed before any hash."""
    add_path(V1 / "src")
    add_path(WS)
    from btransform_unified_v1.bank import TaskBank
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import load_frozen_c2_materializer

    # Explicitly outside the timed path: this is a warm carrier/E0/bank cost.
    materializer = load_frozen_c2_materializer().eval().to("cpu")
    t0 = time.perf_counter()
    carrier = np.ascontiguousarray(frozen_t.copy(), dtype=np.float32)
    e0, direct = materializer.materialize_bank(torch.from_numpy(activity), torch.from_numpy(carrier))
    bank = TaskBank(
        session_id=tag, E0=np.ascontiguousarray(e0.numpy(), dtype=np.float32),
        carrier=np.ascontiguousarray(direct.numpy(), dtype=np.float32),
        unit_mask=np.ones(carrier.shape[0], dtype=np.bool_),
        X_store=np.zeros((0, 300, carrier.shape[0]), dtype=np.float32),
        target_store=np.zeros((0, 7), dtype=np.float32), window_ids=np.zeros(0, dtype=np.int64),
        calibration_meta={"shape": (carrier.shape[0], 700), "trial_count": 3, "budget": 3,
                          "estimator": "frozen C2 M3 q12/lambda10 carrier", "array_sha256": "post_timer_identity"},
    )
    elapsed_seconds = time.perf_counter() - t0
    # All identities are deliberately after the timestamp, so hashing and
    # serialization cannot inflate the reported computation wall time.
    e0_value = np.asarray(bank.E0, dtype=np.float32)
    t_value = np.asarray(bank.carrier, dtype=np.float32)
    return {"elapsed_seconds": elapsed_seconds, "timer_scope": "warm frozen-T copy + C2 E0 materialization + TaskBank construction only; elapsed captured before hashes/serialization", "T": {"sha256": sha_array(t_value), "shape": list(t_value.shape), "nbytes": int(t_value.nbytes)}, "E0": {"sha256": sha_array(e0_value), "shape": list(e0_value.shape), "nbytes": int(e0_value.nbytes)}, "bank": {"session_id": bank.session_id, "unit_mask_true": int(np.asarray(bank.unit_mask).sum())}}


def trace_and_measure(tag: str) -> dict[str, Any]:
    receipt, params = binding_for(tag)
    expected = receipt["array_lineage"]["s0_set_1"]
    session = expected["session"]
    trials = tuple(float(value) for value in expected["calibration_trials"])

    # One cold source-object load. Its timer, object extraction timer, and the
    # later E2E timer are intentionally separate measurements.
    t0 = time.perf_counter()
    source = load_payload(SOURCE_PAYLOAD)
    source_payload_cold_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    row = source["sessions"][tag]
    source_activity = np.ascontiguousarray(row["identity"], dtype=np.float32)
    cached_carrier = np.ascontiguousarray(row["carrier"], dtype=np.float32)
    source_object_extract_seconds = time.perf_counter() - t0
    if tuple(float(x) for x in row["calibration_trials"]) != trials:
        raise RuntimeError("source payload M3 trial binding drift")
    if sha_array(cached_carrier) != expected["source_candidate_T_sha256"]:
        raise RuntimeError("source candidate carrier bytes drift")

    t0 = time.perf_counter()
    rift = load_rift_payload(RIFT_PAYLOAD)
    rift_payload_cold_seconds = time.perf_counter() - t0
    frozen_row = rift["bank_by_dataset_tag"][tag]
    frozen_t = np.ascontiguousarray(frozen_row["T"], dtype=np.float32)
    frozen_e0 = np.ascontiguousarray(frozen_row["E0"], dtype=np.float32)
    if not np.array_equal(cached_carrier, frozen_t):
        raise RuntimeError("source M3 carrier is not byte-identical to frozen RIFT T")
    if sha_array(frozen_t) != expected["b2_T_sha256"]:
        raise RuntimeError("frozen T SHA drift")

    rebuilt_activity, raw = fresh_public_activity(session, trials)
    if not np.array_equal(rebuilt_activity, source_activity):
        raise RuntimeError("fresh public M3 activity differs from immutable source payload")
    rebuilt_e0, embedding = c2_materialize(rebuilt_activity, frozen_t)
    e0_abs = float(np.max(np.abs(rebuilt_e0 - frozen_e0)))
    e0_byte_equal = bool(np.array_equal(rebuilt_e0, frozen_e0))
    receipt_e0 = json.loads(C2_RECEIPT.read_text())["banks"]["per_tag"][tag]["e0_sha256"]
    if sha_array(frozen_e0) != receipt_e0:
        raise RuntimeError("frozen RIFT E0 does not match C2 receipt")

    warm = warm_carrier_e0_bank(rebuilt_activity, frozen_t, tag)
    if warm["E0"]["sha256"] != sha_array(frozen_e0):
        raise RuntimeError("warm carrier/E0/bank path does not reproduce frozen E0")

    t0 = time.perf_counter()
    serialized = pickle.dumps({"activity": rebuilt_activity, "T": frozen_t, "E0": rebuilt_e0}, protocol=4)
    serialization_seconds = time.perf_counter() - t0
    memory_nbytes = int(rebuilt_activity.nbytes + frozen_t.nbytes + rebuilt_e0.nbytes)
    authority_missing = [name for name in ("plan.npz", "arrays.npz") if not (AUTHORITY / name).is_file()]
    solve = {
        "status": "UNTRACEABLE_NO_SEALED_NUMERIC_PLAN_ARRAYS",
        "measured_seconds": None,
        "reason": "The binding receipts preserve q=12/lambda=10/EB/source-RMS metadata, but this checkout lacks the immutable plan NPZ containing mean/scale/pcs/U/mu needed for a fresh exact solve.",
        "authority_dir": str(AUTHORITY), "missing_required_files": authority_missing,
        "cached_frozen_T": {"sha256": sha_array(frozen_t), "shape": list(frozen_t.shape), "nbytes": int(frozen_t.nbytes)},
    }
    return {
        "schema": "h1_frozen_calibration_cost_v1", "status": "COMPLETED_WITH_CARRIER_SOLVE_PROVENANCE_GAP", "utc": utc(),
        "submission": {"id": 582073, "payload": str(RIFT_PAYLOAD), "payload_sha256": sha_file(RIFT_PAYLOAD)},
        "binding": {"receipt": str(BINDING), "receipt_sha256": sha_file(BINDING), "q": params["q"], "ridge_lambda": params["ridge_lambda"], "empirical_bayes_shrinkage": True, "source_rms": params["s_src"], "support_trials": list(trials)},
        "scope": {"cpu_only": True, "official_test_used": False, "target_query_used": False, "optimizer_steps": 0, "repeats": 1, "session": session, "tag": tag},
        "source_payload": {"path": str(SOURCE_PAYLOAD), "sha256": sha_file(SOURCE_PAYLOAD), "cold_load_seconds": source_payload_cold_seconds, "object_extract_seconds": source_object_extract_seconds, "cached_activity": {"sha256": sha_array(source_activity), "shape": list(source_activity.shape)}, "cached_T": {"sha256": sha_array(cached_carrier), "shape": list(cached_carrier.shape)}},
        "public_raw_m3": raw,
        "carrier_solve": solve,
        "frozen_rift_load": {"cold_load_seconds": rift_payload_cold_seconds, "T_sha256": sha_array(frozen_t), "E0_sha256": sha_array(frozen_e0), "E0_receipt_sha256": receipt_e0},
        "embedding_build": {**embedding, "frozen_e0_sha256": sha_array(frozen_e0), "byte_equal": e0_byte_equal, "max_abs_error": e0_abs, "tolerance": 1.0e-6, "passes_tolerance": e0_abs <= 1.0e-6},
        "warm_carrier_e0_bank": warm,
        "serialization_and_memory": {"pickle_seconds": serialization_seconds, "pickle_nbytes": len(serialized), "array_memory_nbytes": memory_nbytes, "array_sha256": {"activity": sha_array(rebuilt_activity), "T": sha_array(frozen_t), "E0": sha_array(rebuilt_e0)}},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--session", default="S0_set_1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CPU-only audit requires CUDA_VISIBLE_DEVICES='' ")
    os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    require_new_dest(args.dest)
    report = trace_and_measure(args.session)
    report["cpu"] = {"torch_threads": torch.get_num_threads(), "affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0)}
    report["mode"] = "preflight" if args.preflight_only else "cost_audit"
    args.dest.mkdir(parents=True)
    (args.dest / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.dest / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
