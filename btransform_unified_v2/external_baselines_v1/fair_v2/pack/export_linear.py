#!/usr/bin/env python3
"""Export label-free fair_v2 linear payloads from sealed models.pkl."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

PACK = Path(__file__).resolve().parent
if str(PACK.parent.parent) not in sys.path:
    sys.path.insert(0, str(PACK.parent.parent))
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))

from fair_v2 import data
from fair_v2.numerics import AlignedFAMap, CoralMap, FactorModel, Standardizer
from common import (
    LINEAR_PROTOCOL,
    LOCAL_SCORES,
    RESULTS,
    SELECTION_BUDGET,
    TASKS,
    prefix_for,
    sha256_file,
)

HISTORY = 10
SCHEMA = "fair_v2_linear_falcon_payload_v1"
ARMS = ("diag_z_wf", "coral_wf", "aligned_fa_stable_wf")


def wf_kernel() -> np.ndarray:
    t = np.arange(12, dtype=np.float64) * 20.0
    kernel = np.exp(-t / 240.0)
    return kernel / kernel.sum()


def np_numeric(x: Any, *, dtype=np.float64) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(x, dtype=dtype))
    if a.dtype == object or not np.isfinite(a).all():
        raise ValueError("payload array is non-numeric or non-finite")
    return a


def compose_affine(adapter: Any, channels: int, feature_dim: int) -> tuple[np.ndarray, np.ndarray, str]:
    if adapter is None:
        return np.eye(channels, dtype=np.float64), np.zeros(channels, dtype=np.float64), "identity_after_z"
    if isinstance(adapter, CoralMap):
        matrix = np_numeric(adapter.matrix_)
        intercept = np_numeric(adapter.source_mean_) - np_numeric(adapter.target_mean_) @ matrix
        return matrix, intercept, "coral"
    if isinstance(adapter, FactorModel):
        matrix = np_numeric(adapter.posterior_matrix().T)
        intercept = -np_numeric(adapter.mean_) @ matrix
        return matrix, intercept, "source_fa_reference"
    if isinstance(adapter, AlignedFAMap):
        matrix = np_numeric(adapter.matrix_)
        intercept = np_numeric(adapter.intercept_)
        if adapter.posterior == "stable":
            full = np.zeros((channels, matrix.shape[1]), dtype=np.float64)
            rows = np.asarray(adapter.stable_rows, dtype=int)
            full[rows] = matrix
            return full, intercept, "aligned_fa_stable"
        return matrix, intercept, "aligned_fa_all"
    raise TypeError("unsupported adapter type: " + type(adapter).__name__)


def falcon_tag(task: str, basename: str) -> str:
    from falcon_challenge.config import FalconConfig, FalconTask

    return str(FalconConfig(getattr(FalconTask, task)).hash_dataset(basename))


def raw_basename(item: dict[str, Any], task: str, session: str, role: str) -> str:
    raw = item.get("support_provenance", {}).get("raw_nwb")
    if raw:
        return Path(raw).stem
    if task == "m2":
        token = "held-in" if role == "heldin" else "held-out"
        return f"sub-MonkeyN-held-{token}-calib_{session}_behavior+ecephys"
    raise ValueError(f"missing raw_nwb for {task}/{session}")


def load_fitted(task: str) -> dict[str, Any]:
    path = RESULTS / f"{task}_v2" / "models.pkl"
    receipt = json.loads((RESULTS / f"{task}_v2" / "receipt.json").read_text())
    if sha256_file(path) != receipt["models_sha256"]:
        raise RuntimeError(f"{task}: models.pkl hash drift")
    with path.open("rb") as handle:
        fitted = pickle.load(handle)
    return fitted


def replay_features(
    raw: np.ndarray,
    endpoints: np.ndarray,
    z_mean: np.ndarray,
    z_scale: np.ndarray,
    matrix: np.ndarray,
    intercept: np.ndarray,
    kernel: np.ndarray,
    history: int,
) -> np.ndarray:
    state = np.zeros((len(kernel), raw.shape[1]), dtype=np.float64)
    hist = np.zeros((history, matrix.shape[1]), dtype=np.float64)
    out = np.empty((len(endpoints), history * matrix.shape[1]), dtype=np.float64)
    raw = np.asarray(raw, np.float64)
    cursor = 0
    for t, row in enumerate(raw):
        state[1:] = state[:-1]
        state[0] = row
        filtered = kernel @ state
        z = (filtered - z_mean) / z_scale
        feat = z @ matrix + intercept
        hist[1:] = hist[:-1]
        hist[0] = feat
        if cursor < len(endpoints) and t == int(endpoints[cursor]):
            out[cursor] = hist.reshape(-1)
            cursor += 1
    if cursor != len(endpoints):
        raise RuntimeError("stream ended before all query endpoints")
    return out


def build_payload(task: str, method: str, destination: Path) -> dict[str, Any]:
    if (destination / "payload.npz").exists():
        raise FileExistsError("refuse to overwrite " + str(destination / "payload.npz"))
    started = time.monotonic()
    fitted = load_fitted(task)
    if method not in fitted["methods"]:
        raise KeyError(method)
    bank = fitted["methods"][method]
    readout = bank["readout"]
    setting = bank["configuration"]
    if int(setting["history_bins"]) != HISTORY:
        raise ValueError("packed linear arms must use 10-bin history")
    loaded = data.load_task(task, include_evaluation=True)
    train, heldout = loaded["train"], loaded["evaluation"]
    context = int(loaded["metadata"]["context"])
    channels = int(loaded["metadata"]["units"])
    kernel = wf_kernel()
    ridge_coef = np_numeric(readout.coef_)
    ridge_intercept = np_numeric(readout.intercept_)
    feature_dim = int(ridge_coef.shape[0] // HISTORY)
    arrays: dict[str, np.ndarray] = {
        "wf_kernel": kernel,
        "ridge_coef": ridge_coef,
        "ridge_intercept": ridge_intercept,
    }
    records: list[tuple[str, str, dict[str, Any], str]] = []
    records += [(s, "heldin", v, raw_basename(v, task, s, "heldin")) for s, v in train.items()]
    records += [(s, "heldout", v, raw_basename(v, task, s, "heldout")) for s, v in heldout.items()]
    if len(records) != TASKS[task]["roster"]:
        raise RuntimeError(f"{task}: roster {len(records)} != {TASKS[task]['roster']}")
    tags = [falcon_tag(task, basename) for *_, basename in records]
    if len(set(tags)) != len(tags):
        raise RuntimeError(f"{task}: Falcon tag collision: {tags}")

    sessions: list[dict[str, Any]] = []
    for session, role, item, basename in records:
        tag = falcon_tag(task, basename)
        pfx = prefix_for(tag)
        if role == "heldin":
            normalizer: Standardizer = fitted["source_normalizers"][session]
            adapter = bank["source_maps"][session]
            kind = "source_map"
        else:
            row = fitted["target_adapters"][session]
            normalizer = row["normalizer"]
            adapter = row["methods"][method]
            kind = "target_adapter"
        matrix, intercept, algebra = compose_affine(adapter, channels, feature_dim)
        if matrix.shape != (channels, feature_dim) or intercept.shape != (feature_dim,):
            raise ValueError(f"{session}: affine shape drift {matrix.shape} {intercept.shape}")
        arrays[f"{pfx}_z_mean"] = np_numeric(normalizer.mean_)
        arrays[f"{pfx}_z_scale"] = np_numeric(normalizer.scale_)
        arrays[f"{pfx}_A"] = matrix
        arrays[f"{pfx}_b"] = intercept
        support_hash = item.get("hashes", {}).get("support") or ""
        sessions.append(
            {
                "tag": tag,
                "source_session": session,
                "raw_basename": basename,
                "role": role,
                "array_prefix": pfx,
                "calibration": {
                    "kind": kind,
                    "algebra": algebra,
                    "target_labels_used": False,
                    "calibration_samples": int(len(item["support"])),
                },
                "support_activity_sha256": support_hash,
            }
        )

    checks: dict[str, Any] = {}
    result_dir = RESULTS / f"{task}_v2"
    for session, role, item, basename in records:
        if role != "heldout":
            continue
        tag = falcon_tag(task, basename)
        pfx = prefix_for(tag)
        raw = np.asarray(item["X"], np.float32)[int(item["pad"]) :]
        endpoints = np.asarray(item["starts"], np.int64) + context - 1 - int(item["pad"])
        features = replay_features(
            raw,
            endpoints,
            arrays[f"{pfx}_z_mean"],
            arrays[f"{pfx}_z_scale"],
            arrays[f"{pfx}_A"],
            arrays[f"{pfx}_b"],
            kernel,
            HISTORY,
        )
        pred = np.asarray(features @ ridge_coef + ridge_intercept, np.float32)
        stored = np.load(result_dir / f"pred_{method}_{session}.npy")
        if pred.shape != stored.shape:
            raise RuntimeError(f"{task}/{method}/{session}: shape {pred.shape} != {stored.shape}")
        err = float(np.max(np.abs(pred.astype(np.float64) - stored.astype(np.float64))))
        checks[session] = {
            "kind": "stored_heldout_query",
            "n": int(len(pred)),
            "max_abs_error": err,
            "tolerance": 0.0,
            "pass": err == 0.0,
        }
    failed = {s: x for s, x in checks.items() if not x["pass"]}
    if failed:
        raise RuntimeError(f"{task}/{method}: export replay mismatch {failed}")

    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "payload.npz", **arrays)
    with np.load(destination / "payload.npz", allow_pickle=False) as zf:
        if any(zf[k].dtype == object for k in zf.files):
            raise RuntimeError("object dtype escaped payload")
        payload_keys = sorted(zf.files)
        forbidden = [
            key
            for key in payload_keys
            if any(word in key.lower() for word in ("label", "target_y", "query", "nwb", "pickle"))
        ]
        if forbidden:
            raise RuntimeError("forbidden payload fields " + str(forbidden))
    receipt = json.loads((result_dir / "receipt.json").read_text())
    selected = receipt["selected"][method]
    manifest = {
        "schema": SCHEMA,
        "task": task,
        "method": method,
        "training_method": method,
        "channels": channels,
        "model_feature_channels": feature_dim,
        "outputs": int(ridge_coef.shape[1]),
        "history": HISTORY,
        "max_batch": TASKS[task]["max_batch"],
        "sessions": {row["tag"]: {k: v for k, v in row.items() if k != "tag"} for row in sessions},
        "expected_prediction_fields": {
            "task": task,
            "channels": channels,
            "outputs": int(ridge_coef.shape[1]),
            "history": HISTORY,
        },
        "source": {
            "reference_session": fitted["reference_source_session"],
            "models_pkl_sha256": receipt["models_sha256"],
            "selection_sha256": receipt["selection_sha256"],
            "ridge": "source-selected unpenalized-intercept ridge; no feature z-score",
            "filter": "FALCON 12-tap causal exponential, tau=240ms, extent=1",
        },
        "selected": selected,
        "local_public_calibration_scores": LOCAL_SCORES[(task, method)],
        "selection_budget_disclosure": SELECTION_BUDGET,
        "protocol": LINEAR_PROTOCOL,
        "h1_channel_correspondence": receipt.get("h1_channel_correspondence"),
        "payload": {
            "file": "payload.npz",
            "sha256": sha256_file(destination / "payload.npz"),
            "keys": payload_keys,
            "numeric_only": True,
            "contains_raw_data": False,
            "contains_target_y": False,
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema": "fair_v2_linear_export_audit_v1",
        "task": task,
        "method": method,
        "payload_sha256": manifest["payload"]["sha256"],
        "payload_bytes": (destination / "payload.npz").stat().st_size,
        "tag_count": len(sessions),
        "tags": [row["tag"] for row in sessions],
        "checks": checks,
        "max_abs_error": max(x["max_abs_error"] for x in checks.values()),
        "elapsed_seconds": time.monotonic() - started,
    }
    (destination / "export_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("m1", "m2", "h1"), required=True)
    parser.add_argument("--method", choices=ARMS, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_payload(args.task, args.method, args.dest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
