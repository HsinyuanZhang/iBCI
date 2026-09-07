"""H1 holdout-date (1925-01-20) exam-face eval of as-shipped Original SPINT.

Runs INSIDE the as-shipped image (read-only mounts; inference only). Level-1
fair-comparison eval per ADDENDUM_FAIR_COMPARISON_H1_20260906.md:

  - face: the LODO holdout date's minival eval_mask (session, end) coordinates
    (2,952 windows) — bitwise-identical coordinate rule as the M-F250 exam
    face (face_inventory.json exam counts 1453 + 1499)
  - window: native W=700 of the Original system (M-F250 used L=250; the two
    systems' window definitions differ — that difference is part of the
    compared system, P0-1)
  - exposure audit: dump the payload ``calib_trial_features`` keys; the 01-20
    sessions are "exposed" iff their reset tags resolve in that dict (reset()
    itself refuses unknown tags)

No training, no GPU, no EvalAI, no overwrite of historical roots.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


HOLDOUT_SESSIONS = ("ses-19250120T115044", "ses-19250120T115537")
RESET_TAG_FMT = "sub-HumanPitt-held-in-minival_{session}"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Per-column-mean centered variance-weighted R2 (Original local convention;
    identical to inner_original_history_mask.py and the C2 c2_reference evaluator)."""
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    denom = np.square(target - target.mean(0)).sum()
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("zero/nonfinite target variance")
    return float(1.0 - np.square(pred - target).sum() / denom)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _load_wrapper(wrapper: Path):
    wrapper = Path(wrapper).resolve()
    image_root = wrapper.parents[2]
    if str(image_root) not in sys.path:
        sys.path.insert(0, str(image_root))
    spec = importlib.util.spec_from_file_location("_orig_wrapper", wrapper)
    if spec is None or spec.loader is None:
        raise RuntimeError("wrapper loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _native_batch(engine, windows: np.ndarray, scale: float) -> np.ndarray:
    raw = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
    calib = engine.local_calib_trial_features[0].unsqueeze(0).expand(len(windows), -1, -1, -1)
    with torch.no_grad():
        out = engine.local_clf(raw, calib_trialized_neural_features=calib.to(raw))[:, -1]
    pred = (out.numpy() / float(scale)).astype(np.float32, copy=True)
    if not np.isfinite(pred).all():
        raise RuntimeError("nonfinite native prediction")
    return pred


def _windows_from_stream(neural: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    n_units = neural.shape[1]
    hist = np.zeros((len(ends), window, n_units), dtype=np.float32)
    for i, end in enumerate(ends):
        start = int(end) - window + 1
        if start >= 0:
            hist[i] = neural[start : int(end) + 1]
        else:
            hist[i, -int(end) - 1 :] = neural[: int(end) + 1]
    return hist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("holdout exam eval is CPU-only inside the image")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA visible inside container")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    from falcon_challenge.config import FalconConfig, FalconTask

    started = time.monotonic()
    wrapper = _load_wrapper(Path("/third_party/falcon_challenge/spint_decoder.py"))
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    minival = cache["minival"]
    engine = wrapper.SpintDecoder(FalconConfig(task=FalconTask.h1), "/data/decoder.pkl", batch_size=1)
    if engine.window_size != 700 or engine.behavior_scaling_factor != 20.0:
        raise RuntimeError("H1 original W/scale drift")

    # ---- exposure audit (payload calib keys; reset() refuses unknown tags) ----
    calib = engine.calib_trial_features
    if not isinstance(calib, dict):
        raise RuntimeError("payload calib_trial_features is not a dict; exposure audit unavailable")
    calib_keys = sorted(str(key) for key in calib.keys())
    task_config = FalconConfig(task=FalconTask.h1)
    exposure_rows = {}
    for session in HOLDOUT_SESSIONS:
        tag = RESET_TAG_FMT.format(session=session)
        # wrapper.reset resolves the tag through task_config.hash_dataset(tag.stem)
        # to the session's ONE payload key (Falcon H1 tag-hashing convention).
        resolved = str(task_config.hash_dataset(tag))
        exposure_rows[session] = {
            "reset_tag": tag,
            "hashed_payload_key": resolved,
            "hashed_payload_key_in_calib_keys": resolved in calib_keys,
            "reset_succeeded": False,
        }

    rows = []
    all_pred: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    arrays_for_export: dict[str, np.ndarray] = {}
    for session in HOLDOUT_SESSIONS:
        tag = RESET_TAG_FMT.format(session=session)
        engine.reset([Path(tag)])
        exposure_rows[session]["reset_succeeded"] = True
        row = minival[session]
        neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
        velocity = np.ascontiguousarray(row["velocity"], dtype=np.float32)
        ends = np.flatnonzero(np.asarray(row["eval_mask"], dtype=np.bool_)).astype(np.int64)
        windows = _windows_from_stream(neural, ends, 700)
        target = velocity[ends]
        preds = []
        for offset in range(0, len(windows), args.batch):
            preds.append(_native_batch(engine, windows[offset : offset + args.batch], 20.0))
        session_pred = np.concatenate(preds)
        if session_pred.shape != target.shape:
            raise RuntimeError(f"{session} pred/target shape mismatch {session_pred.shape} vs {target.shape}")
        all_pred.append(session_pred)
        all_target.append(target)
        arrays_for_export[f"pred__{session}"] = session_pred
        arrays_for_export[f"target__{session}"] = target
        arrays_for_export[f"ends__{session}"] = ends
        rows.append({
            "session": session,
            "reset_tag": tag,
            "n": int(len(ends)),
            "n_stream_bins": int(len(neural)),
            "window": 700,
            "ends_sha256": _array_digest(ends),
            "target_sha256": _array_digest(target),
            "pred_sha256": _array_digest(session_pred),
            "r2_percolumn_mean0": _r2(session_pred, target),
            "calib_shape": list(engine.local_calib_trial_features[0].shape),
        })

    pred = np.concatenate(all_pred)
    target = np.concatenate(all_target)
    per_session = [row["r2_percolumn_mean0"] for row in rows]
    result = {
        "schema": "original_h1_holdout_exam_exposed_eval_v1",
        "system": "as-shipped Original SPINT",
        "task": "h1",
        "surface": "LODO holdout date 1925-01-20 exam (minival eval_mask (session,end) coordinates)",
        "holdout_sessions": list(HOLDOUT_SESSIONS),
        "n_points": int(len(target)),
        "window": 700,
        "window_note": (
            "Original scored with its native W=700 windows; M-F250 exam used L=250. "
            "Window definitions differ across systems (P0-1) — the join locks only (session,end)."
        ),
        "coordinate_rule": "ends = flatnonzero(eval_mask) per session from the frozen source_cache (M-F250 exam rule)",
        "scale": {"behavior_scaling_factor": 20.0, "scoring": "pred/20 vs native velocity"},
        "exposure": {
            "payload_calib_key_count": len(calib_keys),
            "payload_calib_keys": calib_keys,
            "holdout_sessions": exposure_rows,
            "verdict": "exposed" if all(r["hashed_payload_key_in_calib_keys"] for r in exposure_rows.values()) else "check_rows",
            "method": (
                "payload calib_trial_features keys audited in-container; the wrapper resolves "
                "each reset tag via FalconConfig.hash_dataset to the session's one payload key "
                "(inner_original_history_mask.py loading path); reset() raises on a missing key"
            ),
        },
        "r2_convention": "per-column mean(0) centered variance-weighted (Original local convention)",
        "pooled_r2": _r2(pred, target),
        "equal_session_mean_r2": float(np.mean(per_session)),
        "sessions": rows,
        "cache_sha256": _sha(args.cache),
        "elapsed_seconds": time.monotonic() - started,
        "status": "COMPLETE_INFERENCE_ONLY",
        "parameter_updates": 0,
    }
    _atomic_json(args.out, result)
    npz_path = args.out.with_name(args.out.stem + "_arrays.npz")
    np.savez_compressed(npz_path, **arrays_for_export)
    print(json.dumps({
        "status": result["status"], "n_points": result["n_points"],
        "pooled_r2": result["pooled_r2"], "equal_session_mean_r2": result["equal_session_mean_r2"],
        "exposure": result["exposure"]["verdict"], "elapsed_s": result["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
