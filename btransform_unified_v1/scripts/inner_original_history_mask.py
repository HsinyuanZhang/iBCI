"""CPU-only Original-SPINT history-mask scan. Runs INSIDE the as-shipped image.

Zero the oldest (W-k) bins of each scored window, keep the newest k bins,
then native-forward. k=W is the unmasked sanity check against the sealed
reference R². No training, no GPU, no EvalAI, no overwrite of historical roots.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
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
    # Image layout is /third_party/falcon_challenge/spint_decoder.py.
    # `python /in/run.py` puts /in on sys.path[0], so `import third_party` fails
    # unless the image root is inserted first.
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


def _mask_windows(windows: np.ndarray, k: int) -> np.ndarray:
    if k >= windows.shape[1]:
        return windows
    out = windows.copy()
    out[:, : windows.shape[1] - k] = 0.0
    return out


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


def _m2_stem(session: str) -> Path:
    parts = session.removeprefix("ses-").split("-")
    if len(parts) < 4:
        raise RuntimeError(f"unrecognized M2 session {session}")
    return Path(f"sub-MonkeyN{parts[-1]}_{''.join(parts[:3])}_held_in_eval")


def _curve(all_pred: dict, all_target: list, rows: list, ks: list[int]) -> dict:
    target = np.concatenate(all_target)
    curve = {}
    for k in ks:
        pred = np.concatenate(all_pred[k])
        curve[str(k)] = {
            "pooled_r2": _r2(pred, target),
            "equal_session_mean_r2": float(np.mean([r["per_k_r2"][str(k)] for r in rows])),
        }
    return curve, target


def run_h1(out: Path, cache_path: Path, ks: list[int], batch: int) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask

    wrapper = _load_wrapper(Path("/third_party/falcon_challenge/spint_decoder.py"))
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    minival = cache["minival"]
    engine = wrapper.SpintDecoder(FalconConfig(task=FalconTask.h1), "/data/decoder.pkl", batch_size=1)
    if engine.window_size != 700 or engine.behavior_scaling_factor != 20.0:
        raise RuntimeError("H1 original W/scale drift")
    rows = []
    all_target = []
    all_pred = {k: [] for k in ks}
    for session in sorted(minival):
        row = minival[session]
        neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
        velocity = np.ascontiguousarray(row["velocity"], dtype=np.float32)
        ends = np.flatnonzero(np.asarray(row["eval_mask"], dtype=np.bool_)).astype(np.int64)
        engine.reset([Path(f"sub-HumanPitt-held-in-minival_{session}")])
        windows = _windows_from_stream(neural, ends, 700)
        target = velocity[ends]
        preds = {k: [] for k in ks}
        for offset in range(0, len(windows), batch):
            chunk = windows[offset : offset + batch]
            for k in ks:
                preds[k].append(_native_batch(engine, _mask_windows(chunk, k), 20.0))
        session_pred = {k: np.concatenate(v) for k, v in preds.items()}
        all_target.append(target)
        for k in ks:
            all_pred[k].append(session_pred[k])
        rows.append({
            "session": session,
            "n": int(len(ends)),
            "per_k_r2": {str(k): _r2(session_pred[k], target) for k in ks},
            "calib_shape": list(engine.local_calib_trial_features[0].shape),
        })
    curve, target = _curve(all_pred, all_target, rows, ks)
    return {
        "task": "h1",
        "system": "as-shipped Original SPINT",
        "n_points": int(len(target)),
        "window": 700,
        "k": ks,
        "curve": curve,
        "sessions": rows,
    }


def run_m1(out: Path, cache_path: Path, archive_path: Path, ks: list[int], batch: int) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask

    wrapper = _load_wrapper(Path("/third_party/falcon_challenge/spint_decoder.py"))
    archive = np.load(archive_path, allow_pickle=False)
    cache = np.load(cache_path, allow_pickle=False)
    engine = wrapper.SpintDecoder(FalconConfig(task=FalconTask.m1), "/data/decoder.pkl", batch_size=1)
    if engine.window_size != 100 or engine.behavior_scaling_factor != 1.0:
        raise RuntimeError("M1 original W/scale drift")
    sessions = ("ses-20120926", "ses-20120927", "ses-20120928")
    rows = []
    all_target = []
    all_pred = {k: [] for k in ks}
    for session in sessions:
        ids = np.flatnonzero(archive["session"] == session)
        starts = np.asarray(archive["start"][ids], dtype=np.int64)
        target = np.ascontiguousarray(archive["target"][ids], dtype=np.float32)
        raw = np.ascontiguousarray(cache[f"raw_neural/{session}"], dtype=np.float32)
        engine.reset([Path(f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys")])
        windows = np.stack([raw[int(s) : int(s) + 100] for s in starts]).astype(np.float32)
        preds = {k: [] for k in ks}
        for offset in range(0, len(windows), batch):
            chunk = windows[offset : offset + batch]
            for k in ks:
                preds[k].append(_native_batch(engine, _mask_windows(chunk, k), 1.0))
        session_pred = {k: np.concatenate(v) for k, v in preds.items()}
        all_target.append(target)
        for k in ks:
            all_pred[k].append(session_pred[k])
        rows.append({
            "session": session,
            "n": int(len(starts)),
            "per_k_r2": {str(k): _r2(session_pred[k], target) for k in ks},
            "calib_shape": list(engine.local_calib_trial_features[0].shape),
        })
    curve, target = _curve(all_pred, all_target, rows, ks)
    return {
        "task": "m1",
        "system": "as-shipped Original SPINT",
        "n_points": int(len(target)),
        "window": 100,
        "k": ks,
        "curve": curve,
        "sessions": rows,
    }


def run_m2(out: Path, cache_root: Path, ks: list[int], batch: int, surface: str) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask

    wrapper = _load_wrapper(Path("/third_party/falcon_challenge/spint_decoder.py"))
    engine = wrapper.SpintDecoder(FalconConfig(task=FalconTask.m2), "/data/decoder.pkl", batch_size=1)
    if engine.window_size != 50 or engine.behavior_scaling_factor != 5.0:
        raise RuntimeError("M2 original W/scale drift")
    sessions = sorted(p.name for p in (cache_root / surface).iterdir() if p.is_dir() and p.name.startswith("ses-"))
    if not sessions:
        raise RuntimeError(f"no M2 sessions under {cache_root / surface}")
    rows = []
    all_target = []
    all_pred = {k: [] for k in ks}
    for session in sessions:
        dest = cache_root / surface / session
        raw = np.ascontiguousarray(np.load(dest / "X_store.npy"), dtype=np.float32)
        starts = np.asarray(np.load(dest / "eligible_starts.npy"), dtype=np.int64)
        target = np.ascontiguousarray(np.load(dest / "target_store.npy"), dtype=np.float32)
        if raw.ndim != 2:
            raise RuntimeError(f"{session} X_store ndim {raw.ndim}")
        windows = np.stack([raw[int(s) : int(s) + 50] for s in starts]).astype(np.float32)
        if windows.shape[1:] != (50, raw.shape[1]) or len(windows) != len(target):
            raise RuntimeError(f"{session} window/target mismatch {windows.shape} vs {target.shape}")
        engine.reset([_m2_stem(session)])
        preds = {k: [] for k in ks}
        for offset in range(0, len(windows), batch):
            chunk = windows[offset : offset + batch]
            for k in ks:
                preds[k].append(_native_batch(engine, _mask_windows(chunk, k), 5.0))
        session_pred = {k: np.concatenate(v) for k, v in preds.items()}
        all_target.append(target)
        for k in ks:
            all_pred[k].append(session_pred[k])
        rows.append({
            "session": session,
            "n": int(len(starts)),
            "per_k_r2": {str(k): _r2(session_pred[k], target) for k in ks},
            "calib_shape": list(engine.local_calib_trial_features[0].shape),
            "reset_stem": _m2_stem(session).name,
        })
    curve, target = _curve(all_pred, all_target, rows, ks)
    return {
        "task": "m2",
        "surface": surface,
        "system": "as-shipped Original SPINT",
        "n_points": int(len(target)),
        "window": 50,
        "k": ks,
        "curve": curve,
        "sessions": rows,
    }


def run_m1_outer(out: Path, archive_path: Path, batch: int) -> dict:
    from falcon_challenge.config import FalconConfig, FalconTask

    wrapper = _load_wrapper(Path("/third_party/falcon_challenge/spint_decoder.py"))
    engine = wrapper.SpintDecoder(FalconConfig(task=FalconTask.m1), "/data/decoder.pkl", batch_size=1)
    if engine.window_size != 100 or engine.behavior_scaling_factor != 1.0:
        raise RuntimeError("M1 original W/scale drift")
    blob = np.load(archive_path, allow_pickle=False)
    windows = np.ascontiguousarray(blob["windows"], dtype=np.float32)
    target = np.ascontiguousarray(blob["target"], dtype=np.float32)
    session = str(blob["session"][0]) if np.ndim(blob["session"]) else str(blob["session"])
    engine.reset([Path(f"sub-MonkeyL-held-in-calib_{session}_behavior+ecephys")])
    preds = []
    for offset in range(0, len(windows), batch):
        preds.append(_native_batch(engine, windows[offset : offset + batch], 1.0))
    pred = np.concatenate(preds)
    return {
        "task": "m1",
        "system": "as-shipped Original SPINT",
        "surface": f"loso_{session}",
        "n_points": int(len(target)),
        "window": 100,
        "pooled_r2": _r2(pred, target),
        "calib_shape": list(engine.local_calib_trial_features[0].shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("h1", "m1", "m2", "m1_outer"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--surface", default="source_minival")
    parser.add_argument("--ks", default="")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("history-mask scan is CPU-only")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA visible inside history-mask container")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    if args.task == "h1":
        ks = [int(x) for x in (args.ks or "25,50,100,200,350,700").split(",") if x]
        result = run_h1(args.out, args.cache, ks, args.batch)
        result["schema"] = "original_history_mask_scan_v1"
    elif args.task == "m1":
        if args.archive is None:
            raise RuntimeError("M1 requires --archive")
        ks = [int(x) for x in (args.ks or "10,25,50,100").split(",") if x]
        result = run_m1(args.out, args.cache, args.archive, ks, args.batch)
        result["schema"] = "original_history_mask_scan_v1"
    elif args.task == "m2":
        ks = [int(x) for x in (args.ks or "10,25,50").split(",") if x]
        result = run_m2(args.out, args.cache, ks, args.batch, args.surface)
        result["schema"] = "original_history_mask_scan_v1"
    else:
        if args.archive is None:
            raise RuntimeError("m1_outer requires --archive")
        result = run_m1_outer(args.out, args.archive, args.batch)
        result["schema"] = "m1_original_loso_outer20120924_v1"
    result["status"] = "COMPLETE_INFERENCE_ONLY"
    result["parameter_updates"] = 0
    _atomic_json(args.out, result)
    payload = {"task": result["task"], "n": result["n_points"]}
    if "curve" in result:
        payload["curve"] = result["curve"]
    if "pooled_r2" in result:
        payload["pooled_r2"] = result["pooled_r2"]
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
