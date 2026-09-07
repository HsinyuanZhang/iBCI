"""Score the as-shipped original H1 decoder on the sealed minival stream.

This is deliberately a reference exporter, not a training or model-selection
entry point.  It must run inside the old released image named by ``IMAGE``;
the image's wrapper, payload, and SPINT implementation are all hash-bound
before the cache or decoder is loaded.  The only source-data operation is a
direct ``torch.load`` of the already-sealed cache--there is no cache builder
on this import path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


IMAGE = "sha256:f719c4228c345f9a1d6aa7c1e10d63ad7b9aa1f551dc95272b0b7a612d61fac6"
WRAPPER = Path("/third_party/falcon_challenge/spint_decoder.py")
PAYLOAD = Path("/data/decoder.pkl")
SPINT = Path("/src/models/components/spint.py")
DECODE = Path("/decode.py")
EXPECTED = {
    WRAPPER: "e4ae9ce5f51d5050a021b339c4d6b272ce10d580be4ab8c115ac436d1bd12763",
    PAYLOAD: "20a1d41a1d82a8037579caa2e4454f56817e02f021dec2132798c7fc57849298",
    SPINT: "855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519",
    DECODE: "3d17b8097a2804f99380f922500a4a41c7f6c4e5a81a6504a010c6a55ce4260e",
}
CACHE_SHA256 = "51ff9ebfcd10a032f9c173ec426bfb4c421b751502271577582239c51bcc91b4"
AUTHORITY_SHA256 = "92739d9f20d8184e2a24ddc2e4a00545c0b90170796c33f9d189cf90e8c6e41f"
W, UNITS, OUTPUTS, SCALE = 700, 176, 7, 20.0
COUNT, PUBLIC_CALLS, SESSIONS = 20325, 20920, 13
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/family_runtime_v1/original_h1_frozen_same20325_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def tag_for(session: str) -> Path:
    if not isinstance(session, str) or not session.startswith("ses-") or not session[4:]:
        raise RuntimeError("released H1 session tag requires nonempty ses- identifier")
    return Path(f"sub-HumanPitt-held-in-minival_{session}")


def assert_public(value: object) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.shape != (1, OUTPUTS)
            or value.dtype != np.float32 or not value.flags.owndata
            or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise RuntimeError("released original public prediction must be owning native FP32 [1,7]")
    return value


def _raw(engine: Any) -> np.ndarray:
    value = engine.observation_buffer
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    # The released decoder stores [W, batch, units], whereas the independent
    # reference below is [batch, W, units].
    return np.asarray(value).transpose(1, 0, 2)


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(np.float64), target.astype(np.float64)
    denominator = np.square(target - target.mean(0)).sum()
    if denominator <= 0 or not np.isfinite(denominator):
        raise RuntimeError("zero/nonfinite target variance")
    return float(1 - np.square(prediction - target).sum() / denominator)


def replay_session(engine: Any, neural: np.ndarray, velocity: np.ndarray, eval_mask: np.ndarray,
                   session: str, native_forward=None, progress=None) -> dict[str, object]:
    """Reset B1 state and replay every public bin of one sorted minival session."""
    neural, velocity, eval_mask = np.asarray(neural), np.asarray(velocity), np.asarray(eval_mask)
    if (neural.dtype != np.float32 or neural.ndim != 2 or neural.shape[1] != UNITS
            or velocity.shape != (len(neural), OUTPUTS) or eval_mask.dtype != np.bool_
            or eval_mask.shape != (len(neural),) or not np.isfinite(neural).all()
            or not np.isfinite(velocity).all()):
        raise RuntimeError("sealed minival geometry/finite contract drift")
    engine.reset([tag_for(session)])
    if getattr(engine, "window_size", W) != W or getattr(engine, "behavior_scaling_factor", SCALE) != SCALE:
        raise RuntimeError("released original W700/divisor20 contract drift")
    if getattr(engine, "smooth_calibration", False):
        raise RuntimeError("released original H1 must use unsmoothed calibration")
    history = np.zeros((1, W, UNITS), dtype=np.float32)
    predictions: list[np.ndarray] = []
    ends = np.flatnonzero(eval_mask).astype(np.int64)
    subset = {i for i in (0, 4, 699, 700, len(neural)-1) if 0 <= i < len(neural)}
    direct_count, native_error = 0, 0.0
    for index, raw in enumerate(neural):
        history[:, :-1] = history[:, 1:]
        history[:, -1] = raw
        prediction = assert_public(engine.predict(raw[None]))
        if not np.array_equal(_raw(engine), history):
            raise RuntimeError("released original B1 W700 raw history drift")
        if index in subset and native_forward is not None:
            direct = assert_public(native_forward(history.copy()))
            np.testing.assert_allclose(prediction, direct, atol=1e-5, rtol=1e-5)
            native_error = max(native_error, float(np.abs(prediction-direct).max()))
            direct_count += 1
        if eval_mask[index]:
            predictions.append(prediction[0].copy())
        if progress is not None and ((index+1) % 1024 == 0 or index == len(neural)-1):
            progress({"session": session, "public_calls": index+1, "scored_count": len(predictions)})
    prediction = np.stack(predictions) if predictions else np.empty((0, OUTPUTS), np.float32)
    if len(prediction) != len(ends):
        raise RuntimeError("eval-mask prediction cardinality drift")
    return {"prediction": prediction, "target": velocity[ends].astype(np.float32, copy=True),
            "end": ends, "session_id": np.asarray([session] * len(ends)),
            "public_calls": len(neural), "scored_count": len(ends),
            "direct_native_count": direct_count, "max_native_abs_error": native_error,
            "r2_concat_float64": _r2(prediction, velocity[ends])}


def _require_gate(out: Path, container_reference: bool = False, image_digest: str = IMAGE) -> None:
    if out.exists():
        raise FileExistsError(out)
    if (os.environ.get("ORIGINAL_H1_FROZEN_GO") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")
            or not container_reference or image_digest != IMAGE):
        raise RuntimeError("explicit ORIGINAL_H1_FROZEN_GO=1 and CPU-only environment required")


def _audit_immutable(cache: Path, authority: Path) -> dict[str, object]:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError("frozen original H1 image file SHA drift: " + str(path))
    if sha(cache) != CACHE_SHA256 or sha(authority) != AUTHORITY_SHA256:
        raise RuntimeError("sealed H1 cache/authority SHA drift")
    code = [Path(__file__), ROOT/"src/h1_optimized_v2/cache.py", ROOT/"src/h1_optimized_v2/data.py",
            ROOT/"src/h1_temporal_decoder_quick_product_v1/data.py"]
    # All image-owned model source is immutable under the read-only container.
    code += sorted(Path("/src/models").rglob("*.py"))
    return {"image": IMAGE, "image_files": {str(path): expected for path, expected in EXPECTED.items()},
            "code": {str(path): sha(path) for path in code},
            "cache": {"path": str(cache), "sha256": CACHE_SHA256},
            "authority": {"path": str(authority), "sha256": AUTHORITY_SHA256}}


def _load_wrapper() -> Any:
    spec = importlib.util.spec_from_file_location("_original_h1_frozen_wrapper", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("released original H1 wrapper loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(out: Path = OUT, *, container_reference=False, image_digest=IMAGE, threads=2) -> dict[str, object]:
    """Write the one fixed all-20,325-bin original-H1 reference archive."""
    out = Path(out)
    _require_gate(out, container_reference, image_digest)
    if threads not in (1, 2):
        raise RuntimeError("only CPU T1/T2 reference scoring")
    cache_root = ROOT / "results/decoder_validation_v2/20260905_190000/h1"
    CACHE, authority_path = cache_root/"source_cache.pt", cache_root/"source_cache_authority.json"
    pre = _audit_immutable(CACHE, authority_path)
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("released original H1 reference is CPU-only")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    # Bind/import the image model namespace before any host source-route helper.
    import src.models.components.spint as original_module
    if Path(original_module.__file__).resolve() != SPINT:
        raise RuntimeError("not the released image-owned SPINT implementation")
    wrapper = _load_wrapper()
    from tfpd_exploration.src.h1_optimized_v2.cache import validate_authority
    # Intentionally direct: do not call build_or_load(), which may create cache files.
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    validate_authority(cache, json.loads(authority_path.read_text()))
    minival = cache.get("minival")
    if not isinstance(minival, dict) or len(minival) != SESSIONS:
        raise RuntimeError("exactly 13 sealed minival sessions required")
    from falcon_challenge.config import FalconConfig, FalconTask
    engine = wrapper.SpintDecoder(FalconConfig(task=FalconTask.h1), str(PAYLOAD), batch_size=1)
    if (engine.window_size != W or engine.behavior_scaling_factor != SCALE
            or engine.smooth_calibration or any(module.training for module in engine.clf.modules())):
        raise RuntimeError("released original H1 eval/CPU/unsmoothed constructor contract drift")
    out.mkdir(parents=True)
    atomic_json(out / "input_authority.json", {"status": "FIXED_REFERENCE_RUNNING", "pre": pre})
    started, pieces, rows = time.monotonic(), [], []
    def native(history):
        if (engine.device.type != "cpu" or any(module.training for module in engine.local_clf.modules())
                or len(engine.local_calib_trial_features) != 1):
            raise RuntimeError("original post-reset CPU/eval/calibration contract drift")
        raw = torch.from_numpy(history)
        with torch.no_grad():
            prediction = engine.local_clf(raw, calib_trialized_neural_features=engine.local_calib_trial_features[0].unsqueeze(0).to(raw))[:, -1]
        return (prediction.numpy() / SCALE).astype(np.float32, copy=True)
    def progress(body):
        atomic_json(out / "live.json", {"status": "RUNNING", "elapsed_seconds": time.monotonic()-started, **body})
    for session, row in sorted(minival.items()):
        result = replay_session(engine, row["neural"], row["velocity"], row["eval_mask"], session, native, progress)
        result["original_calibration_shape"] = list(engine.local_calib_trial_features[0].shape)
        pieces.append({key: result.pop(key) for key in ("prediction", "target", "end", "session_id")})
        rows.append({"session": session, **result})
    arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in ("prediction", "target", "end", "session_id")}
    arrays["prediction"] = arrays["prediction"].astype(np.float64)
    arrays["target"] = arrays["target"].astype(np.float64)
    if (arrays["prediction"].shape != (COUNT, OUTPUTS) or arrays["target"].shape != (COUNT, OUTPUTS)
            or arrays["end"].dtype != np.int64 or len(arrays["session_id"]) != COUNT
            or sum(row["public_calls"] for row in rows) != PUBLIC_CALLS):
        raise RuntimeError("fixed 20325 endpoint / 20920 public-bin cardinality drift")
    post = _audit_immutable(CACHE, authority_path)
    if post != pre:
        raise RuntimeError("original H1 immutable authority changed during replay")
    archive = out / "original_h1_minival_native_float64.npz"
    atomic_npz(archive, arrays)
    result = {"schema": "original_h1_as_shipped_frozen_minival_reference_v1",
              "status": "PASS_AS_SHIPPED_ORIGINAL_H1_REFERENCE_ONLY", "image": IMAGE,
              "scope": "sorted sealed minival sessions; all chronological public B1 bins; no selection or fitting",
              "pre": pre, "post": post, "sessions": rows, "public_calls": PUBLIC_CALLS,
              "scored_count": COUNT, "archive": {"path": str(archive), "sha256": sha(archive)},
              "window": W, "units": UNITS, "outputs": OUTPUTS, "divisor": SCALE,
              "smooth_calibration": False, "batch": 1, "threads": 2,
              "metrics": {"r2_concat_float64": _r2(arrays["prediction"], arrays["target"]),
                          "equal_session_mean_r2_float64": float(np.mean([row["r2_concat_float64"] for row in rows])),
                          "worst_session_r2_float64": min(row["r2_concat_float64"] for row in rows),
                          "per_session_r2_float64": {row["session"]: row["r2_concat_float64"] for row in rows}},
              "direct_native_count": sum(row["direct_native_count"] for row in rows),
              "max_native_abs_error": max(row["max_native_abs_error"] for row in rows),
              "calibration_disclosure": "original as-shipped calibration, representative array has two trials; family banks use three trials; not a matched-support or historical-training-exposure ablation",
              "not_latency_measurement": True, "no_selection_or_fit": True,
              "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
              "parameter_updates": 0, "elapsed_seconds": time.monotonic() - started}
    result["threads"] = threads
    atomic_json(out / "receipt.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--container-reference", action="store_true")
    parser.add_argument("--image-digest", default=IMAGE)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    result = run(args.out, container_reference=args.container_reference, image_digest=args.image_digest, threads=args.threads)
    print(json.dumps({key: result[key] for key in ("status", "scored_count", "public_calls", "metrics", "max_native_abs_error", "elapsed_seconds")}, sort_keys=True))
