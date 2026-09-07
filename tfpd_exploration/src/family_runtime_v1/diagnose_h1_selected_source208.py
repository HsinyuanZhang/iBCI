"""Read-only, exact-208 source diagnostic for the frozen selected H1 EMAs.

This is deliberately descriptive: it neither reopens minival nor changes the
selected models.  All formal/artifact/code checks happen before torch, model,
or cache imports, and the output directory must not already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ARMS = ("flat", "route")
W, UNITS, OUTPUTS, COUNT, PER_SESSION = 700, 176, 7, 208, 16
IDS_SHA256 = "da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211"
GATE1040_SHA256 = "04aeadccdcb183b050f7006ece4bb15b362a924ae7a7a22012792afe4e8f09b9"
ROOT = Path(__file__).resolve().parents[2]
H1_ROOT = ROOT / "results/decoder_validation_v2/20260905_190000/h1"
IDS = H1_ROOT / "capacity_probe_208_source_v2/frozen_ids.json"
GATE1040 = H1_ROOT / "family_v1/crst_b4_set_v3_localbalanced_source208_preflight_v1/report_1040.json"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(arrays: dict[str, np.ndarray], path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def _require_ids(path: Path = IDS) -> dict[str, np.ndarray]:
    if sha(path) != IDS_SHA256:
        raise RuntimeError("frozen source208 IDs SHA drift")
    body = read(path); raw = body.get("ids")
    if not isinstance(raw, dict) or len(raw) != 13:
        raise RuntimeError("requires exactly 13 frozen source208 sessions")
    fixed = {name: np.asarray(starts, dtype=np.int64) for name, starts in sorted(raw.items())}
    if (sum(len(v) for v in fixed.values()) != COUNT or any(len(v) != PER_SESSION for v in fixed.values())
            or any(v.ndim != 1 or any(not isinstance(x, int) or isinstance(x, bool) for x in raw[k])
                   or not np.array_equal(v, np.asarray(raw[k], dtype=np.int64)) or np.any(v < 0)
                   or np.any(np.diff(v) <= 0) for k, v in fixed.items())):
        raise RuntimeError("invalid exact source208 IDs")
    return fixed


def preflight_audit(formal: Path, ids_path: Path = IDS, gate_path: Path = GATE1040) -> dict:
    """Audit every authority required by this diagnostic before imports/loads."""
    from .complete_h1_family_source import artifact_audit, sha as formal_sha
    audit = artifact_audit(formal)  # COMPLETE only, and binds both selected EMAs.
    fixed = _require_ids(ids_path)
    if sha(gate_path) != GATE1040_SHA256:
        raise RuntimeError("source208 1040 gate SHA drift")
    gate = read(gate_path)
    if (gate.get("status") != "COMPLETE" or gate.get("updates_completed") != 1040
            or set(gate.get("after", {})) != set(ARMS)):
        raise RuntimeError("completed source208 gate receipt drift")
    # These are historical descriptive values, never acceptance thresholds.
    for arm, expected in (("flat", .9536639214291189), ("route", .9641452152933455)):
        if abs(float(gate["after"][arm].get("r2_concat", np.nan)) - expected) > 1e-12:
            raise RuntimeError("historical source208 gate score drift")
    frozen = read(formal / "input_authority.json")
    if frozen.get("bindings", {}).get("gate1040_sha256") != GATE1040_SHA256:
        raise RuntimeError("formal binding does not bind source208 gate")
    selected = audit["selected"]
    for arm in ARMS:
        path = formal / "exports" / f"{arm}_selected_plain_ema.pt"
        # artifact_audit already hash-binds this via the finalizer receipt;
        # repeat its exact selected identity here for a clear local invariant.
        if not path.is_file() or not selected[arm].get("checkpoint_sha256"):
            raise RuntimeError("selected EMA export/identity missing")
    files = {str(ids_path): sha(ids_path), str(gate_path): sha(gate_path), str(formal / "input_authority.json"): formal_sha(formal / "input_authority.json")}
    return {"artifact": audit, "frozen_ids": {"path": str(ids_path), "sha256": IDS_SHA256, "sessions": len(fixed), "windows": COUNT},
            "gate1040": {"path": str(gate_path), "sha256": GATE1040_SHA256,
                         "historical_r2_not_acceptance_threshold": {a: gate["after"][a]["r2_concat"] for a in ARMS}}, "files": files}


def code_source_audit(formal: Path, preflight: dict) -> dict:
    """Bind current executable closure before importing cache/model helpers."""
    source = ROOT / "src"
    # Snapshot auxiliary source208 helpers before their imports, then reuse the
    # complete proof's authoritative frozen closure/cache binding verbatim.
    closure_paths = {"diagnostic": Path(__file__), "source_preflight": source / "h1_family_v1/source_preflight.py",
                     "capacity_probe": source / "h1_optimized_v2/capacity_probe.py", "cache": source / "h1_optimized_v2/cache.py",
                     "complete_source_helper": source / "family_runtime_v1/complete_h1_family_source.py",
                     "replay_helper": source / "family_runtime_v1/h1_replay_contract.py"}
    closure = {name: sha(path) for name, path in closure_paths.items()}
    frozen = read(formal / "input_authority.json")
    if frozen.get("bindings", {}).get("gate1040_sha256") != preflight["gate1040"]["sha256"]:
        raise RuntimeError("formal gate binding changed after preflight")
    from .complete_h1_family_source import code_source_audit as complete_source_audit
    complete = complete_source_audit(formal)
    return {"closure": closure, "complete_source": complete, "formal_input_sha256": sha(formal / "input_authority.json"),
            "frozen_ids_sha256": sha(IDS), "gate1040_sha256": sha(GATE1040)}


def _validate_batch(row: dict, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neural, velocity = np.asarray(row["neural"]), np.asarray(row["velocity"])
    if (starts.dtype != np.int64 or starts.shape != (PER_SESSION,) or np.any(starts < 0) or np.any(np.diff(starts) <= 0)
            or neural.dtype != np.float32 or velocity.dtype != np.float32 or neural.ndim != 2 or neural.shape[1] != UNITS or velocity.shape != (len(neural), OUTPUTS)
            or not np.isfinite(neural).all() or not np.isfinite(velocity).all()
            or np.any(starts + W > len(neural))):
        raise RuntimeError("source208 canonical batch geometry/finite/start contract")
    x = np.stack([neural[s:s + W] for s in starts]).astype(np.float32, copy=False)
    target = np.stack([velocity[s + W - 1] for s in starts]).astype(np.float32, copy=False)
    return np.ascontiguousarray(x), np.ascontiguousarray(target)


def validate_manifest_against_cache(cache: dict, authority: dict, fixed: dict[str, np.ndarray], ids_path: Path = IDS) -> None:
    """Read-only exact agreement of the hard manifest, cache authority and sampler."""
    manifest = read(ids_path)
    if manifest.get("source_authority") != authority:
        raise RuntimeError("source208 manifest embedded authority differs from cache authority")
    from tfpd_exploration.src.h1_optimized_v2.capacity_probe import ids as deterministic_ids
    actual = deterministic_ids(cache)
    if set(actual) != set(fixed) or any(not np.array_equal(np.asarray(actual[k], dtype=np.int64), fixed[k]) for k in fixed):
        raise RuntimeError("source208 manifest differs from deterministic cache sampler")


def _stats(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction, target = prediction.astype(np.float64, copy=False), target.astype(np.float64, copy=False)
    if prediction.shape != target.shape or prediction.shape[1:] != (OUTPUTS,) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise RuntimeError("source208 prediction/target finite shape contract")
    denominator = np.square(target - target.mean(axis=0)).sum()
    if not np.isfinite(denominator) or denominator <= 0: raise RuntimeError("source208 target must have positive finite variance")
    return {"r2_concat_float64": float(1 - np.square(prediction - target).sum() / denominator),
            "prediction_mean_float64": float(prediction.mean()), "prediction_std_float64": float(prediction.std()),
            "target_mean_float64": float(target.mean()), "target_std_float64": float(target.std())}


def evaluate_source208(model, cache: dict, fixed: dict[str, np.ndarray], device, bank_factory) -> tuple[dict, dict[str, np.ndarray]]:
    """Exact native W700 forwards.  Kept injectable so tests need no torch/cache."""
    import torch
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []; sessions: list[np.ndarray] = []; starts_all: list[np.ndarray] = []
    per_session = {}
    model.eval()
    with torch.no_grad():
        for session, starts in sorted(fixed.items()):
            if session not in cache.get("train", {}): raise RuntimeError("frozen source208 train session missing")
            row = cache["train"][session]
            available = np.asarray(row.get("query_starts"))
            if available.dtype.kind not in "iu" or not np.all(np.isin(starts, available)):
                raise RuntimeError("frozen source208 starts are not cache query_starts")
            x, target = _validate_batch(row, starts)
            bank = bank_factory(cache["train"][session], device)
            value = model.forward_last(torch.as_tensor(x, device=device), bank) / 20.0
            prediction = value.detach().cpu().numpy()
            if prediction.dtype != np.float32 or prediction.shape != (PER_SESSION, OUTPUTS) or not np.isfinite(prediction).all():
                raise RuntimeError("native source208 prediction FP32/finite contract")
            p64, t64 = prediction.astype(np.float64), target.astype(np.float64)
            per_session[session] = {"starts": starts.tolist(), **_stats(p64, t64)}
            predictions.append(p64); targets.append(t64); sessions.append(np.asarray([session] * len(starts))); starts_all.append(starts.copy())
    arrays = {"prediction": np.ascontiguousarray(np.concatenate(predictions)), "target": np.ascontiguousarray(np.concatenate(targets)),
              "session_id": np.concatenate(sessions), "start": np.concatenate(starts_all).astype(np.int64)}
    if arrays["prediction"].shape != (COUNT, OUTPUTS) or arrays["target"].shape != (COUNT, OUTPUTS): raise RuntimeError("exact source208 cardinality drift")
    return {"per_session": per_session, "pooled": _stats(arrays["prediction"], arrays["target"]), "windows": COUNT}, arrays


def state_digest(state: dict) -> str:
    """Stable state binding independent of torch serialization container bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def require_fresh_post(formal: Path, preflight: dict, source: dict) -> tuple[dict, dict]:
    """Fresh (not cached) postcondition authority check, testable in isolation."""
    post, post_source = preflight_audit(formal), code_source_audit(formal, preflight)
    if post != preflight or post_source != source:
        raise RuntimeError("fresh post diagnostic closure/artifact drift")
    return post, post_source


def run(formal: Path, out: Path, *, device: str = "cpu", threads: int = 2) -> dict:
    if out.exists(): raise FileExistsError(out)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (os.environ.get("H1_SELECTED_SOURCE208_DIAGNOSTIC_GO") != "1" or device not in ("cpu", "cuda:0") or threads not in (1, 2)
            or (device == "cpu" and visible not in ("", "-1")) or (device == "cuda:0" and visible not in ("0",))):
        raise RuntimeError("explicit source208 diagnostic gate/device/threads required")
    pre = preflight_audit(formal)
    source = code_source_audit(formal, pre)
    import torch
    torch.set_num_threads(threads); torch.set_num_interop_threads(1)
    if device == "cuda:0" and (not torch.cuda.is_available() or torch.cuda.current_device() != 0):
        raise RuntimeError("requested visible cuda:0 unavailable")
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, ROOT as cache_root, validate_authority
    from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    cache = torch.load(CACHE, map_location="cpu", weights_only=False); authority = read(cache_root / "source_cache_authority.json"); validate_authority(cache, authority)
    fixed = _require_ids(); validate_manifest_against_cache(cache, authority, fixed); dev = torch.device(device); results = {}; archives = {}
    for arm, index in (("flat", 0), ("route", 1)):
        pair = make_v2_unscaled_dot_localbalanced_pair(seed=42); model = pair[index].to(dev); del pair
        state_path = formal / "exports" / f"{arm}_selected_plain_ema.pt"
        state = torch.load(state_path, map_location=dev, weights_only=True)
        if not state or any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()): raise RuntimeError("selected EMA must be finite FP32")
        before_state = state_digest(state)
        model.load_state_dict(state, strict=True)
        if state_digest(model.state_dict()) != before_state:
            raise RuntimeError("strict selected EMA load changed tensor contents")
        result, arrays = evaluate_source208(model, cache, fixed, dev, lambda row, d: H1Bank(*[row["bank"][k].to(d) for k in ("E0", "T", "unit_mask")]))
        if state_digest(model.state_dict()) != before_state:
            raise RuntimeError("in-memory selected EMA state changed during diagnostic")
        if state_digest(torch.load(state_path, map_location="cpu", weights_only=True)) != before_state: raise RuntimeError("selected EMA state changed during diagnostic")
        result["selected_plain_ema_state_digest"] = before_state
        results[arm], archives[arm] = result, arrays
    for key in ("target", "session_id", "start"):
        if not np.array_equal(archives["flat"][key], archives["route"][key]):
            raise RuntimeError("FLAT/ROUTE source208 metadata/targets differ")
    post, post_source = require_fresh_post(formal, pre, source)
    out.mkdir(parents=True); archive_paths = {}
    for arm, arrays in archives.items():
        path = out / f"{arm}_selected_source208_native_float64.npz"
        atomic_npz(arrays, path)
        archive_paths[arm] = {"path": str(path), "sha256": sha(path),
                              "typed_array_sha256": {key: array_digest(value) for key, value in arrays.items()}}
    result = {"schema": "h1_selected_ema_exact_source208_descriptive_v1", "status": "COMPLETE_READ_ONLY_DESCRIPTIVE", "preflight": pre,
              "source": source, "postflight": post, "post_source": post_source, "arms": results, "archives": archive_paths,
              "device": device, "threads": threads, "parameter_updates": 0, "selection_changed": False, "minival_forward": False,
              "historical_gate_interpretation": "0.9536639214291189/.9641452152933455 are historical different-training-condition descriptions, not target thresholds"}
    atomic_json(result, out / "receipt.json"); return result


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--formal", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu"); parser.add_argument("--threads", choices=(1, 2), type=int, default=2)
    args = parser.parse_args(argv); return run(args.formal, args.out, device=args.device, threads=args.threads)


if __name__ == "__main__": main()
