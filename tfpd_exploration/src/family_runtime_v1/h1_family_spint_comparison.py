"""Guarded public CPU timing: selected H1 family versus released original SPINT.

This runs *inside the immutable released image*.  It deliberately measures one
fixed known-source train session only; cache targets are deserialized solely by
the existing cache-authority validator and are never scored here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

from .complete_h1_family_source import (
    artifact_audit, atomic_json, code_source_audit, require_same_files, sha,
)

IMAGE = "sha256:f719c4228c345f9a1d6aa7c1e10d63ad7b9aa1f551dc95272b0b7a612d61fac6"
WRAPPER = Path("/third_party/falcon_challenge/spint_decoder.py")
PAYLOAD = Path("/data/decoder.pkl")
MODELS_ROOT = Path("/src/models")
EXPECTED = {
    WRAPPER: "e4ae9ce5f51d5050a021b339c4d6b272ce10d580be4ab8c115ac436d1bd12763",
    PAYLOAD: "20a1d41a1d82a8037579caa2e4454f56817e02f021dec2132798c7fc57849298",
    Path("/src/models/components/spint.py"): "855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519",
    Path("/decode.py"): "3d17b8097a2804f99380f922500a4a41c7f6c4e5a81a6504a010c6a55ce4260e",
}
NAMES = ("ORIGINALdefault", "FLAT", "ROUTE")
SESSION = "ses-19250101T111740"
ORIGINAL_TAG = Path("sub-HumanPitt-held-in-calib_ses-19250101T111740")
W, UNITS, OUT, SCALE = 700, 176, 7, 20.0


def require_proof(path: Path, expected_sha: str, audit: dict, source: dict) -> dict:
    """Bind this timing run to the completed selected-family proof exactly."""
    if len(expected_sha) != 64 or sha(path) != expected_sha:
        raise RuntimeError("full completed H1 proof SHA drift")
    proof = json.loads(path.read_text())
    if (proof.get("status") != "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY"
            or proof.get("parameter_updates") != 0 or proof.get("batch") != 1
            or proof.get("pre_artifact") != audit or proof.get("post_artifact") != audit
            or proof.get("pre_source") != source or proof.get("post_source") != source):
        raise RuntimeError("selected H1 complete proof authority mismatch")
    for arm in ("flat", "route"):
        row = proof.get("arms", {}).get(arm, {})
        archive_error, native_error = row.get("max_archive_abs_error", np.inf), row.get("max_native_subset_abs_error", np.inf)
        if (row.get("selected") != audit["selected"][arm] or row.get("scored_count") != 20325
                or row.get("public_calls") != 20920 or not np.isfinite(archive_error) or not np.isfinite(native_error)
                or not 0 <= archive_error <= 1e-5 or not 0 <= native_error <= 1e-5):
            raise RuntimeError("selected H1 complete chronological proof incomplete")
    require_same_files(proof["pre_source"]["files"])
    return proof


def assert_public(name: str, value: object) -> None:
    if (not isinstance(value, np.ndarray) or value.shape != (1, OUT) or value.dtype != np.float32
            or not value.flags.owndata or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise RuntimeError(name + ": owning native FP32 [1,7] output required")


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or not (array > 0).all():
        raise RuntimeError("invalid timing samples")
    return {"mean_ms": float(array.mean()), "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)), "p99_ms": float(np.percentile(array, 99)),
            "max_ms": float(array.max())}


def image_python_files(root: Path | None = None) -> list[Path]:
    """All image-owned model Python sources, including unimported helpers."""
    root = MODELS_ROOT if root is None else root
    files = sorted(root.rglob("*.py"))
    if not files:
        raise RuntimeError("immutable image model Python source topology drift")
    return files


def immutable_snapshot(proof: Path) -> dict[str, str]:
    """Bind benchmark/proof and every image-owned model source before imports."""
    paths = [*EXPECTED, Path(__file__), Path(__file__).with_name("complete_h1_family_source.py"), proof,
             *image_python_files()]
    return {str(path): sha(path) for path in paths}


def fixed_train_row(cache: dict, count: int) -> tuple[dict, np.ndarray]:
    """Select only the declared train session, never the smaller minival row."""
    if "train" not in cache or SESSION not in cache["train"]:
        raise RuntimeError("fixed H1 train session missing from frozen cache")
    row = cache["train"][SESSION]
    neural = np.asarray(row["neural"])
    if (neural.dtype != np.float32 or neural.ndim != 2 or neural.shape[1] != UNITS
            or len(neural) < count or not np.isfinite(neural).all()):
        raise RuntimeError("fixed source train neural geometry/finite/cardinality contract")
    return row, np.ascontiguousarray(neural[:count])


def _original_oracle(engine, history):
    """Native released-model forward over an independently maintained W700 history."""
    import torch
    raw = torch.from_numpy(history.copy())
    with torch.no_grad():
        value = engine.local_clf(raw, calib_trialized_neural_features=engine.local_calib_trial_features[0].unsqueeze(0).to(raw))[:, -1]
    return (value.numpy() / engine.behavior_scaling_factor).astype(np.float32, copy=True)


def _family_oracle(model, bank, history):
    import torch
    with torch.no_grad():
        return (model.forward_last(torch.from_numpy(history.copy()), bank) / SCALE).numpy().astype(np.float32, copy=True)


def _load_selected(model, path: Path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in state.values()):
        raise RuntimeError("selected plain EMA finite FP32 contract")
    model.load_state_dict(state, strict=True)
    model.eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("selected family model did not enter eval")


def main(args) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    if (os.environ.get("H1_FAMILY_SPINT_COMPARISON_GO") != "1" or not args.container_reference
            or args.image_digest != IMAGE or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")):
        raise RuntimeError("explicit original-image CPU-only comparison gate required")
    if args.calls not in (32, 2048) or args.warmup != 128 or args.threads not in (1, 2):
        raise RuntimeError("only a 32-call contract smoke or 2048-call long run; warmup128 and T1/T2")
    # Complete formal proof authority precedes imports that can load models/cache.
    audit = artifact_audit(args.formal)
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError("frozen original image file SHA drift: " + str(path))
    # Bind the benchmark, proof, and every image model source before any
    # original-wrapper, model, Torch, or cache import can execute code.
    immutable = immutable_snapshot(args.proof)
    # Pin the image-owned module before host cache/data helpers add their
    # historical source roots to sys.path. The immutable preimage is already set.
    import src.models.components.spint as original_module
    if Path(original_module.__file__).resolve() != Path("/src/models/components/spint.py"):
        raise RuntimeError("not the image-owned released SPINT implementation")
    source = code_source_audit(args.formal)
    proof = require_proof(args.proof, args.proof_sha256, audit, source)
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("CPU only")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location("_released_h1_wrapper", WRAPPER)
    wrapper = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("released wrapper loader unavailable")
    spec.loader.exec_module(wrapper)
    from falcon_challenge.config import FalconConfig, FalconTask
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, ROOT as CACHE_ROOT, validate_authority
    from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    from .h1_causal import H1CausalRuntime
    # Direct existing-cache load only: never invoke a build/rebuild path.
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    validate_authority(cache, json.loads((CACHE_ROOT / "source_cache_authority.json").read_text()))
    total = 1 + args.warmup + args.calls
    row, values = fixed_train_row(cache, total)
    bank = H1Bank(*[row["bank"][name] for name in ("E0", "T", "unit_mask")])
    constructor, reset, engines, models = {}, {}, {}, {}
    started = time.perf_counter_ns()
    original = wrapper.SpintDecoder(FalconConfig(task=FalconTask.h1), str(PAYLOAD), batch_size=1)
    constructor["ORIGINALdefault"] = (time.perf_counter_ns() - started) / 1e6
    if any(t.device.type != "cpu" for t in (*original.clf.parameters(), *original.clf.buffers())):
        raise RuntimeError("released original H1 constructor must be CPU before reset")
    started = time.perf_counter_ns(); original.reset([ORIGINAL_TAG]); reset["ORIGINALdefault"] = (time.perf_counter_ns() - started) / 1e6
    local = original.local_calib_trial_features
    if (original.device.type != "cpu" or original.window_size != W or original.behavior_scaling_factor != SCALE
            or original.smooth_calibration or any(module.training for module in original.local_clf.modules())
            or not isinstance(local, list) or len(local) != 1 or tuple(local[0].shape) != (2, 1024, UNITS)
            or local[0].dtype != torch.float32):
        raise RuntimeError("released original H1 constructor/reset contract drift")
    engines["ORIGINALdefault"] = original
    for arm, index in (("FLAT", 0), ("ROUTE", 1)):
        started = time.perf_counter_ns(); pair = make_v2_unscaled_dot_localbalanced_pair(seed=42); model = pair[index]; del pair
        _load_selected(model, args.formal / "exports" / f"{arm.lower()}_selected_plain_ema.pt")
        engine = H1CausalRuntime(model, bank, batch_size=1)
        constructor[arm] = (time.perf_counter_ns() - started) / 1e6
        # Match the original's separately observable reset boundary.
        started = time.perf_counter_ns(); engine.reset(bank, batch=1); reset[arm] = (time.perf_counter_ns() - started) / 1e6
        engines[arm], models[arm] = engine, model
    timing = {name: [] for name in NAMES}; first, errors = {}, {name: 0.0 for name in NAMES}
    oracle = np.zeros((1, W, UNITS), dtype=np.float32)
    direct_indices = {index for index in (0, 4, W - 1, W, len(values) - 1) if 0 <= index < len(values)}
    for index, raw in enumerate(values):
        oracle[:, :-1] = oracle[:, 1:]; oracle[:, -1] = raw
        shift = index % len(NAMES); outputs = {}
        for name in NAMES[shift:] + NAMES[:shift]:
            started = time.perf_counter_ns(); output = engines[name].predict(raw[None]); elapsed = (time.perf_counter_ns() - started) / 1e6
            assert_public(name, output); outputs[name] = output
            if index == 0: first[name] = elapsed
            elif index >= 1 + args.warmup: timing[name].append(elapsed)
        # This comparison is intentionally outside timed predict calls and holds at every bin.
        observed = {"ORIGINALdefault": engines["ORIGINALdefault"].observation_buffer.transpose(1, 0, 2),
                    "FLAT": engines["FLAT"].raw.numpy(), "ROUTE": engines["ROUTE"].raw.numpy()}
        for name in NAMES:
            if not np.array_equal(observed[name], oracle):
                raise RuntimeError(name + ": independent W700 raw history mismatch")
        if index in direct_indices:
            for name in NAMES:
                direct = _original_oracle(original, oracle) if name == "ORIGINALdefault" else _family_oracle(models[name], bank, oracle)
                assert_public(name + " native oracle", direct)
                np.testing.assert_allclose(outputs[name], direct, atol=1e-5, rtol=1e-5)
                errors[name] = max(errors[name], float(np.abs(outputs[name] - direct).max()))
    if any(len(samples) != args.calls for samples in timing.values()):
        raise RuntimeError("exact timed-call cardinality mismatch")
    post_artifact, post_source = artifact_audit(args.formal), code_source_audit(args.formal)
    if post_artifact != audit or post_source != source:
        raise RuntimeError("fresh post-comparison full authority mismatch")
    immutable_post = require_same_files(immutable)
    require_same_files(audit["files"]); require_same_files(source["files"])
    measured = {name: stats(samples) for name, samples in timing.items()}
    result = {
        "schema": "h1_actual_selected_family_vs_released_original_public_b1_v1",
        "status": "PASS_LONG_PUBLIC_TIMING_ONLY" if args.calls == 2048 else "PASS_CONTRACT_SMOKE_NOT_LATENCY_ACCEPTANCE",
        "scope": "fixed known-source H1 train session only; timing, implementation parity, no quality comparison",
        "batch": 1, "threads": args.threads, "interop_threads": 1, "calls": args.calls, "warmup": args.warmup,
        "session": SESSION, "source_raw_bins": total, "historical_spint_image": IMAGE,
        "original_payload": {"path": str(PAYLOAD), "sha256": EXPECTED[PAYLOAD], "window": W, "units": UNITS,
                             "outputs": OUT, "scale": SCALE, "smooth_calibration": False, "calibration_shape": [2, 1024, UNITS]},
        "family": {"arms": ["FLAT", "ROUTE"], "factory": "make_v2_unscaled_dot_localbalanced_pair(seed=42)",
                   "selected_plain_exports": True, "exact_frozen_bank": True,
                   "calibration_method_disclosure": "family uses explicit frozen banks from three calibration trials; released original uses its as-shipped two-trial calibration array [2,1024,176]; this is not a matched-support or quality experiment"},
        "immutable_pre": immutable, "immutable_post": immutable_post, "pre_artifact": audit, "post_artifact": post_artifact,
        "pre_source": source, "post_source": post_source, "proof_sha256": args.proof_sha256, "proof_status": proof["status"],
        "constructor_ms": constructor, "separate_reset_ms": reset, "first_public_call_ms": first,
        "raw_steady_public_ms": timing, "steady_public_call_ms": measured,
        "p95_ratio_to_original_default": {name: measured[name]["p95_ms"] / measured["ORIGINALdefault"]["p95_ms"] for name in ("FLAT", "ROUTE")},
        "first_public_call_excluded_from_warmup_and_steady": True,
        "rotating_three_engine_order": True, "oracle_indices": sorted(direct_indices),
        "w700_rollover_directly_checked": W in direct_indices, "smoke_does_not_reach_w700_rollover": args.calls == 32,
        "max_public_vs_full_native_abs_error": errors,
        "cache_targets_deserialized_for_existing_cache_integrity_only": True, "quality_scoring": False,
        "parameter_updates": 0, "not_official_latency": True, "host_not_exclusive": True,
        "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
    }
    atomic_json(result, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output), "p95": {name: item["p95_ms"] for name, item in result["steady_public_call_ms"].items()}, "oracle_error": errors}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--container-reference", action="store_true")
    parser.add_argument("--image-digest", default=IMAGE)
    main(parser.parse_args())
