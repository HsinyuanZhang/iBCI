"""Gated source-only timing comparison for selected M2 base/static runtimes.

The static-affine adapter remains experimental.  This harness never invokes an
original decoder, opens labels for scoring, fits anything, or changes selected
weights; it only compares paired public B7 calls on the sealed source stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIGEST = "sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f"
GO = "M2_STATIC_AFFINE_COMPARISON_GO"
PROOF = ROOT / "tfpd_exploration/results/family_runtime_v1/m2_family_source_complete_v1/receipt.json"
PROOF_SHA256 = "a76c6cc871a27e5a450998a02d5e299a09992335b1efec85d0b31891890f6a38"
WRAPPER = ROOT / "tfpd_exploration/submissions/evalai_m2_movement_t4_empty_v1/t4_spint_decoder.py"
PAYLOAD = ROOT / "tfpd_exploration/submissions/evalai_m2_movement_t4_empty_v1/artifacts/t4_m2_seed42_movement_t4_empty_identity.pkl"
LOCAL_REFERENCE_FILES = {
    WRAPPER: "fd1d5b203d9c8daccdda2e212c7efd7720f43717dceef876bd136e99bf7224dc",
    PAYLOAD: "4e4dae8f7239582a26d44cdb449e674710f28223523dd691dd4f8758b05220e0",
}
IMAGE_SPINT = Path("/src/models/components/spint.py")
IMAGE_MODELS = Path("/src/models")
IMAGE_SPINT_SHA256 = "855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519"
W, BATCH, ATOL, RTOL = 50, 7, 1e-5, 1e-5
STATUS = "EXPERIMENTAL_NOT_PROMOTED_PUBLIC_STREAM_TIMING_ONLY"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or not bool((array > 0).all()):
        raise RuntimeError("invalid positive timing samples")
    return {"mean_ms": float(array.mean()), "p50_ms": float(np.percentile(array, 50)),
            "p95_ms": float(np.percentile(array, 95)), "p99_ms": float(np.percentile(array, 99)),
            "max_ms": float(array.max())}


def ratio_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or not bool((array > 0).all()):
        raise RuntimeError("invalid positive dimensionless ratio samples")
    return {"mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)), "p99": float(np.percentile(array, 99)),
            "max": float(array.max())}


def paired_ratio_summary(static_samples: list[float], base_samples: list[float]) -> dict[str, object]:
    if len(static_samples) != len(base_samples) or not len(base_samples):
        raise RuntimeError("paired timing samples require equal nonempty cardinality")
    raw = [static / base for static, base in zip(static_samples, base_samples)]
    return {"raw_sample_ratio_static_over_base": raw, "dimensionless_ratio_stats": ratio_stats(raw)}


def assert_public(name: str, output: object) -> None:
    if (not isinstance(output, np.ndarray) or output.shape != (BATCH, 2) or output.dtype != np.float32
            or not output.flags.owndata or not output.flags.c_contiguous or not np.isfinite(output).all()):
        raise RuntimeError(name + ": owning contiguous finite FP32 [7,2] public output required")


def stream_count(calls: int, warmup: int) -> int:
    if calls not in (32, 2048) or warmup != 128:
        raise RuntimeError("fixed comparison stream cardinality requires calls32/2048 and warmup128")
    return 1 + warmup + calls


def _gate(args) -> None:
    if (os.environ.get(GO) != "1" or not args.container_reference or args.image_digest != IMAGE_DIGEST
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        raise RuntimeError("explicit same-image CPU-only static-affine comparison gate required")
    if args.calls not in (32, 2048) or args.warmup != 128 or args.threads not in (1, 2):
        raise RuntimeError("only calls32/2048, warmup128, and T1/T2 are permitted")
    if args.output.exists():
        raise FileExistsError(args.output)


def _preimport_image_audit() -> dict[str, object]:
    local = {str(path): sha(path) for path in LOCAL_REFERENCE_FILES}
    if local != {str(path): expected for path, expected in LOCAL_REFERENCE_FILES.items()}:
        raise RuntimeError("fixed local reference wrapper/payload SHA drift")
    image = {str(path): sha(path) for path in sorted(IMAGE_MODELS.rglob("*.py"))}
    if image.get(str(IMAGE_SPINT)) != IMAGE_SPINT_SHA256:
        raise RuntimeError("fixed image-owned SPINT module SHA drift")
    python = Path(sys.executable).resolve()
    return {"image": IMAGE_DIGEST, "python": {"path": str(python), "sha256": sha(python)},
            "local_reference_files": local, "image_owned_files": image}


def _proof(authority: dict) -> dict:
    if sha(PROOF) != PROOF_SHA256:
        raise RuntimeError("completed source runtime proof SHA drift")
    proof = json.loads(PROOF.read_text())
    if proof.get("status") != "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY" or proof.get("authority_pre") != proof.get("authority_post"):
        raise RuntimeError("completed source runtime proof status/authority drift")
    prior = proof["authority_pre"]
    for key in ("selected", "finalized_model_source", "verified_training_recipe"):
        if prior.get(key) != authority.get(key):
            raise RuntimeError("completed source proof/current selected-source authority disagreement: " + key)
    return proof


def _code_audit() -> dict[str, str]:
    names = ("m2_static_affine_comparison.py", "m2_family_static_affine.py", "m2_family_causal.py",
             "m2_family_spatial.py", "complete_m2_family_source.py", "m2_family_spint_comparison.py",
             "m2_spint_comparison.py", "h1_causal.py", "h1_lifted_frontend.py", "benchmark.py",
             "linear_conv.py", "grouped_value.py")
    return {name: sha(Path(__file__).with_name(name)) for name in names}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _bank_values_digest(bank, values: np.ndarray) -> dict[str, str]:
    def tensor(value):
        return hashlib.sha256(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()).hexdigest()
    return {"E0": tensor(bank.E0), "T": tensor(bank.T), "unit_mask": tensor(bank.unit_mask),
            "source_values": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()}


def run(args) -> dict:
    _gate(args)
    # No torch, decoder, source-bank, or source-proof helper imports precede
    # these exact image/Python/wrapper/payload identity checks.
    image_pre = _preimport_image_audit()
    code_pre = _code_audit()
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only static-affine comparison required")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    from .complete_m2_family_source import require_final
    from .m2_family_causal import M2FamilyCausalRuntime
    from .m2_family_static_affine import M2FamilyStaticAffineRuntime
    from .m2_family_spint_comparison import _strict_source_banks
    from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders
    receipt, authority = require_final()
    proof = _proof(authority)
    count = stream_count(args.calls, args.warmup)
    _tags, bank, values, source_hashes = _strict_source_banks(authority, count)
    if values.shape != (count, BATCH, 96):
        raise RuntimeError("fixed B7 continuous source stream geometry drift")
    source_values_pre = _bank_values_digest(bank, values)
    engines, models, prepare, constructor, reset, model_digest = {}, {}, {}, {}, {}, {}
    for arm, index in (("FLAT", 0), ("ROUTE", 1)):
        begun = time.perf_counter_ns()
        model = make_paired_decoders(42)[index]
        state = torch.load(authority["selected"][arm]["path"], map_location="cpu", weights_only=True)
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in state.values()):
            raise RuntimeError("selected plain EMA finite FP32 state required")
        model.load_state_dict(state, strict=True); model.eval()
        if any(module.training for module in model.modules()):
            raise RuntimeError("selected model eval contract drift")
        prepare[arm] = (time.perf_counter_ns() - begun) / 1e6
        model_digest[arm] = _model_digest(model)
        begun = time.perf_counter_ns()
        base = M2FamilyCausalRuntime(model, bank, batch_size=BATCH)
        constructor[arm + "_base"] = (time.perf_counter_ns() - begun) / 1e6
        begun = time.perf_counter_ns(); static = M2FamilyStaticAffineRuntime(model, bank, batch_size=BATCH)
        constructor[arm + "_static"] = (time.perf_counter_ns() - begun) / 1e6
        for name, engine in ((arm + "_base", base), (arm + "_static", static)):
            begun = time.perf_counter_ns(); engine.reset(bank, batch=BATCH)
            reset[name] = (time.perf_counter_ns() - begun) / 1e6
            engines[name] = engine
        models[arm] = model
    names = ("FLAT_base", "FLAT_static", "ROUTE_base", "ROUTE_static")
    samples, first, errors = {name: [] for name in names}, {}, {name: 0.0 for name in names}
    pair_errors = {arm: {"max_output_abs_error": 0.0, "max_frontend_abs_error": 0.0} for arm in ("FLAT", "ROUTE")}
    native_calls = {arm: 0 for arm in ("FLAT", "ROUTE")}
    independent = np.zeros((BATCH, W, 96), dtype=np.float32)
    oracle_indices = {index for index in (0, 4, 49, 50, len(values) - 1) if index < len(values)}
    for index, raw in enumerate(values):
        independent[:, :-1] = independent[:, 1:]; independent[:, -1] = raw
        outputs = {}; shift = index % len(names)
        for name in names[shift:] + names[:shift]:
            begun = time.perf_counter_ns(); output = engines[name].predict(raw)
            elapsed = (time.perf_counter_ns() - begun) / 1e6
            assert_public(name, output); outputs[name] = output
            if index == 0: first[name] = elapsed
            elif index >= 1 + args.warmup: samples[name].append(elapsed)
        for name in names:
            if not np.array_equal(engines[name].raw.numpy(), independent):
                raise RuntimeError(name + ": independent W50 raw-history drift")
        for arm in ("FLAT", "ROUTE"):
            base, static = outputs[arm + "_base"], outputs[arm + "_static"]
            np.testing.assert_allclose(static, base, atol=ATOL, rtol=RTOL, err_msg=arm + "/static-to-base")
            np.testing.assert_allclose(engines[arm + "_static"].frontend.numpy(), engines[arm + "_base"].frontend.numpy(), atol=ATOL, rtol=RTOL, err_msg=arm + "/static-frontend")
            pair_errors[arm]["max_output_abs_error"] = max(pair_errors[arm]["max_output_abs_error"], float(np.abs(static - base).max()))
            pair_errors[arm]["max_frontend_abs_error"] = max(pair_errors[arm]["max_frontend_abs_error"], float(np.abs(engines[arm + "_static"].frontend.numpy() - engines[arm + "_base"].frontend.numpy()).max()))
        if index in oracle_indices:
            for arm in ("FLAT", "ROUTE"):
                with torch.no_grad():
                    oracle = (models[arm].forward_last(torch.from_numpy(independent.copy()), bank, bank.unit_mask) / 5).numpy().copy()
                native_calls[arm] += 1
                for suffix in ("base", "static"):
                    name = arm + "_" + suffix
                    np.testing.assert_allclose(outputs[name], oracle, atol=ATOL, rtol=RTOL, err_msg=name + "/native-oracle")
                    errors[name] = max(errors[name], float(np.abs(outputs[name] - oracle).max()))
    if (any(len(samples[name]) != args.calls for name in names)
            or any(native_calls[arm] != len(oracle_indices) for arm in models)):
        raise RuntimeError("exact steady-call cardinality drift")
    for arm, model in models.items():
        if _model_digest(model) != model_digest[arm]:
            raise RuntimeError(arm + ": selected model weights mutated")
    _receipt_post, authority_post = require_final()
    if authority_post != authority or _proof(authority_post) != proof:
        raise RuntimeError("selected/source proof authority drifted during comparison")
    if _bank_values_digest(bank, values) != source_values_pre:
        raise RuntimeError("in-memory source bank or neural values mutated during comparison")
    _tags_post, bank_post, values_post, source_hashes_post = _strict_source_banks(authority_post, count)
    source_values_post = _bank_values_digest(bank_post, values_post)
    image_post, code_post = _preimport_image_audit(), _code_audit()
    if (image_post != image_pre or code_post != code_pre or source_hashes_post != source_hashes
            or source_values_post != source_values_pre):
        raise RuntimeError("image/Python/self/helper/runtime authority drifted during comparison")
    timing = {name: stats(samples[name]) for name in names}
    paired = {arm: {**paired_ratio_summary(samples[arm + "_static"], samples[arm + "_base"]),
                    "p95_static_over_base": timing[arm + "_static"]["p95_ms"] / timing[arm + "_base"]["p95_ms"]}
              for arm in ("FLAT", "ROUTE")}
    result = {"schema": "m2_selected_static_affine_same_image_public_b7_v1", "status": STATUS,
              "experimental_candidate": "precomposed E0/T/bias frontend affine term", "promoted": False,
              "batch": BATCH, "calls": args.calls, "warmup": args.warmup, "threads": args.threads,
              "shared_model_prepare_ms": prepare, "runtime_constructor_initial_reset_ms": constructor,
              "separate_reset_ms": reset, "first_public_call_ms": first,
              "raw_steady_public_ms": samples,
              "steady_public_call_ms": timing, "static_to_base_paired_ratios": paired,
              "runtime_state_bytes": {name: engines[name].state_bytes() for name in names},
              "static_affine_bytes": {name: engines[name].static_affine_bytes for name in names if name.endswith("_static")},
              "rotating_four_engine_order": True, "oracle_indices": sorted(oracle_indices), "max_native_abs_error": errors,
              "all_bin_static_to_base_max_abs_error": pair_errors,
              "native_oracle_calls_per_arm": native_calls, "public_bins": len(values), "steady_call_cardinality": {name: len(samples[name]) for name in names},
              "authority_pre": authority, "authority_post": authority_post, "source_train_bank_sha256_pre": source_hashes,
              "source_train_bank_sha256_post": source_hashes_post, "source_bank_values_digest_pre": source_values_pre,
              "source_bank_values_digest_post": source_values_post,
              "completed_source_proof_sha256": PROOF_SHA256, "completed_source_proof_status": proof["status"],
              "image_pre": image_pre, "image_post": image_post, "code_pre": code_pre, "code_post": code_post,
              "selected_model_digest_pre_post_equal": True, "parameter_updates": 0,
              "source_minival_integrity_deserialized_by_require_final": True,
              "source_train_targets_hashed_bytes_only": True, "labels_fitted_or_scored": False,
              "original_decoder_inference": False, "quality_or_selection_claim": False,
              "not_official_latency": True, "host_not_exclusive": True,
              "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__}
    _atomic_json(args.output, result)
    return result


def _model_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--container-reference", action="store_true")
    parser.add_argument("--image-digest", default=IMAGE_DIGEST)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "calls": result["calls"],
                      "p95_ms": {key: value["p95_ms"] for key, value in result["steady_public_call_ms"].items()},
                      "max_native_abs_error": result["max_native_abs_error"]}, sort_keys=True))
