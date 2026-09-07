"""Review-gated, same-host public B7 timing comparison for M2.

This deliberately compares four *actual* fixed implementations on the same
continuous source-only neural stream: the frozen SPINT image's as-shipped and
declared payloads, plus the finalized FLAT and ROUTE family exports.  It is a
latency/streaming-contract experiment, never a quality or selection test.

Nothing in this module runs at import time.  ``main`` refuses to compute unless
the coordinator has supplied the explicit environment gate.
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

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders

from .benchmark import sha, stats
from .complete_m2_family_source import require_final
from .m2_family_causal import M2FamilyCausalRuntime, M2RuntimeBank
from .m2_spint_comparison import AS_SHIPPED, DECLARED, EXPECTED, WRAPPER


ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIGEST = "sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f"
SPINT_SHA256 = "855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519"
GO = "M2_FAMILY_SPINT_COMPARISON_GO"
PROOF_RECEIPT = ROOT / "tfpd_exploration/results/family_runtime_v1/m2_family_source_complete_v1/receipt.json"
PROOF_RECEIPT_SHA256 = "a76c6cc871a27e5a450998a02d5e299a09992335b1efec85d0b31891890f6a38"
PROOF_STATUS = "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY"
W = 50
NATIVE_ATOL = 1.0e-5
NATIVE_RTOL = 1.0e-5


def _file_sha(path: Path) -> str:
    return sha(path)


def _require_gate(args: argparse.Namespace) -> None:
    if os.environ.get(GO) != "1":
        raise RuntimeError(f"review gate required: export {GO}=1")
    if not args.container_reference:
        raise RuntimeError("the frozen SPINT image/container reference must be explicit")
    if args.image_digest != IMAGE_DIGEST:
        raise RuntimeError("frozen SPINT image digest does not match the audited image")
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1") or torch.cuda.is_available():
        raise RuntimeError("this same-host public timing comparison is CPU-only")


def _load_wrapper() -> tuple[object, float]:
    begun = time.perf_counter_ns()
    spec = importlib.util.spec_from_file_location("_m2_spint_frozen_image_reference", WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen SPINT wrapper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, (time.perf_counter_ns() - begun) / 1e6


def _immutable_hashes(spint_module) -> dict[str, object]:
    """Everything outside the family finalizer that must not drift mid-run."""
    paths = {
        "comparison_self": Path(__file__),
        "m2_family_causal": Path(__file__).with_name("m2_family_causal.py"),
        "complete_m2_family_source": Path(__file__).with_name("complete_m2_family_source.py"),
        "spint_module": Path(spint_module.__file__),
        **{f"frozen_{path.name}": path for path in EXPECTED},
        "completed_runtime_proof": PROOF_RECEIPT,
    }
    return {name: _file_sha(path) for name, path in paths.items()}


def _require_proof_receipt() -> dict[str, object]:
    if not PROOF_RECEIPT.is_file() or _file_sha(PROOF_RECEIPT) != PROOF_RECEIPT_SHA256:
        raise RuntimeError("completed independent M2 runtime proof receipt drift")
    receipt = json.loads(PROOF_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("status") != PROOF_STATUS:
        raise RuntimeError("completed independent M2 runtime proof is not PASS")
    return receipt


def _session_tag(session: str) -> str:
    parts = session.removeprefix("ses-").split("-")
    if len(parts) != 4 or not parts[-1].startswith("Run"):
        raise RuntimeError(f"unrecognized source session tag: {session}")
    return f"{parts[-1]}_{''.join(parts[:3])}"


def _strict_source_banks(authority: dict[str, object], count: int) -> tuple[list[str], M2RuntimeBank, np.ndarray, dict[str, dict[str, str]]]:
    """Load source neural/bank inputs; target bytes are hashed but never mapped."""
    recipe = authority["verified_training_recipe"]
    expected = recipe["source_cache_sha256"]
    sessions = list(plan.HELDIN_SESSIONS)
    if set(expected) != set(sessions) or len(sessions) != 7:
        raise RuntimeError("finalized seven-session source-train authority drift")
    e0, t4, raw_rows, actual = [], [], [], {}
    required = ("X_store.npy", "target_store.npy", "eligible_starts.npy", "T.npy", "e0_u.pt", "mapping.json", "provenance.json")
    for session in sessions:
        root = data._session_dir("source_train", session)
        paths = {name: root / name for name in required}
        if any(not path.is_file() for path in paths.values()):
            raise RuntimeError(f"{session}: compact source bank is incomplete")
        # This reads target-store bytes solely for frozen-recipe integrity; it
        # never maps/deserializes a target array.  `require_final` separately
        # performs its existing source-minival integrity deserialization.
        hashes = {name: _file_sha(path) for name, path in paths.items()}
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        hashes["provenance_e0_sha256"] = provenance.get("e0_sha256")
        hashes["provenance_t4_sha256"] = provenance.get("t4_sha256")
        if hashes != expected[session]:
            raise RuntimeError(f"{session}: finalized source-train bank/provenance drift")
        raw = np.load(root / "X_store.npy", mmap_mode="r")
        if raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 96 or not np.all(raw[: W - 1] == 0):
            raise RuntimeError(f"{session}: raw source stream/W49 startup drift")
        row = np.array(raw[W - 1:W - 1 + count], dtype=np.float32, copy=True)
        if row.shape != (count, 96) or not row.flags.c_contiguous or not np.isfinite(row).all():
            raise RuntimeError(f"{session}: insufficient/nonfinite continuous raw source bins")
        payload = torch.load(root / "e0_u.pt", map_location="cpu", weights_only=False)
        local_e0 = payload.get("E0")
        local_t = torch.from_numpy(np.load(root / "T.npy"))
        if (not isinstance(local_e0, torch.Tensor) or local_e0.shape != (96, 50)
                or local_e0.dtype != torch.float32 or local_t.shape != (96, 4)
                or local_t.dtype != torch.float32 or not bool(torch.isfinite(local_e0).all())
                or not bool(torch.isfinite(local_t).all())):
            raise RuntimeError(f"{session}: E0/T geometry, dtype, or finite drift")
        e0.append(local_e0.contiguous())
        t4.append(local_t.contiguous())
        raw_rows.append(row)
        actual[session] = hashes
    bank = M2RuntimeBank(torch.stack(e0), torch.stack(t4), torch.ones(7, 96, dtype=torch.bool))
    return [_session_tag(session) for session in sessions], bank, np.stack(raw_rows, axis=1), actual


def _assert_public(name: str, output: np.ndarray) -> None:
    if (output.shape != (7, 2) or output.dtype != np.float32 or not output.flags.c_contiguous
            or not output.flags.owndata or not np.isfinite(output).all()):
        raise RuntimeError(f"{name}: public whole-B7 native output contract drift")


def _spint_oracle(engine) -> np.ndarray:
    """Independent full SPINT decode from the wrapper's current W50 buffer."""
    neural = torch.as_tensor(engine.observation_buffer.copy().transpose(1, 0, 2), dtype=torch.float32, device=engine.device)
    with torch.inference_mode():
        rows = [engine._decode_with_identity(neural[i:i + 1], identity)[:, -1, :]
                for i, identity in enumerate(engine.local_identities)]
    return (torch.cat(rows, dim=0).cpu().numpy() / engine.behavior_scaling_factor).astype(np.float32, copy=True)


def _family_oracle(engine: M2FamilyCausalRuntime) -> np.ndarray:
    with torch.inference_mode():
        value = (engine.model.forward_last(engine.raw, engine.bank, engine.bank.unit_mask) / 5).numpy()
    return np.ascontiguousarray(value, dtype=np.float32)


def _close(name: str, public: np.ndarray, oracle: np.ndarray) -> float:
    if not np.isfinite(public).all() or not np.isfinite(oracle).all():
        raise RuntimeError(f"{name}: nonfinite public/oracle output")
    np.testing.assert_allclose(public, oracle, atol=NATIVE_ATOL, rtol=NATIVE_RTOL, err_msg=name)
    return float(np.max(np.abs(public - oracle)))


def _build_engines(wrapper, config, authority: dict[str, object], tags: list[str], bank: M2RuntimeBank):
    engines, constructor_ms, reset_ms = {}, {}, {}
    for name, payload in (("ORIGINALdefault", AS_SHIPPED), ("ORIGINALdeclared", DECLARED)):
        begun = time.perf_counter_ns()
        engine = wrapper.T4CachedIdentityDecoder(config, str(payload), batch_size=7)
        constructor_ms[name] = (time.perf_counter_ns() - begun) / 1e6
        begun = time.perf_counter_ns()
        engine.reset([Path(tag) for tag in tags])
        reset_ms[name] = (time.perf_counter_ns() - begun) / 1e6
        if engine.device.type != "cpu" or engine.smooth_observations:
            raise RuntimeError(f"{name}: frozen wrapper CPU/unsmoothed contract drift")
        engines[name] = engine
    for arm, index in (("FLAT", 0), ("ROUTE", 1)):
        item = authority["selected"][arm]
        begun = time.perf_counter_ns()
        model = make_paired_decoders(42)[index]
        state = torch.load(item["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        if any(module.training for module in model.modules()):
            raise RuntimeError(f"{arm}: finalized runtime model not fully eval")
        engine = M2FamilyCausalRuntime(model, bank, batch_size=7)
        constructor_ms[arm] = (time.perf_counter_ns() - begun) / 1e6
        # The family runtime constructor performs one exact zero reset.  Time a
        # distinct subsequent reset, matching the public lifecycle boundary.
        begun = time.perf_counter_ns()
        engine.reset(bank, batch=7)
        reset_ms[arm] = (time.perf_counter_ns() - begun) / 1e6
        engines[arm] = engine
    return engines, constructor_ms, reset_ms


def main(args: argparse.Namespace) -> None:
    _require_gate(args)
    if args.calls != 2048 or args.warmup != 128:
        raise RuntimeError("frozen comparison requires exactly 128 warmup and 2048 timed calls")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    receipt, authority = require_final()
    if not receipt or not authority:
        raise RuntimeError("unreachable finalized authority failure")
    for path, expected in EXPECTED.items():
        if _file_sha(path) != expected:
            raise RuntimeError(f"frozen SPINT wrapper/payload drift: {path}")
    module_import_started = time.perf_counter_ns()
    import src.models.components.spint as spint_module
    module_import_ms = (time.perf_counter_ns() - module_import_started) / 1e6
    if _file_sha(Path(spint_module.__file__)) != SPINT_SHA256:
        raise RuntimeError("actual container SPINT module authority drift")
    proof = _require_proof_receipt()
    immutable_pre = _immutable_hashes(spint_module)
    wrapper, wrapper_import_ms = _load_wrapper()
    from falcon_challenge.config import FalconConfig, FalconTask
    tags, bank, values, source_hashes = _strict_source_banks(authority, 1 + args.warmup + args.calls)
    engines, constructor_ms, reset_ms = _build_engines(wrapper, FalconConfig(task=FalconTask.m2), authority, tags, bank)
    names = ("ORIGINALdefault", "ORIGINALdeclared", "FLAT", "ROUTE")
    times = {name: [] for name in names}
    first, oracle_error = {}, {name: 0.0 for name in names}
    oracle_indices = {0, W - 1, W, len(values) - 1}
    independent_raw = np.zeros((7, W, 96), dtype=np.float32)
    for index, raw in enumerate(values):
        # A separate NumPy W50 record checks public streaming state.  It is
        # intentionally outside each engine's timed predict boundary.
        independent_raw[:, :-1] = independent_raw[:, 1:]
        independent_raw[:, -1] = raw
        outputs = {}
        shift = index % len(names)
        for name in names[shift:] + names[:shift]:
            begun = time.perf_counter_ns()
            output = engines[name].predict(raw)
            elapsed = (time.perf_counter_ns() - begun) / 1e6
            _assert_public(name, output)
            outputs[name] = output
            if index == 0:
                first[name] = elapsed
            elif index >= 1 + args.warmup:
                times[name].append(elapsed)
        if index in oracle_indices:
            for name in names:
                if name.startswith("ORIGINAL"):
                    observed = np.ascontiguousarray(engines[name].observation_buffer.transpose(1, 0, 2))
                else:
                    observed = engines[name].raw.numpy()
                if not np.array_equal(observed, independent_raw):
                    raise RuntimeError(f"{name}: independent NumPy raw W50 history drift at {index}")
                oracle = _spint_oracle(engines[name]) if name.startswith("ORIGINAL") else _family_oracle(engines[name])
                oracle_error[name] = max(oracle_error[name], _close(f"{name}/full-oracle/{index}", outputs[name], oracle))
    if any(len(times[name]) != args.calls for name in names):
        raise RuntimeError("steady timing cardinality drift")
    _, authority_post = require_final()
    if authority != authority_post:
        raise RuntimeError("finalized source/selection authority changed during comparison")
    immutable_post = _immutable_hashes(spint_module)
    if immutable_pre != immutable_post:
        raise RuntimeError("self/helper/SPINT/wrapper/payload/proof authority changed during comparison")
    result = {
        "schema": "m2_family_actual_spint_vs_selected_public_b7_source_train_v1",
        "status": "PASS_PUBLIC_STREAM_TIMING_ONLY",
        "batch": 7, "calls": args.calls, "warmup": args.warmup,
        "source_surface": "source_train_continuous_raw_bins_only",
        "authority_validation_deserializes_existing_minival_targets": True,
        "targets_used_for_timing_or_quality": False, "nwb_opened": False,
        "source_train_target_store_hash_bytes_only": True, "parameter_updates": 0,
        "historical_spint_image": IMAGE_DIGEST, "spint_module_path": str(spint_module.__file__),
        "spint_module_sha256": SPINT_SHA256, "spint_module_import_ms": module_import_ms,
        "wrapper_import_ms": wrapper_import_ms, "constructor_ms": constructor_ms, "reset_ms": reset_ms,
        "first_public_call_ms": first, "steady_public_call_ms": {name: stats(times[name]) for name in names},
        "first_call_excluded_from_warmup_and_steady": True,
        "rotating_four_engine_order": True, "oracle_indices": sorted(oracle_indices),
        "max_public_vs_full_model_oracle_abs_error": oracle_error,
        "completed_runtime_proof_receipt_sha256": PROOF_RECEIPT_SHA256,
        "completed_runtime_proof_status": proof["status"],
        "completed_runtime_proof_scope": "B1 per-session / 1011 endpoints per selected arm / direct error <7e-8",
        "authority_pre": authority, "authority_post": authority_post,
        "immutable_hashes_pre": immutable_pre, "immutable_hashes_post": immutable_post,
        "source_train_bank_sha256": source_hashes,
        "frozen_spint_wrapper_and_payload_sha256": {str(path): expected for path, expected in EXPECTED.items()},
        "code_sha256": {Path(__file__).name: _file_sha(Path(__file__)),
                        "m2_family_causal.py": _file_sha(Path(__file__).with_name("m2_family_causal.py")),
                        "complete_m2_family_source.py": _file_sha(Path(__file__).with_name("complete_m2_family_source.py"))},
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
        "not_official_latency": True, "host_not_exclusive": True,
        "not_quality_generalization_or_selection_claim": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "batch": 7, "calls": args.calls,
                      "p95_ms": {name: result["steady_public_call_ms"][name]["p95_ms"] for name in names},
                      "max_oracle_error": oracle_error, "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--calls", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--container-reference", action="store_true")
    parser.add_argument("--image-digest", default=IMAGE_DIGEST)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
