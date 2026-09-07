"""Post-finalization exact source replay of both selected M2 family runtimes.

Only implementation equivalence is evaluated. Both source-only epoch picks
must already be durable, with complete finalizer archives; this cannot select
an epoch or write to the finalizer, training, cache or submission directories.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_same_query_comparator_v1 import core
from tfpd_exploration.src.m2_family_v1 import finalize_pair as finalizer
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders
from .m2_family_causal import M2FamilyCausalRuntime, M2RuntimeBank

ROOT = Path(__file__).resolve().parents[3]
FINAL = finalizer.OUT_ROOT
RESULT = ROOT / "tfpd_exploration/results/family_runtime_v1/m2_family_source_complete_v1"
COUNTS = (173, 129, 117, 116, 141, 141, 194)
FINAL_STATUS = "SOURCE_MINIVAL_SELECTION_DIAGNOSTIC_NOT_UNTOUCHED_GENERALIZATION"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value))


def runtime_code():
    siblings = ("complete_m2_family_source.py", "m2_family_causal.py", "m2_family_spatial.py",
                "h1_causal.py", "h1_lifted_frontend.py", "linear_conv.py", "grouped_value.py")
    return {name: sha(Path(__file__).with_name(name)) for name in siblings}


def require_final():
    receipt_path, freeze_path = FINAL / "receipt.json", FINAL / "selection_freeze.json"
    if not receipt_path.is_file() or not freeze_path.is_file():
        raise RuntimeError("completed finalizer receipt/freeze absent")
    receipt, freeze = (json.loads(path.read_text()) for path in (receipt_path, freeze_path))
    if receipt.get("status") != FINAL_STATUS or freeze.get("status") != "SELECTION_FROZEN_PRE_SCORE":
        raise RuntimeError("finalizer/freeze status not eligible")
    audit = finalizer.validate_summary()
    audit = canonical({k: v for k, v in audit.items() if k != "summary"})
    if receipt.get("audit") != audit or freeze.get("audit") != audit:
        raise RuntimeError("durable picks/complete24 training audit drift")
    source = finalizer._code_and_source_authority()
    if source != receipt.get("authority_pre") or source != receipt.get("authority_post") or source != freeze.get("authority_pre"):
        raise RuntimeError("finalized model/source authority drift")
    training_recipe = finalizer._verify_pretrain_recipe(json.loads((finalizer.PAIR_ROOT / "pretrain_manifest.json").read_text()))
    if training_recipe != receipt.get("verified_training_recipe") or training_recipe != freeze.get("verified_training_recipe"):
        raise RuntimeError("finalized original training recipe hash drift")
    if set(audit["picks"]) != {"FLAT", "ROUTE"}:
        raise RuntimeError("exactly two finalized selected arms required")
    exports = {}
    for arm in ("FLAT", "ROUTE"):
        epoch = audit["picks"][arm]
        key = f"{arm}_selected_epoch_{epoch:03d}"
        matches = [k for k in receipt["exports"] if k.startswith(arm + "_selected_")]
        if matches != [key]:
            raise RuntimeError("selected export identity drift")
        export, score = receipt["exports"][key], receipt["scores"][key]
        path, archive = FINAL / f"{key}_ema_state.pt", FINAL / f"{key}_source_minival.npz"
        if str(path) != export["export_path"] or sha(path) != export["export_sha256"] or sha(archive) != score["npz_sha256"]:
            raise RuntimeError("selected exported weight/archive hash drift")
        exports[arm] = {"key": key, "path": str(path), "sha256": sha(path),
                        "archive": str(archive), "archive_sha256": sha(archive)}
    authority = {"receipt_sha256": sha(receipt_path), "freeze_sha256": sha(freeze_path),
                 "complete_training_audit": audit, "finalized_model_source": source,
                 "verified_training_recipe": training_recipe,
                 "runtime_code": runtime_code(), "selected": exports}
    return receipt, authority


def close(got, want, label):
    if got.shape != want.shape or not np.isfinite(got).all() or not np.isfinite(want).all():
        raise RuntimeError(f"{label}: nonfinite/shape drift")
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5, err_msg=label)
    return float(np.max(np.abs(got - want)))


def validate_arrays(raw, starts, target, count):
    if (raw.dtype != np.float32 or raw.ndim != 2 or raw.shape[1] != 96 or len(raw) < 50
            or starts.dtype != np.int64 or starts.shape != (count,) or target.dtype != np.float32
            or target.shape != (count, 2) or not np.isfinite(raw).all() or not np.isfinite(target).all()
            or np.any(np.diff(starts) <= 0) or starts[0] < 0 or starts[-1] + 50 > len(raw)
            or not np.all(raw[:49] == 0)):
        raise RuntimeError("source raw/start/count/padding contract drift")


def validate_archive(arrays, reference):
    for name in ("target", "session", "start"):
        if not np.array_equal(arrays[name], reference[name]):
            raise RuntimeError(f"finalizer archive {name} order/value drift")
    return close(arrays["prediction"], reference["prediction"], "runtime/finalizer")


def run_session(model, session, count):
    folder = data._session_dir("source_minival", session)
    raw = np.load(folder / "X_store.npy", mmap_mode="r")
    starts = np.asarray(np.load(folder / "eligible_starts.npy"))
    target = np.asarray(np.load(folder / "target_store.npy", mmap_mode="r"))
    validate_arrays(raw, starts, target, count)
    source_bank = data.load_session_bank("source_minival", session, device="cpu")
    bank = M2RuntimeBank(source_bank.E0, source_bank.T, source_bank.unit_mask)
    runtime = M2FamilyCausalRuntime(model, bank)
    wanted = {int(start + 49): index for index, start in enumerate(starts)}
    prediction, seen = [], []
    for tick in range(49, len(raw)):
        got = runtime.predict(np.array(raw[tick:tick + 1], dtype=np.float32, copy=True))
        if not got.flags.owndata or not got.flags.c_contiguous or got.dtype != np.float32:
            raise RuntimeError("public runtime output ownership/dtype drift")
        if tick in wanted:
            index = wanted[tick]
            if not np.array_equal(runtime.raw[0].numpy(), raw[starts[index]:starts[index] + 50]):
                raise RuntimeError("runtime history not the independent raw W50 window")
            seen.append(index)
            prediction.append(got[0].copy())
    if seen != list(range(count)):
        raise RuntimeError("every selected endpoint must be visited exactly once in order")
    prediction = np.stack(prediction)
    direct = []
    with torch.no_grad():
        for off in range(0, count, 16):
            windows = np.stack([raw[i:i + 50] for i in starts[off:off + 16]])
            direct.append((model.forward_last(torch.from_numpy(windows), bank, bank.unit_mask) / 5).numpy())
    direct = np.concatenate(direct).astype(np.float32)
    error = close(prediction, direct, "runtime/independent direct W50")
    score = core.variance_weighted_r2(target, prediction)
    if not np.isfinite(score):
        raise RuntimeError("nonfinite session native score")
    arrays = {"prediction": prediction, "direct_full": direct, "target": target,
              "start": starts, "session": np.asarray([session] * count)}
    return {"session": session, "endpoint_count": count, "public_bins": len(raw) - 49,
            "native_r2": score, "max_direct_abs_error": error}, arrays


def atomic_npz(path, arrays):
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def run():
    if os.environ.get("M2_FAMILY_SOURCE_COMPLETE") != "1":
        raise RuntimeError("explicit review GO required")
    if RESULT.exists():
        raise FileExistsError(RESULT)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1") or torch.cuda.is_available():
        raise RuntimeError("CUDA must be disabled before process start")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt, authority = require_final()
    RESULT.mkdir(parents=True)
    finalizer._atomic_json(RESULT / "authority_pre.json", authority)
    started, results = time.perf_counter(), {}
    for arm in ("FLAT", "ROUTE"):
        export = authority["selected"][arm]
        model = make_paired_decoders(42)[0 if arm == "FLAT" else 1]
        state = torch.load(export["path"], map_location="cpu", weights_only=True)
        if not all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in state.values()):
            raise RuntimeError("plain exported state dtype/finite drift")
        model.load_state_dict(state, strict=True)
        model.eval()
        rows, pieces = [], []
        for session, count in zip(plan.HELDIN_SESSIONS, COUNTS, strict=True):
            row, arrays = run_session(model, session, count)
            rows.append(row)
            pieces.append(arrays)
        arrays = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
        if arrays["prediction"].shape != (1011, 2):
            raise RuntimeError("global exact1011 endpoint count drift")
        with np.load(export["archive"], allow_pickle=False) as reference:
            final_error = validate_archive(arrays, reference)
        equal = float(np.mean([r["native_r2"] for r in rows]))
        pooled = core.variance_weighted_r2(arrays["target"], arrays["prediction"])
        recorded = receipt["scores"][export["key"]]
        if (not np.isfinite(equal) or not np.isfinite(pooled)
                or abs(equal - recorded["equal_session_r2"]) > 1e-5
                or abs(pooled - recorded["pooled_r2"]) > 1e-5):
            raise RuntimeError("finalized selected native score reproduction drift")
        archive = RESULT / f"{arm}_native.npz"
        atomic_npz(archive, arrays)
        results[arm] = {"key": export["key"], "endpoint_count": 1011,
            "public_bins": sum(row["public_bins"] for row in rows), "per_session": rows,
            "equal_session_r2": equal, "pooled_r2": pooled, "max_finalizer_abs_error": final_error,
            "max_direct_abs_error": max(row["max_direct_abs_error"] for row in rows),
            "native_archive_sha256": sha(archive), "native_array_sha256": {k: core.array_sha256(v) for k, v in arrays.items()}}
        print(json.dumps({"arm": arm, "status": "ARM_REPLAY_COMPLETE", "equal_session_r2": equal}), flush=True)
    _, authority_after = require_final()
    if authority != authority_after:
        raise RuntimeError("runtime/finalizer/source/weight/archive authority changed")
    result = {"schema": "m2_family_source_stream_equivalence_v1", "status": "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY",
        "authority_pre": authority, "authority_post": authority_after, "results": results,
        "elapsed_seconds": time.perf_counter() - started, "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(), "affinity": sorted(os.sched_getaffinity(0)),
        "torch": torch.__version__, "parameter_updates": 0, "new_selection_or_generalization_claim": False,
        "numeric_precision": "CPU FP32 forward/native outputs; float64 R2 aggregation"}
    finalizer._atomic_json(RESULT / "receipt.json", result)
    return result


if __name__ == "__main__":
    run()
