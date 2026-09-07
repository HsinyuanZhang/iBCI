#!/usr/bin/env python3
"""Local-only AOF-S payload and authority validation.

This command deliberately has no EvalAI client or network stage.  ``--audit``
is data-free: it validates AOF-M's held graph and, when given a built payload,
its immutable schema/identity maps.  A later operator-reviewed local minival
may use the evaluator entry point, but cannot change beta or the payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for item in (HERE.parents[2], HERE):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

try:  # route-owned package execution
    from .laws import FROZEN_BETA, sha256_array, validate_aofm_authority, validate_identity_map  # noqa: E402
except ImportError:  # copied flat into the local Docker context
    from laws import FROZEN_BETA, sha256_array, validate_aofm_authority, validate_identity_map  # type: ignore[no-redef] # noqa: E402


def audit_payload(path: Path) -> dict:
    # The serialized decoder refers to ``src.models`` from
    # ``streaming_calibration_exp``.  Establish that namespace before the
    # unpickler imports the decoder, including in a fresh host process.
    from tfpd_exploration.src.pit_m2_v1.trainer import ensure_streaming_paths
    ensure_streaming_paths(HERE.parents[2])
    from aofs_static_decoder import CPUUnpickler, PAYLOAD_ARM, PAYLOAD_SCHEMA

    with path.open("rb") as handle:
        payload = CPUUnpickler(handle).load()
    allowed = {
        "schema_version", "task", "decoder", "native_identity_by_dataset_tag",
        "post_identity_by_dataset_tag", "window_size", "behavior_scaling_factor",
        "smooth_observations", "metadata",
    }
    if set(payload) != allowed or payload.get("schema_version") != PAYLOAD_SCHEMA:
        raise ValueError("AOF-S payload schema/key drift")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("arm") != PAYLOAD_ARM:
        raise ValueError("AOF-S payload arm drift")
    if float(metadata.get("frozen_beta", "nan")) != FROZEN_BETA:
        raise ValueError("AOF-S payload beta drift")
    native = validate_identity_map(payload["native_identity_by_dataset_tag"])
    post = validate_identity_map(payload["post_identity_by_dataset_tag"])
    if set(native) != set(post):
        raise ValueError("AOF-S native/post identity tag drift")
    return {
        "payload": str(path),
        "native_identity_digests": native,
        "post_identity_digests": post,
        "frozen_beta": FROZEN_BETA,
        "network_submission": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_same_window_bridge(payload_path: Path) -> dict:
    """Exercise the actual seven-source-session ``+0`` native contract.

    This is deliberately a *two decoder / same raw-bin stream* check.  The
    c51 decoder is the immutable historical native object; the AOF-S decoder
    is reset independently but receives byte-identical source observations.
    ``+0`` must make only the AOF-S native B-call and its governed prediction
    array must be bitwise equal to c51 on every post-30 window.  The learned
    call is separately executed on those same windows to prove the production
    two-call fusion path is finite without using it for any selection.

    It is source-only: the train data module is the local seven-session
    authority.  No external/EvalAI target, network, optimizer, or GPU path is
    reachable here.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("AOF-S source bridge requires CUDA_VISIBLE_DEVICES='' (CPU only)")
    from tfpd_exploration.src.pit_m2_v1.trainer import ensure_streaming_paths
    ensure_streaming_paths(HERE.parents[2])
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("AOF-S source bridge found CUDA already initialized")
    from falcon_challenge.config import FalconConfig, FalconTask
    from sua_exploration.evalai_t4_m2.export_t4_payload import calibration_file_map, load_frozen_model_and_data
    from sua_exploration.evalai_t4_m2.t4_spint_decoder import T4CachedIdentityDecoder
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
        variance_weighted_r2,
    )
    from .aofs_static_decoder import AofsStaticM2Decoder
    from .export_aofs_static_payload import GOVERNING_PAYLOAD_RELATIVE

    repo_root = HERE.parents[2]
    task = FalconConfig(task=FalconTask.m2)
    historical = T4CachedIdentityDecoder(
        task_config=task, model_path=str(repo_root / GOVERNING_PAYLOAD_RELATIVE), batch_size=1,
    )
    zero = AofsStaticM2Decoder(task_config=task, model_path=str(payload_path), batch_size=1)
    learned = AofsStaticM2Decoder(task_config=task, model_path=str(payload_path), batch_size=1)
    # Count B calls rather than inferring them from an output equality.  The
    # zero decoder must never reach its post identity branch.
    calls = {"zero_native": 0, "zero_post": 0, "learned_native": 0, "learned_post": 0}
    zero_decode = zero._decode_with_identity
    learned_decode = learned._decode_with_identity

    def counted_zero(neural, identity):
        if bool(torch.equal(identity, zero.local_native[0])):
            calls["zero_native"] += 1
        else:
            calls["zero_post"] += 1
        return zero_decode(neural, identity)

    def counted_learned(neural, identity):
        if bool(torch.equal(identity, learned.local_native[0])):
            calls["learned_native"] += 1
        else:
            calls["learned_post"] += 1
        return learned_decode(neural, identity)

    zero._decode_with_identity = counted_zero
    learned._decode_with_identity = counted_learned
    _model, data_module, task_config, _metadata = load_frozen_model_and_data()
    dataset = data_module.train_dataset
    if dataset is None:
        raise ValueError("AOF-S source bridge lacks the prepared source dataset")
    session_to_tag = calibration_file_map(repo_root / "SPINT-main/data/000953", task_config)
    sessions = tuple(sorted(dataset.calib_trialized_neural_features))
    if len(sessions) != 7:
        raise ValueError(f"AOF-S source bridge roster drift: {len(sessions)}")
    rows: dict[str, dict] = {}
    for session in sessions:
        tag = session_to_tag.get(session)
        if not isinstance(tag, str):
            raise ValueError(f"AOF-S source bridge missing dataset tag: {session}")
        dataset_tag = Path(f"sub-MonkeyN-held-in-calib_{session}_behavior+ecephys.nwb")
        historical.reset(dataset_tags=[dataset_tag])
        zero.reset(dataset_tags=[dataset_tag])
        learned.reset(dataset_tags=[dataset_tag])
        neural = np.ascontiguousarray(dataset.neural_data[session], dtype=np.float32)
        starts = np.asarray([start for name, start in dataset.window_indices if name == session], dtype=np.int64)
        starts = np.ascontiguousarray(
            select_common_post30_window_starts(starts, dataset.trial_start_indices[session]), dtype=np.int64,
        )
        if not starts.size:
            raise ValueError(f"AOF-S source bridge has no post30 windows: {session}")
        historical_predictions = np.empty((int(neural.shape[0]), 2), dtype=np.float32)
        zero_predictions = np.empty_like(historical_predictions)
        learned_predictions = np.empty_like(historical_predictions)
        for index in range(int(neural.shape[0])):
            observation = neural[index:index + 1]
            historical_predictions[index] = historical.predict(observation)[0]
            zero_predictions[index] = zero.predict_with_beta_for_validation(observation, 0.0)[0]
            learned_predictions[index] = learned.predict(observation)[0]
        target = np.ascontiguousarray(
            np.stack([np.asarray(dataset.covariate_data[session][int(start) + 49], dtype=np.float32) for start in starts]),
            dtype=np.float32,
        )
        native = np.ascontiguousarray(historical_predictions[starts + 49], dtype=np.float32)
        zero_prediction = np.ascontiguousarray(zero_predictions[starts + 49], dtype=np.float32)
        learned_prediction = np.ascontiguousarray(learned_predictions[starts + 49], dtype=np.float32)
        if not np.array_equal(native, zero_prediction):
            raise ValueError(f"AOF-S source +0 direct-native parity drift: {session}")
        if not (np.isfinite(learned_prediction).all() and np.isfinite(target).all()):
            raise ValueError(f"AOF-S source learned prediction/target nonfinite: {session}")
        rows[session] = {
            "dataset_tag": tag,
            "window_count": int(starts.size),
            "ordered_window_starts_sha256": sha256_array(starts),
            "target_sha256": sha256_array(target),
            "native_prediction_sha256": sha256_array(native),
            "zero_prediction_sha256": sha256_array(zero_prediction),
            "learned_prediction_sha256": sha256_array(learned_prediction),
            "native_zero_bitwise_equal": True,
            "native_r2": float(variance_weighted_r2(target, native)),
            "zero_r2": float(variance_weighted_r2(target, zero_prediction)),
            "learned_r2": float(variance_weighted_r2(target, learned_prediction)),
        }
    if calls["zero_native"] <= 0 or calls["zero_post"] != 0:
        raise ValueError("AOF-S direct +0 branch did not remain native-only")
    if calls["learned_native"] <= 0 or calls["learned_post"] <= 0:
        raise ValueError("AOF-S learned branch did not make independent native/post calls")
    if torch.cuda.is_initialized():
        raise ValueError("AOF-S source bridge initialized CUDA")
    return {
        "source_surface": "within_post30_local_source_only",
        "source_target_access": True,
        "hidden_external_evalai_target_access": False,
        "roster": list(sessions),
        "per_session": rows,
        "same_window_two_decoder_calls": calls,
        "zero_direct_native_all_sessions_bitwise": True,
        "cuda_initialized": False,
    }


def _run_local_minival(payload_path: Path, artifact_root: Path) -> dict:
    """Drive the real host local evaluator, CPU-only and without a network path."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("AOF-S local validation requires CUDA_VISIBLE_DEVICES='' (CPU only)")
    from tfpd_exploration.src.pit_m2_v1.trainer import ensure_streaming_paths
    ensure_streaming_paths(HERE.parents[2])
    import torch
    _ = torch  # explicit import after CVD admission; CUDA must remain untouched
    if torch.cuda.is_initialized():
        raise ValueError("AOF-S CPU local validation found CUDA already initialized")
    import falcon_challenge.evaluator as evaluator_module
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    from aofs_static_decoder import AofsStaticM2Decoder

    os.environ["EVAL_DATA_PATH"] = str(HERE.parents[2] / "SPINT-main/data")
    output_dir = artifact_root / "minival"
    output_dir.mkdir(exist_ok=False)
    prediction_path = output_dir / "prediction.pkl"
    target_path = output_dir / "ground_truth.pkl"
    os.environ["PREDICTION_PATH_LOCAL"] = str(prediction_path)
    os.environ["GT_PATH"] = str(target_path)
    decoder = AofsStaticM2Decoder(FalconConfig(task=FalconTask.m2), str(payload_path), batch_size=7)
    calls = {"reset": 0, "predict": 0, "on_done": 0}
    reset, predict, done = decoder.reset, decoder.predict, decoder.on_done

    def count_reset(*args, **kwargs):
        calls["reset"] += 1
        return reset(*args, **kwargs)

    def count_predict(*args, **kwargs):
        calls["predict"] += 1
        return predict(*args, **kwargs)

    def count_done(*args, **kwargs):
        calls["on_done"] += 1
        return done(*args, **kwargs)

    decoder.reset, decoder.predict, decoder.on_done = count_reset, count_predict, count_done
    started = time.monotonic()
    metrics = FalconEvaluator(eval_remote=False, split="m2").evaluate(decoder, phase="minival")
    elapsed = time.monotonic() - started
    if calls["reset"] <= 0 or calls["predict"] <= 0 or calls["on_done"] != 0:
        raise ValueError("AOF-S local evaluator semantic drift")
    if not prediction_path.is_file() or not target_path.is_file():
        raise ValueError("AOF-S local evaluator did not publish local artifacts")
    if torch.cuda.is_initialized():
        raise ValueError("AOF-S CPU local minival initialized CUDA")
    return {
        "phase": "minival",
        "cpu_only": True,
        "cuda_initialized": False,
        "evaluator_module": str(evaluator_module.__file__),
        "evaluator_sha256": _sha256_file(Path(evaluator_module.__file__)),
        "calls": calls,
        "wall_seconds": elapsed,
        "metrics": metrics,
        "prediction_sha256": _sha256_file(prediction_path),
        "target_sha256": _sha256_file(target_path),
        "network": False,
    }


def run_production_validation(payload_path: Path, artifact_root: Path) -> dict:
    """Route-owned audit + real CPU minival; no selection/refit/update path."""
    audit = audit_payload(payload_path)
    source_bridge = _source_same_window_bridge(payload_path)
    minival = _run_local_minival(payload_path, artifact_root)
    return {
        "schema": "m2_aof_scalar_static_local_validation_v1",
        "status": "LOCAL_VALIDATION_COMPLETE_NOT_SUBMITTED",
        "payload_audit": audit,
        "source_same_window_bridge": source_bridge,
        "official_local_minival": minival,
        "selection_or_refit": False,
        "network_submission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AOF-S local-only validation; no network action")
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit:
        print(json.dumps({"status": "INERT", "requires": "--audit", "network": False}, sort_keys=True))
        return
    result = {"aofm_authority": validate_aofm_authority()}
    if args.payload is not None:
        result["payload"] = audit_payload(args.payload)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
