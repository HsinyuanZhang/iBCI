#!/usr/bin/env python3
"""Local contract validation + reference scoring for the full-block m30 activity-30 M2 image.

Stages (all CPU-only, no network, no submission):

1. ``payload``   - audit the exported payload: exact key set, identity shapes,
                   arm binding, no calibration arrays beyond the identities.
2. ``external``  - replay the LOCAL continual evaluator semantics
                   (reset(dataset_tags) + predict(neural) only, per the sealed
                   audit results/cdm_p1_m2_v1/audit.json) over the six local
                   held-out M2 query sessions and the seven held-in
                   post-30 surfaces; score variance-weighted R2 per session
                   and compare with the sealed static_m30 screen rows
                   (surface: external_official_query / within_post30).
3. ``contract``  - fail-closed assertions: predict never imports the selection
                   law, never reads trial metadata, and mutates no weight or
                   identity across the whole replay; on_done is a no-op that
                   the official evaluator never invokes.
4. ``minival``   - run the real host ``FalconEvaluator`` (local, minival phase,
                   batch 7, the exact evaluator module the sealed audit
                   hashed) end-to-end; record held-in R2 by date + latency.

Writes ``artifacts/local_validation_receipt.json``.
"""

from __future__ import annotations

import argparse
import importlib.abc
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
PAYLOAD_NAME = "t4_m2_seed42_m30_act30_identity.pkl"
DECODER_MODULE = "act30_full_decoder"
DECODER_CLASS = "Act30FullM2Decoder"
PAYLOAD_ARM = "m30_static_act30"
PAYLOAD_LABEL_BUDGET = 30
PAYLOAD_ACTIVITY_BUDGET = 30
SEALED_CELL = "ridge_static_m30"


def bootstrap_runtime_imports() -> None:
    """Pin import roots for host-side stages (dry mode stays import-free).

    ``src`` must resolve to the streaming_calibration_exp tree (the frozen
    checkpoint's module), and is imported first so it stays cached; the
    SPINT-main tree is appended afterwards only for
    ``third_party.falcon_challenge`` (the runtime decoder's smoothing import,
    resolved at ``/`` inside the container).
    """
    streaming_root = ROOT / "streaming_calibration_exp"
    for path in (ROOT, HERE, streaming_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import src.models.streaming_calibration_module  # noqa: F401  - pins `src`

    spint_main = ROOT / "SPINT-main"
    if str(spint_main) not in sys.path:
        sys.path.append(str(spint_main))


SEALED_SCREEN_RELATIVE = "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json"
SEALED_AUDIT_RELATIVE = "tfpd_exploration/results/cdm_p1_m2_v1/audit.json"
EVALUATOR_SHA256 = "2b848f84ef620eac82ce53d331d03727d9ae4ea9a5f2e5edb6e6557537ce4418"

EXPECTED_SESSIONS_EXTERNAL = 6
EXPECTED_SESSIONS_WITHIN = 7
R2_ABS_TOLERANCE = 5.0e-3  # CPU replay vs the sealed GPU screen run


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sealed_rows() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads((ROOT / SEALED_SCREEN_RELATIVE).read_text(encoding="utf-8"))
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        if row["cell"] == SEALED_CELL:
            rows[(row["surface"], row["session"])] = row
    require(len(rows) == 13, "sealed static_m30 row coverage drift")
    return rows


# ---------------------------------------------------------------------------
# Stage 1: payload audit
# ---------------------------------------------------------------------------


def audit_payload(payload_path: Path) -> dict[str, Any]:
    bootstrap_runtime_imports()
    module = __import__(DECODER_MODULE)

    with payload_path.open("rb") as handle:
        payload = module.CPUUnpickler(handle).load()
    allowed = {
        "schema_version",
        "task",
        "decoder",
        "identity_by_dataset_tag",
        "window_size",
        "behavior_scaling_factor",
        "smooth_observations",
        "metadata",
    }
    keys = set(payload)
    require(keys == allowed, f"payload key drift: {sorted(keys ^ allowed)}")
    require(payload["schema_version"] == module.PAYLOAD_SCHEMA, "payload schema drift")
    require(str(payload["task"]) == "FalconTask.m2", "payload task drift")
    require(payload["window_size"] == 50, "payload window drift")
    require(float(payload["behavior_scaling_factor"]) == 5.0, "payload scale drift")
    require(payload["smooth_observations"] is False, "payload smoothing drift")
    metadata = payload["metadata"]
    require(metadata["arm"] == module.PAYLOAD_ARM, "payload arm drift")
    require(
        metadata["label_budget"] == module.PAYLOAD_LABEL_BUDGET
        and metadata["activity_budget"] == module.PAYLOAD_ACTIVITY_BUDGET,
        "budget drift",
    )
    require(metadata["online_backward_pass"] is False, "payload claims online backward pass")
    require(metadata["online_state"] == "cached_E[N,50]", "online state drift")
    identities = payload["identity_by_dataset_tag"]
    require(len(identities) == 13, "identity coverage drift")
    for tag, identity in identities.items():
        array = np.asarray(identity)
        require(array.shape == (96, 50) and array.dtype == np.float32, f"identity shape drift {tag}")
        require(np.isfinite(array).all(), f"identity nonfinite {tag}")
    # Metadata must be pure JSON-serializable build-time provenance: no arrays
    # of calibration data ride along with the runtime.
    json.dumps(metadata, sort_keys=True)
    return {
        "payload_path": str(payload_path),
        "payload_sha256": sha256_file(payload_path),
        "schema_version": payload["schema_version"],
        "arm": metadata["arm"],
        "label_budget": metadata["label_budget"],
        "activity_budget": metadata["activity_budget"],
        "identity_count": len(identities),
        "identity_shapes": sorted({tuple(np.asarray(v).shape) for v in identities.values()}),
        "metadata_is_pure_provenance_json": True,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
    }


# ---------------------------------------------------------------------------
# Stage 2 + 3: evaluator-semantics replay with contract instrumentation
# ---------------------------------------------------------------------------


class ImportBlocker(importlib.abc.MetaPathFinder):
    """Poison the frozen selection/carrier/activity laws so any runtime import fails."""

    BLOCKED = (
        "d_optimal_calibration_design",
        "calibration_budget_comparators_v1",
        "m2_t4_activity_budget_screen_v1",
        "laws",
    )

    def find_module(self, fullname, path=None):  # pragma: no cover - trivial
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        for token in self.BLOCKED:
            if token in fullname:
                raise ValidationError(
                    f"predict-time import of calibration law module blocked: {fullname}"
                )
        return None


def replay_session(
    decoder: Any,
    neural: np.ndarray,
    session: str,
    starts: np.ndarray,
    targets: np.ndarray,
    r2_fn,
) -> dict[str, Any]:
    """Feed one session bin-by-bin (reset + predict only) and score windows."""
    decoder.reset(
        dataset_tags=[
            Path(f"sub-MonkeyN-held-in-calib_{session}_behavior+ecephys.nwb")
        ]
    )
    predictions = np.zeros((int(neural.shape[0]), 2), dtype=np.float32)
    for index in range(int(neural.shape[0])):
        predictions[index] = decoder.predict(neural[index : index + 1])[0]
    last_bin = predictions[starts + 49]
    return {
        "window_count": int(starts.size),
        "r2": float(r2_fn(targets, last_bin)),
    }


def run_replay(payload_path: Path) -> dict[str, Any]:
    bootstrap_runtime_imports()
    import torch

    module = __import__(DECODER_MODULE)
    decoder_class = getattr(module, DECODER_CLASS)
    from falcon_challenge.config import FalconConfig, FalconTask
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        calibration_file_map,
        load_frozen_model_and_data,
    )
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
        variance_weighted_r2,
    )

    torch.set_num_threads(min(8, torch.get_num_threads()))
    task_config = FalconConfig(task=FalconTask.m2)
    decoder = decoder_class(
        task_config=task_config, model_path=str(payload_path), batch_size=1
    )

    model, data_module, _task_config, _metadata = load_frozen_model_and_data()
    session_to_tag = calibration_file_map(ROOT / "SPINT-main/data/000953", task_config)
    sealed_rows = load_sealed_rows()

    # Contract fingerprints before the replay.
    parameter_before = decoder.parameter_fingerprint()
    identity_before = decoder.identity_fingerprint()

    # Poison the calibration-law modules: the runtime must never import them.
    # (All law/scorer imports above resolve before the blocker is armed.)
    blocker = ImportBlocker()
    sys.meta_path.insert(0, blocker)
    law_imports_blocked = list(blocker.BLOCKED)

    surfaces = (
        ("external_official_query", data_module.val_heldout_dataset, EXPECTED_SESSIONS_EXTERNAL),
        ("within_post30", data_module.train_dataset, EXPECTED_SESSIONS_WITHIN),
    )
    replay: dict[str, Any] = {
        "evaluator_semantics": "reset(dataset_tags) once per session, predict(neural) per bin, batch 1",
        "law_imports_blocked_during_replay": law_imports_blocked,
        "surfaces": {},
    }
    try:
        for surface_name, dataset, expected_count in surfaces:
            sessions = sorted(dataset.calib_trialized_neural_features)
            require(len(sessions) == expected_count, f"{surface_name} session count drift")
            per_session: dict[str, dict[str, Any]] = {}
            for session in sessions:
                tag = session_to_tag[session]
                starts = np.asarray(
                    [s for name, s in dataset.window_indices if name == session], dtype=np.int64
                )
                if surface_name == "within_post30":
                    starts = select_common_post30_window_starts(
                        starts, dataset.trial_start_indices[session]
                    )
                else:
                    starts = np.ascontiguousarray(starts)
                require(starts.size > 0, f"{session}: no query windows")
                targets = np.stack(
                    [
                        np.asarray(dataset.covariate_data[session][int(s) + 49], dtype=np.float32)
                        for s in starts
                    ],
                    axis=0,
                )
                neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
                result = replay_session(decoder, neural, session, starts, targets, variance_weighted_r2)
                sealed = sealed_rows[(surface_name, session)]
                delta = float(result["r2"] - float(sealed["r2"]))
                require(
                    abs(delta) <= R2_ABS_TOLERANCE,
                    f"{session} ({surface_name}) r2 drift vs sealed static_m30: "
                    f"{result['r2']!r} vs {sealed['r2']!r} (delta {delta})",
                )
                per_session[session] = {
                    "dataset_tag": tag,
                    "window_count": result["window_count"],
                    "sealed_screen_static_m30_r2": float(sealed["r2"]),
                    "replayed_r2": result["r2"],
                    "delta": delta,
                }
            values = np.asarray([row["replayed_r2"] for row in per_session.values()])
            sealed_values = np.asarray(
                [row["sealed_screen_static_m30_r2"] for row in per_session.values()]
            )
            replay["surfaces"][surface_name] = {
                "per_session": per_session,
                "equal_session_mean": float(values.mean()),
                "equal_session_median": float(np.median(values)),
                "sealed_equal_session_mean": float(sealed_values.mean()),
                "sealed_equal_session_median": float(np.median(sealed_values)),
                "mean_abs_delta": float(np.abs(values - sealed_values).mean()),
                "max_abs_delta": float(np.abs(values - sealed_values).max()),
            }
    finally:
        sys.meta_path.remove(blocker)

    # Contract assertions after the replay.
    parameter_after = decoder.parameter_fingerprint()
    identity_after = decoder.identity_fingerprint()
    require(parameter_after == parameter_before, "decoder weights mutated during predict replay")
    require(identity_after == identity_before, "cached identities mutated during predict replay")
    done_result = decoder.on_done(np.asarray([True]))
    require(done_result is None, "on_done is not a no-op")
    require(
        decoder.parameter_fingerprint() == parameter_before,
        "on_done mutated frozen weights",
    )
    gradients = [p.grad for p in decoder.decoder.parameters() if p.grad is not None]
    require(not gradients, "decoder carries gradient state after replay")
    # The only intended cross-predict state is the frozen decoder weights, the
    # cached identity tensors (selected in reset), and the fixed neural
    # history buffers.  No optimizer/scheduler/training state may exist.
    forbidden = ("optimizer", "optim", "scheduler", "loss", "backward", "trainer", "grad_step")
    matches = [name for name in vars(decoder) if any(token in name.lower() for token in forbidden)]
    require(not matches, f"decoder holds forbidden runtime state: {matches}")
    training_state = [
        name for name, value in vars(decoder).items()
        if hasattr(value, "step") and hasattr(value, "zero_grad")
    ]
    require(not training_state, f"decoder holds optimizer-like objects: {training_state}")

    replay["contract"] = {
        "predict_time_selection_law_imports_blocked": True,
        "weights_unchanged_across_replay": True,
        "identities_unchanged_across_replay": True,
        "parameter_fingerprint_sha256": parameter_before,
        "identity_fingerprint_sha256": identity_before,
        "on_done_is_noop": True,
        "no_gradient_state_after_replay": True,
        "no_optimizer_or_calibration_state_on_decoder": True,
        "trial_metadata_available_at_predict_time": False,
    }
    return replay


# ---------------------------------------------------------------------------
# Stage 4: real host FalconEvaluator minival run
# ---------------------------------------------------------------------------


def run_minival(payload_path: Path, out_dir: Path) -> dict[str, Any]:
    bootstrap_runtime_imports()
    import os

    os.environ["EVAL_DATA_PATH"] = str(ROOT / "SPINT-main/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PREDICTION_PATH_LOCAL"] = str(out_dir / "minival_prediction.pkl")
    os.environ["GT_PATH"] = str(out_dir / "minival_gt.pkl")

    import falcon_challenge.evaluator as evaluator_module
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator

    module = __import__(DECODER_MODULE)
    decoder_class = getattr(module, DECODER_CLASS)
    evaluator_sha = sha256_file(Path(evaluator_module.__file__))
    require(evaluator_sha == EVALUATOR_SHA256, f"evaluator sha drift: {evaluator_sha}")

    task_config = FalconConfig(task=FalconTask.m2)
    decoder = decoder_class(
        task_config=task_config, model_path=str(payload_path), batch_size=7
    )

    calls = {"on_done": 0, "reset": 0, "predict": 0}
    decoder_reset = decoder.reset
    decoder_predict = decoder.predict
    decoder_on_done = decoder.on_done

    def counting_reset(*args, **kwargs):
        calls["reset"] += 1
        return decoder_reset(*args, **kwargs)

    def counting_predict(*args, **kwargs):
        calls["predict"] += 1
        return decoder_predict(*args, **kwargs)

    def counting_on_done(*args, **kwargs):
        calls["on_done"] += 1
        return decoder_on_done(*args, **kwargs)

    decoder.reset = counting_reset  # type: ignore[method-assign]
    decoder.predict = counting_predict  # type: ignore[method-assign]
    decoder.on_done = counting_on_done  # type: ignore[method-assign]

    evaluator = FalconEvaluator(eval_remote=False, split="m2")
    started = time.monotonic()
    result = evaluator.evaluate(decoder, phase="minival")
    elapsed = time.monotonic() - started
    require(calls["on_done"] == 0, "official local evaluator invoked on_done on a continual task")
    require(calls["reset"] > 0 and calls["predict"] > 0, "evaluator never drove the decoder")
    prediction_path = Path(os.environ["PREDICTION_PATH_LOCAL"])
    gt_path = Path(os.environ["GT_PATH"])
    return {
        "phase": "minival",
        "batch_size": 7,
        "evaluator_module": str(evaluator_module.__file__),
        "evaluator_sha256": evaluator_sha,
        "evaluator_semantics_proof": {
            "reset_calls": calls["reset"],
            "predict_calls": calls["predict"],
            "on_done_calls": calls["on_done"],
            "on_done_never_called_by_official_loop": calls["on_done"] == 0,
        },
        "wall_seconds": elapsed,
        "metrics": result,
        "prediction_pkl": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "gt_pkl": str(gt_path),
        "gt_sha256": sha256_file(gt_path),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stages",
        default="payload,external,contract,minival",
        help="comma-separated subset of payload,external,contract,minival",
    )
    parser.add_argument("--payload", type=Path, default=ARTIFACTS / PAYLOAD_NAME)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "local_validation_receipt.json")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "schema": "m2_m30_act30_local_validation_dry_v1",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE",
            "stages": args.stages.split(","),
            "payload": str(args.payload),
            "r2_abs_tolerance_vs_sealed_screen": R2_ABS_TOLERANCE,
        }, sort_keys=True))
        return

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    require(args.payload.exists(), f"payload missing: {args.payload}")
    receipt: dict[str, Any] = {
        "schema_version": "m2_m30_act30_local_validation_v1",
        "status": "LOCAL_VALIDATION_RUNNING",
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "checkpoint_sha256_expectation": "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e",
        "sealed_screen_result": SEALED_SCREEN_RELATIVE,
        "sealed_screen_cell": SEALED_CELL,
        "sealed_audit_contract": SEALED_AUDIT_RELATIVE,
    }
    if "payload" in stages:
        receipt["payload_audit"] = audit_payload(args.payload)
    if "external" in stages or "contract" in stages:
        replay = run_replay(args.payload)
        if "external" in stages:
            receipt["evaluator_semantics_replay"] = replay
        if "contract" in stages:
            receipt["contract"] = replay["contract"]
    if "minival" in stages:
        receipt["official_local_minival"] = run_minival(
            args.payload, ARTIFACTS / "minival_v1"
        )
    receipt["status"] = "LOCAL_VALIDATION_COMPLETE"
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "evaluator_semantics_replay"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
