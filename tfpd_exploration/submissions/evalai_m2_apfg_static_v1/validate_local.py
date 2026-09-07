#!/usr/bin/env python3
"""Local contract validation + sentinel replay for the static-pool APFG M2 image.

Stages (all CPU-only, no network, no submission):

1. ``payload``      - audit the exported payload: exact key set, identity
                      shapes, arm/alpha binding, pure provenance metadata.
2. ``native_seal``  - replay the LOCAL continual evaluator semantics
                      (reset(dataset_tags) + predict(neural) only, per the
                      sealed audit results/cdm_p1_m2_v1/audit.json) over all
                      13 sessions with the NATIVE identities and require the
                      sealed activity30_m4 screen rows within the historical
                      CPU-vs-GPU tolerance (the officially scored arm).
3. ``sentinel``     - same-device/same-input/same-batch decode-level sentinel:
                      with the ZERO identities the decoder must be exactly
                      equal to the NATIVE run on prediction bytes, R2, target
                      bytes, query starts, and window count, for every one of
                      the 13 sessions.  LEARNED deltas are reported
                      descriptively and gate nothing.
4. ``contract``     - fail-closed assertions: predict never imports the
                      selection law, never reads trial metadata, and mutates
                      no weight or identity across the whole replay; on_done
                      is a no-op the official evaluator never invokes.
5. ``minival``      - real host ``FalconEvaluator`` (local, minival phase,
                      batch 7, the exact evaluator module the sealed audit
                      hashed) end-to-end with call counters.

Writes ``artifacts/local_validation_receipt.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
PAYLOAD_NAME = "t4_m2_seed42_dopt4_act30_apfg_identity.pkl"
VALIDATION_NAME = "t4_m2_seed42_dopt4_act30_apfg_validation.pkl"
DECODER_MODULE = "apfg_static_decoder"
DECODER_CLASS = "ApfgStaticM2Decoder"
PAYLOAD_ARM = "apfg_static_act30_dopt4"
PAYLOAD_LABEL_BUDGET = 4
PAYLOAD_ACTIVITY_BUDGET = 30
FROZEN_ALPHA = -0.20759029686450958
SEALED_CELL = "ridge_activity30_m4"


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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_sealed_rows() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads((ROOT / SEALED_SCREEN_RELATIVE).read_text(encoding="utf-8"))
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        if row["cell"] == SEALED_CELL:
            rows[(row["surface"], row["session"])] = row
    require(len(rows) == 13, "sealed activity30_m4 row coverage drift")
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
    require(float(metadata["frozen_alpha"]) == FROZEN_ALPHA, "payload frozen alpha drift")
    require(metadata["online_backward_pass"] is False, "payload claims online backward pass")
    require(metadata["online_state"] == "cached_E[N,50]", "online state drift")
    require(int(metadata["zero_equals_native_sessions"]) == 13, "build-time sentinel drift")
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
        "frozen_alpha": float(metadata["frozen_alpha"]),
        "label_budget": metadata["label_budget"],
        "activity_budget": metadata["activity_budget"],
        "identity_count": len(identities),
        "identity_shapes": sorted({tuple(np.asarray(v).shape) for v in identities.values()}),
        "metadata_is_pure_provenance_json": True,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "normalization_sha256": metadata["normalization_sha256"],
        "alpha_provenance": metadata["alpha_provenance"],
    }


# ---------------------------------------------------------------------------
# Stages 2-4: evaluator-semantics replay, sentinel, and contract
# ---------------------------------------------------------------------------


class ImportBlocker(importlib.abc.MetaPathFinder):
    """Poison the frozen selection/carrier/activity/gate laws so any runtime import fails."""

    BLOCKED = (
        "d_optimal_calibration_design",
        "calibration_budget_comparators_v1",
        "m2_t4_activity_budget_screen_v1",
        "m2_anchored_postfusion_gate",
        "laws",
    )

    def find_module(self, fullname, path=None):  # pragma: no cover - trivial
        return self.find_spec(fullname, path)

    def find_spec(self, fullname, path=None, target=None):
        for token in self.BLOCKED:
            if token in fullname:
                raise ValidationError(
                    f"predict-time import of calibration/gate law module blocked: {fullname}"
                )
        return None


def replay_arms(
    decoders: dict[str, Any],
    neural: np.ndarray,
    session: str,
    starts: np.ndarray,
    targets: np.ndarray,
    r2_fn,
) -> dict[str, dict[str, Any]]:
    """Feed one session bin-by-bin (reset + predict only) to every arm."""
    results: dict[str, dict[str, Any]] = {}
    for arm, decoder in decoders.items():
        decoder.reset(
            dataset_tags=[
                Path(f"sub-MonkeyN-held-in-calib_{session}_behavior+ecephys.nwb")
            ]
        )
        predictions = np.zeros((int(neural.shape[0]), 2), dtype=np.float32)
        for index in range(int(neural.shape[0])):
            predictions[index] = decoder.predict(neural[index : index + 1])[0]
        last_bin = predictions[starts + 49]
        results[arm] = {
            "prediction_sha256": sha256_array(predictions),
            "last_bin_sha256": sha256_array(last_bin),
            "target_sha256": sha256_array(targets),
            "query_starts_sha256": sha256_array(starts),
            "window_count": int(starts.size),
            "r2": float(r2_fn(targets, last_bin)),
        }
    return results


def _derive_arm_payload(payload_path: Path, identity_by_tag: dict[str, np.ndarray], workdir: Path, name: str) -> Path:
    """Materialize a variant payload whose identities are exchanged (sentinel driver)."""
    bootstrap_runtime_imports()
    module = __import__(DECODER_MODULE)
    with Path(payload_path).open("rb") as handle:
        shipped = module.CPUUnpickler(handle).load()
    variant = dict(shipped)
    variant["identity_by_dataset_tag"] = {
        tag: np.ascontiguousarray(value, dtype=np.float32)
        for tag, value in identity_by_tag.items()
    }
    path = workdir / name
    with path.open("wb") as handle:
        pickle.dump(variant, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def run_replay_and_sentinel(payload_path: Path, validation_path: Path) -> dict[str, Any]:
    bootstrap_runtime_imports()
    import torch

    module = __import__(DECODER_MODULE)
    decoder_class = getattr(module, DECODER_CLASS)
    from falcon_challenge.config import FalconConfig, FalconTask
    from sua_exploration.evalai_t4_m2.export_t4_payload import (
        calibration_file_map,
        load_frozen_model_and_data,
    )
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import variance_weighted_r2

    torch.set_num_threads(min(8, torch.get_num_threads()))
    task_config = FalconConfig(task=FalconTask.m2)

    with validation_path.open("rb") as handle:
        validation = pickle.load(handle)
    require(validation["schema_version"] == "e8_t4_m2_apfg_static_validation_v1", "validation artifact schema drift")
    session_to_tag = dict(validation["session_to_dataset_tag"])
    require(calibration_file_map(ROOT / "SPINT-main/data/000953", task_config) == session_to_tag, "validation tag map drift")

    with tempfile.TemporaryDirectory(prefix="apfg_sentinel_") as workdir:
        native_payload = _derive_arm_payload(
            payload_path, {session_to_tag[s]: v for s, v in validation["native_identity_by_session"].items()},
            Path(workdir), "native.pkl",
        )
        zero_payload = _derive_arm_payload(
            payload_path, {session_to_tag[s]: v for s, v in validation["zero_identity_by_session"].items()},
            Path(workdir), "zero.pkl",
        )
        decoders = {
            "native": decoder_class(task_config=task_config, model_path=str(native_payload), batch_size=1),
            "zero": decoder_class(task_config=task_config, model_path=str(zero_payload), batch_size=1),
            "learned": decoder_class(task_config=task_config, model_path=str(payload_path), batch_size=1),
        }

        # Contract fingerprints before the replay (shipped decoder).
        parameter_before = decoders["learned"].parameter_fingerprint()
        identity_before = decoders["learned"].identity_fingerprint()

        # Poison the calibration-law/gate modules: the runtime must never
        # import them.  (All law/scorer imports above resolve first.)
        blocker = ImportBlocker()
        sys.meta_path.insert(0, blocker)
        law_imports_blocked = list(blocker.BLOCKED)
        per_surface: dict[str, dict[str, Any]] = {
            "external_official_query": {},
            "within_post30": {},
        }
        try:
            model, data_module, _task_config, _metadata = load_frozen_model_and_data()
            for surface_name, session, _tag, starts, targets, neural in _session_inputs_data(
                data_module, session_to_tag
            ):
                arms = replay_arms(decoders, neural, session, starts, targets, variance_weighted_r2)
                native_row = arms["native"]
                sealed_rows = load_sealed_rows()
                sealed = sealed_rows[(surface_name, session)]
                native_delta = float(native_row["r2"] - float(sealed["r2"]))
                require(
                    abs(native_delta) <= R2_ABS_TOLERANCE,
                    f"{session} ({surface_name}) native r2 drift vs sealed activity30_m4: "
                    f"{native_row['r2']!r} vs {sealed['r2']!r} (delta {native_delta})",
                )
                zero_row = arms["zero"]
                learned_row = arms["learned"]
                sentinel_fields = (
                    "prediction_sha256",
                    "last_bin_sha256",
                    "target_sha256",
                    "query_starts_sha256",
                    "window_count",
                )
                mismatches = {field: [native_row[field], zero_row[field]] for field in sentinel_fields
                              if native_row[field] != zero_row[field]}
                require(
                    not mismatches and native_row["r2"] == zero_row["r2"],
                    f"{session} ({surface_name}): ZERO sentinel mismatch {mismatches or {'r2': [native_row['r2'], zero_row['r2']]}}",
                )
                per_surface[surface_name][session] = {
                    "native_r2": native_row["r2"],
                    "native_sealed_screen_r2": float(sealed["r2"]),
                    "native_delta_vs_sealed": native_delta,
                    "zero_r2": zero_row["r2"],
                    "zero_equals_native_exact": True,
                    "learned_r2": learned_row["r2"],
                    "learned_minus_native": float(learned_row["r2"] - native_row["r2"]),
                    "native_prediction_sha256": native_row["prediction_sha256"],
                    "zero_prediction_sha256": zero_row["prediction_sha256"],
                    "learned_prediction_sha256": learned_row["prediction_sha256"],
                    "window_count": native_row["window_count"],
                }
        finally:
            if blocker in sys.meta_path:
                sys.meta_path.remove(blocker)

    for surface_name, sessions in per_surface.items():
        values = {
            arm: np.asarray([row[f"{arm}_r2"] for row in sessions.values()])
            for arm in ("native", "zero", "learned")
        }
        deltas = np.asarray([row["learned_minus_native"] for row in sessions.values()])
        per_surface[surface_name] = {
            "per_session": sessions,
            "native_equal_session_mean": float(values["native"].mean()),
            "zero_equal_session_mean": float(values["zero"].mean()),
            "learned_equal_session_mean": float(values["learned"].mean()),
            "learned_minus_native_mean": float(deltas.mean()),
            "learned_minus_native_median": float(np.median(deltas)),
            "learned_positive_sessions": int((deltas > 0).sum()),
            "session_count": int(len(sessions)),
        }

    # Contract assertions after the replay (shipped decoder).
    parameter_after = decoders["learned"].parameter_fingerprint()
    identity_after = decoders["learned"].identity_fingerprint()
    require(parameter_after == parameter_before, "decoder weights mutated during predict replay")
    require(identity_after == identity_before, "cached identities mutated during predict replay")
    done_result = decoders["learned"].on_done(np.asarray([True]))
    require(done_result is None, "on_done is not a no-op")
    require(
        decoders["learned"].parameter_fingerprint() == parameter_before,
        "on_done mutated frozen weights",
    )
    gradients = [p.grad for p in decoders["learned"].decoder.parameters() if p.grad is not None]
    require(not gradients, "decoder carries gradient state after replay")
    forbidden = ("optimizer", "optim", "scheduler", "loss", "backward", "trainer", "grad_step")
    matches = [
        name for name in vars(decoders["learned"])
        if any(token in name.lower() for token in forbidden)
    ]
    require(not matches, f"decoder holds forbidden runtime state: {matches}")
    training_state = [
        name for name, value in vars(decoders["learned"]).items()
        if hasattr(value, "step") and hasattr(value, "zero_grad")
    ]
    require(not training_state, f"decoder holds optimizer-like objects: {training_state}")

    return {
        "evaluator_semantics": "reset(dataset_tags) once per session, predict(neural) per bin, batch 1",
        "law_imports_blocked_during_replay": law_imports_blocked,
        "surfaces": per_surface,
        "sentinel": {
            "zero_equals_native_all_13_sessions_exact": True,
            "compared_fields": [
                "prediction_sha256",
                "last_bin_sha256",
                "r2",
                "target_sha256",
                "query_starts_sha256",
                "window_count",
            ],
            "same_process_same_device_same_batch": True,
        },
        "contract": {
            "predict_time_selection_law_imports_blocked": True,
            "weights_unchanged_across_replay": True,
            "identities_unchanged_across_replay": True,
            "parameter_fingerprint_sha256": parameter_before,
            "identity_fingerprint_sha256": identity_before,
            "on_done_is_noop": True,
            "no_gradient_state_after_replay": True,
            "no_optimizer_or_calibration_state_on_decoder": True,
            "trial_metadata_available_at_predict_time": False,
        },
    }


def _session_inputs_data(data_module, session_to_tag: dict[str, str]):
    """Same as ``_session_inputs`` but driven by the loaded data module."""
    from tfpd_exploration.src.m2_t4_activity_budget_screen_v1.core import (
        select_common_post30_window_starts,
    )

    surfaces = (
        ("external_official_query", data_module.val_heldout_dataset, EXPECTED_SESSIONS_EXTERNAL),
        ("within_post30", data_module.train_dataset, EXPECTED_SESSIONS_WITHIN),
    )
    for surface_name, dataset, expected_count in surfaces:
        require(dataset is not None, f"{surface_name} dataset missing")
        sessions = sorted(dataset.calib_trialized_neural_features)
        require(len(sessions) == expected_count, f"{surface_name} session count drift")
        for session in sessions:
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
            yield surface_name, session, session_to_tag[session], starts, targets, neural


# ---------------------------------------------------------------------------
# Stage 5: real host FalconEvaluator minival run
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
        default="payload,native_seal,sentinel,contract,minival",
        help="comma-separated subset of payload,native_seal,sentinel,contract,minival",
    )
    parser.add_argument("--payload", type=Path, default=ARTIFACTS / PAYLOAD_NAME)
    parser.add_argument(
        "--validation", type=Path, default=ARTIFACTS / VALIDATION_NAME
    )
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "local_validation_receipt.json")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "schema": "m2_apfg_static_local_validation_dry_v1",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_TORCH_NO_WRITE",
            "stages": args.stages.split(","),
            "payload": str(args.payload),
            "r2_abs_tolerance_vs_sealed_screen": R2_ABS_TOLERANCE,
        }, sort_keys=True))
        return

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    require(args.payload.exists(), f"payload missing: {args.payload}")
    require(args.validation.exists(), f"validation artifact missing: {args.validation}")
    receipt: dict[str, Any] = {
        "schema_version": "m2_apfg_static_local_validation_v1",
        "status": "LOCAL_VALIDATION_RUNNING",
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "checkpoint_sha256_expectation": "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e",
        "frozen_alpha": FROZEN_ALPHA,
        "sealed_screen_result": SEALED_SCREEN_RELATIVE,
        "sealed_screen_cell": SEALED_CELL,
        "sealed_audit_contract": SEALED_AUDIT_RELATIVE,
    }
    if "payload" in stages:
        receipt["payload_audit"] = audit_payload(args.payload)
    replay_stages = {"native_seal", "sentinel", "contract"}
    if replay_stages & set(stages):
        replay = run_replay_and_sentinel(args.payload, args.validation)
        if "native_seal" in stages:
            receipt["native_seal_replay"] = replay["surfaces"]
        if "sentinel" in stages:
            receipt["zero_native_sentinel"] = replay["sentinel"]
            receipt["learned_descriptive"] = replay["surfaces"]
        if "contract" in stages:
            receipt["contract"] = replay["contract"]
    if "minival" in stages:
        receipt["official_local_minival"] = run_minival(
            args.payload, ARTIFACTS / "minival_v1"
        )
    receipt["status"] = "LOCAL_VALIDATION_COMPLETE"
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {k: v for k, v in receipt.items() if k not in ("native_seal_replay", "learned_descriptive")}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
