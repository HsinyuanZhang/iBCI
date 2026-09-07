#!/usr/bin/env python3
"""Forward-only authority for the M1 compact-B3S fold0/seed42 diagnostic.

This evaluator opens only the four native ``held-in-calib`` M1 files already
bound by the Version-B source-LOSO artifacts.  It never constructs a Trainer
or optimizer and evaluates the fixed epoch-11 B0 and B3S-Zero4 checkpoints on
one shared ordered query loader.  The resulting receipt distinguishes analytic
identity-encoder MACs from observed CPU wall time; no profiler-derived hardware
latency claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
STREAM_ROOT = ROOT / "streaming_calibration_exp"
for entry in (str(ROOT), str(STREAM_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from sua_exploration.scripts import aggregate_m1_version_b_pilot as pilot  # noqa: E402
from src.models.components.streaming_encoders import (  # noqa: E402
    BatchReferenceEncoder,
    SideFeatureEarlyPoolEncoder,
)


SCHEMA = "m1_compact_b3s_fold0_forward_authority_v1"
STATUS = "PASS_M1_COMPACT_B3S_FOLD0_FORWARD_ONLY_AUTHORITY"
TARGET_SESSION = "ses-20120924"
EXPECTED_EPOCH = 11
EXPECTED_SEED = 42
EXPECTED_QUERY_SAMPLES = 26_496
PARITY_ATOL = 1.0e-6

DEFAULT_B0_RUN = ROOT / (
    "outputs/streaming_calibration/"
    "m1_version_b_hs_continuation_f0_s42_f0_s42_20260809_231214"
)
DEFAULT_B3S_RUN = ROOT / (
    "outputs/streaming_calibration/"
    "m1_version_b_c0_f0_s42_f0_s42_20260809_222207"
)
DEFAULT_PREFLIGHT = ROOT / "sua_exploration/results/m1_version_b_preflight/receipt_v2.json"
DEFAULT_AGGREGATE = ROOT / (
    "sua_exploration/results/m1_version_b_pilot_aggregate_remote_f0_s42_v3/"
    "M1_VERSION_B_F0_S42_AGGREGATE_v1.json"
)
DEFAULT_OUTPUT = ROOT / (
    "sua_exploration/m1_compact_replication/results/"
    "M1_COMPACT_B3S_F0_S42_FORWARD_AUTHORITY_v1.json"
)


class ForwardAuthorityError(ValueError):
    """The fixed forward-only authority contract was not satisfied."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ForwardAuthorityError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ForwardAuthorityError(f"{label} is unreadable: {path}") from error
    _need(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _load_config(path: Path) -> Any:
    _need(path.is_file() and not path.is_symlink(), f"resolved config is missing or symlinked: {path}")
    config = OmegaConf.load(path)
    # Exported configs retain Hydra runtime resolvers for logger/trainer paths.
    # This authority instantiates neither logger nor Trainer, so bind those two
    # runtime-only leaves explicitly before resolving the model/data subtrees.
    config.paths.output_dir = str(path.parent.resolve())
    config.paths.work_dir = str(ROOT.resolve())
    OmegaConf.resolve(config)
    return config


def ordered_sampler_receipt(sampler: Any, dataset: Any, *, label: str) -> dict[str, Any]:
    """Hash actual frozen batches and their ordered semantic query identities."""

    batches = getattr(sampler, "batched_indices", None)
    windows = getattr(dataset, "window_indices", None)
    _need(isinstance(batches, Sequence) and batches, f"{label} sampler exposes no frozen batches")
    _need(isinstance(windows, Sequence) and windows, f"{label} dataset exposes no window identities")
    index_digest = hashlib.sha256()
    identity_digest = hashlib.sha256()
    observed: list[int] = []
    first_identity: list[Any] | None = None
    last_identity: list[Any] | None = None
    for batch_number, batch in enumerate(batches):
        indices = np.asarray(batch, dtype=np.int64).reshape(-1)
        _need(indices.size > 0, f"{label} batch {batch_number} is empty")
        index_digest.update(np.asarray([batch_number, indices.size], dtype="<i8").tobytes())
        index_digest.update(indices.astype("<i8", copy=False).tobytes())
        for position, raw_index in enumerate(indices.tolist()):
            index = int(raw_index)
            _need(0 <= index < len(windows), f"{label} sampler index out of bounds: {index}")
            identity = windows[index]
            _need(
                isinstance(identity, Sequence) and len(identity) >= 2,
                f"{label} window identity has unexpected shape: {identity!r}",
            )
            row = [batch_number, position, index, str(identity[0]), int(identity[1])]
            identity_digest.update(
                json.dumps(row, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
            )
            first_identity = row if first_identity is None else first_identity
            last_identity = row
            observed.append(index)
    _need(len(observed) == len(set(observed)), f"{label} sampler repeats dataset indices")
    return {
        "batches": len(batches),
        "samples": len(observed),
        "ordered_sampler_indices_sha256": index_digest.hexdigest(),
        "ordered_window_identity_sha256": identity_digest.hexdigest(),
        "first_identity": first_identity,
        "last_identity": last_identity,
        "unique_dataset_indices": True,
    }


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    _need(prediction64.shape == target64.shape, "prediction/target shape mismatch")
    _need(prediction64.ndim == 2 and prediction64.shape[0] > 1, "regression arrays must be [samples, outputs]")
    _need(np.isfinite(prediction64).all() and np.isfinite(target64).all(), "non-finite prediction or target")
    residual = target64 - prediction64
    centered = target64 - target64.mean(axis=0, keepdims=True)
    sse = np.sum(residual * residual, axis=0, dtype=np.float64)
    tss = np.sum(centered * centered, axis=0, dtype=np.float64)
    _need(bool(np.all(tss > 0.0)), "one or more outputs have zero TSS")
    r2 = 1.0 - sse / tss
    pooled = 1.0 - float(sse.sum(dtype=np.float64) / tss.sum(dtype=np.float64))
    _need(math.isfinite(pooled) and np.isfinite(r2).all(), "non-finite R2")
    return {
        "samples": int(prediction64.shape[0]),
        "outputs": int(prediction64.shape[1]),
        "sse_float64_per_output": sse.tolist(),
        "tss_float64_per_output": tss.tolist(),
        "r2_per_output": r2.tolist(),
        "pooled_variance_weighted_r2": pooled,
        "definition": "1-sum_output(SSE)/sum_output(TSS); float64 accumulation over the shared ordered query",
    }


def _linear_stack_mac_per_vector(module: nn.Module, input_dim: int) -> int:
    current = int(input_dim)
    mac = 0
    for layer in module.modules():
        if layer is module:
            continue
        if isinstance(layer, nn.Linear):
            # The archived B0 was constructed through a lazy input path.  Its
            # restored first Linear retains metadata ``in_features=0`` even
            # though the strictly loaded weight is [1024,1024] and forward
            # consumes 1024 values.  The tensor shape is the actual affine
            # operation and is therefore authoritative for MAC accounting.
            weight_out, weight_in = (int(value) for value in layer.weight.shape)
            _need(weight_in == current, "linear-stack weight dimension drift")
            _need(weight_out == int(layer.out_features), "linear-stack output metadata drift")
            mac += weight_in * weight_out
            current = weight_out
    return mac


def analytic_identity_mac(
    encoder: nn.Module, *, num_units: int, calibration_trials: int, trial_length: int,
) -> dict[str, Any]:
    _need(num_units > 0 and calibration_trials > 0 and trial_length > 0, "invalid MAC dimensions")
    if isinstance(encoder, BatchReferenceEncoder):
        per_trial_per_unit = _linear_stack_mac_per_vector(encoder.fc_id_in, trial_length)
        finalize_per_unit = _linear_stack_mac_per_vector(
            encoder.fc_id_out, encoder.fc_id_in[0].out_features,
        )
    elif isinstance(encoder, SideFeatureEarlyPoolEncoder):
        per_trial_per_unit = _linear_stack_mac_per_vector(encoder.pre_pool, trial_length)
        finalize_per_unit = _linear_stack_mac_per_vector(
            encoder.post_pool, encoder.hidden_dim + encoder.side_dim + encoder.electrode_embed_dim,
        )
    else:
        raise ForwardAuthorityError(f"unsupported identity encoder for analytic MAC: {type(encoder).__name__}")
    per_trial = num_units * per_trial_per_unit
    per_session = calibration_trials * per_trial + num_units * finalize_per_unit
    _need(per_trial > 0 and per_session > 0, "analytic identity MAC must be positive")
    return {
        "method": "exact affine-layer algebra; multiply-accumulate counts only",
        "scope": "identity encoder only; excludes decoder, interpolation, data movement, and activation operations",
        "num_units": num_units,
        "calibration_trials": calibration_trials,
        "trial_length": trial_length,
        "mac_per_trial": int(per_trial),
        "mac_per_session": int(per_session),
        "profiler_measurement_collected": False,
    }


def active_parameter_receipt(student: nn.Module, *, prune_zero4_columns: bool) -> dict[str, Any]:
    decoder = student.decoder
    identity = student.id_encoder
    identity_parameters = sum(parameter.numel() for parameter in identity.parameters())
    decoder_parameters = sum(parameter.numel() for parameter in decoder.parameters())
    legacy_parameters = sum(parameter.numel() for parameter in decoder.fc_id_in.parameters()) + sum(
        parameter.numel() for parameter in decoder.fc_id_out.parameters()
    )
    student_parameters = sum(parameter.numel() for parameter in student.parameters())
    _need(student_parameters == decoder_parameters + identity_parameters, "student parameter accounting drift")
    active_shared_decoder = decoder_parameters - legacy_parameters
    pruned_columns = 0
    if prune_zero4_columns:
        _need(isinstance(identity, SideFeatureEarlyPoolEncoder), "Zero4 pruning requires B3S")
        pruned_columns = identity.side_dim * identity.post_pool[0].out_features
    deployed_identity = identity_parameters - pruned_columns
    return {
        "encoder_only_parameters_checkpoint_topology": int(identity_parameters),
        "decoder_parameters_checkpoint_topology": int(decoder_parameters),
        "unused_legacy_decoder_identity_parameters": int(legacy_parameters),
        "student_parameters_checkpoint_topology": int(student_parameters),
        "active_shared_decoder_parameters_after_legacy_removal": int(active_shared_decoder),
        "zero4_columns_pruned": int(pruned_columns),
        "active_identity_parameters_after_zero4_pruning": int(deployed_identity),
        "active_whole_model_parameters": int(active_shared_decoder + deployed_identity),
        "teacher_copy_excluded_from_deployment_counts": True,
    }


def _pruned_zero4_identity(encoder: SideFeatureEarlyPoolEncoder, calibration: torch.Tensor) -> torch.Tensor:
    trials = calibration.permute(0, 1, 3, 2)
    mean_features = encoder.pre_pool(trials).mean(dim=1)
    first = encoder.post_pool[0]
    _need(isinstance(first, nn.Linear), "B3S first post-pool layer is not Linear")
    value = torch.nn.functional.linear(
        mean_features, first.weight[:, : encoder.hidden_dim], first.bias,
    )
    for layer in list(encoder.post_pool.children())[1:]:
        value = layer(value)
    return value


def _decode_with_legacy_identity_removed(student: nn.Module, neural: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    decoder = student.decoder
    original_in = decoder.fc_id_in
    original_out = decoder.fc_id_out
    decoder.fc_id_in = nn.Identity()
    decoder.fc_id_out = nn.Identity()
    try:
        return student.decode_with_identity(neural, identity)
    finally:
        decoder.fc_id_in = original_in
        decoder.fc_id_out = original_out


def _move_batch(batch: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(value.cpu() if isinstance(value, torch.Tensor) else value for value in batch)


def _restore_model(config: Any, checkpoint: Path, *, expected_variant: str) -> tuple[Any, dict[str, Any]]:
    _need(str(config.model.variant) == expected_variant, f"expected {expected_variant} config")
    _need(config.model.freeze_decoder is False, "authority requires jointly trained decoder")
    module = hydra.utils.instantiate(config.model)
    module.setup("test")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _need(payload.get("epoch") == EXPECTED_EPOCH, "checkpoint is not fixed epoch 11")
    missing, unexpected = module.load_state_dict(payload["state_dict"], strict=True)
    _need(not missing and not unexpected, "strict checkpoint restoration failed")
    module.cpu().eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    _need(module.student is not None, "restored module has no student")
    return module, payload


def _validate_existing_artifacts(b0_run: Path, b3s_run: Path, preflight_path: Path) -> dict[str, Any]:
    preflight = _read_json(preflight_path, label="Version-B preflight")
    _need(preflight.get("schema") == "m1_version_b_preflight_v2", "Version-B preflight schema drift")
    digest_cache: dict[Path, str] = {}
    expected_teacher = pilot.EXPECTED_TEACHER_CHECKPOINT_SHA256
    b0 = pilot._load_arm(
        b0_run, pilot.ARM_SPECS["hs"], preflight, digest_cache, expected_teacher,
    )
    b3s = pilot._load_arm(
        b3s_run, pilot.ARM_SPECS["bc0"], preflight, digest_cache, expected_teacher,
    )
    _need(b0["query_pairing_sha256"] == b3s["query_pairing_sha256"], "archived query pairing drift")
    _need(b0["source_manifest"] == b3s["source_manifest"], "archived source manifest drift")
    return {"b0": b0, "b3s": b3s, "preflight": preflight}


def evaluate(
    *, b0_run: Path = DEFAULT_B0_RUN, b3s_run: Path = DEFAULT_B3S_RUN,
    preflight_path: Path = DEFAULT_PREFLIGHT, aggregate_path: Path = DEFAULT_AGGREGATE,
) -> dict[str, Any]:
    """Run the shared-query CPU authority without optimization or selection."""

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    _need(visible_devices in {"", "-1"}, "CUDA_VISIBLE_DEVICES must fail closed to empty or -1")
    _need(not torch.cuda.is_available(), "CUDA must be unavailable to the forward-only authority")
    b0_run = b0_run.resolve()
    b3s_run = b3s_run.resolve()
    validated = _validate_existing_artifacts(b0_run, b3s_run, preflight_path.resolve())
    aggregate = _read_json(aggregate_path.resolve(), label="frozen fold0 three-arm aggregate")
    _need(aggregate.get("schema") == "m1_version_b_pilot_aggregate_v1", "aggregate schema drift")
    _need(aggregate.get("status") == "valid_terminal_three_arm_result", "aggregate is not terminal-valid")
    _need(aggregate.get("scope", {}).get("fold") == 0, "aggregate fold drift")
    _need(aggregate.get("scope", {}).get("seed") == EXPECTED_SEED, "aggregate seed drift")
    for field in ("formal_test_opened", "heldout_opened", "minival_opened"):
        _need(aggregate.get("scope", {}).get(field) is False, f"aggregate {field} scope drift")

    b0_config_path = b0_run / "resolved_config.yaml"
    b3s_config_path = b3s_run / "resolved_config.yaml"
    b0_checkpoint = b0_run / "checkpoints/best.ckpt"
    b3s_checkpoint = b3s_run / "checkpoints/best.ckpt"
    for key, run, checkpoint in (
        ("hs", b0_run, b0_checkpoint),
        ("bc0", b3s_run, b3s_checkpoint),
    ):
        arm = aggregate.get("arms", {}).get(key, {})
        _need(Path(str(arm.get("run_dir", ""))).resolve() == run, f"aggregate {key} run binding drift")
        _need(arm.get("checkpoint_sha256") == _file_sha256(checkpoint), f"aggregate {key} checkpoint binding drift")
    b0_config = _load_config(b0_config_path)
    b3s_config = _load_config(b3s_config_path)
    _need(b0_config.data.afc4_arm == "none", "B0 data arm drift")
    _need(b3s_config.data.afc4_arm == "zero4", "B3S data arm drift")
    _need(int(b0_config.seed) == int(b3s_config.seed) == EXPECTED_SEED, "seed drift")

    b0_module, _ = _restore_model(b0_config, b0_checkpoint, expected_variant="B0")
    b3s_module, _ = _restore_model(b3s_config, b3s_checkpoint, expected_variant="B3S")
    b0_student = b0_module.student
    b3s_student = b3s_module.student
    assert b0_student is not None and b3s_student is not None

    datamodule = hydra.utils.instantiate(b3s_config.data)
    datamodule.setup("test")
    _need(datamodule.outer_left_out == TARGET_SESSION, "outer target drift")
    train_sampler = ordered_sampler_receipt(
        datamodule.train_batch_sampler, datamodule.train_dataset, label="source-train",
    )
    query_sampler = ordered_sampler_receipt(
        datamodule.val_heldin_batch_sampler, datamodule.val_heldin_dataset, label="target-query",
    )
    _need(query_sampler["samples"] == EXPECTED_QUERY_SAMPLES, "target query sample count drift")

    before = {
        "b0": _module_state_sha256(b0_module),
        "b3s_zero4": _module_state_sha256(b3s_module),
    }
    b0_predictions: list[np.ndarray] = []
    b3s_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    identity_parity_max = 0.0
    prediction_parity_max = {"b0_legacy_removed": 0.0, "b3s_zero4_and_legacy_removed": 0.0}
    all_parity_exact = {key: True for key in prediction_parity_max}
    side_abs_max = 0.0
    num_units: int | None = None
    latency = {
        "b0_identity_seconds": 0.0,
        "b0_decoder_seconds": 0.0,
        "b3s_identity_seconds": 0.0,
        "b3s_decoder_seconds": 0.0,
    }

    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            batch = _move_batch(batch)
            _need(len(batch) == 5, "shared Zero4 query batch must contain side features")
            neural, target, calibration, session_names, side = batch
            _need(set(str(name) for name in session_names) == {TARGET_SESSION}, "query session drift")
            _need(side.shape[-1] == 4, "Zero4 side width drift")
            side_abs_max = max(side_abs_max, float(side.abs().max().item()))
            _need(torch.count_nonzero(side).item() == 0, "B-C0 query side input is not exact Zero4")
            current_units = int(calibration.shape[-1])
            num_units = current_units if num_units is None else num_units
            _need(num_units == current_units, "unit count changed within target query")

            start = time.perf_counter()
            b0_identity = b0_student.compute_identity(calibration)
            latency["b0_identity_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
            b0_prediction = b0_student.decode_with_identity(neural, b0_identity)
            latency["b0_decoder_seconds"] += time.perf_counter() - start

            start = time.perf_counter()
            b3s_identity = b3s_student.compute_identity(calibration, side_features=side)
            latency["b3s_identity_seconds"] += time.perf_counter() - start
            start = time.perf_counter()
            b3s_prediction = b3s_student.decode_with_identity(neural, b3s_identity)
            latency["b3s_decoder_seconds"] += time.perf_counter() - start

            pruned_identity = _pruned_zero4_identity(b3s_student.id_encoder, calibration)
            identity_delta = float((pruned_identity - b3s_identity).abs().max().item())
            identity_parity_max = max(identity_parity_max, identity_delta)
            stripped_b0 = _decode_with_legacy_identity_removed(b0_student, neural, b0_identity)
            stripped_b3s = _decode_with_legacy_identity_removed(b3s_student, neural, pruned_identity)
            for key, candidate, reference in (
                ("b0_legacy_removed", stripped_b0, b0_prediction),
                ("b3s_zero4_and_legacy_removed", stripped_b3s, b3s_prediction),
            ):
                delta = float((candidate - reference).abs().max().item())
                prediction_parity_max[key] = max(prediction_parity_max[key], delta)
                all_parity_exact[key] = all_parity_exact[key] and torch.equal(candidate, reference)

            b0_prediction, sliced_target = b0_module._slice_last_timestep(b0_prediction, target)
            b3s_prediction, sliced_target_b3s = b3s_module._slice_last_timestep(b3s_prediction, target)
            _need(torch.equal(sliced_target, sliced_target_b3s), "arm target tensors differ")
            b0_predictions.append(b0_prediction.flatten(0, 1).cpu().numpy())
            b3s_predictions.append(b3s_prediction.flatten(0, 1).cpu().numpy())
            targets.append(sliced_target.flatten(0, 1).cpu().numpy())

    _need(num_units is not None, "query produced no batches")
    _need(identity_parity_max <= PARITY_ATOL, "Zero4-column-pruned identity parity failed")
    for key, delta in prediction_parity_max.items():
        _need(delta <= PARITY_ATOL, f"{key} forward parity failed: {delta}")
    after = {
        "b0": _module_state_sha256(b0_module),
        "b3s_zero4": _module_state_sha256(b3s_module),
    }
    _need(before == after, "model state changed during forward authority")

    prediction_arrays = {
        "b0": np.concatenate(b0_predictions, axis=0),
        "b3s_zero4": np.concatenate(b3s_predictions, axis=0),
    }
    target_array = np.concatenate(targets, axis=0)
    _need(target_array.shape[0] == EXPECTED_QUERY_SAMPLES, "evaluated sample count drift")
    metrics = {name: regression_metrics(value, target_array) for name, value in prediction_arrays.items()}
    delta = (
        metrics["b3s_zero4"]["pooled_variance_weighted_r2"]
        - metrics["b0"]["pooled_variance_weighted_r2"]
    )
    archived = aggregate["arms"]
    for name, key in (("b0", "hs"), ("b3s_zero4", "bc0")):
        _need(
            abs(metrics[name]["pooled_variance_weighted_r2"] - float(archived[key]["R2"])) <= 2.0e-5,
            f"{name} forward authority does not reproduce archived R2",
        )

    first_layer = b3s_student.id_encoder.post_pool[0]
    _need(isinstance(first_layer, nn.Linear), "B3S fusion layer drift")
    side_columns = first_layer.weight[:, b3s_student.id_encoder.hidden_dim :]
    _need(torch.count_nonzero(side_columns).item() == 0, "trained B-C0 side columns are nonzero")
    calibration_trials = int(b3s_config.data.calibration_n_trials)
    trial_length = int(b3s_config.data.max_trial_length)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "scope": {
            "task": "m1",
            "fold": 0,
            "seed": EXPECTED_SEED,
            "target_session": TARGET_SESSION,
            "support_trials": [0, 10],
            "query_trials": [10, 210],
            "device": "cpu",
            "cuda_visible_devices": visible_devices,
            "torch_cuda_available": False,
            "trainer_constructed": False,
            "optimizer_constructed": False,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
            "target_checkpoint_selection": False,
            "formal_opened": False,
            "minival_opened": False,
            "evalai_opened": False,
        },
        "bindings": {
            "authority_script": {"path": str(Path(__file__).resolve()), "sha256": _file_sha256(Path(__file__))},
            "pilot_preflight": {"path": str(preflight_path.resolve()), "sha256": _file_sha256(preflight_path)},
            "pilot_aggregate": {"path": str(aggregate_path.resolve()), "sha256": _file_sha256(aggregate_path)},
            "b0": {
                "run_dir": str(b0_run),
                "config_sha256": _file_sha256(b0_config_path),
                "checkpoint_sha256": _file_sha256(b0_checkpoint),
            },
            "b3s_zero4": {
                "run_dir": str(b3s_run),
                "config_sha256": _file_sha256(b3s_config_path),
                "checkpoint_sha256": _file_sha256(b3s_checkpoint),
            },
            "source_manifest_equal": validated["b0"]["source_manifest"] == validated["b3s"]["source_manifest"],
        },
        "ordered_indices": {"source_train": train_sampler, "target_query": query_sampler},
        "arrays": {
            "target": {"shape": list(target_array.shape), "dtype": str(target_array.dtype), "sha256": _array_sha256(target_array)},
            "b0_prediction": {
                "shape": list(prediction_arrays["b0"].shape),
                "dtype": str(prediction_arrays["b0"].dtype),
                "sha256": _array_sha256(prediction_arrays["b0"]),
            },
            "b3s_zero4_prediction": {
                "shape": list(prediction_arrays["b3s_zero4"].shape),
                "dtype": str(prediction_arrays["b3s_zero4"].dtype),
                "sha256": _array_sha256(prediction_arrays["b3s_zero4"]),
            },
        },
        "metrics": {
            **metrics,
            "b3s_zero4_minus_b0": float(delta),
            "archived_rounded_b3s_zero4_minus_b0": float(aggregate["paired_deltas"]["bc0_minus_hs"]),
        },
        "zero4_and_pruning": {
            "side_input_abs_max": side_abs_max,
            "side_input_exact_zero": side_abs_max == 0.0,
            "trained_side_column_nonzero_count": int(torch.count_nonzero(side_columns).item()),
            "pruned_identity_max_abs_error": identity_parity_max,
            "prediction_max_abs_error": prediction_parity_max,
            "prediction_bit_exact": all_parity_exact,
            "parity_tolerance": PARITY_ATOL,
            "scope": "all ordered target-query batches",
        },
        "model_state": {
            "before_sha256": before,
            "after_sha256": after,
            "unchanged": before == after,
        },
        "parameters": {
            "b0": active_parameter_receipt(b0_student, prune_zero4_columns=False),
            "b3s_zero4": active_parameter_receipt(b3s_student, prune_zero4_columns=True),
            "counting_rule": "numel; encoder-only and active whole-model counts are reported separately",
        },
        "compute": {
            "analytic_identity_encoder_mac": {
                "b0": analytic_identity_mac(
                    b0_student.id_encoder, num_units=num_units,
                    calibration_trials=calibration_trials, trial_length=trial_length,
                ),
                "b3s_zero4": analytic_identity_mac(
                    b3s_student.id_encoder, num_units=num_units,
                    calibration_trials=calibration_trials, trial_length=trial_length,
                ),
            },
            "observed_cpu_wall_clock": {
                **{key: float(value) for key, value in latency.items()},
                "host": platform.node(),
                "processor": platform.processor(),
                "torch_version": torch.__version__,
                "torch_num_threads": torch.get_num_threads(),
                "claim_limit": "observed current-host wall time; not profiler FLOPs and not deployment hardware latency",
            },
        },
    }
    result["canonical_content_sha256"] = _canonical_sha256(result)
    return result


def write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    output = path.resolve()
    _need(not output.exists() and not output.is_symlink(), f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=str(output.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, output)
        _need(stat.S_IMODE(output.stat().st_mode) == 0o444, "authority output mode drift")
        return _file_sha256(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b0-run", type=Path, default=DEFAULT_B0_RUN)
    parser.add_argument("--b3s-zero4-run", type=Path, default=DEFAULT_B3S_RUN)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(
        b0_run=args.b0_run,
        b3s_run=args.b3s_zero4_run,
        preflight_path=args.preflight,
        aggregate_path=args.aggregate,
    )
    output_sha = write_immutable(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "sha256": output_sha}, sort_keys=True))


if __name__ == "__main__":
    main()
