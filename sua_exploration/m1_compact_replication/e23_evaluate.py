#!/usr/bin/env python3
"""CPU-only terminal evaluator for the M1 compact-B3S e23 pair.

The evaluator consumes the two terminal epoch-23 checkpoints indexed by the
pair runner.  It opens one shared ordered query loader, computes predictions
without a Trainer/optimizer/backward step, and emits an immutable paired gate
receipt with prediction/target hashes and per-output SSE/TSS/R2.  It never
reads the target scores emitted by the training command and cannot select a
checkpoint from a target metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
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

from sua_exploration.m1_compact_replication import fold0_forward_authority as authority  # noqa: E402


SCHEMA = "m1_compact_b3s_f0_s42_e23_gate_v1"
STATUS_PASS = "PASS_M1_COMPACT_B3S_F0_E23_NONINFERIORITY"
STATUS_FAIL = "FAIL_M1_COMPACT_B3S_F0_E23_NONINFERIORITY"
EXPECTED_EPOCH = 23
EXPECTED_STEP = 118_824
EXPECTED_SEED = 42
EXPECTED_QUERY_SAMPLES = 26_496
GATE_THRESHOLD = -0.03
PARITY_ATOL = 1.0e-6
EXPECTED_AUTHORITY_SHA = "7ec281631b8e96d0311e029ca227fc7c47b01d9d095f3015399952de2d1f315a"
EXPECTED_DATA_BINDING_SHA = "c17e4291cf29ba2884aff9f91f8e0ef44f68b574663221bd233010a57789d8c8"


class E23EvaluationError(ValueError):
    """Terminal e23 receipt invariant failed."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise E23EvaluationError(message)


def sha256_file(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing or symlinked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def model_state_sha(module: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def restore_e23(config: Any, checkpoint: Path, expected_variant: str) -> tuple[Any, dict[str, Any]]:
    need(str(config.model.variant) == expected_variant, f"{expected_variant} variant drift")
    need(config.model.freeze_decoder is False, "e23 requires jointly trained decoder")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == EXPECTED_EPOCH, f"{expected_variant} terminal epoch drift")
    need(payload.get("global_step") == EXPECTED_STEP, f"{expected_variant} terminal global step drift")
    need(payload.get("pytorch-lightning_version") == "2.6.5", f"{expected_variant} Lightning drift")
    need(isinstance(payload.get("state_dict"), Mapping) and payload["state_dict"], f"{expected_variant} terminal state missing")
    module = hydra.utils.instantiate(config.model)
    module.setup("test")
    missing, unexpected = module.load_state_dict(payload["state_dict"], strict=True)
    need(not missing and not unexpected, f"{expected_variant} strict restoration failed")
    module.cpu().eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    need(module.student is not None, f"{expected_variant} student missing")
    return module, payload


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    need(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[0] > 1, "prediction/target shape drift")
    need(np.isfinite(prediction).all() and np.isfinite(target).all(), "non-finite prediction or target")
    residual = target - prediction
    centered = target - target.mean(axis=0, keepdims=True)
    sse = np.sum(residual * residual, axis=0, dtype=np.float64)
    tss = np.sum(centered * centered, axis=0, dtype=np.float64)
    need(bool(np.all(tss > 0.0)), "zero target variance")
    r2 = 1.0 - sse / tss
    pooled = 1.0 - float(sse.sum(dtype=np.float64) / tss.sum(dtype=np.float64))
    need(math.isfinite(pooled) and np.isfinite(r2).all(), "non-finite R2")
    return {
        "definition": "1-sum_output(SSE)/sum_output(TSS); float64 accumulation over the shared ordered query",
        "samples": int(prediction.shape[0]),
        "outputs": int(prediction.shape[1]),
        "sse_float64_per_output": sse.tolist(),
        "tss_float64_per_output": tss.tolist(),
        "r2_per_output": r2.tolist(),
        "pooled_variance_weighted_r2": pooled,
    }


def _load_config(path: Path) -> Any:
    need(path.is_file() and not path.is_symlink(), f"resolved config missing: {path}")
    config = OmegaConf.load(path)
    config.paths.output_dir = str(path.parent.resolve())
    config.paths.work_dir = str(ROOT.resolve())
    OmegaConf.resolve(config)
    return config


def _write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        need(stat.S_IMODE(path.stat().st_mode) == 0o444, "gate receipt mode drift")
        return sha256_file(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_state(state_path: Path) -> dict[str, Any]:
    state = read_json(state_path, "pair execution state")
    need(state.get("status") == "PASS_M1_COMPACT_B3S_F0_E23_PAIR_TERMINAL", "pair did not terminate cleanly")
    need(state.get("exit_codes") == {"b0": 0, "b3s_zero4": 0}, "pair process status is not zero")
    terminals = state.get("terminal_checkpoints", {})
    need(set(terminals) == {"b0", "b3s_zero4"}, "terminal checkpoint map incomplete")
    for key, variant in (("b0", "B0"), ("b3s_zero4", "B3S")):
        item = terminals[key]
        path = Path(str(item.get("path", "")))
        need(item.get("variant") == variant and item.get("epoch") == EXPECTED_EPOCH and item.get("global_step") == EXPECTED_STEP, f"{key} terminal metadata drift")
        need(sha256_file(path) == item.get("sha256"), f"{key} terminal checkpoint SHA drift")
        need(item.get("resume_parent_sha256"), f"{key} parent SHA missing")
        need(item.get("launch_command"), f"{key} launch command missing")
        need(item.get("log", {}).get("sha256") == sha256_file(Path(item["log"]["path"])), f"{key} log SHA drift")
        need(item.get("restored_all_states", {}).get("present") is True, f"{key} strict restore evidence missing")
    return state


def evaluate(state_path: Path, output: Path) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    need(visible in {"", "-1"}, "e23 evaluator must hide CUDA")
    need(not torch.cuda.is_available(), "e23 evaluator must run CPU-only")
    state = _validate_state(state_path)
    launch_receipt = Path(str(state.get("launch_receipt", {}).get("path", "")))
    launch_sha = str(state.get("launch_receipt", {}).get("sha256", ""))
    need(sha256_file(launch_receipt) == launch_sha, "launch receipt SHA drift")
    launch = read_json(launch_receipt, "launch receipt")
    need(launch.get("scope", {}).get("target_checkpoint_selection") is False, "launch target selection drift")
    authority_path = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F0_S42_FORWARD_AUTHORITY_v1.json"
    need(sha256_file(authority_path) == EXPECTED_AUTHORITY_SHA, "forward authority SHA drift")
    frozen_authority = read_json(authority_path, "fold0 forward authority")
    need(frozen_authority.get("canonical_content_sha256") == canonical_sha({k: v for k, v in frozen_authority.items() if k != "canonical_content_sha256"}), "forward authority canonical hash drift")
    data_binding_path = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F0_S42_E23_DATA_BINDING_v1.json"
    need(sha256_file(data_binding_path) == EXPECTED_DATA_BINDING_SHA, "data binding SHA drift")
    data_binding = read_json(data_binding_path, "M1 data binding")
    need(data_binding.get("canonical_content_sha256") == canonical_sha({k: v for k, v in data_binding.items() if k != "canonical_content_sha256"}), "data binding canonical hash drift")
    data_inventory = data_binding.get("data", {})
    need(data_inventory.get("file_count") == 12 and len(data_inventory.get("files_sha256", {})) == 12, "M1 data binding file count drift")
    data_root = Path(str(data_inventory.get("path", "")))
    for rel, expected in data_inventory.get("files_sha256", {}).items():
        need(sha256_file(data_root / rel) == expected, f"M1 data byte drift: {rel}")
    terminals = state["terminal_checkpoints"]
    b0_checkpoint = Path(terminals["b0"]["path"])
    b3s_checkpoint = Path(terminals["b3s_zero4"]["path"])
    # Artifact paths are recorded by the train command; use direct parent
    # discovery rather than guessing the timestamp suffix.
    artifact_root = ROOT / "outputs/streaming_calibration"
    def find_artifact(prefix: str) -> Path:
        candidates = sorted(artifact_root.glob(prefix + "_f0_s42_*"))
        need(len(candidates) == 1, f"expected one artifact for {prefix}, found {len(candidates)}")
        return candidates[0]
    b0_run = find_artifact("m1_compact_b0_f0_s42_resume_e11_to_e23")
    b3s_run = find_artifact("m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23")
    b0_config_path, b3s_config_path = b0_run / "resolved_config.yaml", b3s_run / "resolved_config.yaml"
    b0_config, b3s_config = _load_config(b0_config_path), _load_config(b3s_config_path)
    need(int(b0_config.seed) == int(b3s_config.seed) == EXPECTED_SEED, "seed drift")
    need(int(b0_config.data.loso_fold) == int(b3s_config.data.loso_fold) == 0, "fold drift")
    need(str(b0_config.data.afc4_arm) == "none" and str(b3s_config.data.afc4_arm) == "zero4", "M1 arm drift")
    b0_module, _ = restore_e23(b0_config, b0_checkpoint, "B0")
    b3s_module, _ = restore_e23(b3s_config, b3s_checkpoint, "B3S")
    datamodule = hydra.utils.instantiate(b3s_config.data)
    datamodule.setup("test")
    need(datamodule.outer_left_out == "ses-20120924", "target session drift")
    train_receipt = authority.ordered_sampler_receipt(datamodule.train_batch_sampler, datamodule.train_dataset, label="source-train")
    query_receipt = authority.ordered_sampler_receipt(datamodule.val_heldin_batch_sampler, datamodule.val_heldin_dataset, label="target-query")
    need(query_receipt["samples"] == EXPECTED_QUERY_SAMPLES, "query sample count drift")
    frozen_query = frozen_authority["ordered_indices"]["target_query"]
    for field in ("samples", "batches", "ordered_sampler_indices_sha256", "ordered_window_identity_sha256", "first_identity", "last_identity", "unique_dataset_indices"):
        need(query_receipt.get(field) == frozen_query.get(field), f"e23 query {field} differs from e11 authority")
    before = {"b0": model_state_sha(b0_module), "b3s_zero4": model_state_sha(b3s_module)}
    b0_predictions: list[np.ndarray] = []
    b3s_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    side_abs_max = 0.0
    zero4_identity_parity = 0.0
    zero4_prediction_parity = 0.0
    zero4_prediction_bit_exact = True
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            need(len(batch) == 5, "M1 test batch shape drift")
            neural, target, calibration, session_names, side = authority._move_batch(batch)
            need(set(str(x) for x in session_names) == {"ses-20120924"}, "query session drift")
            need(side.shape[-1] == 4, "side width drift")
            side_abs_max = max(side_abs_max, float(side.abs().max().item()))
            need(torch.count_nonzero(side).item() == 0, "B3S Zero4 target side is not exact zero")
            b0_identity = b0_module.student.compute_identity(calibration)
            b3s_identity = b3s_module.student.compute_identity(calibration, side_features=side)
            b0_prediction = b0_module.student.decode_with_identity(neural, b0_identity)
            b3s_prediction = b3s_module.student.decode_with_identity(neural, b3s_identity)
            pruned_identity = authority._pruned_zero4_identity(b3s_module.student.id_encoder, calibration)
            zero4_identity_parity = max(zero4_identity_parity, float((pruned_identity - b3s_identity).abs().max().item()))
            pruned_prediction = b3s_module.student.decode_with_identity(neural, pruned_identity)
            prediction_error = float((pruned_prediction - b3s_prediction).abs().max().item())
            zero4_prediction_parity = max(zero4_prediction_parity, prediction_error)
            zero4_prediction_bit_exact = zero4_prediction_bit_exact and torch.equal(pruned_prediction, b3s_prediction)
            b0_prediction, sliced_target = b0_module._slice_last_timestep(b0_prediction, target)
            b3s_prediction, sliced_target_b3s = b3s_module._slice_last_timestep(b3s_prediction, target)
            need(torch.equal(sliced_target, sliced_target_b3s), "B0/B3S target tensors differ")
            b0_predictions.append(b0_prediction.flatten(0, 1).cpu().numpy())
            b3s_predictions.append(b3s_prediction.flatten(0, 1).cpu().numpy())
            targets.append(sliced_target.flatten(0, 1).cpu().numpy())
    after = {"b0": model_state_sha(b0_module), "b3s_zero4": model_state_sha(b3s_module)}
    need(before == after, "model state changed during terminal evaluation")
    b0_array = np.concatenate(b0_predictions, axis=0)
    b3s_array = np.concatenate(b3s_predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    need(target_array.shape[0] == EXPECTED_QUERY_SAMPLES, "evaluated query count drift")
    frozen_target = frozen_authority["arrays"]["target"]
    need(frozen_target.get("shape") == list(target_array.shape), "e23 target shape differs from e11 authority")
    need(frozen_target.get("dtype") == str(target_array.dtype), "e23 target dtype differs from e11 authority")
    need(frozen_target.get("sha256") == array_sha(target_array), "e23 target array SHA differs from e11 authority")
    b0_metrics, b3s_metrics = regression_metrics(b0_array, target_array), regression_metrics(b3s_array, target_array)
    delta = float(b3s_metrics["pooled_variance_weighted_r2"] - b0_metrics["pooled_variance_weighted_r2"])
    status = STATUS_PASS if delta >= GATE_THRESHOLD else STATUS_FAIL
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "scope": {
            "task": "m1", "fold": 0, "seed": EXPECTED_SEED,
            "target_session": "ses-20120924", "support_trials": [0, 10], "query_trials": [10, 210],
            "formal_opened": False, "minival_opened": False, "heldout_opened": False,
            "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False,
            "device": "cpu", "cuda_visible_devices": visible,
        },
        "launch": {"state_path": str(state_path.resolve()), "state_sha256": sha256_file(state_path), "launch_receipt": {"path": str(launch_receipt.resolve()), "sha256": launch_sha}, "authority": {"path": str(authority_path.resolve()), "sha256": EXPECTED_AUTHORITY_SHA}, "data_binding": {"path": str(data_binding_path.resolve()), "sha256": EXPECTED_DATA_BINDING_SHA}},
        "bindings": {
            "b0": {"artifact_dir": str(b0_run.resolve()), "config_sha256": sha256_file(b0_config_path), "terminal_checkpoint": terminals["b0"]},
            "b3s_zero4": {"artifact_dir": str(b3s_run.resolve()), "config_sha256": sha256_file(b3s_config_path), "terminal_checkpoint": terminals["b3s_zero4"]},
            "shared_query": query_receipt,
        },
        "ordered_indices": {"source_train": train_receipt, "target_query": query_receipt},
        "arrays": {
            "target": {"shape": list(target_array.shape), "dtype": str(target_array.dtype), "sha256": array_sha(target_array)},
            "b0_prediction": {"shape": list(b0_array.shape), "dtype": str(b0_array.dtype), "sha256": array_sha(b0_array)},
            "b3s_zero4_prediction": {"shape": list(b3s_array.shape), "dtype": str(b3s_array.dtype), "sha256": array_sha(b3s_array)},
        },
        "metrics": {
            "b0": b0_metrics,
            "b3s_zero4": b3s_metrics,
            "b3s_zero4_minus_b0": delta,
            "gate_threshold": GATE_THRESHOLD,
            "gate_definition": "PASS iff terminal B3S-Zero4 minus B0 pooled variance-weighted R2 >= -0.03 on the shared ordered query",
        },
        "model_state": {"before_sha256": before, "after_sha256": after, "unchanged": before == after},
        "parity": {"side_input_abs_max": side_abs_max, "side_input_exact_zero": side_abs_max == 0.0, "zero4_identity_max_abs_error": zero4_identity_parity, "zero4_prediction_max_abs_error": zero4_prediction_parity, "zero4_prediction_bit_exact": zero4_prediction_bit_exact, "tolerance": PARITY_ATOL},
        "parameters": {
            "b0": authority.active_parameter_receipt(b0_module.student, prune_zero4_columns=False),
            "b3s_zero4": authority.active_parameter_receipt(b3s_module.student, prune_zero4_columns=True),
            "counting_rule": "numel; encoder-only and active whole-model counts are reported separately",
        },
        "evaluation_policy": {
            "forward_only": True,
            "trainer_constructed": False,
            "optimizer_constructed": False,
            "intermediate_target_metric_read": False,
            "claim_limit": "fixed fold0/seed42 development convergence gate; not formal held-out superiority",
        },
        "created_at_epoch": time.time(),
    }
    body["canonical_content_sha256"] = canonical_sha(body)
    _write_immutable(output.resolve(), body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body = evaluate(args.state.resolve(), args.output.resolve())
    print(json.dumps({"status": body["status"], "output": str(args.output.resolve()), "sha256": sha256_file(args.output.resolve()), "delta": body["metrics"]["b3s_zero4_minus_b0"]}, sort_keys=True))


if __name__ == "__main__":
    main()
