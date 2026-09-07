#!/usr/bin/env python3
"""Independent fixed epoch-23 target-development evaluator for folds 1/2.

The training runner writes only source-only terminal state.  This evaluator is
the sole process that switches to the ordinary LOSO data module and opens the
left-out target query, exactly once after both epoch-23 checkpoints exist.  It
constructs no Trainer or optimizer and never performs a target backward step.
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
from typing import Any, Mapping

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[2]
STREAM_ROOT = ROOT / "streaming_calibration_exp"
for entry in (str(ROOT), str(STREAM_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from sua_exploration.m1_compact_replication import fold0_forward_authority as authority  # noqa: E402
from sua_exploration.m1_compact_replication import e23_continuation as runner  # noqa: E402


FULL_TARGET = "src.data.m1_version_b_source_loso_datamodule.M1VersionBSourceLOSODataModule"
THRESHOLD = -0.03


class EvaluationError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def model_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def regression(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    need(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[0] > 1, "prediction/target shape drift")
    need(np.isfinite(prediction).all() and np.isfinite(target).all(), "non-finite prediction/target")
    residual = target - prediction
    centered = target - target.mean(axis=0, keepdims=True)
    sse = np.sum(residual * residual, axis=0, dtype=np.float64)
    tss = np.sum(centered * centered, axis=0, dtype=np.float64)
    need(bool(np.all(tss > 0)), "zero target variance")
    r2 = 1.0 - sse / tss
    pooled = 1.0 - float(sse.sum(dtype=np.float64) / tss.sum(dtype=np.float64))
    need(math.isfinite(pooled) and np.isfinite(r2).all(), "non-finite R2")
    return {"definition": "1-sum_output(SSE)/sum_output(TSS); float64 accumulation over the shared ordered query", "samples": int(prediction.shape[0]), "outputs": int(prediction.shape[1]), "sse_float64_per_output": sse.tolist(), "tss_float64_per_output": tss.tolist(), "r2_per_output": r2.tolist(), "pooled_variance_weighted_r2": pooled}


def read_json(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing/symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"{label} not an object")
    return value


def write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        need(stat.S_IMODE(path.stat().st_mode) == 0o444, "gate mode drift")
        return sha(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def state_for(fold: int) -> Path:
    return runner.execution_path(fold)


def gate_for(fold: int) -> Path:
    return ROOT / "sua_exploration/m1_compact_replication/results" / f"M1_COMPACT_B3S_F{fold}_S42_E23_GATE_v1.json"


def restore(config: Any, checkpoint: Path, variant: str, expected_step: int) -> Any:
    need(str(config.model.variant) == variant and config.model.freeze_decoder is False, f"{variant} model policy drift")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == 23 and payload.get("global_step") == expected_step and payload.get("pytorch-lightning_version") == "2.6.5", f"{variant} checkpoint drift")
    module = hydra.utils.instantiate(config.model)
    module.setup("test")
    missing, unexpected = module.load_state_dict(payload["state_dict"], strict=True)
    need(not missing and not unexpected, f"{variant} strict restoration failed")
    module.cpu().eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def evaluate(fold: int) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    need(visible in {"", "-1"} and not torch.cuda.is_available(), "e23 evaluator must be CPU-only")
    state_path = state_for(fold)
    state = read_json(state_path, f"fold-{fold} continuation state")
    need(state_path.stat().st_mode & 0o777 == 0o444, f"fold-{fold} continuation state mutable")
    need(state.get("status") == runner.EXEC_STATUS.format(fold=fold), f"fold-{fold} pair not terminal")
    need(state.get("exit_codes") == {"b0": 0, "b3s_zero4": 0}, f"fold-{fold} process status drift")
    need(state.get("canonical_content_sha256") == canonical({k: v for k, v in state.items() if k != "canonical_content_sha256"}), f"fold-{fold} state canonical drift")
    receipt_path = Path(state["launch_receipt"]["path"])
    receipt = read_json(receipt_path, f"fold-{fold} launch receipt")
    need(sha(receipt_path) == state["launch_receipt"]["sha256"], f"fold-{fold} launch receipt SHA drift")
    need(receipt.get("scope", {}).get("target_backward_steps") == 0 and receipt.get("scope", {}).get("target_optimizer_steps") == 0 and receipt.get("scope", {}).get("target_checkpoint_selection") is False, f"fold-{fold} continuation target policy drift")
    terminals = state.get("terminal_checkpoints", {})
    need(set(terminals) == {"b0", "b3s_zero4"}, f"fold-{fold} terminal map incomplete")
    expected_step = int(receipt["execution_policy"]["terminal_global_step"])
    configs: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    for key, variant in (("b0", "B0"), ("b3s_zero4", "B3S")):
        item = terminals[key]
        checkpoint = Path(item["path"])
        need(sha(checkpoint) == item["sha256"], f"fold-{fold} {key} checkpoint SHA drift")
        config_path = Path(item["artifact"]["resolved_config"]["path"])
        need(sha(config_path) == item["artifact"]["resolved_config"]["sha256"], f"fold-{fold} {key} config SHA drift")
        cfg = OmegaConf.load(config_path)
        cfg.paths.output_dir = str(config_path.parent.resolve())
        cfg.paths.work_dir = str(ROOT.resolve())
        OmegaConf.resolve(cfg)
        need(cfg.get("test") is False and cfg.get("optimized_metric") is None and int(cfg.get("seed")) == 42, f"fold-{fold} {key} training config policy drift")
        need(str(cfg.data._target_) == runner.SOURCE_ONLY_TARGET and int(cfg.data.loso_fold) == fold and list(cfg.data.source_session_names) == list(runner.SOURCES[fold]), f"fold-{fold} {key} source-only config drift")
        need(int(cfg.data.heldin_query_start_trial) == 10 and int(cfg.data.heldin_query_end_trial) == 210, f"fold-{fold} {key} query window drift")
        need(str(cfg.data.afc4_arm) == ("none" if key == "b0" else "zero4"), f"fold-{fold} {key} arm config drift")
        need(int(cfg.trainer.min_epochs) == 24 and int(cfg.trainer.max_epochs) == 24 and int(cfg.trainer.limit_val_batches) == 0 and int(cfg.trainer.num_sanity_val_steps) == 0, f"fold-{fold} {key} trainer config drift")
        cfg.data._target_ = FULL_TARGET
        configs[key] = cfg
        modules[key] = restore(cfg, checkpoint, variant, expected_step)
    datamodule = hydra.utils.instantiate(configs["b3s_zero4"].data)
    datamodule.setup("test")
    need(datamodule.outer_left_out == runner.TARGETS[fold], f"fold-{fold} target session drift")
    query = authority.ordered_sampler_receipt(datamodule.val_heldin_batch_sampler, datamodule.val_heldin_dataset, label=f"fold{fold}-e23-target-query")
    need(query.get("support_trials") in (None, [0, 10]) or query.get("support_trials") == [0, 10], f"fold-{fold} query support metadata drift")
    before = {key: model_sha(module) for key, module in modules.items()}
    predictions = {"b0": [], "b3s_zero4": [], "target": []}
    side_abs_max = 0.0
    identity_max = 0.0
    prediction_max = 0.0
    prediction_exact = True
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            need(len(batch) == 5, f"fold-{fold} test batch arity drift")
            neural, target, calibration, names, side = authority._move_batch(batch)
            need(set(str(x) for x in names) == {runner.TARGETS[fold]}, f"fold-{fold} query session drift")
            side_abs_max = max(side_abs_max, float(side.abs().max().item()))
            need(torch.count_nonzero(side).item() == 0, f"fold-{fold} B3S Zero4 side is not zero")
            b0_identity = modules["b0"].student.compute_identity(calibration)
            b3s_identity = modules["b3s_zero4"].student.compute_identity(calibration, side_features=side)
            b0_prediction = modules["b0"].student.decode_with_identity(neural, b0_identity)
            b3s_prediction = modules["b3s_zero4"].student.decode_with_identity(neural, b3s_identity)
            pruned_identity = authority._pruned_zero4_identity(modules["b3s_zero4"].student.id_encoder, calibration)
            identity_max = max(identity_max, float((pruned_identity - b3s_identity).abs().max().item()))
            pruned_prediction = modules["b3s_zero4"].student.decode_with_identity(neural, pruned_identity)
            prediction_error = float((pruned_prediction - b3s_prediction).abs().max().item())
            prediction_max = max(prediction_max, prediction_error)
            prediction_exact = prediction_exact and torch.equal(pruned_prediction, b3s_prediction)
            b0_prediction, sliced_target = modules["b0"]._slice_last_timestep(b0_prediction, target)
            b3s_prediction, sliced_target_b3s = modules["b3s_zero4"]._slice_last_timestep(b3s_prediction, target)
            need(torch.equal(sliced_target, sliced_target_b3s), f"fold-{fold} arm target mismatch")
            predictions["b0"].append(b0_prediction.flatten(0, 1).cpu().numpy())
            predictions["b3s_zero4"].append(b3s_prediction.flatten(0, 1).cpu().numpy())
            predictions["target"].append(sliced_target.flatten(0, 1).cpu().numpy())
    after = {key: model_sha(module) for key, module in modules.items()}
    need(before == after, f"fold-{fold} target forward changed model state")
    arrays = {key: np.concatenate(value, axis=0) for key, value in predictions.items()}
    need(arrays["b0"].shape == arrays["b3s_zero4"].shape == arrays["target"].shape, f"fold-{fold} evaluated shapes differ")
    metrics = {key: regression(arrays[key], arrays["target"]) for key in ("b0", "b3s_zero4")}
    delta = float(metrics["b3s_zero4"]["pooled_variance_weighted_r2"] - metrics["b0"]["pooled_variance_weighted_r2"])
    status = f"PASS_M1_COMPACT_B3S_F{fold}_S42_E23_NONINFERIORITY" if delta >= THRESHOLD else f"STOP_M1_COMPACT_B3S_F{fold}_S42_E23_NONINFERIORITY"
    output = gate_for(fold)
    body = {"schema": f"m1_compact_b3s_f{fold}_s42_e23_gate_v1", "status": status, "scope": {"task": "m1", "fold": fold, "seed": 42, "target_session": runner.TARGETS[fold], "support_trials": [0, 10], "query_trials": [10, 210], "formal_opened": False, "minival_opened": False, "heldout_opened": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False, "device": "cpu", "cuda_visible_devices": visible}, "bindings": {"launch_receipt": {"path": str(receipt_path.resolve()), "sha256": sha(receipt_path)}, "execution": {"path": str(state_path.resolve()), "sha256": sha(state_path)}, "b0": {"terminal_checkpoint": terminals["b0"], "config": {"path": str(Path(terminals["b0"]["artifact"]["resolved_config"]["path"]).resolve()), "sha256": terminals["b0"]["artifact"]["resolved_config"]["sha256"]}}, "b3s_zero4": {"terminal_checkpoint": terminals["b3s_zero4"], "config": {"path": str(Path(terminals["b3s_zero4"]["artifact"]["resolved_config"]["path"]).resolve()), "sha256": terminals["b3s_zero4"]["artifact"]["resolved_config"]["sha256"]}}, "query": query}, "arrays": {key: {"shape": list(arrays[key].shape), "dtype": str(arrays[key].dtype), "sha256": array_sha(arrays[key])} for key in ("target", "b0", "b3s_zero4")}, "metrics": {"b0": metrics["b0"], "b3s_zero4": metrics["b3s_zero4"], "b3s_zero4_minus_b0": delta, "gate_threshold": THRESHOLD, "gate_definition": "PASS iff terminal B3S-Zero4 minus B0 pooled variance-weighted R2 >= -0.03 on one shared ordered target query"}, "parity": {"side_input_abs_max": side_abs_max, "side_input_exact_zero": side_abs_max == 0.0, "zero4_identity_max_abs_error": identity_max, "zero4_prediction_max_abs_error": prediction_max, "zero4_prediction_bit_exact": prediction_exact}, "model_state": {"before_sha256": before, "after_sha256": after, "unchanged": before == after}, "evaluation_policy": {"forward_only": True, "trainer_constructed": False, "optimizer_constructed": False, "target_opened_here_after_both_terminal_checkpoints": True, "training_target_opened": False, "intermediate_target_metric_read": False, "claim_limit": f"development fold{fold} source-LOSO e23 replication; not formal held-out superiority"}, "created_at_epoch": time.time()}
    write_immutable(output, body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2))
    args = parser.parse_args()
    body = evaluate(args.fold)
    print(json.dumps({"status": body["status"], "delta": body["metrics"]["b3s_zero4_minus_b0"], "output": str(gate_for(args.fold).resolve()), "sha256": sha(gate_for(args.fold))}, sort_keys=True))


if __name__ == "__main__":
    main()
