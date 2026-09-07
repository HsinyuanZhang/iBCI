#!/usr/bin/env python3
"""One-shot CPU-only evaluator for a terminal M1 fold-2 v2 pair.

The target session is opened here, and only here, after the fold-1 PASS gate
and both fold-2 source-only terminal checkpoints have been verified.  No
trainer, optimizer, target-session checkpoint selection, or target backward
step is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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
from sua_exploration.m1_compact_replication import fold1_runner as base  # noqa: E402
from sua_exploration.m1_compact_replication import fold1_v2_contract as fold1_contract  # noqa: E402
from sua_exploration.m1_compact_replication import fold2_runner_v2 as runner  # noqa: E402
from sua_exploration.m1_compact_replication import fold2_sampler_audit as sampler_audit  # noqa: E402
from sua_exploration.m1_compact_replication import fold2_v2_contract as contract  # noqa: E402


SCHEMA = "m1_compact_b3s_f2_s42_gate_v2"
PASS = "PASS_M1_COMPACT_B3S_F2_S42_NONINFERIORITY"
STOP = "STOP_M1_COMPACT_B3S_F2_S42_NONINFERIORITY"
THRESHOLD = -0.03


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256(
        str(value.dtype).encode()
        + json.dumps(list(value.shape), separators=(",", ":")).encode()
        + value.tobytes()
    )
    return digest.hexdigest()


def regression(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    need(
        pred.shape == target.shape
        and pred.ndim == 2
        and np.isfinite(pred).all()
        and np.isfinite(target).all(),
        "prediction/target drift",
    )
    residual = target - pred
    centered = target - target.mean(axis=0, keepdims=True)
    sse = np.sum(residual * residual, axis=0, dtype=np.float64)
    tss = np.sum(centered * centered, axis=0, dtype=np.float64)
    need(np.all(tss > 0), "zero target variance")
    r2 = 1.0 - sse / tss
    return {
        "samples": int(pred.shape[0]),
        "outputs": int(pred.shape[1]),
        "sse_float64_per_output": sse.tolist(),
        "tss_float64_per_output": tss.tolist(),
        "r2_per_output": r2.tolist(),
        "pooled_variance_weighted_r2": float(1.0 - sse.sum() / tss.sum()),
        "definition": "1-sum_output(SSE)/sum_output(TSS); float64 accumulation over the shared ordered query",
    }


def load_config(path: Path) -> Any:
    cfg = OmegaConf.load(path)
    cfg.paths.output_dir = str(path.parent.resolve())
    cfg.paths.work_dir = str(ROOT.resolve())
    OmegaConf.resolve(cfg)
    return cfg


def restore(cfg: Any, ckpt: Path, variant: str, expected_global_step: int) -> Any:
    need(str(cfg.model.variant) == variant and cfg.model.freeze_decoder is False, f"{variant} config drift")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == 11 and payload.get("global_step") == expected_global_step, f"{variant} checkpoint drift")
    optimizer_states = payload.get("optimizer_states")
    need(isinstance(optimizer_states, list) and len(optimizer_states) == 1, f"{variant} optimizer count drift")
    state_steps = set()
    for slot in optimizer_states[0].get("state", {}).values():
        step = slot.get("step")
        if hasattr(step, "detach"):
            step = step.detach().cpu().item()
        state_steps.add(int(step))
    need(state_steps == {expected_global_step}, f"{variant} optimizer step drift")
    module = hydra.utils.instantiate(cfg.model)
    module.setup("test")
    missing, unexpected = module.load_state_dict(payload["state_dict"], strict=True)
    need(not missing and not unexpected, f"{variant} strict restoration failed")
    module.cpu().eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp = Path(name)
    try:
        value = dict(body)
        value["canonical_content_sha256"] = canonical(value)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(0o444)
        temp.replace(path)
        return sha(path)
    finally:
        if temp.exists():
            temp.unlink()


def _validate_source_manifest(spec: Mapping[str, Any]) -> None:
    path = Path(str(spec.get("path", "")))
    need(path.is_file() and sha(path) == spec.get("sha256"), "fold-2 source-only fit manifest changed")
    body = json.loads(path.read_text(encoding="utf-8"))
    need(body.get("schema") == "m1_version_b_source_only_fit_v2", "source-only fit manifest schema drift")
    need(
        body.get("source_only") is True
        and body.get("target_path_resolved_during_fit") is False
        and body.get("target_query_values_read_by_fit") is False,
        "source-only fit manifest target policy drift",
    )
    need(
        body.get("outer_fold") == 2
        and body.get("outer_left_out") == contract.TARGET
        and body.get("validation_sessions") == [],
        "fold-2 source-only fit manifest scope drift",
    )
    for forbidden in ("target_file", "target_path", "query_window_audit", "query_sampler_sha256", "query_scored_windows"):
        need(forbidden not in body, f"source-only fit manifest leaked {forbidden}")
    need(body.get("train_sessions") == list(contract.SOURCES), "fold-2 source session order drift")
    for name in contract.SOURCES:
        file_spec = body.get("source_files", {}).get(name, {})
        source_path = Path(str(file_spec.get("path", "")))
        need(
            source_path.is_file()
            and not source_path.is_symlink()
            and sha(source_path) == file_spec.get("sha256"),
            f"fold-2 source hash drift: {name}",
        )


def evaluate(state_path: Path, output: Path, receipt_path: Path | None = None) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    need(visible in {"", "-1"} and not torch.cuda.is_available(), "fold-2 evaluator must be CPU-only")
    need(state_path.is_file() and not state_path.is_symlink(), "fold-2 execution state missing")
    need(state_path.stat().st_mode & 0o777 == 0o444, "fold-2 execution state must be immutable")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    need(state.get("schema") == "m1_compact_b3s_f2_s42_execution_v2", "fold-2 execution schema drift")
    need(
        state.get("canonical_content_sha256")
        == canonical({k: v for k, v in state.items() if k != "canonical_content_sha256"}),
        "fold-2 execution state canonical drift",
    )
    need(
        state.get("status") == runner.STATUS_TERMINAL
        and state.get("exit_codes") == {"b0": 0, "b3s_zero4": 0},
        "fold-2 pair not terminal",
    )
    receipt = (receipt_path or contract.V2_RECEIPT).resolve()
    staged = fold1_contract._read_immutable(
        receipt,
        "fold-2 v2 staged receipt",
        contract.RECEIPT_SCHEMA,
        contract.RECEIPT_STATUS,
    )
    need(sha(receipt) == state.get("receipt_sha256"), "fold-2 staged receipt SHA drift")
    proposal = json.loads(Path(staged["proposal"]["path"]).read_text(encoding="utf-8"))
    fold1_contract._read_immutable(
        Path(staged["proposal"]["path"]),
        "fold-2 proposal",
        contract.V2_SCHEMA,
        contract.V2_STATUS,
    )
    contract._validate_proposal(proposal)
    need(sha(Path(staged["proposal"]["path"])) == staged["proposal"]["sha256"], "fold-2 proposal SHA drift")
    need(state.get("proposal_sha256") == staged["proposal"]["sha256"], "fold-2 state proposal binding drift")
    current_gate = runner.validate_fold1_gate(staged)
    need(state.get("fold1_gate") == current_gate, "fold-1 gate changed after fold-2 pair")
    need(contract.validate_teacher() == staged["teacher"], "fold-2 teacher changed")
    current_inventory = contract.source_only_inventory()
    need(current_inventory == staged["inventory"], "fold-2 source/data inventory changed")
    source_sampler = runner._validate_sampler_audit(staged)
    expected_global_step = int(source_sampler["trainer"]["expected_global_step"])
    need(state.get("source_sampler") == source_sampler, "fold-2 source sampler audit changed")
    need(
        state.get("source_sampler_audit")
        == {"path": str(sampler_audit.RECEIPT.resolve()), "sha256": sha(sampler_audit.RECEIPT), "body": source_sampler},
        "fold-2 source sampler audit binding drift",
    )
    need(
        state.get("inventory_before") == state.get("inventory_after") == current_inventory,
        "fold-2 execution inventory revalidation failed",
    )
    need(state.get("target_opened_by_training") is False and state.get("target_metrics_not_read_by_runner") is True, "fold-2 target policy drift")

    terminals = state.get("terminal_checkpoints", {})
    need(set(terminals) == {"b0", "b3s_zero4"}, "fold-2 terminal map incomplete")
    for terminal in terminals.values():
        manifest_spec = terminal.get("source_only_manifest")
        need(isinstance(manifest_spec, dict), "fold-2 terminal source-only manifest missing")
        _validate_source_manifest(manifest_spec)
    for key, variant in (("b0", "B0"), ("b3s_zero4", "B3S")):
        resolved = terminals[key].get("resolved_config")
        need(isinstance(resolved, dict), f"fold-2 {key} resolved config receipt missing")
        current_resolved = runner._resolved_config_for_run(
            "m1_compact_b0_f2_s42_fresh_e11" if key == "b0" else "m1_compact_b3s_zero4_f2_s42_fresh_e11",
            variant,
            staged["teacher"]["checkpoint"]["path"],
        )
        need(current_resolved == resolved, f"fold-2 {key} resolved config changed after terminal")

    b0_artifacts = sorted((ROOT / "outputs/streaming_calibration").glob("m1_compact_b0_f2_s42_fresh_e11_f2_s42_*"))
    b3s_artifacts = sorted((ROOT / "outputs/streaming_calibration").glob("m1_compact_b3s_zero4_f2_s42_fresh_e11_f2_s42_*"))
    need(len(b0_artifacts) == 1 and len(b3s_artifacts) == 1, "fold-2 artifact directories ambiguous")
    b0_cfg = load_config(b0_artifacts[0] / "resolved_config.yaml")
    b3s_cfg = load_config(b3s_artifacts[0] / "resolved_config.yaml")
    source_only_target = contract.SOURCE_ONLY_TARGET
    full_target = "src.data.m1_version_b_source_loso_datamodule.M1VersionBSourceLOSODataModule"
    need(
        str(b0_cfg.data._target_) == source_only_target
        and str(b3s_cfg.data._target_) == source_only_target,
        "fold-2 artifacts did not bind source-only fit module",
    )
    # This explicit evaluator-local transition is the only place that opens
    # the left-out held-in query after both source-only terminals exist.
    b0_cfg.data._target_ = full_target
    b3s_cfg.data._target_ = full_target
    b0_module = restore(b0_cfg, Path(terminals["b0"]["path"]), "B0", expected_global_step)
    b3s_module = restore(b3s_cfg, Path(terminals["b3s_zero4"]["path"]), "B3S", expected_global_step)
    need(int(b0_cfg.data.loso_fold) == int(b3s_cfg.data.loso_fold) == 2, "fold-2 fold drift")
    datamodule = hydra.utils.instantiate(b3s_cfg.data)
    datamodule.setup("test")
    need(datamodule.outer_left_out == contract.TARGET, "fold-2 target drift")
    query = authority.ordered_sampler_receipt(
        datamodule.val_heldin_batch_sampler,
        datamodule.val_heldin_dataset,
        label="fold2-target-query",
    )
    before = {
        "b0": authority._module_state_sha256(b0_module),
        "b3s_zero4": authority._module_state_sha256(b3s_module),
    }
    b0_pred, b3s_pred, target = [], [], []
    side_max, identity_max, prediction_max, prediction_exact = 0.0, 0.0, 0.0, True
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            need(len(batch) == 5, "fold-2 batch arity drift")
            neural, target_batch, calibration, names, side = authority._move_batch(batch)
            need(set(str(x) for x in names) == {contract.TARGET}, "fold-2 query session drift")
            side_max = max(side_max, float(side.abs().max().item()))
            need(torch.count_nonzero(side).item() == 0, "fold-2 B3S Zero4 side is not zero")
            b0_id = b0_module.student.compute_identity(calibration)
            b3s_id = b3s_module.student.compute_identity(calibration, side_features=side)
            p0 = b0_module.student.decode_with_identity(neural, b0_id)
            p3 = b3s_module.student.decode_with_identity(neural, b3s_id)
            pruned = authority._pruned_zero4_identity(b3s_module.student.id_encoder, calibration)
            identity_max = max(identity_max, float((pruned - b3s_id).abs().max().item()))
            pp = b3s_module.student.decode_with_identity(neural, pruned)
            prediction_max = max(prediction_max, float((pp - p3).abs().max().item()))
            prediction_exact = prediction_exact and torch.equal(pp, p3)
            p0, target_s = b0_module._slice_last_timestep(p0, target_batch)
            p3, target_s3 = b3s_module._slice_last_timestep(p3, target_batch)
            need(torch.equal(target_s, target_s3), "fold-2 arm target mismatch")
            b0_pred.append(p0.flatten(0, 1).numpy())
            b3s_pred.append(p3.flatten(0, 1).numpy())
            target.append(target_s.flatten(0, 1).numpy())
    after = {
        "b0": authority._module_state_sha256(b0_module),
        "b3s_zero4": authority._module_state_sha256(b3s_module),
    }
    need(before == after, "fold-2 target forward changed model state")
    a0, a3, at = np.concatenate(b0_pred), np.concatenate(b3s_pred), np.concatenate(target)
    m0, m3 = regression(a0, at), regression(a3, at)
    delta = float(m3["pooled_variance_weighted_r2"] - m0["pooled_variance_weighted_r2"])
    body = {
        "schema": SCHEMA,
        "status": PASS if delta >= THRESHOLD else STOP,
        "scope": {
            "task": "m1",
            "fold": 2,
            "seed": 42,
            "target_session": contract.TARGET,
            "formal_opened": False,
            "minival_opened": False,
            "heldout_opened": False,
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "target_checkpoint_selection": False,
            "device": "cpu",
            "cuda_visible_devices": visible,
        },
        "bindings": {
            "receipt": {"path": str(receipt.resolve()), "sha256": sha(receipt)},
            "fold1_gate": current_gate,
            "b0": {"artifact": str(b0_artifacts[0].resolve()), "terminal_checkpoint": terminals["b0"]},
            "b3s_zero4": {"artifact": str(b3s_artifacts[0].resolve()), "terminal_checkpoint": terminals["b3s_zero4"]},
            "query": query,
        },
        "arrays": {
            "target": {"shape": list(at.shape), "dtype": str(at.dtype), "sha256": array_sha(at)},
            "b0_prediction": {"shape": list(a0.shape), "dtype": str(a0.dtype), "sha256": array_sha(a0)},
            "b3s_zero4_prediction": {"shape": list(a3.shape), "dtype": str(a3.dtype), "sha256": array_sha(a3)},
        },
        "metrics": {
            "b0": m0,
            "b3s_zero4": m3,
            "b3s_zero4_minus_b0": delta,
            "gate_threshold": THRESHOLD,
        },
        "parity": {
            "side_input_abs_max": side_max,
            "side_input_exact_zero": side_max == 0.0,
            "zero4_identity_max_abs_error": identity_max,
            "zero4_prediction_max_abs_error": prediction_max,
            "zero4_prediction_bit_exact": prediction_exact,
        },
        "model_state": {"before_sha256": before, "after_sha256": after, "unchanged": before == after},
        "evaluation_policy": {
            "forward_only": True,
            "trainer_constructed": False,
            "optimizer_constructed": False,
            "intermediate_target_metric_read": False,
            "training_target_opened": False,
            "target_opened_here_after_terminal_pair": True,
            "claim_limit": "development fold2 source-LOSO replication; not formal held-out superiority",
        },
        "created_at_epoch": time.time(),
    }
    immutable(output.resolve(), body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    body = evaluate(args.state.resolve(), args.output.resolve(), args.receipt.resolve() if args.receipt else None)
    print(
        json.dumps(
            {"status": body["status"], "delta": body["metrics"]["b3s_zero4_minus_b0"], "output": str(args.output.resolve()), "sha256": sha(args.output.resolve())},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
