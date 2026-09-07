#!/usr/bin/env python3
"""One-shot CPU-only evaluator for a terminal M1 fold-1 pair."""
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
from sua_exploration.m1_compact_replication import fold1_runner as runner  # noqa: E402
from sua_exploration.m1_compact_replication import fold1_v2_contract as v2_contract  # noqa: E402
from sua_exploration.m1_compact_replication import fold1_runner_v2 as v2_runner  # noqa: E402
from sua_exploration.m1_compact_replication import fold1_recovery as recovery  # noqa: E402


SCHEMA = "m1_compact_b3s_f1_s42_gate_v2"
PASS = "PASS_M1_COMPACT_B3S_F1_S42_NONINFERIORITY"
STOP = "STOP_M1_COMPACT_B3S_F1_S42_NONINFERIORITY"
THRESHOLD = -0.03


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    d = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value))
    d = hashlib.sha256(str(value.dtype).encode() + json.dumps(list(value.shape), separators=(",", ":")).encode() + value.tobytes())
    return d.hexdigest()


def regression(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    need(pred.shape == target.shape and pred.ndim == 2 and np.isfinite(pred).all() and np.isfinite(target).all(), "prediction/target drift")
    residual = target - pred
    centered = target - target.mean(axis=0, keepdims=True)
    sse = np.sum(residual * residual, axis=0, dtype=np.float64)
    tss = np.sum(centered * centered, axis=0, dtype=np.float64)
    need(np.all(tss > 0), "zero target variance")
    r2 = 1.0 - sse / tss
    return {"samples": int(pred.shape[0]), "outputs": int(pred.shape[1]), "sse_float64_per_output": sse.tolist(), "tss_float64_per_output": tss.tolist(), "r2_per_output": r2.tolist(), "pooled_variance_weighted_r2": float(1.0 - sse.sum() / tss.sum()), "definition": "1-sum_output(SSE)/sum_output(TSS); float64 accumulation over the shared ordered query"}


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
    need(path.is_file() and sha(path) == spec.get("sha256"), "source-only fit manifest changed")
    body = json.loads(path.read_text(encoding="utf-8"))
    need(body.get("schema") == "m1_version_b_source_only_fit_v2", "source-only fit manifest schema drift")
    need(body.get("source_only") is True and body.get("target_path_resolved_during_fit") is False, "source-only fit manifest target policy drift")
    need(body.get("validation_sessions") == [] and body.get("target_query_values_read_by_fit") is False, "source-only fit manifest validation drift")
    for forbidden in ("target_file", "target_path", "query_window_audit", "query_sampler_sha256", "query_scored_windows"):
        need(forbidden not in body, f"source-only fit manifest leaked {forbidden}")
    source_names = ["ses-20120924", "ses-20120927", "ses-20120928"]
    need(body.get("train_sessions") == source_names, "source-only fit manifest source order drift")
    for name in source_names:
        file_spec = body.get("source_files", {}).get(name, {})
        source_path = Path(str(file_spec.get("path", "")))
        need(source_path.is_file() and sha(source_path) == file_spec.get("sha256"), f"source-only fit source hash drift: {name}")


def evaluate(state_path: Path, output: Path, receipt_path: Path | None = None) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    need(visible in {"", "-1"} and not torch.cuda.is_available(), "fold1 evaluator must be CPU-only")
    need(state_path.is_file() and not state_path.is_symlink(), "fold1 execution state missing")
    need(state_path.stat().st_mode & 0o777 == 0o444, "fold1 execution state must be immutable")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    need(state.get("schema") in {"m1_compact_b3s_f1_s42_execution_v2", recovery.STATE_SCHEMA}, "fold1 execution schema drift")
    need(state.get("canonical_content_sha256") == canonical({k: v for k, v in state.items() if k != "canonical_content_sha256"}), "fold1 execution state canonical drift")
    need(
        state.get("status") in {"PASS_M1_COMPACT_B3S_F1_S42_V2_PAIR_TERMINAL", recovery.STATE_PASS}
        and (state.get("exit_codes") == {"b0": 0, "b3s_zero4": 0} or state.get("exit_codes") == {"b3s_zero4": 0}),
        "fold1 pair not terminal",
    )
    receipt = (receipt_path or v2_contract.V2_RECEIPT).resolve()
    staged = v2_contract._read_immutable(receipt, "fold1 v2 staged receipt", v2_contract.RECEIPT_SCHEMA, v2_contract.RECEIPT_STATUS)
    need(sha(receipt) == state.get("receipt_sha256"), "fold1 staged receipt SHA drift")
    v2_contract._validate_v2_proposal(json.loads(Path(staged["proposal"]["path"]).read_text(encoding="utf-8")))
    need(runner.sha(Path(staged["proposal"]["path"])) == staged["proposal"]["sha256"], "fold1 v2 proposal SHA drift")
    need(runner.validate_gate()["sha256"] == staged["gate"]["sha256"], "fold0/e23 gate changed")
    need(runner.validate_teacher() == staged["teacher"], "fold1 teacher changed")
    need(runner.validate_preflight() == staged["preflight"], "fold1 preflight changed")
    current_inventory = v2_contract.source_only_inventory()
    need(current_inventory == staged["inventory"], "fold1 source/data inventory changed")
    equivalence_binding = v2_contract.validate_equivalence_binding(receipt)
    # Derive the expected checkpoint step from the bound fold-1 source sampler,
    # not from the fold-0 literal (59412) that caused the original executor to
    # fail closed.  The recovered state records the same derivation for audit.
    _, b0_config = recovery._resolved_config("m1_compact_b0_f1_s42_fresh_e11", "B0")
    sampler_contract = recovery.derive_sampler_contract(
        json.loads(recovery.EQUIVALENCE_RECEIPT.read_text(encoding="utf-8")),
        batch_size=int(b0_config["data"]["batch_size"]),
        epochs=int(b0_config["trainer"]["max_epochs"]),
    )
    expected_global_step = int(sampler_contract["expected_global_step"])
    if state.get("schema") == recovery.STATE_SCHEMA:
        need(state.get("source_sampler") == sampler_contract, "recovered source sampler derivation drift")
    need(state.get("inventory_before") == state.get("inventory_after") == current_inventory, "fold1 execution inventory revalidation failed")
    terminals = state.get("terminal_checkpoints", {})
    need(set(terminals) == {"b0", "b3s_zero4"}, "fold1 terminal map incomplete")
    for terminal in terminals.values():
        manifest_spec = terminal.get("source_only_manifest")
        need(isinstance(manifest_spec, dict), "terminal source-only manifest missing")
        _validate_source_manifest(manifest_spec)
    for key, variant in (("b0", "B0"), ("b3s_zero4", "B3S")):
        resolved = terminals[key].get("resolved_config")
        need(isinstance(resolved, dict), f"{key} resolved config receipt missing")
        current_resolved = v2_runner._resolved_config_for_run(
            "m1_compact_b0_f1_s42_fresh_e11" if key == "b0" else "m1_compact_b3s_zero4_f1_s42_fresh_e11",
            variant,
            staged["teacher"]["checkpoint"]["path"],
        )
        need(current_resolved == resolved, f"{key} resolved config changed after terminal")
    b0_artifacts = sorted((ROOT / "outputs/streaming_calibration").glob("m1_compact_b0_f1_s42_fresh_e11_f1_s42_*"))
    b3s_artifacts = sorted((ROOT / "outputs/streaming_calibration").glob("m1_compact_b3s_zero4_f1_s42_fresh_e11_f1_s42_*"))
    need(len(b0_artifacts) == 1 and len(b3s_artifacts) == 1, "fold1 artifact directories ambiguous")
    b0_cfg, b3s_cfg = load_config(b0_artifacts[0] / "resolved_config.yaml"), load_config(b3s_artifacts[0] / "resolved_config.yaml")
    # Training resolves the source-only-fit class.  The independent evaluator
    # is the only stage allowed to switch to the full LOSO data module and
    # materialize the target held-in query.
    source_only_target = (
        "src.data.m1_version_b_source_loso_datamodule."
        "M1VersionBSourceOnlyFitDataModule"
    )
    full_target = "src.data.m1_version_b_source_loso_datamodule.M1VersionBSourceLOSODataModule"
    need(str(b0_cfg.data._target_) == source_only_target and str(b3s_cfg.data._target_) == source_only_target, "fold1 artifact did not bind source-only fit module")
    b0_cfg.data._target_ = full_target
    b3s_cfg.data._target_ = full_target
    b0_module = restore(b0_cfg, Path(terminals["b0"]["path"]), "B0", expected_global_step)
    b3s_module = restore(b3s_cfg, Path(terminals["b3s_zero4"]["path"]), "B3S", expected_global_step)
    need(int(b0_cfg.data.loso_fold) == int(b3s_cfg.data.loso_fold) == 1, "fold1 fold drift")
    datamodule = hydra.utils.instantiate(b3s_cfg.data)
    datamodule.setup("test")
    need(datamodule.outer_left_out == "ses-20120926", "fold1 target drift")
    query = authority.ordered_sampler_receipt(datamodule.val_heldin_batch_sampler, datamodule.val_heldin_dataset, label="fold1-target-query")
    before = {"b0": authority._module_state_sha256(b0_module), "b3s_zero4": authority._module_state_sha256(b3s_module)}
    b0_pred, b3s_pred, target = [], [], []
    side_max, identity_max, prediction_max, prediction_exact = 0.0, 0.0, 0.0, True
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            need(len(batch) == 5, "fold1 batch arity drift")
            neural, target_batch, calibration, names, side = authority._move_batch(batch)
            need(set(str(x) for x in names) == {"ses-20120926"}, "fold1 query session drift")
            side_max = max(side_max, float(side.abs().max().item()))
            need(torch.count_nonzero(side).item() == 0, "fold1 B3S Zero4 side is not zero")
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
            need(torch.equal(target_s, target_s3), "fold1 arm target mismatch")
            b0_pred.append(p0.flatten(0, 1).numpy())
            b3s_pred.append(p3.flatten(0, 1).numpy())
            target.append(target_s.flatten(0, 1).numpy())
    after = {"b0": authority._module_state_sha256(b0_module), "b3s_zero4": authority._module_state_sha256(b3s_module)}
    need(before == after, "fold1 target forward changed model state")
    a0, a3, at = np.concatenate(b0_pred), np.concatenate(b3s_pred), np.concatenate(target)
    m0, m3 = regression(a0, at), regression(a3, at)
    delta = float(m3["pooled_variance_weighted_r2"] - m0["pooled_variance_weighted_r2"])
    body = {"schema": SCHEMA, "status": PASS if delta >= THRESHOLD else STOP, "scope": {"task": "m1", "fold": 1, "seed": 42, "target_session": "ses-20120926", "formal_opened": False, "minival_opened": False, "heldout_opened": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False, "device": "cpu", "cuda_visible_devices": visible}, "bindings": {"receipt": {"path": str(receipt.resolve()), "sha256": sha(receipt)}, "source_fit_equivalence": equivalence_binding, "b0": {"artifact": str(b0_artifacts[0].resolve()), "terminal_checkpoint": terminals["b0"]}, "b3s_zero4": {"artifact": str(b3s_artifacts[0].resolve()), "terminal_checkpoint": terminals["b3s_zero4"]}, "query": query}, "arrays": {"target": {"shape": list(at.shape), "dtype": str(at.dtype), "sha256": array_sha(at)}, "b0_prediction": {"shape": list(a0.shape), "dtype": str(a0.dtype), "sha256": array_sha(a0)}, "b3s_zero4_prediction": {"shape": list(a3.shape), "dtype": str(a3.dtype), "sha256": array_sha(a3)}}, "metrics": {"b0": m0, "b3s_zero4": m3, "b3s_zero4_minus_b0": delta, "gate_threshold": THRESHOLD}, "parity": {"side_input_abs_max": side_max, "side_input_exact_zero": side_max == 0.0, "zero4_identity_max_abs_error": identity_max, "zero4_prediction_max_abs_error": prediction_max, "zero4_prediction_bit_exact": prediction_exact}, "model_state": {"before_sha256": before, "after_sha256": after, "unchanged": before == after}, "evaluation_policy": {"forward_only": True, "trainer_constructed": False, "optimizer_constructed": False, "intermediate_target_metric_read": False, "training_target_opened": False, "target_opened_here_after_terminal_pair": True, "claim_limit": "development fold1 source-LOSO replication; not formal held-out superiority"}, "created_at_epoch": time.time()}
    immutable(output.resolve(), body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    body = evaluate(args.state.resolve(), args.output.resolve(), args.receipt.resolve() if args.receipt else None)
    print(json.dumps({"status": body["status"], "delta": body["metrics"]["b3s_zero4_minus_b0"], "output": str(args.output.resolve()), "sha256": sha(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
