#!/usr/bin/env python3
"""Strict source-only epoch-11 -> epoch-23 continuation for M1 folds 1/2.

This is deliberately a new, append-only runner.  The already accepted fold-0
runner is not modified and the fresh fold-1/fold-2 epoch-11 receipts remain
immutable parents.  Preparation only validates the complete Lightning
checkpoint state and writes a read-only launch receipt.  Execution consumes
the exact two command vectors in that receipt (B0 then B3S-Zero4), never
opens the target and never reads a target metric.  A separate CPU evaluator
must be used after the pair terminal receipt exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
TRAIN = ROOT / "streaming_calibration_exp/src/train.py"
SOURCE_ONLY_TARGET = (
    "src.data.m1_version_b_source_loso_datamodule."
    "M1VersionBSourceOnlyFitDataModule"
)
SOURCES = {
    1: ("ses-20120924", "ses-20120927", "ses-20120928"),
    2: ("ses-20120924", "ses-20120926", "ses-20120928"),
}
TARGETS = {1: "ses-20120926", 2: "ses-20120927"}
TEACHERS = {
    1: ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold1/runs/2026-08-06-16-20-17-515482_rid-m1_afc4_source_decoder_fold1_dev20_f1_s42/checkpoints/best_ckpt/epoch_019.ckpt",
    2: ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold2_remote/runs/remote_fold2_source_epoch019/checkpoints/best_ckpt/epoch_019.ckpt",
}
PARENT_EXECUTIONS = {
    1: ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_EXECUTION_RECOVERED_v1.json",
    2: ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_EXECUTION_v2.json",
}
PARENT_GATES = {
    1: ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_GATE_v2.json",
    2: ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F2_S42_GATE_v2.json",
}

# Hash the code/config surface that can affect the fit.  The target NWB is
# intentionally absent; source-only preparation must not even hash it.
CODE_FILES = (
    "streaming_calibration_exp/src/train.py",
    "streaming_calibration_exp/src/data/m1_version_b_source_loso_datamodule.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/configs/experiment/m1_version_b_hs_continuation.yaml",
    "streaming_calibration_exp/configs/experiment/m1_version_b_c0.yaml",
    "streaming_calibration_exp/configs/data/m1_version_b_source_loso.yaml",
    "streaming_calibration_exp/configs/model/m1_version_b_b0.yaml",
    "streaming_calibration_exp/configs/model/m1_version_b_b3s.yaml",
    "streaming_calibration_exp/configs/callbacks/m1_version_b_fixed_epoch.yaml",
    "streaming_calibration_exp/configs/hydra/default.yaml",
)

SCHEMA_PREFIX = "m1_compact_b3s_f{fold}_s42_e23_continuation"
RECEIPT_STATUS = "PASS_M1_COMPACT_B3S_F{fold}_S42_E23_LAUNCH_PREPARED_NOT_LAUNCHED"
EXEC_STATUS = "PASS_M1_COMPACT_B3S_F{fold}_S42_E23_PAIR_TERMINAL"
FAIL_STATUS = "FAIL_M1_COMPACT_B3S_F{fold}_S42_E23_PAIR_EXECUTION"


class ContractError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked path: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing/symlinked: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"{label} is not an object")
    return value


def write_immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(body)
    value["canonical_content_sha256"] = canonical(value)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
        need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"immutable mode drift: {path}")
        return sha(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def receipt_path(fold: int) -> Path:
    return ROOT / "sua_exploration/m1_compact_replication/results" / f"M1_COMPACT_B3S_F{fold}_S42_E23_CONTINUATION_LAUNCH_v2.json"


def execution_path(fold: int) -> Path:
    return ROOT / "sua_exploration/m1_compact_replication/results" / f"M1_COMPACT_B3S_F{fold}_S42_E23_CONTINUATION_EXECUTION_v2.json"


def parent_checkpoint(fold: int, arm: str) -> Path:
    run = f"m1_compact_{'b0' if arm == 'B0' else 'b3s_zero4'}_f{fold}_s42_fresh_e11"
    candidates = sorted(
        p for p in (ROOT / "logs").rglob("epoch_epoch=011.ckpt")
        if run in str(p) and "fixed_last" in str(p)
    )
    need(len(candidates) == 1, f"expected one parent checkpoint for {run}, found {len(candidates)}")
    return candidates[0]


def expected_parent_step(fold: int) -> int:
    # Bound to the observed source sampler, and checked again from the
    # immutable parent execution receipt.  Fold 1 and fold 2 intentionally
    # have different source batch cardinalities.
    parent = read_json(PARENT_EXECUTIONS[fold], f"fold-{fold} e11 execution")
    need(parent.get("status", "").endswith("PAIR_TERMINAL"), f"fold-{fold} e11 parent is not terminal")
    terminals = parent.get("terminal_checkpoints", {})
    steps = {int(item.get("global_step", -1)) for item in terminals.values() if isinstance(item, dict)}
    need(len(steps) == 1 and next(iter(steps)) > 0, f"fold-{fold} e11 parent step missing")
    return next(iter(steps))


def validate_parent(fold: int) -> dict[str, Any]:
    parent_receipt = PARENT_EXECUTIONS[fold]
    parent = read_json(parent_receipt, f"fold-{fold} e11 execution")
    need(parent_receipt.stat().st_mode & 0o777 == 0o444, f"fold-{fold} e11 execution mutable")
    need(parent.get("canonical_content_sha256") == canonical({k: v for k, v in parent.items() if k != "canonical_content_sha256"}), f"fold-{fold} e11 canonical drift")
    need(parent.get("status", "").endswith("PAIR_TERMINAL"), f"fold-{fold} e11 execution not terminal")
    expected = expected_parent_step(fold)
    terminals = parent.get("terminal_checkpoints", {})
    need(set(terminals) == {"b0", "b3s_zero4"}, f"fold-{fold} e11 terminal map incomplete")
    result: dict[str, Any] = {"path": str(parent_receipt.resolve()), "sha256": sha(parent_receipt), "status": parent["status"], "global_step": expected, "terminals": {}}
    for key, variant in (("b0", "B0"), ("b3s_zero4", "B3S")):
        item = terminals[key]
        path = Path(str(item.get("path", "")))
        need(path.is_file() and sha(path) == item.get("sha256"), f"fold-{fold} parent {key} checkpoint binding drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        need(payload.get("epoch") == 11 and payload.get("global_step") == expected, f"fold-{fold} parent {key} epoch/step drift")
        need(payload.get("pytorch-lightning_version") == "2.6.5", f"fold-{fold} parent {key} Lightning drift")
        need(isinstance(payload.get("state_dict"), Mapping) and payload["state_dict"], f"fold-{fold} parent {key} state_dict absent")
        opt = payload.get("optimizer_states")
        need(isinstance(opt, list) and len(opt) == 1, f"fold-{fold} parent {key} optimizer count drift")
        groups = opt[0].get("param_groups", [])
        need(len(groups) == 1 and float(groups[0].get("lr")) == 1.0e-4 and float(groups[0].get("weight_decay")) == 0.0, f"fold-{fold} parent {key} optimizer hyperparameter drift")
        state_steps = set()
        for slot in opt[0].get("state", {}).values():
            step = slot.get("step")
            if hasattr(step, "detach"):
                step = step.detach().cpu().item()
            if step is not None:
                state_steps.add(int(step))
        need(state_steps == {expected}, f"fold-{fold} parent {key} optimizer step drift")
        fit = payload.get("loops", {}).get("fit_loop", {})
        need(fit.get("epoch_loop.batch_progress", {}).get("total", {}).get("completed") == expected, f"fold-{fold} parent {key} batch progress drift")
        need(fit.get("epoch_progress", {}).get("total", {}).get("processed") == 12, f"fold-{fold} parent {key} epoch progress drift")
        callbacks = payload.get("callbacks", {})
        need(len(callbacks) == 1 and "every_n_epochs': 12" in str(next(iter(callbacks))), f"fold-{fold} parent {key} callback state drift")
        result["terminals"][key] = {"path": str(path.resolve()), "sha256": sha(path), "variant": variant, "epoch": 11, "global_step": expected, "optimizer_state_count": len(opt[0].get("state", {})), "strict_resume_possible": True}
    return result


def source_inventory(fold: int) -> dict[str, Any]:
    code = {rel: sha(ROOT / rel) for rel in CODE_FILES}
    data_root = ROOT / "SPINT-main/data/000941"
    source_dir = data_root / "sub-MonkeyL-held-in-calib"
    files: dict[str, str] = {}
    for session in SOURCES[fold]:
        suffix = session.removeprefix("ses-")
        path = source_dir / f"sub-MonkeyL-held-in-calib_ses-{suffix}_behavior+ecephys.nwb"
        need(path.is_file() and not path.is_symlink(), f"fold-{fold} source NWB missing: {session}")
        files[path.relative_to(data_root).as_posix()] = sha(path)
    return {"code_files_sha256": code, "data_root": str(data_root.resolve()), "source_sessions": list(SOURCES[fold]), "source_data_files_sha256": files, "source_data_file_count": len(files), "target_session_excluded": TARGETS[fold], "target_data_hashed_by_prelaunch": False}


def environment() -> dict[str, Any]:
    def run(argv: list[str]) -> str:
        try:
            return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:  # noqa: BLE001
            return f"unavailable:{exc}"
    import lightning
    return {"hostname": platform.node(), "python": run([sys.executable, "--version"]), "lightning": str(lightning.__version__), "torch": str(torch.__version__), "torch_cuda_available": bool(torch.cuda.is_available()), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])}


def gate_binding(fold: int) -> dict[str, Any]:
    path = PARENT_GATES[fold]
    body = read_json(path, f"fold-{fold} e11 gate")
    need(path.stat().st_mode & 0o777 == 0o444, f"fold-{fold} e11 gate mutable")
    need(body.get("status", "").endswith("NONINFERIORITY"), f"fold-{fold} e11 gate is not PASS")
    need(body.get("scope", {}).get("target_backward_steps") == 0 and body.get("scope", {}).get("target_optimizer_steps") == 0 and body.get("scope", {}).get("target_checkpoint_selection") is False, f"fold-{fold} e11 gate target policy drift")
    return {"path": str(path.resolve()), "sha256": sha(path), "schema": body.get("schema"), "status": body.get("status"), "delta": body.get("metrics", {}).get("b3s_zero4_minus_b0")}


def command_vectors(fold: int, parent: Mapping[str, Any]) -> dict[str, list[str]]:
    teacher = TEACHERS[fold]
    need(teacher.is_file() and not teacher.is_symlink(), f"fold-{fold} teacher missing")
    commands: dict[str, list[str]] = {}
    for arm, key, experiment in (("B0", "b0", "m1_version_b_hs_continuation"), ("B3S", "b3s_zero4", "m1_version_b_c0")):
        run_id = f"m1_compact_{key}_f{fold}_s42_resume_e11_to_e23"
        ckpt = parent["terminals"][key]["path"]
        # Hydra's override grammar treats an unescaped '=' in the checkpoint
        # filename (epoch_epoch=011.ckpt) as a second assignment.  Keep the
        # exact bound path while escaping only that delimiter in the argv
        # token; Hydra unescapes it back to the filesystem path.
        ckpt_arg = str(ckpt).replace("epoch_epoch=011.ckpt", r"epoch_epoch\=011.ckpt")
        commands[key] = [PYTHON, str(TRAIN.resolve()), f"experiment={experiment}", f"run_id={run_id}", f"data.loso_fold={fold}", f"data.source_session_names=[{','.join(SOURCES[fold])}]", "seed=42", "trainer.min_epochs=24", "trainer.max_epochs=24", "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0", f"data._target_={SOURCE_ONLY_TARGET}", "test=false", "optimized_metric=null", f"model.teacher_ckpt_path={teacher.resolve()}", f"ckpt_path={ckpt_arg}"]
    for key, argv in commands.items():
        need(argv[0] == PYTHON and argv[1] == str(TRAIN.resolve()), f"fold-{fold} {key} train entrypoint drift")
        for token in ("seed=42", "trainer.min_epochs=24", "trainer.max_epochs=24", "trainer.limit_val_batches=0", "trainer.num_sanity_val_steps=0", f"data._target_={SOURCE_ONLY_TARGET}", "test=false", "optimized_metric=null"):
            need(argv.count(token) == 1, f"fold-{fold} {key} command token drift: {token}")
        need("test=true" not in argv and "ckpt_path=null" not in argv, f"fold-{fold} {key} command opens test/null resume")
    return commands


def prepare(fold: int) -> dict[str, Any]:
    need(fold in (1, 2), "fold must be 1 or 2")
    parent = validate_parent(fold)
    inventory = source_inventory(fold)
    gate = gate_binding(fold)
    commands = command_vectors(fold, parent)
    body = {"schema": SCHEMA_PREFIX.format(fold=fold) + "_launch_v2", "status": RECEIPT_STATUS.format(fold=fold), "scope": {"task": "m1", "fold": fold, "seed": 42, "target_session": TARGETS[fold], "source_sessions": list(SOURCES[fold]), "support_trials": [0, 10], "query_trials": [10, 210], "formal_or_minival_or_heldout": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False}, "parent_e11_execution": parent, "parent_e11_gate": gate, "teacher": {"path": str(TEACHERS[fold].resolve()), "sha256": sha(TEACHERS[fold])}, "commands": commands, "fixed_order": ["b0", "b3s_zero4"], "source_inventory": inventory, "environment": environment(), "execution_policy": {"train_test_flag": False, "source_only_fit_data_module": SOURCE_ONLY_TARGET, "target_opened_by_training": False, "target_metrics_not_read_by_launcher": True, "resume_all_states_required": True, "terminal_epoch": 23, "terminal_global_step": parent["global_step"] * 2, "gpu_index": 0, "tmux_required": True, "launched": False}}
    path = receipt_path(fold)
    if path.exists():
        existing = read_json(path, f"fold-{fold} continuation launch receipt")
        need(path.stat().st_mode & 0o777 == 0o444, f"fold-{fold} launch receipt mutable")
        need(existing.get("canonical_content_sha256") == canonical({k: v for k, v in existing.items() if k != "canonical_content_sha256"}), f"fold-{fold} launch receipt canonical drift")
        need(canonical(body) == existing.get("canonical_content_sha256"), f"fold-{fold} existing launch receipt differs")
        return existing
    write_immutable(path, body)
    return read_json(path, f"fold-{fold} continuation launch receipt")


def _validate_receipt(fold: int, path: Path) -> dict[str, Any]:
    body = read_json(path, f"fold-{fold} continuation launch receipt")
    need(path.stat().st_mode & 0o777 == 0o444, f"fold-{fold} launch receipt mutable")
    need(body.get("schema") == SCHEMA_PREFIX.format(fold=fold) + "_launch_v2", f"fold-{fold} launch schema drift")
    need(body.get("status") == RECEIPT_STATUS.format(fold=fold), f"fold-{fold} launch receipt already consumed/invalid")
    need(body.get("canonical_content_sha256") == canonical({k: v for k, v in body.items() if k != "canonical_content_sha256"}), f"fold-{fold} launch receipt canonical drift")
    need(body.get("scope", {}).get("source_sessions") == list(SOURCES[fold]) and body.get("scope", {}).get("target_session") == TARGETS[fold], f"fold-{fold} launch scope drift")
    need(body.get("execution_policy", {}).get("train_test_flag") is False and body.get("execution_policy", {}).get("target_opened_by_training") is False, f"fold-{fold} target policy drift")
    need(body.get("fixed_order") == ["b0", "b3s_zero4"], f"fold-{fold} command order drift")
    current = source_inventory(fold)
    need(current == body.get("source_inventory"), f"fold-{fold} code/data inventory changed after prepare")
    need(sha(TEACHERS[fold]) == body.get("teacher", {}).get("sha256"), f"fold-{fold} teacher changed after prepare")
    return body


def _validate_terminal(fold: int, key: str, run_id: str, log_path: Path, parent_step: int, command: list[str]) -> dict[str, Any]:
    expected_step = parent_step * 2
    candidates = sorted(p for p in (ROOT / "logs").rglob("epoch_epoch=023.ckpt") if run_id in str(p) and "fixed_last" in str(p))
    need(len(candidates) == 1, f"fold-{fold} {key} expected one epoch-23 checkpoint, found {len(candidates)}")
    path = candidates[0]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == 23 and payload.get("global_step") == expected_step, f"fold-{fold} {key} terminal epoch/step drift")
    need(payload.get("pytorch-lightning_version") == "2.6.5", f"fold-{fold} {key} Lightning drift")
    need(isinstance(payload.get("state_dict"), Mapping) and payload["state_dict"], f"fold-{fold} {key} terminal state missing")
    opt = payload.get("optimizer_states")
    need(isinstance(opt, list) and len(opt) == 1, f"fold-{fold} {key} terminal optimizer count drift")
    groups = opt[0].get("param_groups", [])
    need(len(groups) == 1 and float(groups[0].get("lr")) == 1.0e-4 and float(groups[0].get("weight_decay")) == 0.0, f"fold-{fold} {key} terminal optimizer hyperparameter drift")
    state_steps = set()
    for slot in opt[0].get("state", {}).values():
        step = slot.get("step")
        if hasattr(step, "detach"):
            step = step.detach().cpu().item()
        if step is not None:
            state_steps.add(int(step))
    need(state_steps == {expected_step}, f"fold-{fold} {key} terminal optimizer step drift")
    fit = payload.get("loops", {}).get("fit_loop", {})
    need(fit.get("epoch_loop.batch_progress", {}).get("total", {}).get("completed") == expected_step, f"fold-{fold} {key} terminal batch progress drift")
    need(fit.get("epoch_progress", {}).get("total", {}).get("processed") == 24, f"fold-{fold} {key} terminal epoch progress drift")
    callbacks = payload.get("callbacks", {})
    need(len(callbacks) == 1 and "every_n_epochs': 12" in str(next(iter(callbacks))), f"fold-{fold} {key} terminal callback drift")
    need(log_path.is_file(), f"fold-{fold} {key} training log missing")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    need("Starting training!" in text and "max_epochs=24" in text, f"fold-{fold} {key} training completion evidence missing")
    run_candidates = sorted(p for p in (ROOT / "outputs/streaming_calibration").glob(run_id + f"_f{fold}_s42_*"))
    need(len(run_candidates) == 1, f"fold-{fold} {key} artifact directory ambiguity")
    config_path = run_candidates[0] / "resolved_config.yaml"
    need(config_path.is_file(), f"fold-{fold} {key} resolved config missing")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    need(config.get("run_id") == run_id and config.get("seed") == 42 and config.get("test") is False and config.get("optimized_metric") is None, f"fold-{fold} {key} resolved test/seed policy drift")
    data = config.get("data", {})
    need(data.get("_target_") == SOURCE_ONLY_TARGET and data.get("loso_fold") == fold and data.get("source_session_names") == list(SOURCES[fold]) and data.get("heldin_query_start_trial") == 10 and data.get("heldin_query_end_trial") == 210, f"fold-{fold} {key} resolved data policy drift")
    need(data.get("afc4_arm") == ("none" if key == "b0" else "zero4"), f"fold-{fold} {key} resolved arm drift")
    trainer = config.get("trainer", {})
    need(trainer.get("min_epochs") == 24 and trainer.get("max_epochs") == 24 and trainer.get("limit_val_batches") == 0 and trainer.get("num_sanity_val_steps") == 0, f"fold-{fold} {key} resolved trainer policy drift")
    model = config.get("model", {})
    need(model.get("teacher_ckpt_path") == str(TEACHERS[fold].resolve()), f"fold-{fold} {key} resolved teacher drift")
    manifests = sorted(p for p in (ROOT / "logs").rglob("m1_version_b_source_only_fit_manifest.json") if run_id in str(p))
    need(len(manifests) == 1, f"fold-{fold} {key} source-only manifest ambiguity")
    manifest_path = manifests[0]
    manifest = read_json(manifest_path, f"fold-{fold} {key} source-only manifest")
    need(manifest.get("schema") == "m1_version_b_source_only_fit_v2" and manifest.get("source_only") is True and manifest.get("target_path_resolved_during_fit") is False and manifest.get("target_query_values_read_by_fit") is False and manifest.get("validation_sessions") == [] and manifest.get("train_sessions") == list(SOURCES[fold]), f"fold-{fold} {key} source-only manifest policy drift")
    for forbidden in ("target_file", "target_path", "query_window_audit", "query_sampler_sha256", "query_scored_windows"):
        need(forbidden not in manifest, f"fold-{fold} {key} source-only manifest leaked {forbidden}")
    parent_path = Path(command[-1].replace(r"\=", "=").split("ckpt_path=", 1)[1])
    return {"path": str(path.resolve()), "sha256": sha(path), "variant": "B0" if key == "b0" else "B3S", "epoch": 23, "global_step": expected_step, "optimizer": {"count": 1, "state_count": len(opt[0].get("state", {})), "state_steps": [expected_step], "lr": 1.0e-4, "weight_decay": 0.0}, "fit_loop": {"batch_completed": expected_step, "epoch_processed": 24}, "callback_key": str(next(iter(callbacks))), "launch_command": command, "resume_parent_sha256": sha(parent_path), "strict_terminal_state": True, "log": {"path": str(log_path.resolve()), "sha256": sha(log_path)}, "artifact": {"path": str(run_candidates[0].resolve()), "resolved_config": {"path": str(config_path.resolve()), "sha256": sha(config_path)}}, "source_only_manifest": {"path": str(manifest_path.resolve()), "sha256": sha(manifest_path)}}


def execute(fold: int, authorization: str) -> int:
    need(authorization == f"M1_COMPACT_F{fold}_E23_APPROVED_BY_ROOT", "explicit root authorization token required")
    path = receipt_path(fold)
    receipt = _validate_receipt(fold, path)
    state_path = execution_path(fold)
    need(not state_path.exists() and not state_path.is_symlink(), f"fold-{fold} execution receipt already exists")
    parent_step = int(receipt["parent_e11_execution"]["global_step"])
    log_dir = ROOT / "sua_exploration/m1_compact_replication/results" / f"f{fold}_e23_continuation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    exit_codes: dict[str, int] = {}
    terminals: dict[str, Any] = {}
    for key in ("b0", "b3s_zero4"):
        log_path = log_dir / f"{key}.log"
        command = list(receipt["commands"][key])
        with log_path.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, stdout=handle, stderr=subprocess.STDOUT, check=False)
        exit_codes[key] = proc.returncode
        if proc.returncode != 0:
            break
        terminals[key] = _validate_terminal(fold, key, f"m1_compact_{key}_f{fold}_s42_resume_e11_to_e23", log_path, parent_step, command)
    inventory_after = source_inventory(fold)
    need(inventory_after == receipt["source_inventory"], f"fold-{fold} code/data inventory changed during continuation")
    complete = exit_codes == {"b0": 0, "b3s_zero4": 0} and set(terminals) == {"b0", "b3s_zero4"}
    body = {"schema": SCHEMA_PREFIX.format(fold=fold) + "_execution_v2", "status": EXEC_STATUS.format(fold=fold) if complete else FAIL_STATUS.format(fold=fold), "launch_receipt": {"path": str(path.resolve()), "sha256": sha(path)}, "parent_e11_execution": receipt["parent_e11_execution"], "parent_e11_gate": receipt["parent_e11_gate"], "commands": receipt["commands"], "fixed_order": ["b0", "b3s_zero4"], "exit_codes": exit_codes, "terminal_checkpoints": terminals, "source_inventory_before": receipt["source_inventory"], "source_inventory_after": inventory_after, "target_metrics_not_read_by_launcher": True, "target_opened_by_training": False, "created_at_epoch": time.time()}
    write_immutable(state_path, body)
    return 0 if complete else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=(1, 2))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-root-launch")
    args = parser.parse_args()
    need(args.prepare ^ args.execute, "choose exactly one --prepare/--execute")
    if args.prepare:
        body = prepare(args.fold)
        print(json.dumps({"status": body["status"], "path": str(receipt_path(args.fold).resolve()), "sha256": sha(receipt_path(args.fold))}, sort_keys=True))
    else:
        raise SystemExit(execute(args.fold, args.allow_root_launch or ""))


if __name__ == "__main__":
    main()
