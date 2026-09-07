#!/usr/bin/env python3
"""Fail-closed launcher for the M1 compact-B3S fold-0/e23 pair.

This module deliberately has two separate phases:

``--prepare``
    Revalidates the immutable proposal, authority, resume checkpoints, source
    tree, data tree, and environment, then writes a read-only launch receipt.
    No CUDA import, Trainer construction, or subprocess is performed.

``--execute``
    Revalidates the read-only receipt and executes the *exact* two argv vectors
    recorded in the proposal, in the fixed order B0 then B3S.  The commands'
    output is redirected to private logs; this launcher never parses a target
    score and never selects a checkpoint from an intermediate score.  A small
    terminal index records only process status and terminal checkpoint paths.

The script is intended to run on the 5070Ti host.  It does not modify the
proposal or forward-authority files and refuses to reuse a prior launch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
PROPOSAL = ROOT / "sua_exploration/m1_compact_replication/proposals/M1_COMPACT_B3S_GPU_PROPOSALS_v1.json"
AUTHORITY = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F0_S42_FORWARD_AUTHORITY_v1.json"
DATA_BINDING = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F0_S42_E23_DATA_BINDING_v1.json"
EXPECTED_PROPOSAL_SHA = "403d12e613fc3937580db718ae9cef9d78e1c6ed1680cb4fcc459c9305896fb4"
EXPECTED_AUTHORITY_SHA = "7ec281631b8e96d0311e029ca227fc7c47b01d9d095f3015399952de2d1f315a"
SCHEMA = "m1_compact_b3s_f0_s42_e23_pair_launch_v1"
PREPARED_STATUS = "PASS_M1_COMPACT_B3S_F0_E23_LAUNCH_PREPARED_NOT_LAUNCHED"
STARTED_STATUS = "PASS_M1_COMPACT_B3S_F0_E23_LAUNCH_STARTED"
TERMINAL_STATUS = "PASS_M1_COMPACT_B3S_F0_E23_PAIR_TERMINAL"
EXPECTED_LIGHTNING = "2.6.5"
EXPECTED_SEED = 42
EXPECTED_EPOCH = 11
EXPECTED_STEP = 59_412
EXPECTED_TERMINAL_EPOCH = 23
EXPECTED_TERMINAL_STEP = 118_824

CODE_FILES = (
    "streaming_calibration_exp/src/train.py",
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


class LaunchContractError(RuntimeError):
    """A launch prerequisite or terminal invariant was not satisfied."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchContractError(message)


def sha256_file(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing or symlinked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchContractError(f"{label} unreadable: {path}") from exc
    need(isinstance(value, dict), f"{label} must be an object")
    return value


def _check_immutable(path: Path, label: str, expected_sha: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing")
    need(sha256_file(path) == expected_sha, f"{label} SHA drift")
    need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{label} mode is not 0444")
    return read_json(path, label)


def _validate_authority(body: Mapping[str, Any]) -> None:
    need(body.get("schema") == "m1_compact_b3s_fold0_forward_authority_v1", "authority schema drift")
    need(body.get("status") == "PASS_M1_COMPACT_B3S_FOLD0_FORWARD_ONLY_AUTHORITY", "authority status drift")
    scope = body.get("scope", {})
    need(scope.get("task") == "m1" and scope.get("fold") == 0 and scope.get("seed") == EXPECTED_SEED, "authority scope drift")
    for field in ("formal_opened", "minival_opened", "evalai_opened", "target_checkpoint_selection"):
        need(scope.get(field) is False, f"authority {field} must be false")
    need(scope.get("target_optimizer_steps") == 0 and scope.get("target_backward_steps") == 0, "authority target adaptation drift")
    need(body.get("canonical_content_sha256") == canonical_sha({k: v for k, v in body.items() if k != "canonical_content_sha256"}), "authority canonical hash drift")


def _validate_checkpoint(path: Path, expected_sha: str, expected_variant: str) -> dict[str, Any]:
    """Validate the e11 parent without importing CUDA or Lightning."""
    need(sha256_file(path) == expected_sha, f"{expected_variant} parent checkpoint SHA drift")
    # Use the configured environment's Python/torch only at execution-time on
    # the remote host.  Keeping preparation JSON-only makes the static phase
    # safe on a CPU-only root host.
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == EXPECTED_EPOCH, f"{expected_variant} parent epoch drift")
    need(payload.get("global_step") == EXPECTED_STEP, f"{expected_variant} parent global_step drift")
    need(payload.get("pytorch-lightning_version") == EXPECTED_LIGHTNING, f"{expected_variant} Lightning version drift")
    need(isinstance(payload.get("state_dict"), Mapping) and payload["state_dict"], f"{expected_variant} parent state missing")
    need(isinstance(payload.get("optimizer_states"), list) and len(payload["optimizer_states"]) == 1, f"{expected_variant} parent optimizer drift")
    need(payload.get("lr_schedulers") == [], f"{expected_variant} parent scheduler drift")
    loops = payload.get("loops")
    need(isinstance(loops, Mapping) and isinstance(loops.get("fit_loop"), Mapping), f"{expected_variant} parent loop state missing")
    fit = loops["fit_loop"]
    need(fit.get("epoch_loop.batch_progress", {}).get("total", {}).get("completed") == EXPECTED_STEP, f"{expected_variant} batch progress drift")
    need(fit.get("epoch_progress", {}).get("total", {}).get("processed") == 12, f"{expected_variant} epoch progress drift")
    callbacks = payload.get("callbacks")
    need(isinstance(callbacks, Mapping) and len(callbacks) == 1 and "every_n_epochs': 12" in str(next(iter(callbacks))), f"{expected_variant} callback state drift")
    return {
        "path": str(path.resolve()),
        "sha256": expected_sha,
        "variant": expected_variant,
        "epoch": EXPECTED_EPOCH,
        "global_step": EXPECTED_STEP,
        "pytorch_lightning_version": payload.get("pytorch-lightning_version"),
        "optimizer_state_count": len(payload["optimizer_states"][0].get("state", {})),
        "strict_resume_possible": True,
    }


def _proposal_commands(proposal: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    cont = proposal.get("fold0_e23_continuation", {})
    need(cont.get("launched") is False, "proposal is already marked launched")
    need(cont.get("evaluation") == "one terminal e23 target query per arm; no intermediate target metric", "proposal evaluation contract drift")
    gate = cont.get("gate", {})
    need(gate.get("threshold") == -0.03 and gate.get("paired_same_ordered_query_required") is True, "proposal gate drift")
    commands = cont.get("commands")
    need(isinstance(commands, Mapping), "proposal command map missing")
    b0, b3s = commands.get("b0"), commands.get("b3s_zero4")
    need(isinstance(b0, list) and isinstance(b3s, list), "proposal command vectors malformed")
    for argv, label in ((b0, "B0"), (b3s, "B3S")):
        need(all(isinstance(item, str) for item in argv), f"{label} argv is not string-only")
        need("trainer.min_epochs=24" in argv and "trainer.max_epochs=24" in argv, f"{label} e23 epoch override drift")
        need("test=true" in argv and "trainer.limit_val_batches=0" in argv and "trainer.num_sanity_val_steps=0" in argv, f"{label} evaluation override drift")
    need(b0[3] == "run_id=m1_compact_b0_f0_s42_resume_e11_to_e23", "B0 run id drift")
    need(b3s[3] == "run_id=m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23", "B3S run id drift")
    return list(b0), list(b3s)


def _source_inventory() -> dict[str, Any]:
    files: dict[str, str] = {}
    for rel in CODE_FILES:
        path = ROOT / rel
        files[rel] = sha256_file(path)
    git_head = "unknown"
    git_status = "unavailable"
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception as exc:  # noqa: BLE001
        git_status = repr(exc)
    return {
        "root": str(ROOT.resolve()),
        "files": files,
        "git_head": git_head,
        "git_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
        "git_status_line_count": len(git_status.splitlines()),
    }


def _env_inventory() -> dict[str, Any]:
    python = subprocess.check_output([sys.executable, "--version"], text=True).strip()
    lightning = "unknown"
    torch_version = "unknown"
    cuda = False
    try:
        import lightning
        import torch

        lightning = str(getattr(lightning, "__version__", "unknown"))
        torch_version = str(torch.__version__)
        cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        lightning = f"import_error:{exc}"
    nvidia = ""
    try:
        nvidia = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        nvidia = f"unavailable:{exc}"
    return {
        "hostname": platform.node(),
        "python": python,
        "lightning": lightning,
        "torch": torch_version,
        "torch_cuda_available": cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": nvidia,
    }


def _data_inventory(proposal: Mapping[str, Any]) -> dict[str, Any]:
    # The resolved M1 config is immutable in the command contract; bind the
    # data directory and only lightweight filesystem facts, not the data bytes.
    command = proposal["fold0_e23_continuation"]["commands"]["b0"]
    need("data.source_session_names=[ses-20120926,ses-20120927,ses-20120928]" in command, "source session binding drift")
    data_root = ROOT / "SPINT-main/data/000941"
    need(data_root.is_dir(), f"M1 data root missing: {data_root}")
    files = sorted(path.relative_to(data_root).as_posix() for path in data_root.rglob("*") if path.is_file())
    need(files, "M1 data root is empty")
    file_digest = hashlib.sha256("\n".join(files).encode()).hexdigest()
    byte_hashes = {rel: sha256_file(data_root / rel) for rel in files}
    return {
        "path": str(data_root.resolve()),
        "file_count": len(files),
        "relative_file_list_sha256": file_digest,
        "files_sha256": byte_hashes,
    }


def write_data_binding(path: Path = DATA_BINDING) -> dict[str, Any]:
    """Write a separate immutable byte-level binding for all M1 source files.

    The launch receipt is intentionally immutable once prepared.  This
    supplement therefore carries the stronger byte hashes requested by the
    terminal receipt without mutating a running launch's contract.
    """
    proposal = _check_immutable(PROPOSAL, "proposal", EXPECTED_PROPOSAL_SHA)
    authority = _check_immutable(AUTHORITY, "authority", EXPECTED_AUTHORITY_SHA)
    inventory = _data_inventory(proposal)
    body: dict[str, Any] = {
        "schema": "m1_compact_b3s_f0_s42_e23_data_binding_v1",
        "status": "PASS_M1_COMPACT_B3S_F0_E23_DATA_BYTE_BINDING",
        "proposal_sha256": EXPECTED_PROPOSAL_SHA,
        "authority_sha256": EXPECTED_AUTHORITY_SHA,
        "scope": {"task": "m1", "fold": 0, "seed": EXPECTED_SEED, "target_session": "ses-20120924", "source_sessions": ["ses-20120926", "ses-20120927", "ses-20120928"]},
        "data": inventory,
    }
    body["canonical_content_sha256"] = canonical_sha(body)
    _write_immutable(path.resolve(), body)
    return body


def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp, 0o444)
        os.replace(temp, path)
        need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"{path} mode drift")
        return sha256_file(path)
    finally:
        if temp.exists():
            temp.unlink()


def prepare(receipt: Path) -> dict[str, Any]:
    proposal = _check_immutable(PROPOSAL, "proposal", EXPECTED_PROPOSAL_SHA)
    authority = _check_immutable(AUTHORITY, "authority", EXPECTED_AUTHORITY_SHA)
    _validate_authority(authority)
    b0_argv, b3s_argv = _proposal_commands(proposal)
    resume = proposal["fold0_e23_continuation"]["resume_validation"]
    b0_parent = Path(str(resume["b0"]["path"]))
    b3s_parent = Path(str(resume["b3s_zero4"]["path"]))
    b0_info = _validate_checkpoint(b0_parent, str(resume["b0"]["sha256"]), "B0")
    b3s_info = _validate_checkpoint(b3s_parent, str(resume["b3s_zero4"]["sha256"]), "B3S")
    need(b0_info["variant"] == "B0" and b3s_info["variant"] == "B3S", "resume variant drift")
    artifact_root = ROOT / "outputs/streaming_calibration"
    run_ids = {
        "b0": "m1_compact_b0_f0_s42_resume_e11_to_e23",
        "b3s_zero4": "m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23",
    }
    for run_id in run_ids.values():
        # The artifact writer adds a timestamp suffix; a pre-existing prefix
        # would make the pair ambiguous, so fail closed before launch.
        need(not any(artifact_root.glob(f"{run_id}_f0_s42_*")), f"prior artifact with run id prefix exists: {run_id}")
    need(not any((ROOT / "logs").glob("*m1_compact_b0_f0_s42_resume_e11_to_e23*")), "prior B0 log run exists")
    need(not any((ROOT / "logs").glob("*m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23*")), "prior B3S log run exists")
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": PREPARED_STATUS,
        "proposal": {"path": str(PROPOSAL.resolve()), "sha256": EXPECTED_PROPOSAL_SHA},
        "authority": {"path": str(AUTHORITY.resolve()), "sha256": EXPECTED_AUTHORITY_SHA, "status": authority["status"]},
        "scope": {
            "task": "m1", "fold": 0, "seed": EXPECTED_SEED,
            "outer_target": "ses-20120924",
            "source_sessions": ["ses-20120926", "ses-20120927", "ses-20120928"],
            "support_trials": [0, 10], "query_trials": [10, 210],
            "formal_or_minival_or_heldout": False,
            "target_backward_steps": 0, "target_optimizer_steps": 0,
            "target_checkpoint_selection": False,
        },
        "resume_parents": {"b0": b0_info, "b3s_zero4": b3s_info},
        "commands": {"b0": b0_argv, "b3s_zero4": b3s_argv},
        "source_inventory": _source_inventory(),
        "data_inventory": _data_inventory(proposal),
        "environment": _env_inventory(),
        "execution": {
            "fixed_order": ["b0", "b3s_zero4"],
            "target_metrics_not_read_by_launcher": True,
            "intermediate_checkpoint_selection": False,
            "tmux_required": True,
            "gpu_index": 0,
        },
    }
    body["canonical_content_sha256"] = canonical_sha(body)
    _write_immutable(receipt, body)
    return body


def _load_receipt(receipt: Path) -> dict[str, Any]:
    body = _check_immutable(receipt, "launch receipt", sha256_file(receipt))
    expected = body.get("canonical_content_sha256")
    need(expected == canonical_sha({k: v for k, v in body.items() if k != "canonical_content_sha256"}), "launch receipt canonical hash drift")
    need(body.get("schema") == SCHEMA, "launch receipt schema drift")
    need(body.get("status") == PREPARED_STATUS, "launch receipt is not prepared")
    need(body.get("execution", {}).get("target_metrics_not_read_by_launcher") is True, "launcher metric policy drift")
    return body


def _terminal_checkpoint_index(run_id: str, *, expected_variant: str) -> dict[str, Any]:
    """Find only the fixed e23 checkpoint; never reads its metrics."""
    candidates = []
    for base in (ROOT / "logs", ROOT / "outputs/streaming_calibration"):
        if not base.exists():
            continue
        for path in base.rglob("*.ckpt"):
            if run_id not in str(path):
                continue
            if "fixed_last" not in str(path) and "/checkpoints/last" not in str(path):
                continue
            candidates.append(path)
    need(candidates, f"no terminal checkpoint found for {run_id}")
    # Inspect only checkpoint metadata and state, not predictions/metrics.
    import torch

    terminal = []
    for path in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("epoch") == EXPECTED_TERMINAL_EPOCH and payload.get("global_step") == EXPECTED_TERMINAL_STEP:
            terminal.append((path, _terminal_payload_summary(payload, expected_variant=expected_variant)))
    need(len(terminal) == 1, f"expected exactly one e23 checkpoint for {run_id}, found {len(terminal)}")
    path, summary = terminal[0]
    return {"path": str(path.resolve()), "sha256": sha256_file(path), **summary}


def _terminal_payload_summary(payload: Mapping[str, Any], *, expected_variant: str) -> dict[str, Any]:
    """Validate the complete Lightning continuation state at epoch 23."""
    need(payload.get("epoch") == EXPECTED_TERMINAL_EPOCH, f"{expected_variant} terminal epoch drift")
    need(payload.get("global_step") == EXPECTED_TERMINAL_STEP, f"{expected_variant} terminal global_step drift")
    need(payload.get("pytorch-lightning_version") == EXPECTED_LIGHTNING, f"{expected_variant} terminal Lightning drift")
    need(isinstance(payload.get("state_dict"), Mapping) and payload["state_dict"], f"{expected_variant} terminal state missing")
    optimizers = payload.get("optimizer_states")
    need(isinstance(optimizers, list) and len(optimizers) == 1, f"{expected_variant} terminal optimizer count drift")
    optimizer = optimizers[0]
    groups = optimizer.get("param_groups")
    slots = optimizer.get("state")
    need(isinstance(groups, list) and len(groups) == 1, f"{expected_variant} terminal param groups drift")
    need(isinstance(slots, Mapping) and slots, f"{expected_variant} terminal optimizer state missing")
    group = groups[0]
    need(float(group.get("lr")) == 1.0e-4 and float(group.get("weight_decay")) == 0.0, f"{expected_variant} terminal optimizer hyperparameter drift")
    steps: set[int] = set()
    for slot in slots.values():
        need(isinstance(slot, Mapping), f"{expected_variant} terminal optimizer slot malformed")
        step = slot.get("step")
        if hasattr(step, "detach"):
            step = step.detach().cpu().item()
        steps.add(int(step))
    need(steps == {EXPECTED_TERMINAL_STEP}, f"{expected_variant} terminal optimizer step drift: {sorted(steps)}")
    loops = payload.get("loops")
    need(isinstance(loops, Mapping) and isinstance(loops.get("fit_loop"), Mapping), f"{expected_variant} terminal loop state missing")
    fit = loops["fit_loop"]
    batch_total = fit.get("epoch_loop.batch_progress", {}).get("total", {})
    epoch_total = fit.get("epoch_progress", {}).get("total", {})
    need(batch_total.get("completed") == EXPECTED_TERMINAL_STEP, f"{expected_variant} terminal batch-progress drift")
    need(epoch_total.get("processed") == 24, f"{expected_variant} terminal epoch-progress drift")
    callbacks = payload.get("callbacks")
    need(isinstance(callbacks, Mapping) and len(callbacks) == 1, f"{expected_variant} terminal callback state drift")
    callback_key = next(iter(callbacks))
    need("every_n_epochs': 12" in str(callback_key), f"{expected_variant} fixed callback state missing")
    return {
        "epoch": EXPECTED_TERMINAL_EPOCH,
        "global_step": EXPECTED_TERMINAL_STEP,
        "variant": expected_variant,
        "pytorch_lightning_version": EXPECTED_LIGHTNING,
        "optimizer": {"count": 1, "param_group_count": 1, "state_count": len(slots), "lr": float(group["lr"]), "weight_decay": float(group["weight_decay"]), "all_state_steps": sorted(steps)},
        "lr_schedulers": 0,
        "fit_loop": {"batch_completed": batch_total.get("completed"), "epoch_processed": epoch_total.get("processed")},
        "callback_key": str(callback_key),
        "strict_terminal_state": True,
    }


def _restore_evidence(log_path: Path, parent_path: Path) -> dict[str, Any]:
    need(log_path.is_file(), f"run log missing: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if "Restored all states from the checkpoint at" in line]
    matching = [line for line in lines if str(parent_path) in line]
    need(matching, f"missing strict restore evidence for {parent_path}")
    line = matching[-1]
    return {"present": True, "line_sha256": hashlib.sha256(line.encode()).hexdigest(), "line_count": len(matching)}


def execute(receipt: Path, *, state_path: Path) -> int:
    body = _load_receipt(receipt)
    state_path = state_path.resolve()
    need(not state_path.exists(), f"refusing existing execution state: {state_path}")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    logs = state_path.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # Recheck parent hashes from the immutable receipt immediately before GPU launch.
    for key, expected in (("b0", body["resume_parents"]["b0"]["sha256"]), ("b3s_zero4", body["resume_parents"]["b3s_zero4"]["sha256"])):
        path = Path(body["resume_parents"][key]["path"])
        need(sha256_file(path) == expected, f"{key} parent changed after prepare")
    started = {
        "schema": "m1_compact_b3s_f0_s42_e23_execution_v1",
        "status": STARTED_STATUS,
        "launch_receipt": {"path": str(receipt.resolve()), "sha256": sha256_file(receipt)},
        "started_at_epoch": time.time(),
        "commands": body["commands"],
        "fixed_order": ["b0", "b3s_zero4"],
        "target_metrics_not_read_by_launcher": True,
    }
    state_path.write_text(json.dumps(started, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    exit_codes: dict[str, int] = {}
    for key in ("b0", "b3s_zero4"):
        argv = body["commands"][key]
        log_path = logs / f"{key}.log"
        # The exact proposal argv is preserved.  CUDA visibility is an
        # environment binding, not an argv modification.
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("# exact argv: " + shlex.join(argv) + "\n")
            handle.flush()
            proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        exit_codes[key] = int(proc.returncode)
    terminal: dict[str, Any] = {}
    if exit_codes["b0"] == 0:
        terminal["b0"] = _terminal_checkpoint_index("m1_compact_b0_f0_s42_resume_e11_to_e23", expected_variant="B0")
        terminal["b0"]["resume_parent_sha256"] = body["resume_parents"]["b0"]["sha256"]
        terminal["b0"]["launch_command"] = body["commands"]["b0"]
        terminal["b0"]["log"] = {"path": str((logs / "b0.log").resolve()), "sha256": sha256_file(logs / "b0.log")}
        terminal["b0"]["restored_all_states"] = _restore_evidence(logs / "b0.log", Path(body["resume_parents"]["b0"]["path"]))
    if exit_codes["b3s_zero4"] == 0:
        terminal["b3s_zero4"] = _terminal_checkpoint_index("m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23", expected_variant="B3S")
        terminal["b3s_zero4"]["resume_parent_sha256"] = body["resume_parents"]["b3s_zero4"]["sha256"]
        terminal["b3s_zero4"]["launch_command"] = body["commands"]["b3s_zero4"]
        terminal["b3s_zero4"]["log"] = {"path": str((logs / "b3s_zero4.log").resolve()), "sha256": sha256_file(logs / "b3s_zero4.log")}
        terminal["b3s_zero4"]["restored_all_states"] = _restore_evidence(logs / "b3s_zero4.log", Path(body["resume_parents"]["b3s_zero4"]["path"]))
    result = {
        **started,
        "status": TERMINAL_STATUS if all(code == 0 for code in exit_codes.values()) and len(terminal) == 2 else "FAIL_M1_COMPACT_B3S_F0_E23_EXECUTION",
        "ended_at_epoch": time.time(),
        "exit_codes": exit_codes,
        "logs": {key: str((logs / f"{key}.log").resolve()) for key in exit_codes},
        "terminal_checkpoints": terminal,
    }
    state_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["status"] == TERMINAL_STATUS else 1


def enrich_terminal_state(receipt: Path, state_path: Path) -> dict[str, Any]:
    """Attach strict terminal-state/log evidence after an older runner exits.

    This is intentionally a post-run metadata operation.  It never launches
    a training subprocess and never reads a target metric.  The compatibility
    path is needed because a runner process already loaded before a source
    update cannot see newly added receipt fields.
    """
    body = _load_receipt(receipt)
    state = read_json(state_path, "pair execution state")
    need(state.get("status") == TERMINAL_STATUS, "pair is not terminal-success")
    need(state.get("exit_codes") == {"b0": 0, "b3s_zero4": 0}, "pair exit codes are not zero")
    logs = state_path.parent / "logs"
    enriched = dict(state)
    terminals: dict[str, Any] = {}
    for key, run_id, variant in (
        ("b0", "m1_compact_b0_f0_s42_resume_e11_to_e23", "B0"),
        ("b3s_zero4", "m1_compact_b3s_zero4_f0_s42_resume_e11_to_e23", "B3S"),
    ):
        terminal = _terminal_checkpoint_index(run_id, expected_variant=variant)
        parent_key = key
        log_path = logs / f"{key}.log"
        terminal["resume_parent_sha256"] = body["resume_parents"][parent_key]["sha256"]
        terminal["launch_command"] = body["commands"][key]
        terminal["log"] = {"path": str(log_path.resolve()), "sha256": sha256_file(log_path)}
        terminal["restored_all_states"] = _restore_evidence(log_path, Path(body["resume_parents"][parent_key]["path"]))
        terminals[key] = terminal
    enriched["terminal_checkpoints"] = terminals
    enriched["metadata_enriched_by"] = {"script": str(Path(__file__).resolve()), "script_sha256": sha256_file(Path(__file__)), "target_metrics_read": False}
    state_path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enrich-state", action="store_true")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--data-binding", type=Path)
    args = parser.parse_args()
    need(sum(bool(x) for x in (args.prepare, args.execute, args.enrich_state, args.data_binding is not None)) == 1, "choose exactly one launcher mode")
    if args.prepare:
        body = prepare(args.receipt.resolve())
        print(json.dumps({"status": body["status"], "receipt": str(args.receipt.resolve()), "sha256": sha256_file(args.receipt.resolve())}, sort_keys=True))
    elif args.execute:
        need(args.state is not None, "--execute requires --state")
        raise SystemExit(execute(args.receipt.resolve(), state_path=args.state))
    elif args.enrich_state:
        need(args.state is not None, "--enrich-state requires --state")
        enriched = enrich_terminal_state(args.receipt.resolve(), args.state.resolve())
        print(json.dumps({"status": enriched["status"], "state": str(args.state.resolve()), "target_metrics_read": False}, sort_keys=True))
    else:
        body = write_data_binding(args.data_binding.resolve())
        print(json.dumps({"status": body["status"], "output": str(args.data_binding.resolve()), "sha256": sha256_file(args.data_binding.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
