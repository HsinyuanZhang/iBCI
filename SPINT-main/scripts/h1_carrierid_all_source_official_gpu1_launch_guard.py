#!/usr/bin/env python3
"""One-shot fail-closed GPU1 launch guard for the H1 all-source candidate.

The guard is intentionally separate from the H-C0 workers.  It waits forever
for both GPU1 H-C0 terminal checkpoints, verifies their immutable provenance,
verifies the frozen all-source asset/launch receipt and current code hashes,
records a nonce-bound launch reservation atomically, and starts one fresh
``h1_all_source_gpu1`` tmux session.  It never restarts or signals a training
process and refuses to overwrite any launch marker or log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import uuid

import torch

# Resolve the repository before importing project modules when this file is
# executed directly by its path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.h1_m4_cce_contract import canonical_sha256, sha256_file, write_immutable_json
from src.models.h1_carrierid_all_source_official_module import ALL_SOURCE_CHECKPOINT_SCHEMA


ART = ROOT / "pilot_artifacts/h1_carrierid_all_source_official_v1"
ASSET = ART / "H1_CARRIERID_ALL_SOURCE_ASSET_PREFLIGHT_v1.json"
PREPARED_RECEIPT = ART / "H1_CARRIERID_ALL_SOURCE_LAUNCH_RECEIPT_v1.json"
EXECUTION_RECEIPT = ART / "H1_CARRIERID_ALL_SOURCE_GPU1_EXECUTION_LAUNCH_v1.json"
LOG_DIR = ART / "logs"
LOG = LOG_DIR / "h1_all_source_gpu1.log"
START = LOG_DIR / "h1_all_source_gpu1_start.txt"
RUN_ROOT = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hc0_fivedate/gpu_runs"
GPU1_LOG = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hc0_fivedate/logs/gpu1.log"
ALL_SOURCE_RUNS = ROOT / "logs/h1_carrierid_all_source_official/runs"
GUARD_SESSION = "h1_all_source_gpu1_watch"
TARGET_SESSION = "h1_all_source_gpu1"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
ASSET_REL = "/home/xinyuan/Work_host/SPINT/SPINT-main/pilot_artifacts/h1_carrierid_all_source_official_v1/H1_CARRIERID_ALL_SOURCE_ASSET_PREFLIGHT_v1.json"

CODE_FILES = {
    "launcher": ROOT / "scripts/h1_carrierid_all_source_official_launcher.py",
    "data": ROOT / "src/data/h1_carrierid_all_source_official.py",
    "model": ROOT / "src/models/h1_carrierid_all_source_official_module.py",
    "experiment": ROOT / "configs/experiment/h1_carrierid_all_source_official.yaml",
    "data_config": ROOT / "configs/data/falcon_h1_carrierid_all_source_official.yaml",
    "model_config": ROOT / "configs/model/falcon_h1_carrierid_all_source_official.yaml",
}
DATES = ("19250113", "19250119")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _immutable(path: Path) -> bool:
    return path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _tmux_alive(session: str) -> bool:
    # ``tmux has-session -t NAME`` accepts unique prefixes.  The guard itself
    # is named ``h1_all_source_gpu1_watch``; prefix matching would therefore
    # falsely report the target ``h1_all_source_gpu1`` as already running.
    result = subprocess.run(["tmux", "list-sessions", "-F", "#S"], text=True, capture_output=True)
    return session in {line.strip() for line in result.stdout.splitlines()}


def _gpu1_compute_pids() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _current_code_hashes() -> dict[str, str]:
    _need(all(path.is_file() for path in CODE_FILES.values()), "all-source code/config file missing")
    return {name: sha256_file(path) for name, path in CODE_FILES.items()}


def _planned_command() -> list[str]:
    return [
        PYTHON,
        str(ROOT / "src/train.py"),
        "experiment=h1_carrierid_all_source_official",
        f"official_candidate.asset_manifest_path={ASSET_REL}",
        "ckpt_path=null",
        "test=false",
    ]


def _validate_static() -> tuple[dict, dict, dict[str, str], list[str]]:
    _need(_immutable(ASSET), f"all-source asset manifest is not immutable: {ASSET}")
    _need(_immutable(PREPARED_RECEIPT), f"prepared launch receipt is not immutable: {PREPARED_RECEIPT}")
    _need(not EXECUTION_RECEIPT.exists(), f"refusing to overwrite execution launch marker: {EXECUTION_RECEIPT}")
    _need(not LOG.exists() and not START.exists(), "refusing to reuse an existing all-source log/start marker")
    if _tmux_alive(TARGET_SESSION):
        raise RuntimeError(f"refusing duplicate all-source tmux session: {TARGET_SESSION}")
    receipt = json.loads(PREPARED_RECEIPT.read_text(encoding="utf-8"))
    _need(receipt.get("schema") == "h1_carrierid_all_public_source_official_launch_receipt_v1", "prepared receipt schema mismatch")
    _need(receipt.get("status") == "PASS_ALL_SOURCE_HC_PREPARED_NOT_LAUNCHED", "prepared receipt already consumed or failed")
    _need(receipt.get("planned_command") == _planned_command(), "prepared receipt command mismatch")
    _need(receipt.get("candidate", {}).get("asset_manifest_path") == ASSET_REL, "prepared receipt asset path mismatch")
    _need(receipt.get("candidate", {}).get("asset_manifest_sha256") == sha256_file(ASSET), "prepared receipt asset SHA mismatch")
    current_hashes = _current_code_hashes()
    _need(receipt.get("code_sha256") == current_hashes, "current all-source code hashes differ from prepared receipt")
    asset = json.loads(ASSET.read_text(encoding="utf-8"))
    _need(asset.get("schema") == "h1_carrierid_all_public_source_assets_v1", "asset manifest schema mismatch")
    _need(asset.get("status") == "PASS_ALL_PUBLIC_HELDIN_ASSETS_FROZEN_NO_GPU_NO_FORMAL", "asset manifest status mismatch")
    _need(asset.get("scope", {}).get("formal_test_labels_opened") == 0 and asset.get("scope", {}).get("evalai_accessed") is False, "asset manifest formal/EvalAI scope mismatch")
    preexisting = sorted(str(path.relative_to(ALL_SOURCE_RUNS)) for path in ALL_SOURCE_RUNS.iterdir() if path.is_dir()) if ALL_SOURCE_RUNS.is_dir() else []
    return receipt, asset, current_hashes, preexisting


def _validate_hc0_checkpoint(date: str) -> dict:
    run = RUN_ROOT / date
    ckpt = run / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    config = run / ".hydra/config.yaml"
    pair = ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase2" / f"H1_CARRIERID_DATE_LODO_PHASE2_{date}_PAIR_CPU_PREFLIGHT_v1.json"
    _need(ckpt.is_file() and config.is_file() and _immutable(pair), f"{date}: terminal checkpoint/config/pair missing or pair mutable")
    pair_body = json.loads(pair.read_text(encoding="utf-8"))
    _need(pair_body.get("schema") == "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1", f"{date}: pair schema mismatch")
    _need(pair_body.get("status") == "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED", f"{date}: pair status mismatch")
    _need(pair_body.get("outer_date") == date and pair_body.get("source_binding", {}).get("outer_date") == date, f"{date}: pair date binding mismatch")
    _need(pair_body.get("source_binding", {}).get("target_recordings_opened") == 0 and pair_body.get("source_binding", {}).get("target_bytes_read") == 0, f"{date}: pair target scope drift")
    _need(pair_body.get("source_binding", {}).get("warm_start_forbidden") is True, f"{date}: pair warm-start drift")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    _need(isinstance(payload, dict), f"{date}: checkpoint not a mapping")
    metadata = payload.get("h1_carrierid_date_lodo_phase2")
    expected = {
        "schema": "h1_carrierid_date_lodo_phase2_terminal_checkpoint_v1", "arm": "H-C", "outer_date": date,
        "fresh_seed": 42, "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
        "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection", "target_optimizer_steps": 0,
        "target_backward_steps": 0, "checkpoint_warm_start": False,
    }
    _need(isinstance(metadata, dict) and all(metadata.get(key) == value for key, value in expected.items()), f"{date}: checkpoint metadata mismatch")
    _need(metadata.get("phase2_source_binding_sha256") == pair_body.get("source_binding_sha256"), f"{date}: checkpoint/pair source binding mismatch")
    _need(metadata.get("phase1_source_manifest_sha256") == pair_body.get("source_binding", {}).get("source_manifest_sha256"), f"{date}: checkpoint/source manifest mismatch")
    _need(metadata.get("config_sha256") == hashlib.sha256(config.read_bytes()).hexdigest(), f"{date}: checkpoint/config SHA mismatch")
    state = payload.get("state_dict")
    _need(isinstance(state, dict) and state, f"{date}: checkpoint state missing")
    for name, tensor in state.items():
        _need(torch.is_tensor(tensor), f"{date}: non-tensor state {name}")
        if tensor.is_floating_point():
            _need(bool(torch.isfinite(tensor).all()), f"{date}: nonfinite state {name}")
    return {
        "date": date, "checkpoint_path": str(ckpt.resolve()), "checkpoint_sha256": sha256_file(ckpt),
        "checkpoint_size_bytes": ckpt.stat().st_size, "config_path": str(config.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(), "pair_path": str(pair.resolve()),
        "pair_sha256": sha256_file(pair), "phase2_source_binding_sha256": metadata["phase2_source_binding_sha256"],
        "phase1_source_manifest_sha256": metadata["phase1_source_manifest_sha256"], "state_tensor_count": len(state),
    }


def _wait_and_launch() -> None:
    while True:
        if _tmux_alive(TARGET_SESSION):
            raise RuntimeError(f"duplicate target session already exists: {TARGET_SESSION}")
        # A worker session is allowed to remain active while this guard waits.
        if _tmux_alive("h1_hc0_gpu1"):
            time.sleep(30)
            continue
        checkpoint_paths = [RUN_ROOT / date / "checkpoints/fixed_epoch50/epoch_049.ckpt" for date in DATES]
        if not all(path.is_file() for path in checkpoint_paths):
            if GPU1_LOG.is_file():
                tail = GPU1_LOG.read_text(errors="ignore")[-250000:]
                if any(token in tail for token in ("Traceback", "out of memory", "CUDA error", "did not create required e49")):
                    raise RuntimeError("GPU1 H-C0 worker exited with a terminal error before both e49 checkpoints")
            time.sleep(30)
            continue
        # Once both files exist, an integrity failure is terminal: never keep
        # retrying a potentially tampered or mismatched checkpoint.
        try:
            ckpts = [_validate_hc0_checkpoint(date) for date in DATES]
        except Exception as error:
            raise RuntimeError(f"GPU1 H-C0 checkpoint audit failed: {error}") from error
        if _gpu1_compute_pids():
            time.sleep(30)
            continue
        receipt, asset, code_hashes, preexisting = _validate_static()
        # Re-check all gates immediately before publishing the one-shot marker.
        _need(not _tmux_alive("h1_hc0_gpu1"), "GPU1 H-C0 worker reappeared before launch")
        _need(not _gpu1_compute_pids(), "GPU1 compute process appeared before launch")
        nonce = uuid.uuid4().hex
        start_time_ns = time.time_ns()
        command = _planned_command()
        command_text = "CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 " + " ".join(command)
        reservation = {
            "schema": "h1_carrierid_all_public_source_gpu1_execution_launch_v1",
            "status": "PASS_ALL_SOURCE_GPU1_LAUNCH_RESERVED",
            "nonce": nonce,
            "start_time_ns": start_time_ns,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "tmux_session": TARGET_SESSION,
            "guard_session": GUARD_SESSION,
            "workdir": str(ROOT),
            "expected_command": command,
            "expected_command_text": command_text,
            "asset_manifest": {"path": str(ASSET.resolve()), "sha256": sha256_file(ASSET)},
            "prepared_launch_receipt": {"path": str(PREPARED_RECEIPT.resolve()), "sha256": sha256_file(PREPARED_RECEIPT)},
            "prepared_candidate_config_sha256": receipt["candidate"]["config_sha256"],
            "current_code_sha256": code_hashes,
            "launch_guard_sha256": sha256_file(Path(__file__).resolve()),
            "preexisting_all_source_run_dirs": preexisting,
            "expected_new_run_root": str(ALL_SOURCE_RUNS.resolve()),
            "h_c0_gpu1_terminal_checkpoints": ckpts,
            "scope": {"formal_test_labels_opened": 0, "target_optimizer_steps": 0, "target_backward_steps": 0, "evalai_submission_authorized": False},
        }
        write_immutable_json(EXECUTION_RECEIPT, reservation)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _need(not LOG.exists() and not START.exists(), "launch log/start marker appeared after reservation")
        start_text = "\n".join([
            f"nonce={nonce}", f"start_time_ns={start_time_ns}", f"start_time={reservation['start_time']}",
            f"workdir={ROOT}", f"tmux_session={TARGET_SESSION}", f"guard_session={GUARD_SESSION}",
            f"command={command_text}", f"stdout_stderr_log={LOG}", f"execution_receipt={EXECUTION_RECEIPT}",
        ]) + "\n"
        temporary = START.parent / f".{START.name}.{nonce}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(start_text); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, START)
        temporary.unlink(missing_ok=True)
        tmux_command = (
            f"cd {ROOT!s} && CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 {command[0]} "
            f"{command[1]} {command[2]} {command[3]} {command[4]} {command[5]} 2>&1 | tee {LOG!s}"
        )
        result = subprocess.run(["tmux", "new-session", "-d", "-s", TARGET_SESSION, tmux_command], capture_output=True, text=True)
        _need(result.returncode == 0 and _tmux_alive(TARGET_SESSION), f"tmux launch failed: {result.stderr.strip()}")
        print(json.dumps({"launch": "PASS", "execution_receipt": str(EXECUTION_RECEIPT), "start": str(START), "log": str(LOG), "nonce": nonce, "start_time_ns": start_time_ns}, sort_keys=True), flush=True)
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate static receipt/code/asset gates without waiting or launching")
    args = parser.parse_args()
    receipt, asset, code_hashes, preexisting = _validate_static()
    if args.dry_run:
        print(json.dumps({"dry_run": "PASS", "prepared_receipt_sha256": sha256_file(PREPARED_RECEIPT), "asset_sha256": sha256_file(ASSET), "code_sha256": code_hashes, "preexisting_all_source_run_dirs": preexisting}, sort_keys=True), flush=True)
        return
    _wait_and_launch()


if __name__ == "__main__":
    main()
