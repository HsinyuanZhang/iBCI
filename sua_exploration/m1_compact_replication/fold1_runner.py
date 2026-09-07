#!/usr/bin/env python3
"""Prepare (and, only with explicit root authorization, run) M1 fold-1.

Fold 1 is conditional on the sealed fold-0/e23 non-inferiority receipt.  The
default operation is a CPU-only dry-run that validates the source-only
teacher and writes a read-only staged receipt.  Execution is deliberately
guarded by ``--allow-root-launch FOLD1_APPROVED_BY_ROOT`` so preparing this
file cannot accidentally consume the 5070Ti.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
STREAM_ROOT = ROOT / "streaming_calibration_exp"
for _entry in (str(ROOT), str(STREAM_ROOT)):
    if _entry not in os.sys.path:
        os.sys.path.insert(0, _entry)
PROPOSAL = ROOT / "sua_exploration/m1_compact_replication/proposals/M1_COMPACT_B3S_GPU_PROPOSALS_v1.json"
GATE_MAIN = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F0_S42_E23_GATE_v1.json"
GATE_PULL = ROOT / "sua_exploration/m1_compact_replication/results/m1_e23_remote_pull_20260810_035000/receipts/M1_COMPACT_B3S_F0_S42_E23_GATE_v1.json"
PREFLIGHT = ROOT / "sua_exploration/results/m1_version_b_preflight/receipt_v2.json"
EXPECTED_PROPOSAL_SHA = "403d12e613fc3937580db718ae9cef9d78e1c6ed1680cb4fcc459c9305896fb4"
EXPECTED_GATE_SHA = "3fcc510cfec1f9ea4be123de4dbf8f27039158a3b65aa91212546c2d7047036c"
EXPECTED_PREFLIGHT_SHA = "02068703574309bcb50df249f58cf07c9cb91aca15447a1beaf926f150b9ee7c"
SCHEMA = "m1_compact_b3s_f1_s42_staged_v1"
STATUS_DRY = "PASS_M1_COMPACT_B3S_F1_S42_STAGED_NOT_LAUNCHED"
STATUS_STARTED = "PASS_M1_COMPACT_B3S_F1_S42_LAUNCH_STARTED"
EXPECTED_SEED = 42
EXPECTED_TEACHER_SHA = "b15edc9f66ff9acded6b78fe0b7a2041b359f8f831db22f3ab87f6083e8f50f3"
EXPECTED_MANIFEST_SHA = "9d67b56bdc354d1a8b0b7bee83b6c5203e566adb982496289a0122723e130551"
TEACHER = ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold1/runs/2026-08-06-16-20-17-515482_rid-m1_afc4_source_decoder_fold1_dev20_f1_s42/checkpoints/best_ckpt/epoch_019.ckpt"
TEACHER_MANIFEST = ROOT / "streaming_calibration_exp/logs/m1_afc4_source_decoder_fold1/runs/2026-08-06-16-20-17-515482_rid-m1_afc4_source_decoder_fold1_dev20_f1_s42/source_only_decoder_manifest.json"
DEFAULT_RECEIPT = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_STAGED_RECEIPT_v1.json"
DEFAULT_STATE = ROOT / "sua_exploration/m1_compact_replication/results/M1_COMPACT_B3S_F1_S42_EXECUTION_v1.json"


class Fold1ContractError(RuntimeError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Fold1ContractError(message)


def sha(path: Path) -> str:
    need(path.is_file() and not path.is_symlink(), f"missing/symlinked file: {path}")
    d = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path: Path, label: str) -> dict[str, Any]:
    need(path.is_file() and not path.is_symlink(), f"{label} missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"{label} not object")
    return value


def immutable(path: Path, body: Mapping[str, Any]) -> str:
    need(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(name)
    try:
        value = dict(body)
        value["canonical_content_sha256"] = canonical(value)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.chmod(0o444)
        tmp.replace(path)
        return sha(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_teacher() -> dict[str, Any]:
    need(sha(TEACHER) == EXPECTED_TEACHER_SHA, "fold1 teacher checkpoint SHA drift")
    need(sha(TEACHER_MANIFEST) == EXPECTED_MANIFEST_SHA, "fold1 teacher manifest SHA drift")
    manifest = read(TEACHER_MANIFEST, "fold1 teacher manifest")
    need(manifest.get("task") == "m1" and manifest.get("outer_fold") == 1, "fold1 teacher task/fold drift")
    need(manifest.get("outer_left_out") == "ses-20120926", "fold1 teacher target drift")
    need(manifest.get("source_only") is True and manifest.get("target_backpropagation") is False, "fold1 teacher source-only drift")
    for field in ("heldout_opened", "minival_opened", "formal", "evalai"):
        need(manifest.get(field) is False, f"fold1 teacher {field} drift")
    sources = ("ses-20120924", "ses-20120927", "ses-20120928")
    need(tuple(manifest.get("train_sessions", ())) == sources and manifest.get("validation_sessions") == [], "fold1 teacher source list drift")
    need("ses-20120926" not in manifest.get("source_files", {}), "fold1 target appears in teacher files")
    source_hashes = {}
    for session in ("ses-20120924", "ses-20120927", "ses-20120928"):
        spec = manifest.get("source_files", {}).get(session, {})
        source_path = Path(str(spec.get("path", "")))
        need(source_path.is_file() and sha(source_path) == spec.get("sha256"), f"fold1 teacher source NWB SHA drift: {session}")
        source_hashes[session] = {"path": str(source_path.resolve()), "sha256": str(spec["sha256"]), "bytes": source_path.stat().st_size}
    import torch
    payload = torch.load(TEACHER, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == 19, "fold1 teacher terminal epoch drift")
    return {"checkpoint": {"path": str(TEACHER.resolve()), "sha256": EXPECTED_TEACHER_SHA, "epoch": 19}, "manifest": {"path": str(TEACHER_MANIFEST.resolve()), "sha256": EXPECTED_MANIFEST_SHA}, "source_files": source_hashes, "outer_target": "ses-20120926", "source_sessions": list(sources), "source_only": True}


def validate_preflight() -> dict[str, Any]:
    need(sha(PREFLIGHT) == EXPECTED_PREFLIGHT_SHA, "M1 preflight SHA drift")
    body = read(PREFLIGHT, "M1 preflight")
    need(body.get("schema") == "m1_version_b_preflight_v2", "M1 preflight schema drift")
    scope = body.get("scope", {})
    need(scope.get("task") == "m1" and scope.get("target_backpropagation") is False and scope.get("target_query_values_read_by_preflight") is False, "M1 preflight target disclosure drift")
    need(scope.get("formal_test_opened") is False and scope.get("heldout_files_opened") is False and scope.get("minival_files_opened") is False, "M1 preflight scope drift")
    source_hashes = {}
    for session in ("ses-20120924", "ses-20120927", "ses-20120928"):
        spec = body.get("source_files", {}).get(session, {})
        path = Path(str(spec.get("path", "")))
        need(path.is_file() and sha(path) == spec.get("sha256"), f"M1 preflight source NWB SHA drift: {session}")
        source_hashes[session] = {"path": str(path.resolve()), "sha256": str(spec["sha256"]), "bytes": path.stat().st_size}
    return {"path": str(PREFLIGHT.resolve()), "sha256": EXPECTED_PREFLIGHT_SHA, "schema": body["schema"], "source_files": source_hashes, "scope": {"target_backpropagation": False, "target_query_values_read_by_preflight": False, "formal_test_opened": False, "heldout_files_opened": False, "minival_files_opened": False}}


def validate_gate() -> dict[str, Any]:
    path = GATE_MAIN if GATE_MAIN.is_file() else GATE_PULL
    need(sha(path) == EXPECTED_GATE_SHA, "fold0/e23 gate SHA drift")
    body = read(path, "fold0/e23 gate")
    need(body.get("status") == "PASS_M1_COMPACT_B3S_F0_E23_NONINFERIORITY", "fold0/e23 gate status is not PASS")
    need(float(body.get("metrics", {}).get("b3s_zero4_minus_b0")) >= -0.03, "fold0/e23 gate delta below threshold")
    need(body.get("scope", {}).get("formal_opened") is False and body.get("scope", {}).get("target_checkpoint_selection") is False, "fold0/e23 gate scope drift")
    need(body.get("canonical_content_sha256") == canonical({k: v for k, v in body.items() if k != "canonical_content_sha256"}), "fold0/e23 gate canonical hash drift")
    return {"path": str(path.resolve()), "sha256": EXPECTED_GATE_SHA, "status": body["status"], "delta": float(body["metrics"]["b3s_zero4_minus_b0"]), "threshold": -0.03}


def _commands(proposal: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    cells = proposal["folds1_2_cross_session"]["cells"]
    selected = [cell for cell in cells if cell.get("fold") == 1]
    need(len(selected) == 2, "proposal fold1 cell count drift")
    selected.sort(key=lambda cell: (0 if cell.get("arm") == "b0" else 1))
    need([cell.get("arm") for cell in selected] == ["b0", "b3s_zero4"], "proposal fold1 order drift")
    commands = [list(cell["argv"]) for cell in selected]
    for argv in commands:
        need("data.loso_fold=1" in argv and "seed=42" in argv and "trainer.min_epochs=12" in argv and "trainer.max_epochs=12" in argv, "fold1 command budget drift")
        need("test=true" in argv and "ckpt_path=null" in argv, "fold1 test/checkpoint policy drift")
        need("model.teacher_ckpt_path=" + str(TEACHER.resolve()) in argv, "fold1 teacher command binding drift")
        need("data.source_session_names=[ses-20120924,ses-20120927,ses-20120928]" in argv, "fold1 source session binding drift")
    need(commands[0][2] == "experiment=m1_version_b_hs_continuation" and commands[1][2] == "experiment=m1_version_b_c0", "fold1 experiment binding drift")
    need(commands[0][3] == "run_id=m1_compact_b0_f1_s42_fresh_e11" and commands[1][3] == "run_id=m1_compact_b3s_zero4_f1_s42_fresh_e11", "fold1 run id binding drift")
    for argv in commands:
        need("trainer.limit_val_batches=0" in argv and "trainer.num_sanity_val_steps=0" in argv, "fold1 validation suppression drift")
    return commands[0], commands[1]


def _inventory() -> dict[str, Any]:
    files = {}
    for rel in ("streaming_calibration_exp/src/train.py", "streaming_calibration_exp/src/data/m1_version_b_source_loso_datamodule.py", "streaming_calibration_exp/src/models/streaming_calibration_module.py", "streaming_calibration_exp/src/models/components/streaming_spint.py", "streaming_calibration_exp/src/models/components/streaming_encoders.py", "streaming_calibration_exp/configs/experiment/m1_version_b_hs_continuation.yaml", "streaming_calibration_exp/configs/experiment/m1_version_b_c0.yaml", "streaming_calibration_exp/configs/data/m1_version_b_source_loso.yaml", "streaming_calibration_exp/configs/model/m1_version_b_b0.yaml", "streaming_calibration_exp/configs/model/m1_version_b_b3s.yaml", "streaming_calibration_exp/configs/callbacks/m1_version_b_fixed_epoch.yaml"):
        files[rel] = sha(ROOT / rel)
    data = ROOT / "SPINT-main/data/000941"
    need(data.is_dir(), "M1 data root missing")
    data_files = {path.relative_to(data).as_posix(): sha(path) for path in sorted(data.rglob("*")) if path.is_file()}
    need(len(data_files) == 12, f"M1 data file count drift: {len(data_files)}")
    return {"code_files_sha256": files, "data_root": str(data.resolve()), "data_files_sha256": data_files, "data_file_count": len(data_files)}


def prior_artifact_exists(artifact_root: Path, run_base: str) -> bool:
    """Match the exact Hydra ``run_id + _f1_s42_`` prefix only."""
    prefix = run_base + "_f1_s42_"
    return any(path.is_dir() and path.name.startswith(prefix) for path in artifact_root.iterdir())


def prepare(output: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    proposal = read(PROPOSAL, "M1 compact proposal")
    need(sha(PROPOSAL) == EXPECTED_PROPOSAL_SHA, "proposal SHA drift")
    need(proposal.get("status") == "PASS_M1_COMPACT_B3S_PROPOSALS_PREPARED_NOT_LAUNCHED", "proposal status drift")
    gate = validate_gate()
    teacher = validate_teacher()
    preflight = validate_preflight()
    b0, b3s = _commands(proposal)
    artifact = ROOT / "outputs/streaming_calibration"
    for base in ("m1_compact_b0_f1_s42_fresh_e11", "m1_compact_b3s_zero4_f1_s42_fresh_e11"):
        need(not prior_artifact_exists(artifact, base), f"prior fold1 artifact exists: {base}_f1_s42_")
    body = {
        "schema": SCHEMA,
        "status": STATUS_DRY,
        "proposal": {"path": str(PROPOSAL.resolve()), "sha256": EXPECTED_PROPOSAL_SHA},
        "gate": gate,
        "teacher": teacher,
        "preflight": preflight,
        "scope": {"task": "m1", "fold": 1, "seed": EXPECTED_SEED, "outer_target": "ses-20120926", "source_sessions": ["ses-20120924", "ses-20120927", "ses-20120928"], "support_trials": [0, 10], "query_trials": [10, 210], "formal_or_minival_or_heldout": False, "target_backward_steps": 0, "target_optimizer_steps": 0, "target_checkpoint_selection": False},
        "commands": {"b0": b0, "b3s_zero4": b3s},
        "inventory": _inventory(),
        "execution": {"fixed_order": ["b0", "b3s_zero4"], "target_metrics_not_read_by_runner": True, "intermediate_checkpoint_selection": False, "launched": False, "gpu_index": 0},
    }
    immutable(output.resolve(), body)
    return body


def execute(receipt: Path, state: Path, authorization: str | None) -> int:
    need(authorization == "FOLD1_APPROVED_BY_ROOT", "fold1 launch requires explicit root authorization token")
    body = read(receipt, "fold1 staged receipt")
    need(body.get("status") == STATUS_DRY and body.get("canonical_content_sha256") == canonical({k: v for k, v in body.items() if k != "canonical_content_sha256"}), "fold1 staged receipt drift")
    proposal = read(PROPOSAL, "M1 compact proposal")
    need(sha(PROPOSAL) == body["proposal"]["sha256"] == EXPECTED_PROPOSAL_SHA, "fold1 proposal changed after prepare")
    b0, b3s = _commands(proposal)
    need(body["commands"] == {"b0": b0, "b3s_zero4": b3s}, "fold1 command vector changed after prepare")
    need(validate_gate()["sha256"] == body["gate"]["sha256"], "fold1 gate changed after prepare")
    need(validate_teacher() == body["teacher"], "fold1 teacher inputs changed after prepare")
    need(validate_preflight() == body["preflight"], "fold1 preflight changed after prepare")
    need(_inventory() == body["inventory"], "fold1 source/data inventory changed after prepare")
    artifact = ROOT / "outputs/streaming_calibration"
    for base in ("m1_compact_b0_f1_s42_fresh_e11", "m1_compact_b3s_zero4_f1_s42_fresh_e11"):
        need(not prior_artifact_exists(artifact, base), f"prior fold1 artifact exists at launch: {base}_f1_s42_")
    need(not state.exists(), "fold1 execution state already exists")
    state.write_text(json.dumps({"schema": "m1_compact_b3s_f1_s42_execution_v1", "status": STATUS_STARTED, "receipt_sha256": sha(receipt), "commands": body["commands"], "fixed_order": ["b0", "b3s_zero4"], "target_metrics_not_read_by_runner": True}, indent=2) + "\n", encoding="utf-8")
    log_dir = state.parent / "fold1_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    exit_codes = {}
    for key in ("b0", "b3s_zero4"):
        with (log_dir / f"{key}.log").open("w", encoding="utf-8") as handle:
            proc = subprocess.run(body["commands"][key], cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, stdout=handle, stderr=subprocess.STDOUT, check=False)
        exit_codes[key] = proc.returncode
    terminal = {}
    if exit_codes["b0"] == 0:
        terminal["b0"] = terminal_checkpoint("m1_compact_b0_f1_s42_fresh_e11", "B0", log_dir / "b0.log", body["teacher"]["checkpoint"]["path"])
    if exit_codes["b3s_zero4"] == 0:
        terminal["b3s_zero4"] = terminal_checkpoint("m1_compact_b3s_zero4_f1_s42_fresh_e11", "B3S", log_dir / "b3s_zero4.log", body["teacher"]["checkpoint"]["path"])
    state.write_text(json.dumps({"schema": "m1_compact_b3s_f1_s42_execution_v1", "status": "PASS_M1_COMPACT_B3S_F1_S42_PAIR_TERMINAL" if exit_codes == {"b0": 0, "b3s_zero4": 0} and len(terminal) == 2 else "FAIL_M1_COMPACT_B3S_F1_S42_EXECUTION", "receipt_sha256": sha(receipt), "exit_codes": exit_codes, "commands": body["commands"], "terminal_checkpoints": terminal, "target_metrics_not_read_by_runner": True}, indent=2) + "\n", encoding="utf-8")
    return 0 if exit_codes == {"b0": 0, "b3s_zero4": 0} else 1


def terminal_checkpoint(run_id: str, variant: str, log_path: Path, teacher_path: str) -> dict[str, Any]:
    """Locate and validate the fresh fixed epoch-11 checkpoint, score-free."""
    candidates = []
    for path in (ROOT / "logs").rglob("*.ckpt"):
        if run_id in str(path) and "fixed_last" in str(path):
            candidates.append(path)
    need(len(candidates) == 1, f"expected one terminal checkpoint for {run_id}, found {len(candidates)}")
    path = candidates[0]
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    need(payload.get("epoch") == 11 and payload.get("global_step") == 59412, f"{variant} terminal epoch/step drift")
    need(payload.get("pytorch-lightning_version") == "2.6.5", f"{variant} Lightning drift")
    need(isinstance(payload.get("optimizer_states"), list) and len(payload["optimizer_states"]) == 1, f"{variant} optimizer count drift")
    optimizer = payload["optimizer_states"][0]
    need(len(optimizer.get("param_groups", [])) == 1 and float(optimizer["param_groups"][0].get("lr")) == 1.0e-4 and float(optimizer["param_groups"][0].get("weight_decay")) == 0.0, f"{variant} optimizer hyperparameter drift")
    steps = set()
    for slot in optimizer.get("state", {}).values():
        step = slot.get("step")
        if hasattr(step, "detach"):
            step = step.detach().cpu().item()
        steps.add(int(step))
    need(steps == {59412}, f"{variant} optimizer step drift")
    need(payload.get("lr_schedulers") == [], f"{variant} scheduler drift")
    fit = payload.get("loops", {}).get("fit_loop", {})
    need(fit.get("epoch_loop.batch_progress", {}).get("total", {}).get("completed") == 59412 and fit.get("epoch_progress", {}).get("total", {}).get("processed") == 12, f"{variant} fit-loop drift")
    callbacks = payload.get("callbacks", {})
    need(len(callbacks) == 1 and "every_n_epochs': 12" in str(next(iter(callbacks))), f"{variant} callback drift")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if "Restored all states from the checkpoint at" in line and teacher_path in line]
    need(lines, f"{variant} teacher restoration evidence missing")
    return {"path": str(path.resolve()), "sha256": sha(path), "variant": variant, "epoch": 11, "global_step": 59412, "optimizer": {"count": 1, "state_steps": sorted(steps), "lr": 1.0e-4, "weight_decay": 0.0}, "lr_schedulers": 0, "fit_loop": {"batch_completed": 59412, "epoch_processed": 12}, "callback_key": str(next(iter(callbacks))), "resume_teacher_sha256": EXPECTED_TEACHER_SHA, "restored_all_states": {"present": True, "line_sha256": hashlib.sha256(lines[-1].encode()).hexdigest()}, "log": {"path": str(log_path.resolve()), "sha256": sha(log_path)}, "launch_command_bound": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--allow-root-launch")
    args = parser.parse_args()
    need(args.prepare ^ args.execute, "choose exactly one of --prepare/--execute")
    if args.prepare:
        body = prepare(args.receipt)
        print(json.dumps({"status": body["status"], "receipt": str(args.receipt.resolve()), "sha256": sha(args.receipt.resolve())}, sort_keys=True))
    else:
        raise SystemExit(execute(args.receipt.resolve(), args.state.resolve(), args.allow_root_launch))


if __name__ == "__main__":
    main()
