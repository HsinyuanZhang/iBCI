#!/usr/bin/env python3
"""Run or finalize one immutable fresh paired-view C1 cell.

The runner is portable across hosts: code lives under ``--workspace`` while
data, teacher, cache, and result roots may be host-local.  Every entry verifies
the sealed source graph and the 33-file data inventory before training.  The
v3 repair receipt owns all 12 fresh cells; no v2 checkpoint, status, or metric
artifact is imported after the TS4 input-pipeline failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_CELLS = {
    "separate_sua_t4",
    "separate_pseudo_mua_t4",
    "shared_t4",
    "shared_ts4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    # Serialize before claiming the write-once path.  A runtime-only object
    # (for example PyTorch 2.5's ``_CUuuid``) must fail without leaving a
    # truncated status file that looks like an experiment boundary.
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialized)


def runtime_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        payload.update(
            {
                "pytorch": torch.__version__,
                "pytorch_cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            payload["torch_gpus"] = [
                {
                    "logical_index": index,
                    "name": torch.cuda.get_device_name(index),
                    # Some PyTorch builds expose a private ``_CUuuid`` object
                    # instead of a Python string.  Provenance stores its text,
                    # never the extension object itself.
                    "uuid": (
                        str(getattr(torch.cuda.get_device_properties(index), "uuid"))
                        if getattr(torch.cuda.get_device_properties(index), "uuid", None)
                        is not None
                        else None
                    ),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                }
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # runtime receipt must survive diagnostic import failure
        payload["torch_error"] = repr(exc)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        payload["nvidia_smi"] = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
    except Exception as exc:
        payload["nvidia_smi_error"] = repr(exc)
    return payload


def _matrix_row(receipt: dict[str, Any], cell: str, seed: int) -> dict[str, Any]:
    rows = [
        row
        for row in receipt["fresh_matrix"]
        if row.get("cell") == cell and row.get("seed") == seed
    ]
    if len(rows) != 1:
        raise ValueError(f"receipt has {len(rows)} rows for {cell}/seed{seed}")
    row = rows[0]
    prelaunch_id = str(receipt.get("prelaunch_id", ""))
    expected_artifacts = (
        [f"artifacts/{cell}_s{seed}.json"]
        if cell.startswith("separate_")
        else [
            f"artifacts/{cell}_s{seed}_sua.json",
            f"artifacts/{cell}_s{seed}_pseudo_mua.json",
        ]
    )
    expected = {
        "logical_checkpoint_dir": f"checkpoints/{prelaunch_id}_{cell}_s{seed}",
        "logical_status": f"status/{cell}_s{seed}.complete.json",
        "logical_closure_dir": f"closure/{cell}_s{seed}",
        "logical_artifacts": expected_artifacts,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"canonical C1 matrix path drift for {cell}/s{seed}/{key}")
    return row


def _verify_or_create_data_stamp(
    *,
    receipt_path: Path,
    workspace: Path,
    teacher: Path,
    data_dir: Path,
    result_root: Path,
    python: Path,
) -> Path:
    receipt = load_json(receipt_path)
    manifest_sha = receipt["portable_data_manifest"]["sha256"]
    stamp = result_root / "runtime_verification" / f"data_{manifest_sha}.json"
    lock = stamp.with_suffix(".lock")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    if stamp.is_file():
        row = load_json(stamp)
        if (
            row.get("status") == "passed"
            and row.get("data_manifest_sha256") == manifest_sha
            and row.get("data_dir") == str(data_dir)
            and row.get("hostname") == socket.gethostname()
        ):
            return stamp
        raise ValueError(f"runtime data-verification stamp drift: {stamp}")
    acquired = False
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(descriptor)
        acquired = True
    except FileExistsError:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if stamp.is_file():
                return _verify_or_create_data_stamp(
                    receipt_path=receipt_path,
                    workspace=workspace,
                    teacher=teacher,
                    data_dir=data_dir,
                    result_root=result_root,
                    python=python,
                )
            time.sleep(0.5)
        raise TimeoutError(f"timed out waiting for data verification: {lock}")
    if acquired:
        verifier = workspace / "sua_exploration/scripts/verify_t4_paired_view_c1_fresh_prelaunch.py"
        command = [
            str(python),
            str(verifier),
            "--receipt",
            str(receipt_path),
            "--workspace",
            str(workspace),
            "--teacher",
            str(teacher),
            "--data-dir",
            str(data_dir),
            "--verify-data-content",
        ]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, check=True)
            verifier_result = json.loads(completed.stdout.strip().splitlines()[-1])
            write_json_exclusive(
                stamp,
                {
                    "schema_version": 1,
                    "status": "passed",
                    "created_at": dt.datetime.now().astimezone().isoformat(),
                    "hostname": socket.gethostname(),
                    "data_dir": str(data_dir),
                    "data_manifest_sha256": manifest_sha,
                    "verifier_result": verifier_result,
                },
            )
        finally:
            lock.unlink(missing_ok=True)
    return stamp


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(json.dumps({"exec": command}, sort_keys=True), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _validate_metadata(
    metadata_path: Path,
    *,
    cell: str,
    seed: int,
    teacher_sha: str,
    manifest_sha: str,
    expected_train_source_sha: str,
) -> tuple[dict[str, Any], Path]:
    metadata = load_json(metadata_path)
    shared = cell.startswith("shared_")
    side_group = "ts4" if cell == "shared_ts4" else "t4"
    expected_view = (
        "sua" if cell == "separate_sua_t4" else "pseudo_mua"
        if cell == "separate_pseudo_mua_t4"
        else "paired_sua_pseudo_mua"
    )
    if (
        metadata.get("status"),
        metadata.get("seed"),
        metadata.get("task"),
        metadata.get("signal_view"),
        metadata.get("teacher_sha256"),
        metadata.get("train_val_manifest_sha256"),
        metadata.get("held_out_test_evaluated"),
    ) != ("completed", seed, "CO", expected_view, teacher_sha, manifest_sha, False):
        raise ValueError(f"metadata identity/provenance drift: {metadata_path}")
    training = metadata.get("training") or {}
    if shared:
        no_backprop = metadata.get("no_heldout_backprop_contract") or {}
        if (
            metadata.get("training_kind") != "shared_paired_view"
            or metadata.get("formal_sua_files_opened") is not False
            or metadata.get("paired_objective", {}).get("lambda_consistency") != 0.0
            or metadata.get("paired_objective", {}).get("sua_task_loss_weight") != 0.5
            or metadata.get("paired_objective", {}).get("pseudo_mua_task_loss_weight") != 0.5
            or any(
                row.get("side_features", {}).get("group") != side_group
                for row in metadata.get("view_configs", {}).values()
            )
            or no_backprop.get("optimizer_and_backward_scope") != "source_train_27_only"
            or no_backprop.get("development_enters_train_dataloader") is not False
            or no_backprop.get("development_enters_loss") is not False
            or no_backprop.get("development_enters_optimizer") is not False
            or no_backprop.get("development_uses_backward_gradients") is not False
            or no_backprop.get("formal_paths_resolved") is not False
            or no_backprop.get("formal_files_opened") is not False
        ):
            raise ValueError(f"shared C1 objective/descriptor drift: {metadata_path}")
        expected_training = (
            training.get("max_epochs"),
            training.get("source_activity_calibration_n_trials"),
            training.get("t4_label_rate_pool_size"),
            training.get("evaluation_forward_calibration_n"),
            training.get("evaluation_start_trial"),
            training.get("random_calibration"),
            training.get("loss_mode"),
            training.get("freeze_decoder"),
        )
        if expected_training != (12, 10, 50, 30, 50, False, "task_only", False):
            raise ValueError(f"shared C1 budget/training drift: {metadata_path}")
    else:
        side = metadata.get("side_features") or {}
        if (
            side.get("group") != "t4"
            or side.get("pool_size") != 50
            or training.get("max_epochs") != 12
            or training.get("calibration_n_trials") != 10
            or training.get("random_calibration") is not False
            or training.get("calibration_selection") != "chronological_first_n"
            or training.get("no_early_stopping") is not True
            or training.get("checkpoint_every_epoch") is not True
            or training.get("loss_mode") != "task_only"
            or training.get("freeze_decoder") is not False
        ):
            raise ValueError(f"separate C1 budget/training drift: {metadata_path}")
    cost_path = metadata_path.parent / "post_run_cost_receipt.json"
    cost = load_json(cost_path)
    if (
        cost.get("status") != "completed"
        or cost.get("run_metadata_sha256") != sha256_file(metadata_path)
        or cost.get("accelerator") != "gpu"
    ):
        raise ValueError(f"cost/metadata closure drift: {cost_path}")
    if not shared and cost.get("train_variant_source_sha256") != expected_train_source_sha:
        raise ValueError("fresh separate control train source differs from frozen receipt")
    return metadata, cost_path


def _exact_student_parameter_count(
    metadata: dict[str, Any], *, teacher: Path, workspace: Path
) -> int:
    checkpoints = metadata.get("epoch_checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 12:
        epoch_dir = Path(metadata["output_dir"]) / "epoch_ckpts"
        checkpoints = sorted(str(path) for path in epoch_dir.glob("epoch_*.ckpt"))
    if len(checkpoints) != 12:
        raise ValueError("exact parameter audit requires all 12 epoch checkpoints")
    sys.path.insert(0, str(workspace / "sua_exploration/scripts"))
    sys.path.insert(0, str(workspace / "sua_exploration"))
    from select_gradient_free_protocol_dandi688 import load_frozen_model
    import torch

    model = load_frozen_model(
        Path(checkpoints[-1]), teacher, "B3S", torch.device("cpu")
    )
    if model.student is None:
        raise RuntimeError("C1 parameter audit loaded no student")
    return int(sum(parameter.numel() for parameter in model.student.parameters()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=sorted(EXPECTED_CELLS), required=True)
    parser.add_argument("--seed", choices=[42, 43, 44], type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("T4_PAIRED_VIEW_C1_AUTHORIZED") != "YES":
        raise RuntimeError("set T4_PAIRED_VIEW_C1_AUTHORIZED=YES only after root review")
    workspace = args.workspace.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    teacher = args.teacher.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    receipt = load_json(receipt_path)
    program_receipt_sha = sha256_file(receipt_path)
    shared = args.cell.startswith("shared_")
    row = _matrix_row(receipt, args.cell, args.seed)
    cell_receipt_path = receipt_path
    cell_receipt_sha = program_receipt_sha
    control_cache_by_view = {
        "sua": cache_root / "sua",
        "pseudo_mua": cache_root / "pseudo_mua",
    }
    for view, path in control_cache_by_view.items():
        if not path.is_dir():
            raise FileNotFoundError(f"C1 {view} cache namespace missing: {path}")

    verifier = workspace / "sua_exploration/scripts/verify_t4_paired_view_c1_fresh_prelaunch.py"
    _run(
        [
            str(python), str(verifier), "--receipt", str(receipt_path),
            "--workspace", str(workspace), "--teacher", str(teacher),
        ],
        cwd=workspace,
        env=dict(os.environ),
    )
    data_stamp = _verify_or_create_data_stamp(
        receipt_path=receipt_path,
        workspace=workspace,
        teacher=teacher,
        data_dir=data_dir,
        result_root=result_root,
        python=python,
    )

    status_dir = result_root / "status"
    started = status_dir / f"{args.cell}_s{args.seed}.started.json"
    completed = status_dir / f"{args.cell}_s{args.seed}.complete.json"
    failed = status_dir / f"{args.cell}_s{args.seed}.failed.json"
    if completed.exists() or failed.exists() or started.exists():
        raise FileExistsError(f"C1 cell status already exists for {args.cell}/s{args.seed}")
    started_at = dt.datetime.now().astimezone().isoformat()
    write_json_exclusive(
        started,
        {
            "schema_version": 1,
            "status": "started",
            "cell": args.cell,
            "seed": args.seed,
            "started_at": started_at,
            "receipt_path": str(receipt_path),
            "program_receipt_sha256": program_receipt_sha,
            "cell_receipt_path": str(cell_receipt_path),
            "cell_receipt_sha256": cell_receipt_sha,
            "data_verification_stamp": str(data_stamp),
            "data_verification_stamp_sha256": sha256_file(data_stamp),
            "runtime_environment": runtime_environment(),
            "formal_sua_files_opened": False,
        },
    )

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    checkpoint_name = Path(row["logical_checkpoint_dir"]).name
    run_dir = checkpoint_root / checkpoint_name
    artifact_paths = [result_root / relative for relative in row["logical_artifacts"]]
    try:
        for path in artifact_paths:
            if path.exists():
                raise FileExistsError(f"C1 result collision: {path}")
        for path in artifact_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        if shared:
            if run_dir.exists():
                raise FileExistsError(f"shared C1 run directory exists: {run_dir}")
            side = "ts4" if args.cell == "shared_ts4" else "t4"
            _run(
                [
                    str(python),
                    str(workspace / "sua_exploration/scripts/train_paired_view_c1_dandi688.py"),
                    "--teacher-ckpt", str(teacher),
                    "--out-name", checkpoint_name,
                    "--checkpoint-root", str(checkpoint_root),
                    "--data-dir", str(data_dir),
                    "--train-val-manifest", str(workspace / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"),
                    "--data-manifest", str(receipt_path.parent / receipt["portable_data_manifest"]["filename"]),
                    "--sua-cache-dir", str(control_cache_by_view["sua"]),
                    "--pseudo-mua-cache-dir", str(control_cache_by_view["pseudo_mua"]),
                    "--side-features", side,
                    "--seed", str(args.seed),
                    "--max-epochs", "12",
                    "--batch-size", "32",
                    "--num-workers", "4",
                    "--learning-rate", "0.0001",
                    "--accelerator", "gpu",
                    "--require-gpu",
                    "--disable-progress-bar",
                ],
                cwd=workspace,
                env=environment,
            )
        else:
            if run_dir.exists():
                raise FileExistsError(f"fresh control run directory exists: {run_dir}")
            view = "sua" if args.cell == "separate_sua_t4" else "pseudo_mua"
            separate_cache = control_cache_by_view[view]
            _run(
                [
                    str(python),
                    str(workspace / "sua_exploration/scripts/train_variant_dandi688.py"),
                    "--variant", "B3S", "--out_name", checkpoint_name,
                    "--data_dir", str(data_dir),
                    "--train_val_manifest", str(workspace / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"),
                    "--task", "CO", "--split_counts", "27,6,6",
                    "--max_units_exclusive", "100", "--max_epochs", "12",
                    "--no_early_stopping", "--checkpoint_every_epoch",
                    "--lr", "0.0001", "--seed", str(args.seed),
                    "--batch_size", "32", "--num_workers", "4",
                    "--cache_dir", str(separate_cache),
                    "--signal_view", view, "--require_gpu", "--accelerator", "gpu",
                    "--side_features", "t4", "--side_feature_pool_size", "50",
                    "--calibration_n_trials", "10", "--chronological_calibration",
                    "--disable_progress_bar",
                ],
                cwd=workspace,
                env=environment,
            )

        metadata_path = run_dir / "run_metadata.json"
        metadata, cost_path = _validate_metadata(
            metadata_path,
            cell=args.cell,
            seed=args.seed,
            teacher_sha=receipt["teacher"]["sha256"],
            manifest_sha=receipt["strict_loader_manifest"]["sha256"],
            expected_train_source_sha=receipt["source_map"][
                "sua_exploration/scripts/train_variant_dandi688.py"
            ]["sha256"],
        )
        student_parameter_count = _exact_student_parameter_count(
            metadata, teacher=teacher, workspace=workspace
        )
        if shared:
            for view, artifact in zip(("sua", "pseudo_mua"), artifact_paths, strict=True):
                _run(
                    [
                        str(python),
                        str(workspace / "sua_exploration/scripts/eval_paired_view_c1_epoch_window.py"),
                        "--run-dir", str(run_dir), "--signal-view", view,
                        "--out-path", str(artifact), "--total-epochs", "12",
                        "--burn-in", "4", "--calibration-n", "30", "--pool-size", "50",
                        "--train-val-manifest", str(workspace / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"),
                    ],
                    cwd=workspace,
                    env=environment,
                )
        else:
            _run(
                [
                    str(python),
                    str(workspace / "sua_exploration/scripts/eval_epoch_window_generic_dandi688.py"),
                    "--run_dir", str(run_dir), "--total_epochs", "12", "--burn_in", "4",
                    "--calibration_n", "30", "--pool_size", "50",
                    "--train_val_manifest", str(workspace / "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"),
                    "--out_path", str(artifact_paths[0]),
                ],
                cwd=workspace,
                env=environment,
            )

        for artifact in artifact_paths:
            result = load_json(artifact)
            if (
                result.get("seed") != args.seed
                or result.get("no_test_files_evaluated") is not True
                or result.get("uses_backward_gradients") is not False
                or result.get("protocol", {}).get("calibration_n") != 30
                or result.get("protocol", {}).get("pool_size") != 50
                or result.get("epoch_list") != list(range(5, 13))
            ):
                raise ValueError(f"C1 scorer artifact drift: {artifact}")

        closure_dir = result_root / row["logical_closure_dir"]
        if closure_dir.exists():
            raise FileExistsError(f"C1 closure collision: {closure_dir}")
        closure_dir.mkdir(parents=True, exist_ok=False)
        copied: list[dict[str, Any]] = []
        for source in (metadata_path, cost_path, *artifact_paths):
            destination = closure_dir / source.name
            shutil.copy2(source, destination)
            copied.append(
                {
                    "source": str(source),
                    "copy": str(destination),
                    "copy_relative": str(destination.relative_to(result_root)),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        closure_manifest = closure_dir / "closure_manifest.json"
        write_json_exclusive(
            closure_manifest,
            {
                "schema_version": 1,
                "status": "completed",
                "cell": args.cell,
                "seed": args.seed,
                "program_receipt_sha256": program_receipt_sha,
                "cell_receipt_path": str(cell_receipt_path),
                "cell_receipt_sha256": cell_receipt_sha,
                "fresh_control_finalization_only": False,
                "files": copied,
                "runtime_environment": runtime_environment(),
                "student_parameter_count": student_parameter_count,
                "fp32_student_weight_bytes": student_parameter_count * 4,
                "formal_sua_files_opened": False,
            },
        )
        write_json_exclusive(
            completed,
            {
                "schema_version": 1,
                "status": "completed",
                "cell": args.cell,
                "seed": args.seed,
                "started_status": str(started),
                "started_status_sha256": sha256_file(started),
                "completed_at": dt.datetime.now().astimezone().isoformat(),
                "program_receipt_sha256": program_receipt_sha,
                "cell_receipt_path": str(cell_receipt_path),
                "cell_receipt_sha256": cell_receipt_sha,
                "checkpoint_dir": str(run_dir),
                "metadata_sha256": sha256_file(metadata_path),
                "cost_sha256": sha256_file(cost_path),
                "student_parameter_count": student_parameter_count,
                "fp32_student_weight_bytes": student_parameter_count * 4,
                "artifacts": [
                    {
                        "path": str(path),
                        "relative": str(path.relative_to(result_root)),
                        "sha256": sha256_file(path),
                    }
                    for path in artifact_paths
                ],
                "closure_manifest": str(closure_manifest),
                "closure_relative": str(closure_manifest.relative_to(result_root)),
                "closure_manifest_sha256": sha256_file(closure_manifest),
                "formal_sua_files_opened": False,
            },
        )
    except Exception as exc:
        if not failed.exists():
            write_json_exclusive(
                failed,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "cell": args.cell,
                    "seed": args.seed,
                    "failed_at": dt.datetime.now().astimezone().isoformat(),
                    "error": repr(exc),
                    "started_status": str(started),
                    "started_status_sha256": sha256_file(started),
                    "program_receipt_sha256": program_receipt_sha,
                    "cell_receipt_path": str(cell_receipt_path),
                    "cell_receipt_sha256": cell_receipt_sha,
                    "formal_sua_files_opened": False,
                },
            )
        raise
    print(json.dumps({"status": "completed", "cell": args.cell, "seed": args.seed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
