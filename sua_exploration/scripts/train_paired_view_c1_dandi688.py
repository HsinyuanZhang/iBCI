#!/usr/bin/env python3
"""Train one fresh C1 shared SUA/pseudo-MUA model, with no validation selection.

This is intentionally a narrow companion to ``train_variant_dandi688.py``.
The latter remains the exact, tested trainer for the six fresh *separate* C1
references.  This entry point is only needed for the six shared-weight runs.

Training uses source-session chronological Q=10 activity calibration, T4/T S4
features fitted from the first 50 labelled rewarded trials, and one shared
optimizer step for each matched pair.  Development evaluation is external and
predeclared: Q=30, pool=50, epochs 5--12.  No test session is resolved or
opened in this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))
sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))

from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
from mc_maze.paired_view_c1 import PairedViewC1DataModule, normalizer_hashes_are_distinct
from mc_maze.unit_side_features import side_feature_stats_sha256
from src.models.paired_view_c1_module import PairedViewC1LitModule


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        payload["torch_visible_devices"] = [
            {
                "logical_index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        payload["nvidia_smi_gpus"] = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        payload["nvidia_smi_error"] = str(exc)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-val-manifest", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--sua-cache-dir", required=True)
    parser.add_argument("--pseudo-mua-cache-dir", required=True)
    parser.add_argument("--side-features", choices=["t4", "ts4"], required=True)
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], required=True)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    parser.add_argument("--limit-train-batches", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_epochs != 12:
        raise ValueError("C1 freezes max_epochs=12")
    if args.batch_size != 32:
        raise ValueError("C1 freezes batch_size=32")
    if args.learning_rate != 1e-4:
        raise ValueError("C1 freezes learning_rate=1e-4")
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("--require-gpu set but CUDA is unavailable")

    teacher = Path(args.teacher_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    train_val_manifest = Path(args.train_val_manifest).expanduser().resolve()
    data_manifest = Path(args.data_manifest).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_dir = checkpoint_root / args.out_name
    sua_cache = Path(args.sua_cache_dir).expanduser().resolve()
    pseudo_cache = Path(args.pseudo_mua_cache_dir).expanduser().resolve()
    for label, path, kind in (
        ("teacher", teacher, "file"),
        ("data_dir", data_dir, "dir"),
        ("train_val_manifest", train_val_manifest, "file"),
        ("data_manifest", data_manifest, "file"),
    ):
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            raise FileNotFoundError(f"C1 {label} is missing: {path}")
    if output_dir.exists():
        raise FileExistsError(f"C1 refuses to reuse output directory: {output_dir}")
    if sua_cache == pseudo_cache:
        raise ValueError("C1 requires separate SUA and pseudo-MUA cache namespaces")

    pl.seed_everything(args.seed, workers=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = output_dir / "run_metadata.json"

    common_dm_kwargs = dict(
        data_dir=str(data_dir),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=args.batch_size,
        window_size=50,
        # Historical source10 semantics, but now fixed and explicitly recorded.
        calibration_n_trials=10,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=args.num_workers,
        # Exact first-Q support is part of C1's pairing contract; do not call
        # the legacy random calibration path even if future records grow longer.
        random_calibration=False,
        seed=args.seed,
        max_units_exclusive=100,
        side_feature_group=args.side_features,
        side_feature_pool_size=50,
        side_permutation_seed=args.seed if args.side_features == "ts4" else None,
        train_val_manifest_path=str(train_val_manifest),
    )
    sua_dm = Dandi688MultiSessionDataModule(
        **common_dm_kwargs, cache_dir=str(sua_cache), signal_view="sua"
    )
    pseudo_dm = Dandi688MultiSessionDataModule(
        **common_dm_kwargs, cache_dir=str(pseudo_cache), signal_view="pseudo_mua"
    )
    paired_dm = PairedViewC1DataModule(sua_dm, pseudo_dm)
    paired_dm.setup()

    sua_mean, sua_std = sua_dm._get_side_feature_stats()  # train-only, view-specific
    pseudo_mean, pseudo_std = pseudo_dm._get_side_feature_stats()  # train-only, view-specific
    assert sua_mean is not None and sua_std is not None
    assert pseudo_mean is not None and pseudo_std is not None
    sua_normalizer_sha = side_feature_stats_sha256(sua_mean, sua_std)
    pseudo_normalizer_sha = side_feature_stats_sha256(pseudo_mean, pseudo_std)
    if not normalizer_hashes_are_distinct(sua_normalizer_sha, pseudo_normalizer_sha):
        raise RuntimeError("C1 normalizer isolation failed: SUA/pseudo-MUA hashes collide")

    optimizer = partial(torch.optim.Adam, lr=args.learning_rate, weight_decay=0.0)
    model = PairedViewC1LitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path=str(teacher),
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        pad_value=-1.0,
        freeze_decoder=False,
        loss_mode="task_only",
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        identity_mode="calibrated",
        side_dim=4,
        electrode_embed_dim=0,
        num_electrodes=0,
        optimizer=optimizer,
        scheduler=None,
        compile=False,
        lambda_consistency=0.0,
        sua_task_loss_weight=0.5,
        pseudo_mua_task_loss_weight=0.5,
    )
    model.setup("fit")
    assert model.student is not None
    encoder_cost = model.student.id_encoder.cost_profile(
        num_neurons=64, trial_length=100, num_trials=10
    )
    decoder_cost = model.student.decoder_cost_comparison_receipt(
        batch_size=1, num_neurons=64
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "initialized",
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "paired_view_c1_fresh",
        "training_kind": "shared_paired_view",
        "variant": "B3S",
        "seed": args.seed,
        "task": "CO",
        "signal_view": "paired_sua_pseudo_mua",
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": sha256_file(teacher),
        "data_dir": str(data_dir),
        "train_val_manifest": str(train_val_manifest),
        "train_val_manifest_sha256": sha256_file(train_val_manifest),
        "data_manifest": str(data_manifest),
        "data_manifest_sha256": sha256_file(data_manifest),
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "output_dir": str(output_dir),
        "held_out_test_evaluated": False,
        "formal_sua_files_opened": False,
        "no_heldout_backprop_contract": {
            "source_train_sessions": 27,
            "development_heldout_sessions": 6,
            "formal_test_sessions": 6,
            "optimizer_and_backward_scope": "source_train_27_only",
            "development_enters_train_dataloader": False,
            "development_enters_loss": False,
            "development_enters_optimizer": False,
            "development_uses_backward_gradients": False,
            "development_use": "post-training forward-only fixed epoch-5-through-12 scoring",
            "formal_paths_resolved": False,
            "formal_files_opened": False,
        },
        "session_splits": paired_dm.session_splits,
        "session_files": {
            "train": [str(path) for path in sua_dm.session_files["train"]],
            "val": [str(path) for path in sua_dm.session_files["val"]],
            "test": [],
        },
        "view_configs": {
            "sua": {
                "signal_view": "sua",
                "cache_dir": str(sua_cache),
                "side_features": {
                    "group": args.side_features,
                    "pool_size": 50,
                    "side_dim": 4,
                    "normalization_sha256": sua_normalizer_sha,
                    "normalization_scope": "source_train_27_only",
                    "permutation_seed": args.seed if args.side_features == "ts4" else None,
                },
            },
            "pseudo_mua": {
                "signal_view": "pseudo_mua",
                "cache_dir": str(pseudo_cache),
                "side_features": {
                    "group": args.side_features,
                    "pool_size": 50,
                    "side_dim": 4,
                    "normalization_sha256": pseudo_normalizer_sha,
                    "normalization_scope": "source_train_27_only",
                    "permutation_seed": args.seed if args.side_features == "ts4" else None,
                    "semantic_contract": "pool_electrode_trial_rates_then_refit_T4_never_average_unit_T4",
                },
            },
        },
        "cache_isolation": {
            "distinct_paths": True,
            "sua_cache_dir": str(sua_cache),
            "pseudo_mua_cache_dir": str(pseudo_cache),
            "normalizer_hashes_distinct": True,
        },
        "paired_exposure": paired_dm.exposure_receipt(),
        "paired_objective": model.c1_objective_receipt(),
        "training": {
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "epoch_checkpoints_dir": str(output_dir / "epoch_ckpts"),
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "source_activity_calibration_n_trials": 10,
            "t4_label_rate_pool_size": 50,
            "evaluation_forward_calibration_n": 30,
            "evaluation_pool_size": 50,
            "evaluation_start_trial": 50,
            "loss_mode": "task_only",
            "freeze_decoder": False,
            "shared_optimizer_steps": True,
            "random_calibration": False,
        },
        "cost_receipt_reference": {
            "model_parameter_count": sum(parameter.numel() for parameter in model.student.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.student.parameters() if parameter.requires_grad
            ),
            "fp32_weight_bytes": 4 * sum(parameter.numel() for parameter in model.student.parameters()),
            "descriptor_persistent_bytes_per_channel": 16,
            "encoder": asdict(encoder_cost),
            "decoder": decoder_cost,
            "shared_training_view_forwards_per_optimizer_step": 2,
            "deployment_weight_copies": 1,
            "deployment_extra_state_vs_separate": 0,
        },
        "runtime_environment_at_initialization": runtime_environment(),
    }
    write_json(metadata_path, metadata)

    epoch_ckpt_dir = output_dir / "epoch_ckpts"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(epoch_ckpt_dir),
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=12,
        accelerator=args.accelerator,
        devices=1,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_checkpointing=True,
        deterministic=True,
        enable_progress_bar=not args.disable_progress_bar,
        log_every_n_steps=50,
        default_root_dir=str(output_dir),
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        num_sanity_val_steps=0,
    )
    using_gpu = trainer.accelerator.__class__.__name__.lower().startswith("cuda")
    if args.require_gpu and not using_gpu:
        raise RuntimeError("C1 requires a CUDA trainer")
    if using_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model, train_dataloaders=paired_dm.train_dataloader())
    if using_gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    expected_checkpoints = [epoch_ckpt_dir / f"epoch_{epoch:03d}.ckpt" for epoch in range(12)]
    missing = [str(path) for path in expected_checkpoints if not path.is_file()]
    if missing:
        raise RuntimeError(f"C1 failed to write all fixed epoch checkpoints: {missing}")

    metadata.update(
        {
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "epoch_checkpoints": [str(path) for path in expected_checkpoints],
            "held_out_test_evaluated": False,
            "formal_sua_files_opened": False,
            "runtime_environment_at_completion": runtime_environment(),
        }
    )
    write_json(metadata_path, metadata)
    cost_path = output_dir / "post_run_cost_receipt.json"
    if cost_path.exists():
        raise FileExistsError(f"C1 post-run cost path already exists: {cost_path}")
    cost = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_metadata_path": str(metadata_path),
        "run_metadata_sha256": sha256_file(metadata_path),
        "fit_wall_clock_seconds": elapsed,
        "accelerator": "gpu" if using_gpu else "cpu",
        "cuda_peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if using_gpu else 0,
        "cuda_peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if using_gpu else 0,
        "runtime_environment": runtime_environment(),
        "cost_receipt_reference": metadata["cost_receipt_reference"],
        "formal_sua_files_opened": False,
    }
    write_json(cost_path, cost)
    print(json.dumps({"run_metadata": str(metadata_path), "cost": str(cost_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
