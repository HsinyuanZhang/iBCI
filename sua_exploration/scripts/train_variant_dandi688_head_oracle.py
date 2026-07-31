"""Standalone strict trainer for the teacher-head-preserving K/V oracle.

The oracle is an attribution experiment, not the final hardware candidate.  It
keeps the teacher's full read-in, 64 independent attention heads, Q/K/V/out
projections, norms, FFN and output projection.  The only topology change is to
derive a session-static key from ``E`` and an online value from activity.

This entrypoint is deliberately isolated from the active v1/v2 trainers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

_SUA_ROOT = Path(__file__).resolve().parents[1]
_SCE_ROOT = _SUA_ROOT.parent / "streaming_calibration_exp"
sys.path.insert(0, str(_SUA_ROOT))
sys.path.insert(0, str(_SCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mc_maze.multisession_datamodule import (  # noqa: E402
    Dandi688MultiSessionDataModule,
)
from mc_maze.unit_side_features import (  # noqa: E402
    feature_semantics_version,
    side_feature_stats_sha256,
)
from src.metrics.run_artifacts import assert_run_dir_is_fresh  # noqa: E402
from src.models.head_oracle_module import (  # noqa: E402
    TeacherHeadOracleLitModule,
)
from train_variant_dandi688_decoupled_v2 import (  # noqa: E402
    configure_multisession_metrics,
    sha256_file,
    sha256_tensor_state,
    write_json,
)


DEFAULT_TEACHER = (
    _SUA_ROOT
    / "checkpoints/teacher_mc_maze/"
    "best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)
KEY_MODES = {"e_t4", "e_ts4"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher_ckpt", default=str(DEFAULT_TEACHER))
    parser.add_argument("--variant", default="B3S")
    parser.add_argument("--side_features", default="t4")
    parser.add_argument("--side_feature_pool_size", type=int, default=50)
    parser.add_argument("--calibration_n_trials", type=int, default=30)
    parser.add_argument("--decoder_mode", default="coupled")
    parser.add_argument(
        "--oracle_key_mode", choices=sorted(KEY_MODES), required=True
    )
    parser.add_argument(
        "--oracle_key_permutation_seed", type=int, default=None
    )
    parser.add_argument("--out_name", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--train_val_manifest", required=True)
    parser.add_argument("--signal_view", default="sua")
    parser.add_argument("--task", default="CO")
    parser.add_argument("--split_counts", default="27,6,6")
    parser.add_argument("--max_units_exclusive", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=12)
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--checkpoint_every_epoch", action="store_true")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss_mode", default="task_only")
    parser.add_argument("--identity_mode", default="calibrated")
    parser.add_argument(
        "--accelerator", choices=["auto", "cpu", "gpu"], default="auto"
    )
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--disable_progress_bar", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    expected = {
        "variant": (args.variant, "B3S"),
        "side_features": (args.side_features, "t4"),
        "side_feature_pool_size": (args.side_feature_pool_size, 50),
        "calibration_n_trials": (args.calibration_n_trials, 30),
        "decoder_mode": (args.decoder_mode, "coupled"),
        "signal_view": (args.signal_view, "sua"),
        "task": (args.task, "CO"),
        "split_counts": (args.split_counts, "27,6,6"),
        "max_units_exclusive": (args.max_units_exclusive, 100),
        "max_epochs": (args.max_epochs, 12),
        "loss_mode": (args.loss_mode, "task_only"),
        "identity_mode": (args.identity_mode, "calibrated"),
    }
    for name, (observed, required) in expected.items():
        if observed != required:
            raise ValueError(
                f"head-oracle protocol requires {name}={required!r}, "
                f"got {observed!r}"
            )
    if args.oracle_key_mode == "e_ts4":
        if args.oracle_key_permutation_seed != args.seed:
            raise ValueError(
                "oracle e_ts4 permutation seed must equal training seed"
            )
    elif args.oracle_key_permutation_seed is not None:
        raise ValueError(
            "oracle e_t4 forbids --oracle_key_permutation_seed"
        )
    if not args.no_early_stopping or not args.checkpoint_every_epoch:
        raise ValueError(
            "head-oracle protocol requires --no_early_stopping and "
            "--checkpoint_every_epoch"
        )
    if args.lr <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError(
            "lr/batch_size must be positive and workers non-negative"
        )
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("--require_gpu was set but CUDA is unavailable")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator gpu was set but CUDA is unavailable")


def _permutation_sha(seed: int, unit_count: int) -> str:
    permutation = (
        np.random.RandomState(seed)
        .permutation(unit_count)
        .astype(np.int64)
    )
    return hashlib.sha256(permutation.tobytes()).hexdigest()


def _oracle_decoder_metadata(
    model: TeacherHeadOracleLitModule,
    dm: Dandi688MultiSessionDataModule,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if model.student is None:
        raise RuntimeError("oracle model has no student")
    initialization = model.oracle_initialization_receipt
    checkpoint_receipt = model.active_oracle_checkpoint_receipt()
    cost = model.decoupled_cost_receipt(
        batch_size=1, num_neurons=64
    )
    metadata: dict[str, Any] = {
        "architecture_family": (
            "teacher_head_preserving_decoupled_kv_oracle"
        ),
        "base_decoder_mode_argument": "coupled",
        "active_decoder_mode": (
            "teacher_head_preserving_decoupled_oracle"
        ),
        "key_mode": args.oracle_key_mode,
        "key_width": 512,
        "value_width": 512,
        "attention_heads": 64,
        "head_dim": 8,
        "teacher_readin_space": "decoder.fc_in",
        "query_source": "decoder.fc_in(decoder.rep)",
        "key_source": "decoder.fc_in(identity_E)",
        "value_source": "decoder.fc_in(activity)",
        "direct_t4_branch": "none",
        "encoder_side_input": "aligned_real_t4",
        "decoder_ts4_control": (
            "fixed_E_row_permutation_only"
            if args.oracle_key_mode == "e_ts4"
            else "none"
        ),
        "key_permutation_seed": args.oracle_key_permutation_seed,
        "fixed_slot_count": 0,
        "fresh_common_teacher_fit": True,
        "headwise_softmax_preserved": True,
        "low_rank_factorization_used": False,
        "head_averaging_used": False,
        "legacy_decoder_transformer_active": False,
        "legacy_decoder_transformer_trainable": False,
        "oracle_initialization_receipt_at_start": initialization,
        "oracle_checkpoint_receipt_at_start": checkpoint_receipt,
        "online_cost_receipt_reference_n64": cost,
        "decoder_cost_comparison_receipt_reference_n64": {
            "schema_version": 1,
            "active_mode": (
                "teacher_head_preserving_decoupled_oracle"
            ),
            "coupled": cost["coupled_reference"],
            "teacher_head_preserving_decoupled_oracle": cost,
        },
        "shared_decoder_base_sha256_at_start": sha256_tensor_state({
            name: tensor
            for name, tensor in model.student.decoder.state_dict().items()
            if (
                name.startswith("fc_in.")
                or name.startswith("fc_out.")
                or name == "rep"
            )
        }),
        "parameter_counts": {
            "student_total": sum(
                parameter.numel()
                for parameter in model.student.parameters()
            ),
            "optimizer_trainable": sum(
                parameter.numel()
                for parameter in model.student.parameters()
                if parameter.requires_grad
            ),
            "identity_encoder": sum(
                parameter.numel()
                for parameter in model.student.id_encoder.parameters()
            ),
            "head_oracle": sum(
                parameter.numel()
                for parameter in model.student.head_oracle.parameters()
            ),
            "identity_key_active_for_task": True,
        },
    }
    if args.oracle_key_mode == "e_ts4":
        metadata["key_permutation_sha256_by_session"] = {
            session: _permutation_sha(args.seed, unit_count)
            for session, unit_count
            in sorted(dm.session_unit_counts.items())
        }
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    pl.seed_everything(args.seed, workers=True)

    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_path = Path(
        args.train_val_manifest
    ).expanduser().resolve()
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else None
    )
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(
            f"missing teacher checkpoint: {teacher_ckpt}"
        )
    if not data_dir.is_dir():
        raise FileNotFoundError(f"missing data directory: {data_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing strict manifest: {manifest_path}"
        )

    output_dir = _SUA_ROOT / "checkpoints" / args.out_name
    results_dir = _SUA_ROOT / "results"
    assert_run_dir_is_fresh(output_dir)
    output_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"

    metadata: dict[str, Any] = {
        "schema_version": 2,
        "runner_family": "teacher_head_preserving_kv_oracle",
        "lightning_module_class": (
            "src.models.head_oracle_module."
            "TeacherHeadOracleLitModule"
        ),
        "generated_by": "train_variant_dandi688_head_oracle.py",
        "status": "initialized",
        "created_at": datetime.now().astimezone().isoformat(),
        "variant": "B3S",
        "seed": args.seed,
        "teacher_checkpoint": str(teacher_ckpt),
        "teacher_sha256": sha256_file(teacher_ckpt),
        "data_dir": str(data_dir),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "train_val_manifest": str(manifest_path),
        "train_val_manifest_sha256": sha256_file(manifest_path),
        "signal_view": "sua",
        "side_features": {
            "group": "t4",
            "pool_size": 50,
            "side_dim": 4,
            "permutation_seed": None,
        },
        "task": "CO",
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "output_dir": str(output_dir.resolve()),
        "checkpoint_reconstruction": {
            "strict_state_dict": True,
            "checkpoint_receipt_key": "teacher_head_oracle_receipt",
            "requires_setup_before_load": True,
            "requires_post_load_active_factor_sha_validation": True,
        },
        "training": {
            "max_epochs": 12,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "epoch_checkpoints_dir": str(
                (output_dir / "epoch_ckpts").resolve()
            ),
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "window_size": 50,
            "calibration_n_trials": 30,
            "trial_length": 100,
            "bin_size_ms": 20,
            "loss_mode": "task_only",
            "identity_mode": "calibrated",
            "freeze_decoder": False,
            "decode_last_timestep_only": True,
            "behavior_scaling_factor": 5.0,
            "deterministic": True,
            "world_size": 1,
            "distributed_supported": False,
        },
        "validation_protocol": {
            "activity_calibration": "chronological trials[0:30]",
            "t4_label_pool": (
                "chronological rewarded labelled trials[0:50]"
            ),
            "evaluation_start": 50,
            "trial_disjoint": True,
        },
        "held_out_evaluation_protocol": {
            "held_out_test_evaluated": False,
            "formal_test_sessions_loaded_during_fit": False,
            "model_weights": "validation-only architecture selection",
        },
        "held_out_test_evaluated": False,
    }
    write_json(metadata_path, metadata)

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(data_dir),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=args.batch_size,
        window_size=50,
        calibration_n_trials=30,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=args.num_workers,
        seed=args.seed,
        max_units_exclusive=100,
        cache_dir=str(cache_dir) if cache_dir else None,
        signal_view="sua",
        side_feature_group="t4",
        side_feature_pool_size=50,
        side_permutation_seed=None,
        train_val_manifest_path=str(manifest_path),
    )
    dm.setup("fit")
    side_mean, side_std = dm._get_side_feature_stats()
    if side_mean is None or side_std is None:
        raise RuntimeError(
            "strict oracle T4 side-feature normalization is missing"
        )
    metadata["side_features"].update({
        "feature_version": feature_semantics_version("t4"),
        "normalization_sha256": side_feature_stats_sha256(
            side_mean, side_std
        ),
    })
    metadata["session_splits"] = dm.session_splits
    metadata["session_unit_counts"] = dm.session_unit_counts
    metadata["session_channel_counts"] = dm.session_channel_counts
    metadata["session_files"] = {
        split: [str(path) for path in paths]
        for split, paths in dm.session_files.items()
    }
    metadata["trainer_fit_validation_loader_contract"] = {
        "loader_0_sessions": dm.session_splits["val"],
        "loader_1_sessions": dm.session_splits["val"],
        "legacy_metric_names": ["val_heldin/*", "val_heldout/*"],
        "formal_test_sessions_loaded_during_fit": False,
    }

    optimizer = partial(
        torch.optim.Adam, lr=args.lr, weight_decay=0.0
    )
    model = TeacherHeadOracleLitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path=str(teacher_ckpt),
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        pad_value=-1.0,
        freeze_decoder=False,
        freeze_encoder_base=False,
        loss_mode="task_only",
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        identity_mode="calibrated",
        fixed_slot_count=0,
        fixed_slot_dim=32,
        fixed_slot_mode="soft",
        fixed_slot_fusion="film",
        fixed_slot_temperature=1.0,
        decoder_mode="coupled",
        side_dim=4,
        electrode_embed_dim=0,
        num_electrodes=0,
        encoder_warmstart_path=None,
        optimizer=optimizer,
        scheduler=None,
        compile=False,
        oracle_key_mode=args.oracle_key_mode,
        oracle_key_permutation_seed=(
            args.oracle_key_permutation_seed
        ),
    )
    model.setup("fit")
    if model.student is None:
        raise RuntimeError("oracle setup failed to construct a student")
    encoder_cost = model.student.id_encoder.cost_profile(
        num_neurons=64,
        trial_length=100,
        num_trials=30,
    )
    metadata["encoder_cost_profile_reference"] = {
        "reference_shape": {
            "num_neurons": 64,
            "trial_length": 100,
            "num_trials": 30,
        },
        "supports_bin_streaming": bool(
            model.student.id_encoder.supports_bin_streaming
        ),
        **asdict(encoder_cost),
    }
    metadata["decoder_architecture"] = _oracle_decoder_metadata(
        model, dm, args
    )
    metadata["status"] = "trainer_ready"
    write_json(metadata_path, metadata)
    configure_multisession_metrics(model, dm)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch:03d}-{val_heldin/r2_mean:.4f}",
        monitor="val_heldin/r2_mean",
        mode="max",
        save_top_k=3,
    )
    epoch_ckpt_dir = output_dir / "epoch_ckpts"
    epoch_cb = ModelCheckpoint(
        dirpath=str(epoch_ckpt_dir),
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=12,
        accelerator=(
            "gpu"
            if args.accelerator == "gpu"
            or (
                args.accelerator == "auto"
                and torch.cuda.is_available()
            )
            else "cpu"
        ),
        devices=1,
        callbacks=[checkpoint_cb, epoch_cb],
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        default_root_dir=str(output_dir),
        deterministic=True,
        enable_progress_bar=not args.disable_progress_bar,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )
    trainer.fit(model, datamodule=dm)

    epoch_checkpoints = sorted(epoch_ckpt_dir.glob("epoch_*.ckpt"))
    if len(epoch_checkpoints) != 12:
        raise RuntimeError(
            "oracle protocol requires 12 epoch checkpoints, got "
            f"{len(epoch_checkpoints)}"
        )
    final_checkpoint = epoch_ckpt_dir / "epoch_011.ckpt"
    final_state = torch.load(
        str(final_checkpoint), map_location="cpu", weights_only=False
    )
    saved_receipt = final_state.get("teacher_head_oracle_receipt")
    if not isinstance(saved_receipt, dict):
        raise ValueError(
            "final oracle checkpoint is missing its receipt"
        )
    active_receipt = model.active_oracle_checkpoint_receipt()
    if saved_receipt.get("active_factor_sha256") != (
        active_receipt["active_factor_sha256"]
    ):
        raise ValueError(
            "live final oracle factors differ from final checkpoint"
        )

    metadata.update({
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "best_checkpoint": str(
            Path(checkpoint_cb.best_model_path).resolve()
        ),
        "best_checkpoint_sha256": sha256_file(
            Path(checkpoint_cb.best_model_path)
        ),
        "best_checkpoint_validation_r2": float(
            checkpoint_cb.best_model_score
        ),
        "epoch_checkpoints": [
            str(path.resolve()) for path in epoch_checkpoints
        ],
        "final_epoch_checkpoint": str(final_checkpoint.resolve()),
        "final_epoch_checkpoint_sha256": sha256_file(final_checkpoint),
        "oracle_final_active_checkpoint_receipt": active_receipt,
        "held_out_test_evaluated": False,
        "next_action": (
            "validation_only_fixed_epoch_window_scoring_via_"
            "eval_epoch_window_head_oracle_dandi688"
        ),
    })
    metadata["decoder_architecture"][
        "oracle_initialization_receipt_after_training"
    ] = model.oracle_initialization_receipt
    write_json(metadata_path, metadata)
    summary_path = (
        results_dir / f"p3_{args.out_name}_seed{args.seed}.json"
    )
    write_json(summary_path, metadata)
    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(
        "Best val R2: "
        f"{float(checkpoint_cb.best_model_score):.4f}"
    )
    print(f"Run metadata: {metadata_path}")
    print("No formal held-out test was opened or evaluated.")


if __name__ == "__main__":
    main()
