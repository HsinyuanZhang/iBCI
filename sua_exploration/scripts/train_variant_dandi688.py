"""Train encoder variant on DANDI 000688 multi-session data with cross-session splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics.regression import R2Score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

_sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_sce_root))
from src.metrics.run_artifacts import assert_run_dir_is_fresh
from src.models.streaming_calibration_module import StreamingCalibrationLitModule

DEFAULT_TEACHER = (
    Path(__file__).resolve().parents[1]
    / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)
def configure_multisession_metrics(
    model: StreamingCalibrationLitModule,
    dm: Dandi688MultiSessionDataModule,
) -> None:
    """Register per-session R² metrics for cross-session evaluation."""
    val_sessions = dm.session_splits["val"]
    test_sessions = dm.session_splits["test"]
    model.val_heldin_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in val_sessions}
    )
    model.val_heldout_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in val_sessions}
    )
    model.test_heldin_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in test_sessions}
    )
    model.test_heldout_r2 = nn.ModuleDict(
        {name: R2Score(multioutput="variance_weighted") for name in test_sessions}
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_split_counts(text: str) -> tuple[int, int, int]:
    raw_parts = text.split(",")
    if len(raw_parts) != 3:
        raise ValueError("split_counts must be three comma-separated non-negative integers")
    try:
        parts = [int(part.strip()) for part in raw_parts]
    except ValueError as exc:
        raise ValueError(
            "split_counts must be three comma-separated non-negative integers"
        ) from exc
    if any(part < 0 for part in parts):
        raise ValueError("split_counts must be three comma-separated non-negative integers")
    return parts[0], parts[1], parts[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, default=str(DEFAULT_TEACHER))
    parser.add_argument(
        "--variant",
        type=str,
        default="B3",
        choices=[
            "B0", "B3", "B3S", "B3T", "B3TS", "B3A", "B15P", "B15D", "B15", "B16",
            # T4-substrate electrode designs (docs/ELECTRODE_ANCHOR_DESIGNS.md):
            # B3SEG = design D (per-electrode reliability gate on T4's identity output),
            # B3SEA = design C (additive per-electrode anchor on T4's identity output).
            # Design A (learned electrode embedding concatenated alongside T4) needs no new
            # variant: it is plain B3S with --side_features t4e.
            "B3SEG", "B3SEA", "B3SCF", "B3SCFS", "B3SCFA",
            # Stage-0 same-electrode membership relation / parameter-matched no-group control.
            "B3SER", "B3SERN",
        ],
    )
    parser.add_argument(
        "--fixed_slot_count",
        type=int,
        default=0,
        help="Project variable session units into this many fixed decoder tokens after calibration.",
    )
    parser.add_argument(
        "--fixed_slot_dim",
        type=int,
        default=32,
        help="NeuronID-to-slot router latent width when --fixed_slot_count is positive.",
    )
    parser.add_argument(
        "--fixed_slot_mode",
        choices=["soft", "top1"],
        default="soft",
        help="Use dense soft routing or straight-through top-1 routing.",
    )
    parser.add_argument(
        "--fixed_slot_fusion",
        choices=["additive", "film"],
        default="film",
        help="Fuse routed spikes with calibration state before the fixed decoder.",
    )
    parser.add_argument(
        "--fixed_slot_temperature",
        type=float,
        default=1.0,
        help="Positive softmax temperature for the calibration-derived slot routing.",
    )
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sua_exploration/data/dandi_000688/sub-C",
    )
    parser.add_argument(
        "--train_val_manifest",
        type=str,
        default=None,
        help=(
            "Strict 27-train/6-validation manifest. Test entries are receipt names only; "
            "no test NWB is discovered or opened. Required by strict relation pilots."
        ),
    )
    parser.add_argument("--task", type=str, default="CO", choices=["CO", "RT"])
    parser.add_argument("--split_counts", type=str, default="37,8,8")
    parser.add_argument(
        "--max_units_exclusive",
        type=int,
        default=None,
        help="Optionally retain only sessions with strictly fewer units than this value.",
    )
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--no_early_stopping",
        action="store_true",
        help=(
            "M2 fixed-epoch-budget mode: disable EarlyStopping so training always runs "
            "exactly --max_epochs epochs. Does not change the default (early-stopping) "
            "path; --patience is ignored when this is set. See "
            "sua_exploration/docs/CURRENT_RESULTS.md section H.2 (unequal max-of-N "
            "selection bias from variants training different numbers of epochs)."
        ),
    )
    parser.add_argument(
        "--checkpoint_every_epoch",
        action="store_true",
        help=(
            "M3 deterministic-checkpoint mode: in addition to the existing best-3 "
            "checkpoint callback, save a checkpoint at every epoch under "
            "<output_dir>/epoch_ckpts/epoch_{epoch:03d}.ckpt (Lightning-native "
            "0-indexed epoch numbers). Intended to be combined with "
            "--no_early_stopping and --max_epochs 12 so "
            "scripts/eval_epoch_window_dandi688.py can score epochs 5-12 without ever "
            "selecting a checkpoint by argmax over the noisy validation metric. See "
            "sua_exploration/docs/CURRENT_RESULTS.md section H.3."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional shared disk cache for preprocessed session tensors.",
    )
    parser.add_argument(
        "--signal_view",
        choices=["sua", "pseudo_mua"],
        default="sua",
        help="Use sorted units directly or pool sorted units by their NWB electrode id.",
    )
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="Fail instead of silently falling back to CPU when CUDA is unavailable.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Trainer accelerator. cpu is intended for bounded wiring smokes only.",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=None,
        help="Optional Lightning batch limit for a bounded wiring smoke; never use for a pilot.",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=None,
        help="Optional Lightning validation-batch limit for a bounded wiring smoke; never use for a pilot.",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Disable Lightning's per-batch terminal progress bar for batch runs.",
    )
    parser.add_argument(
        "--freeze_decoder",
        action="store_true",
        help="Freeze teacher decoder (distillation mode). Default False = end-to-end.",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="task_only",
        help="Loss mode passed to StreamingCalibrationLitModule. Default task_only.",
    )
    parser.add_argument(
        "--identity_mode",
        type=str,
        default="calibrated",
        choices=["calibrated", "learned_prior"],
        help="Use identity source: calibration encoder (calibrated) or learned prior identity.",
    )
    parser.add_argument(
        "--side_features",
        choices=[
            "none", "f1", "f2", "f3", "fs1", "fs2", "fs3", "t4", "t8", "ts4", "ts8",
            # T4-substrate electrode designs (docs/ELECTRODE_ANCHOR_DESIGNS.md), variant
            # B3S (design A) or B3SEG/B3SEA (designs D/C) only -- see the variant/side_features
            # cross-validation below.
            "t4e", "t4e_shuffled", "t4gate", "t4gate_shuffled", "t4anchor", "t4anchor_shuffled",
            "t4rel", "t4rel_membership_shuffled", "t4rel_nogroup",
            "t4cf", "t4cf_ts4", "t4cf_confidence_shuffled",
        ],
        default="none",
        help=(
            "Per-unit side feature group for B3S/B3SEG/B3SEA (none disables side features). "
            "f1/f2 are waveform amplitude/shape scalars with fs1/fs2 shuffled controls; f3 "
            "adds a learned electrode-index embedding on top of f2 with fs3 as the shuffle "
            "control (UNIT_SIDE_FEATURE_ABLATION.md); t4/t8 are E3 directional tuning "
            "features with ts4/ts8 shuffled controls (E3_E4_ENCODER_PROGRAM.md section 1). "
            "t4e/t4e_shuffled (variant B3S, design A), t4gate/t4gate_shuffled (variant "
            "B3SEG, design D), t4anchor/t4anchor_shuffled (variant B3SEA, design C) each add "
            "an electrode-indexed mechanism on top of T4 (docs/ELECTRODE_ANCHOR_DESIGNS.md)."
        ),
    )
    parser.add_argument(
        "--side_feature_pool_size",
        type=int,
        default=50,
        help="Rewarded-trial pool size used to compute waveform side features.",
    )
    parser.add_argument(
        "--calibration_n_trials",
        type=int,
        default=10,
        help=(
            "Chronological activity-calibration prefix used by every arm during training. "
            "The matched SPINT/T4 mainline sets this equal to --side_feature_pool_size."
        ),
    )
    parser.add_argument(
        "--encoder_warmstart_path",
        type=str,
        default=None,
        help=(
            "Selected ordinary B3S/T4 full Lightning checkpoint. Required for B3SCF/B3SCFS/B3SCFA "
            "so decoder and T4 substrate both start exactly from the selected baseline."
        ),
    )
    args = parser.parse_args()
    if args.max_epochs <= 0 or args.patience < 0:
        raise ValueError("--max_epochs must be positive and --patience must be non-negative")
    if args.fixed_slot_count < 0:
        raise ValueError("--fixed_slot_count must be >= 0")
    if args.fixed_slot_count > 0 and args.fixed_slot_dim <= 0:
        raise ValueError("--fixed_slot_dim must be positive when fixed slots are enabled")
    if args.fixed_slot_temperature <= 0.0:
        raise ValueError("--fixed_slot_temperature must be positive")
    if args.side_feature_pool_size <= 0:
        raise ValueError("--side_feature_pool_size must be positive")
    if args.calibration_n_trials <= 0:
        raise ValueError("--calibration_n_trials must be positive")
    # B3S (design A / F1-F3 / T4-T8) plus B3SEG (design D, gate) / B3SEA (design C, anchor) --
    # docs/ELECTRODE_ANCHOR_DESIGNS.md -- are the only variants that consume --side_features.
    SIDE_FEATURE_VARIANTS = {"B3S", "B3TS", "B3SEG", "B3SEA", "B3SCF", "B3SCFS", "B3SCFA", "B3SER", "B3SERN"}
    if args.side_features != "none" and args.variant not in SIDE_FEATURE_VARIANTS:
        raise ValueError(f"--side_features requires --variant in {sorted(SIDE_FEATURE_VARIANTS)}")
    # B3SEG/B3SEA are always built on a T4 substrate (docs/ELECTRODE_ANCHOR_DESIGNS.md): each
    # requires exactly its own side_features token (real or shuffled-control), never "none" and
    # never the other design's token, so a misconfigured run can never silently train a
    # gate/anchor encoder without the T4 content it sits on top of. B3S must not be paired with
    # a gate/anchor token either -- that token's mechanism only exists on B3SEG/B3SEA; pairing
    # it with plain B3S would silently ignore the mechanism rather than raising.
    GATE_ANCHOR_VARIANT_SIDE_FEATURES = {
        "B3SEG": {"t4gate", "t4gate_shuffled"},
        "B3SEA": {"t4anchor", "t4anchor_shuffled"},
        "B3SCF": {"t4cf", "t4cf_ts4"},
        "B3SCFS": {"t4cf_confidence_shuffled"},
        "B3SCFA": {"t4cf"},
        "B3SER": {"t4rel", "t4rel_membership_shuffled"},
        "B3SERN": {"t4rel_nogroup"},
    }
    if args.variant in GATE_ANCHOR_VARIANT_SIDE_FEATURES:
        allowed = GATE_ANCHOR_VARIANT_SIDE_FEATURES[args.variant]
        if args.side_features not in allowed:
            raise ValueError(
                f"--variant {args.variant} requires --side_features in {sorted(allowed)}, "
                f"got {args.side_features!r}"
            )
    elif args.variant == "B3S" and args.side_features in {
        token for tokens in GATE_ANCHOR_VARIANT_SIDE_FEATURES.values() for token in tokens
    }:
        raise ValueError(
            f"--side_features {args.side_features!r} requires --variant B3SEG/B3SEA (its "
            "gate/anchor mechanism does not exist on plain B3S), not B3S"
        )
    if args.variant == "B3TS" and args.side_features not in {"t4", "ts4"}:
        raise ValueError(
            "B3TS is the predeclared temporal-functional screen and requires "
            "--side_features t4 or ts4"
        )
    if args.side_features == "none":
        side_dim = 0
        electrode_embed_dim = 0
        num_electrodes = 0
    else:
        from mc_maze.unit_side_features import (
            ELECTRODE_EMBED_DIM,
            SIDE_FEATURE_DIMS,
            uses_electrode_embedding,
            uses_electrode_relation_membership,
        )

        side_dim = SIDE_FEATURE_DIMS[args.side_features]
        electrode_embed_dim = ELECTRODE_EMBED_DIM if uses_electrode_embedding(args.side_features) else 0
        num_electrodes = 0
        relation_membership = uses_electrode_relation_membership(args.side_features)
    if args.variant in {"B3SCF", "B3SCFS", "B3SCFA"} and args.encoder_warmstart_path is None:
        raise ValueError(
            "B3SCF/B3SCFS/B3SCFA require --encoder_warmstart_path from the selected ordinary T4 checkpoint; "
            "refusing a confounded from-scratch FiLM run"
        )
    if args.encoder_warmstart_path is not None and not Path(args.encoder_warmstart_path).expanduser().is_file():
        raise FileNotFoundError(f"Encoder warm-start does not exist: {args.encoder_warmstart_path}")
    if args.side_features == "none":
        relation_membership = False
    from mc_maze.unit_side_features import confidence_component_shuffle, is_shuffled_control

    # fs1/fs2 permute their dimension-matched base feature group (f1/f2 respectively)
    # along the unit axis; the permutation itself is seeded from --seed so it is
    # reproducible per run but independent across seeds (UNIT_SIDE_FEATURE_ABLATION.md
    # section 6). Computed once and reused for both run_metadata provenance and the
    # datamodule constructor below so the two can never silently drift apart.
    side_permutation_seed = (
        args.seed if is_shuffled_control(args.side_features)
        or confidence_component_shuffle(args.side_features) is not None else None
    )
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("--require_gpu was set but CUDA is unavailable")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("--accelerator gpu was set but CUDA is unavailable")

    pl.seed_everything(args.seed, workers=True)

    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_ckpt}")
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    manifest_path = (
        Path(args.train_val_manifest).expanduser().resolve()
        if args.train_val_manifest else None
    )
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(f"Strict train/validation manifest does not exist: {manifest_path}")

    split_counts = parse_split_counts(args.split_counts)
    out_name = args.out_name or f"{args.variant.lower()}_dandi688_{args.task.lower()}"
    output_dir = Path(__file__).resolve().parents[1] / "checkpoints" / out_name
    results_dir = Path(__file__).resolve().parents[1] / "results"
    # M1 hard assertion (sua_exploration/docs/CURRENT_RESULTS.md section H.4): refuse to
    # reuse a checkpoint directory that already holds another run's checkpoints or
    # tfevents. --out_name defaults to a seed-agnostic name, so re-invoking this script
    # twice without a fresh --out_name (e.g. two seeds) would otherwise silently
    # commingle both runs' checkpoints the way the MUA Hydra run dir once did.
    assert_run_dir_is_fresh(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata_path = output_dir / "run_metadata.json"

    run_metadata = {
        "schema_version": 1,
        "status": "initialized",
        "created_at": datetime.now().astimezone().isoformat(),
        "variant": args.variant,
        "fixed_slot": {
            "enabled": args.fixed_slot_count > 0,
            "count": args.fixed_slot_count,
            "router_dim": args.fixed_slot_dim,
            "routing_mode": args.fixed_slot_mode,
            "fusion": args.fixed_slot_fusion,
            "temperature": args.fixed_slot_temperature,
            "deployment_contract": (
                "Calibration derives session-local routing; online decoding projects all "
                "variable unit windows into fixed slot_count tokens before decoder.fc_in."
                if args.fixed_slot_count > 0
                else None
            ),
        },
        "seed": args.seed,
        "teacher_checkpoint": str(teacher_ckpt),
        "teacher_sha256": sha256_file(teacher_ckpt),
        "encoder_warmstart_path": (
            str(Path(args.encoder_warmstart_path).expanduser().resolve())
            if args.encoder_warmstart_path else None
        ),
        "encoder_warmstart_sha256": (
            sha256_file(Path(args.encoder_warmstart_path).expanduser())
            if args.encoder_warmstart_path else None
        ),
        "data_dir": str(data_dir),
        "train_val_manifest": str(manifest_path) if manifest_path else None,
        "train_val_manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()) if args.cache_dir else None,
        "signal_view": args.signal_view,
        "side_features": {
            "group": args.side_features,
            "pool_size": args.side_feature_pool_size,
            "side_dim": side_dim,
            "electrode_embed_dim": electrode_embed_dim,
            "num_electrodes": num_electrodes,
            "uses_equality_only_relation_membership": relation_membership,
            "permutation_seed": side_permutation_seed,
        },
        "task": args.task,
        "split_counts": list(split_counts),
        "max_units_exclusive": args.max_units_exclusive,
        "output_dir": str(output_dir.resolve()),
        "held_out_evaluation_protocol": {
            "name": "frozen_gradient_free_streaming_calibration",
            "held_out_test_evaluated": False,
            "formal_test_entrypoint": "scripts/eval_adaptation_dandi688.py",
            "model_weights": "best checkpoint must be frozen during held-out evaluation",
            "calibration_input": "held-out session spikes only",
            "held_out_behavior_labels_used_for_updates": False,
            "backward_gradients_on_held_out_sessions": False,
        },
        "validation_protocol": {
            "calibration_trials": "trials[0:calibration_n_trials]",
            "evaluation_windows": "trials[calibration_n_trials:] only",
            "trial_disjoint": True,
        },
        "training": {
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "no_early_stopping": args.no_early_stopping,
            "checkpoint_every_epoch": args.checkpoint_every_epoch,
            "epoch_checkpoints_dir": (
                str((output_dir / "epoch_ckpts").resolve()) if args.checkpoint_every_epoch else None
            ),
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "window_size": 50,
            "calibration_n_trials": args.calibration_n_trials,
            "trial_length": 100,
            "bin_size_ms": 20,
            "loss_mode": args.loss_mode,
            "identity_mode": args.identity_mode,
            "lambda_y": 1.0,
            "lambda_E": 0.1,
            "freeze_decoder": args.freeze_decoder,
            "decode_last_timestep_only": True,
            "behavior_scaling_factor": 5.0,
            "deterministic": True,
            "progress_bar": not args.disable_progress_bar,
            "accelerator": args.accelerator,
            "limit_train_batches": args.limit_train_batches,
            "limit_val_batches": args.limit_val_batches,
        },
    }
    write_json(run_metadata_path, run_metadata)

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(data_dir),
        task=args.task,
        split_counts=split_counts,
        batch_size=args.batch_size,
        window_size=50,
        calibration_n_trials=args.calibration_n_trials,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=args.num_workers,
        seed=args.seed,
        max_units_exclusive=args.max_units_exclusive,
        cache_dir=args.cache_dir,
        signal_view=args.signal_view,
        side_feature_group=None if args.side_features == "none" else args.side_features,
        side_feature_pool_size=args.side_feature_pool_size,
        side_permutation_seed=side_permutation_seed,
        train_val_manifest_path=args.train_val_manifest,
    )
    dm.setup("fit")
    if args.side_features != "none":
        from mc_maze.unit_side_features import (
            compute_electrode_vocab_size,
            feature_semantics_version,
            side_feature_stats_sha256,
            uses_electrode_ids,
            uses_electrode_relation_membership,
        )

        side_mean, side_std = dm._get_side_feature_stats()
        assert side_mean is not None and side_std is not None
        # uses_electrode_ids (not the narrower uses_electrode_embedding) so B3SEG/B3SEA's own
        # gate/anchor tables -- which do not use the concat-embedding mechanism -- still get a
        # correctly-sized vocabulary (docs/ELECTRODE_ANCHOR_DESIGNS.md).
        if uses_electrode_ids(args.side_features) and not uses_electrode_relation_membership(args.side_features):
            train_paths = [Path(path) for path in dm.session_files["train"]]
            num_electrodes = compute_electrode_vocab_size(train_paths)
            run_metadata["side_features"]["num_electrodes"] = num_electrodes
        run_metadata["side_features"].update({
            "feature_version": feature_semantics_version(args.side_features),
            "normalization_sha256": side_feature_stats_sha256(side_mean, side_std),
        })
    run_metadata["session_splits"] = dm.session_splits
    run_metadata["trainer_fit_validation_loader_contract"] = {
        "loader_0_sessions": dm.session_splits["val"],
        "loader_1_sessions": dm.session_splits["val"],
        "legacy_metric_names": ["val_heldin/*", "val_heldout/*"],
        "formal_test_sessions_loaded_during_fit": False,
        "formal_test_entrypoint": "scripts/eval_adaptation_dandi688.py",
    }
    run_metadata["session_unit_counts"] = dm.session_unit_counts
    run_metadata["session_channel_counts"] = dm.session_channel_counts
    run_metadata["session_files"] = {
        split: [str(path) for path in files]
        for split, files in dm.session_files.items()
    }
    write_json(run_metadata_path, run_metadata)

    optimizer = partial(torch.optim.Adam, lr=args.lr, weight_decay=0.0)
    model = StreamingCalibrationLitModule(
        task="mc_maze",
        variant=args.variant,
        teacher_ckpt_path=str(teacher_ckpt),
        window_size=50,
        trial_length=100,
        id_hidden_dim=128,
        hidden_dim=64,
        pad_value=-1.0,
        freeze_decoder=args.freeze_decoder,
        loss_mode=args.loss_mode,
        lambda_y=1.0,
        lambda_E=0.1,
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=5.0,
        identity_mode=args.identity_mode,
        fixed_slot_count=args.fixed_slot_count,
        fixed_slot_dim=args.fixed_slot_dim,
        fixed_slot_mode=args.fixed_slot_mode,
        fixed_slot_fusion=args.fixed_slot_fusion,
        fixed_slot_temperature=args.fixed_slot_temperature,
        side_dim=side_dim,
        electrode_embed_dim=electrode_embed_dim,
        num_electrodes=num_electrodes if args.side_features != "none" else 0,
        encoder_warmstart_path=args.encoder_warmstart_path,
        optimizer=optimizer,
        scheduler=None,
        compile=False,
    )
    # Build the student once before Trainer.fit so the exact instantiated
    # encoder—not a hand-maintained estimate—can be receipted in metadata.
    # Lightning's later setup("fit") is idempotent.
    model.setup("fit")
    assert model.student is not None
    encoder_cost = model.student.id_encoder.cost_profile(
        num_neurons=64,
        trial_length=100,
        num_trials=args.calibration_n_trials,
    )
    run_metadata["encoder_cost_profile_reference"] = {
        "reference_shape": {
            "num_neurons": 64,
            "trial_length": 100,
            "num_trials": args.calibration_n_trials,
        },
        **asdict(encoder_cost),
    }
    write_json(run_metadata_path, run_metadata)
    configure_multisession_metrics(model, dm)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="best-{epoch:03d}-{val_heldin/r2_mean:.4f}",
        monitor="val_heldin/r2_mean",
        mode="max",
        save_top_k=3,
    )
    callbacks: list = [checkpoint_cb]

    # M2 fixed-epoch-budget mode: default (False) leaves the existing early-stopping
    # path unchanged; --no_early_stopping drops it so every variant trains exactly
    # --max_epochs epochs (sua_exploration/docs/CURRENT_RESULTS.md section H.2).
    if not args.no_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_heldin/r2_mean",
                mode="max",
                patience=args.patience,
            )
        )

    # M3 deterministic-checkpoint mode: save every epoch (Lightning-native 0-indexed
    # filenames) instead of relying only on the argmax-selected best_checkpoint above.
    # scripts/eval_epoch_window_dandi688.py hardcodes which epoch files it needs
    # (sua_exploration/docs/CURRENT_RESULTS.md section H.3).
    epoch_ckpt_dir = output_dir / "epoch_ckpts"
    if args.checkpoint_every_epoch:
        callbacks.append(
            # monitor intentionally left unset: save_top_k=-1 already saves every
            # trigger unconditionally, and Lightning derives ModelCheckpoint.state_key
            # from (monitor, mode, every_n_train_steps, every_n_epochs,
            # train_time_interval) -- reusing checkpoint_cb's monitor/mode here would
            # give both callbacks an identical state_key and Lightning would refuse to
            # start ("Found more than one stateful callback of type `ModelCheckpoint`").
            ModelCheckpoint(
                dirpath=str(epoch_ckpt_dir),
                filename="epoch_{epoch:03d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=-1,
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=(
            "gpu" if args.accelerator == "gpu" or (args.accelerator == "auto" and torch.cuda.is_available())
            else "cpu"
        ),
        devices=1,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        default_root_dir=str(output_dir),
        deterministic=True,
        enable_progress_bar=not args.disable_progress_bar,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
    )

    trainer.fit(model, datamodule=dm)
    epoch_checkpoints = (
        sorted(str(path.resolve()) for path in epoch_ckpt_dir.glob("epoch_*.ckpt"))
        if args.checkpoint_every_epoch
        else []
    )
    run_metadata.update({
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(),
        "best_checkpoint": str(Path(checkpoint_cb.best_model_path).resolve()),
        "best_checkpoint_sha256": sha256_file(Path(checkpoint_cb.best_model_path)),
        "best_checkpoint_validation_r2": float(checkpoint_cb.best_model_score),
        "best_checkpoint_selection_caveat": (
            "best_checkpoint is an argmax over a noisy per-epoch validation metric "
            "(sua_exploration/docs/CURRENT_RESULTS.md section H.1-H.3); for the M3 "
            "deterministic protocol use epoch_checkpoints with "
            "scripts/eval_epoch_window_dandi688.py instead, never this field."
        ),
        "epoch_checkpoints": epoch_checkpoints,
        "held_out_test_evaluated": False,
        "next_action": (
            "lock_best_checkpoint_then_run_the_single_formal_frozen_gradient_free_"
            "disjoint_test_via_eval_adaptation_dandi688"
        ),
        "held_out_test_usage": (
            "Not evaluated by this training run. After validation-only model selection, "
            "evaluate exactly once via eval_adaptation_dandi688.py; never select variants, "
            "tune hyperparameters, initiate rerun searches, or update weights."
        ),
    })
    write_json(run_metadata_path, run_metadata)
    summary_path = results_dir / f"p3_{out_name}_seed{args.seed}.json"
    write_json(summary_path, run_metadata)

    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val R2: {checkpoint_cb.best_model_score:.4f}")
    print(f"Run metadata: {run_metadata_path}")
    print(
        "No held-out test was run. Lock this validation-selected checkpoint, then run "
        "the single formal frozen, gradient-free disjoint test via "
        "eval_adaptation_dandi688.py."
    )


if __name__ == "__main__":
    main()
