"""Dedicated, narrow source trainer for the sole fresh A1 H/T4 family.

This is intentionally separate from the SHA-sealed A2 trainer.  Its fixed
surface admits only the matched B3S/M30/task-only/joint-decoder A1 cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.regression import R2Score

REPO = Path(__file__).resolve().parents[2]
SUA = REPO / "sua_exploration"
STREAMING = REPO / "streaming_calibration_exp"
for root in (SUA, STREAMING):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from a1_hidden_carrier.a2_anchors import (  # noqa: E402
    DEV_SESSIONS,
    FORMAL_SESSIONS,
    MANIFEST,
    MANIFEST_SHA256,
    TEACHER,
    TEACHER_SHA256,
)
from a1_hidden_carrier.artifacts import canonical_json_sha256, sha256_file  # noqa: E402
from a1_hidden_carrier.contract import PILOT_SEED, SCREEN_ID  # noqa: E402
from a1_hidden_carrier.evidence import load_verified_preflight  # noqa: E402
from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule  # noqa: E402
from mc_maze.unit_side_features import (  # noqa: E402
    feature_semantics_version,
    side_feature_stats_sha256,
)
from src.models.a1_hidden_carrier_module import A1HiddenCarrierLitModule  # noqa: E402
from src.metrics.run_artifacts import assert_run_dir_is_fresh  # noqa: E402


def _write_mutable_run_json(path: Path, payload: dict) -> None:
    # Training metadata is updated once after fit and is not a scientific
    # receipt.  Immutable launch/score/aggregate evidence lives separately.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_metrics(model: A1HiddenCarrierLitModule, dm: Dandi688MultiSessionDataModule) -> None:
    val = dm.session_splits["val"]
    test = dm.session_splits["test"]
    model.val_heldin_r2 = nn.ModuleDict({name: R2Score(multioutput="variance_weighted") for name in val})
    model.val_heldout_r2 = nn.ModuleDict({name: R2Score(multioutput="variance_weighted") for name in val})
    model.test_heldin_r2 = nn.ModuleDict({name: R2Score(multioutput="variance_weighted") for name in test})
    model.test_heldout_r2 = nn.ModuleDict({name: R2Score(multioutput="variance_weighted") for name in test})


def _optimizer_coverage(model: A1HiddenCarrierLitModule) -> dict[str, object]:
    configured = model.configure_optimizers()
    optimizer = configured["optimizer"]
    expected = [(name, p) for name, p in model.student.named_parameters() if p.requires_grad]
    observed = [p for group in optimizer.param_groups for p in group["params"]]
    ids = [id(p) for p in observed]
    p_weight = model.student.hidden_carrier_map.weight
    return {
        "trainable_tensor_count": len(expected),
        "optimizer_tensor_count": len(ids),
        "duplicate_parameter_count": len(ids) - len(set(ids)),
        "all_trainable_parameters_exactly_once": set(ids) == {id(p) for _, p in expected} and len(ids) == len(set(ids)),
        "hidden_carrier_parameter_present": id(p_weight) in set(ids),
        "hidden_carrier_parameter_name": "student.hidden_carrier_map.weight",
        "hidden_carrier_shape": list(p_weight.shape),
        "decoder_tensor_count": sum(name.startswith("decoder.") for name, _ in expected),
        "identity_encoder_tensor_count": sum(name.startswith("id_encoder.") for name, _ in expected),
        "hidden_carrier_tensor_count": sum(name.startswith("hidden_carrier_map.") for name, _ in expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-preflight", required=True, type=Path)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=SUA / "data/dandi_000688/sub-C")
    parser.add_argument("--cache-dir", type=Path, default=SUA / "cache/dandi688_subc_co_v1")
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    # Smoke limits are integer batch counts.  Fractions are deliberately
    # forbidden here: on this very large dataloader an apparently tiny
    # fraction can still execute thousands of batches, while a fraction below
    # 1 / len(loader) fails before the first step.  Exact integer counts make
    # the production-format wiring check bounded and auditable.
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--wiring-smoke", action="store_true")
    args = parser.parse_args()
    if args.seed != PILOT_SEED:
        raise ValueError(f"pilot trainer accepts only seed {PILOT_SEED}")
    if args.accelerator != "gpu" and not args.wiring_smoke:
        raise ValueError("production A1 training requires GPU; CPU is wiring-smoke only")
    if args.wiring_smoke and (args.limit_train_batches is None or args.limit_val_batches is None):
        raise ValueError("wiring smoke requires explicit train/val batch limits")
    if args.wiring_smoke and (args.limit_train_batches < 1 or args.limit_val_batches < 1):
        raise ValueError("wiring smoke batch limits must be positive integer counts")
    if not args.wiring_smoke and (args.limit_train_batches is not None or args.limit_val_batches is not None):
        raise ValueError("pilot forbids batch limits")

    preflight, preflight_sha = load_verified_preflight(args.official_preflight)
    from a1_hidden_carrier.artifacts import load_verified_immutable_json

    launch, launch_sha = load_verified_immutable_json(args.launch_receipt, label="A1 pretraining launch receipt")
    expected_launch = {
        "schema_version": 2,
        "receipt_kind": "a1_hidden_space_carrier_pretraining_launch",
        "screen_id": SCREEN_ID,
        "cell": "H/T4",
        "seed": args.seed,
        "implementation_bindings_sha256": preflight["implementation_bindings_sha256"],
        "training_started": False,
        "formal_subc_test_nwb_opened": False,
        "no_test_files_evaluated": True,
    }
    for key, expected in expected_launch.items():
        if type(launch.get(key)) is not type(expected) or launch.get(key) != expected:
            raise ValueError(f"A1 launch receipt {key} drift")
    expected_status = (
        "CPU_WIRING_SMOKE_ONLY"
        if args.wiring_smoke
        else "ROOT_GO_AND_GPU_AUTHORIZED_PRETRAINING_ENVIRONMENT_VERIFIED"
    )
    if launch.get("status") != expected_status:
        raise ValueError("A1 launch receipt status does not match execution mode")
    if launch.get("official_preflight_sha256") != preflight_sha:
        raise ValueError("A1 launch/preflight pairing drift")
    if args.accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("A1 production trainer requires a visible CUDA GPU")
    if sha256_file(TEACHER) != TEACHER_SHA256 or sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("teacher or strict manifest byte drift")
    run_dir = args.run_dir.expanduser().resolve()
    assert_run_dir_is_fresh(run_dir)
    run_dir.mkdir(parents=True)
    metadata_path = run_dir / "run_metadata.json"
    epoch_dir = run_dir / "epoch_ckpts"

    pl.seed_everything(args.seed, workers=True)
    dm = Dandi688MultiSessionDataModule(
        data_dir=str(args.data_dir.resolve()), task="CO", split_counts=(27, 6, 6),
        batch_size=args.batch_size, window_size=50, calibration_n_trials=30,
        max_trial_length=100, bin_size_ms=20, num_workers=args.num_workers,
        random_calibration=False, seed=args.seed, max_units_exclusive=100,
        cache_dir=str(args.cache_dir.resolve()), signal_view="sua",
        side_feature_group="t4", side_feature_pool_size=30,
        side_permutation_seed=None, train_val_manifest_path=str(MANIFEST.resolve()),
    )
    dm.setup("fit")
    if tuple(dm.session_splits["val"]) != DEV_SESSIONS:
        raise ValueError("A1 development session roster drift")
    if tuple(dm.session_splits["test"]) != FORMAL_SESSIONS:
        raise ValueError("A1 formal exclusion roster drift")
    side_mean, side_std = dm._get_side_feature_stats()
    normalizer_sha = side_feature_stats_sha256(side_mean, side_std)
    if normalizer_sha != preflight["a2_reuse_evidence"]["t4_normalizer_sha256"]:
        raise ValueError("A1 T4 normalizer differs from sealed A2 source-only authority")

    optimizer_factory = partial(torch.optim.Adam, lr=1e-4, weight_decay=0.0)
    model = A1HiddenCarrierLitModule(
        task="mc_maze", variant="B3S", teacher_ckpt_path=str(TEACHER.resolve()),
        window_size=50, trial_length=100, id_hidden_dim=128, hidden_dim=64,
        pad_value=-1.0, freeze_decoder=False, freeze_encoder_base=False,
        loss_mode="task_only", lambda_y=0.0, lambda_E=0.0,
        decode_last_timestep_only=True, predict_scaled_behavior=True,
        behavior_scaling_factor=5.0, identity_mode="calibrated",
        fixed_slot_count=0, decoder_mode="coupled", side_dim=4,
        electrode_embed_dim=0, num_electrodes=0, optimizer=optimizer_factory,
        scheduler=None, compile=False, a1_attachment_mode="aligned",
        a1_attachment_permutation_seed=None, a1_carrier_dim=4,
    )
    model.setup("fit")
    coverage = _optimizer_coverage(model)
    if not coverage["all_trainable_parameters_exactly_once"] or not coverage["hidden_carrier_parameter_present"]:
        raise RuntimeError("A1 optimizer coverage proof failed before fit")
    _configure_metrics(model, dm)
    encoder_cost = asdict(model.student.id_encoder.cost_profile(64, 100, 30))

    metadata = {
        "schema_version": 2, "status": "initialized", "screen_id": SCREEN_ID,
        "receipt_kind": "a1_hidden_space_carrier_source_run_metadata",
        "created_at": datetime.now().astimezone().isoformat(), "seed": args.seed,
        "task": "CO", "variant": "B3S", "data_dir": str(args.data_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()), "train_val_manifest": str(MANIFEST.resolve()),
        "train_val_manifest_sha256": MANIFEST_SHA256, "teacher_checkpoint": str(TEACHER.resolve()),
        "teacher_sha256": TEACHER_SHA256, "session_splits": dm.session_splits,
        "held_out_test_evaluated": False, "formal_subc_test_nwb_opened": False,
        "official_preflight_sha256": preflight_sha, "launch_receipt_sha256": launch_sha,
        "side_features": {
            "group": "t4", "pool_size": 30, "side_dim": 4,
            "normalization_base_feature_group": "t4", "normalization_sha256": normalizer_sha,
            "feature_version": feature_semantics_version("t4"), "permutation_seed": None,
        },
        "decoder_architecture": {"mode": "coupled", "fixed_slot_count": 0},
        "a1_hidden_carrier": {
            "add_site": "hidden", "carrier": "t4", "activity_identity_carrier": "z4",
            "attachment_mode_during_training": "aligned", "carrier_dim": 4,
            "projection_parameter_name": "student.hidden_carrier_map.weight",
            "optimizer_coverage": coverage,
        },
        "training": {
            "max_epochs": 12, "no_early_stopping": True, "checkpoint_every_epoch": True,
            "learning_rate": 1e-4, "batch_size": args.batch_size, "window_size": 50,
            "calibration_n_trials": 30, "random_calibration": False,
            "calibration_selection": "chronological_first_n", "trial_length": 100,
            "bin_size_ms": 20, "loss_mode": "task_only", "lambda_y": 0.0,
            "lambda_E": 0.0, "identity_mode": "calibrated", "freeze_decoder": False,
            "freeze_encoder_base": False, "wiring_smoke": args.wiring_smoke,
        },
        "validation_protocol": {
            "calibration_trials": "trials[0:calibration_n_trials]",
            "evaluation_windows": "trials[calibration_n_trials:] only", "trial_disjoint": True,
        },
        "encoder_cost_profile_reference": encoder_cost,
    }
    _write_mutable_run_json(metadata_path, metadata)

    if args.wiring_smoke:
        # A one-batch validation smoke cannot visit every session required by
        # the aggregate R2 metric.  Monitor the always-emitted held-in loss so
        # that the smoke still exercises Lightning's production checkpoint
        # writer and yields a real-format checkpoint for the strict-loader
        # proof below.
        best = ModelCheckpoint(
            dirpath=str(run_dir), filename="best-smoke-{epoch:03d}",
            monitor="val_heldin/loss", mode="min", save_top_k=1,
        )
    else:
        best = ModelCheckpoint(
            dirpath=str(run_dir), filename="best-{epoch:03d}-{val_heldin/r2_mean:.4f}",
            monitor="val_heldin/r2_mean", mode="max", save_top_k=3,
        )
    every = ModelCheckpoint(
        dirpath=str(epoch_dir), filename="epoch_{epoch:03d}", auto_insert_metric_name=False,
        every_n_epochs=1, save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=1 if args.wiring_smoke else 12, accelerator=args.accelerator,
        devices=1, callbacks=[best, every], check_val_every_n_epoch=1,
        log_every_n_steps=50, default_root_dir=str(run_dir), deterministic=True,
        enable_progress_bar=False,
        num_sanity_val_steps=0 if args.wiring_smoke else 2,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
    )
    if trainer.accelerator.__class__.__name__.lower().startswith("cuda"):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    if torch.cuda.is_available() and args.accelerator == "gpu":
        torch.cuda.synchronize()
    checkpoints = sorted(epoch_dir.glob("epoch_*.ckpt"))
    expected_count = 1 if args.wiring_smoke else 12
    if len(checkpoints) != expected_count:
        raise RuntimeError(f"A1 epoch checkpoint count drift: {len(checkpoints)} != {expected_count}")
    bundle = {str(index + 1): sha256_file(path) for index, path in enumerate(checkpoints)}
    metadata.update(
        {
            "status": "wiring_smoke_completed" if args.wiring_smoke else "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "fit_wall_clock_seconds": time.perf_counter() - started,
            "epoch_checkpoints": [str(path.resolve()) for path in checkpoints],
            "epoch_checkpoint_sha256_bundle": bundle,
            "epoch_checkpoint_sha256_bundle_sha256": canonical_json_sha256(bundle),
            "best_checkpoint": str(Path(best.best_model_path).resolve()),
            "best_checkpoint_sha256": sha256_file(Path(best.best_model_path)),
            "held_out_test_evaluated": False,
            "formal_subc_test_nwb_opened": False,
        }
    )
    _write_mutable_run_json(metadata_path, metadata)
    print(json.dumps({"status": metadata["status"], "run_dir": str(run_dir), "bundle_sha256": metadata["epoch_checkpoint_sha256_bundle_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
