#!/usr/bin/env python3
"""Run one real-data, GPU-only TS4 paired-view C1 microfit audit.

This is an operational v3 prelaunch gate, not a training run or scorer.  It
resolves only the 27 source and 6 development sessions named by the strict
train/validation manifest.  Formal-test identifiers remain names in the
manifest; this script never resolves or opens a formal-test path.

The audit builds the two independent TS4 datamodules, validates descriptor and
pairing contracts, performs exactly one 0.5 * SUA + 0.5 * pseudo-MUA task-loss
optimizer step on one real paired GPU batch, then round-trips a temporary
checkpoint through the standard frozen-model loader.  The temporary checkpoint
is hashed, reloaded, and deleted before the write-once JSON receipt is emitted.
No evaluation loop or R-squared metric is invoked.
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import lightning.pytorch as pl
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SUA_ROOT = ROOT / "sua_exploration"
SCE_ROOT = ROOT / "streaming_calibration_exp"
sys.path.insert(0, str(SUA_ROOT))
sys.path.insert(0, str(SUA_ROOT / "scripts"))
sys.path.insert(0, str(SCE_ROOT))

from mc_maze.multisession_datamodule import (  # noqa: E402
    Dandi688MultiSessionDataModule,
    load_frozen_train_val_manifest,
)
from mc_maze.paired_view_c1 import (  # noqa: E402
    PairedViewC1DataModule,
    normalizer_hashes_are_distinct,
    validate_pair_batch,
)
from mc_maze.unit_side_features import (  # noqa: E402
    base_feature_group,
    fit_side_feature_stats,
    permute_side_feature_rows,
    side_feature_stats_sha256,
)
from select_gradient_free_protocol_dandi688 import load_frozen_model  # noqa: E402
from src.models.paired_view_c1_module import PairedViewC1LitModule  # noqa: E402


EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_PARAMETER_COUNT = 4_613_178
DEFAULT_DATA_DIR = SUA_ROOT / "data/dandi_000688/sub-C"
DEFAULT_MANIFEST = SUA_ROOT / "configs/subc_co_27_6_strict_train_val_manifest.json"
DEFAULT_TEACHER = (
    SUA_ROOT
    / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)
DEFAULT_CACHE_ROOT = (
    SUA_ROOT / "cache/t4_paired_view_c1_fresh_controls_prelaunch_v1_20260804"
)
DEFAULT_RECEIPT_DIR = (
    SUA_ROOT / "results/t4_paired_view_c1_v3_ts4_microfit_audit_20260804"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Claim the receipt path only after JSON serialization succeeds."""
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialized)


def runtime_environment(gpu: int) -> dict[str, Any]:
    """JSON-safe, score-free runtime provenance."""
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_gpu": gpu,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available() and gpu < torch.cuda.device_count():
        properties = torch.cuda.get_device_properties(gpu)
        payload["requested_gpu_properties"] = {
            "logical_index": gpu,
            "name": str(properties.name),
            "uuid": (
                str(getattr(properties, "uuid"))
                if getattr(properties, "uuid", None) is not None
                else None
            ),
            "total_memory_bytes": int(properties.total_memory),
        }
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
    except Exception as exc:  # provenance must not fail due to a diagnostic command
        payload["nvidia_smi_error"] = repr(exc)
    return payload


def source_hashes() -> dict[str, str]:
    """Hash the implementation closure used by this microfit only."""
    relatives = (
        "sua_exploration/mc_maze/multisession_datamodule.py",
        "sua_exploration/mc_maze/unit_side_features.py",
        "sua_exploration/mc_maze/paired_view_c1.py",
        "sua_exploration/scripts/audit_t4_paired_view_c1_v3_ts4_microfit.py",
        "sua_exploration/scripts/train_paired_view_c1_dandi688.py",
        "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
        "sua_exploration/scripts/eval_adaptation_dandi688.py",
        "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        "streaming_calibration_exp/src/models/paired_view_c1_module.py",
        "streaming_calibration_exp/src/models/components/streaming_encoders.py",
        "streaming_calibration_exp/src/models/components/streaming_spint.py",
    )
    result: dict[str, str] = {}
    for relative in relatives:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"microfit source missing: {path}")
        result[relative] = sha256_file(path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def _permitted_manifest_receipt(
    manifest: Path, data_dir: Path
) -> tuple[dict[str, Any], list[Path], list[Path]]:
    """Resolve exactly the permitted 27+6 paths and retain formal names only."""
    train_files, val_files, test_names = load_frozen_train_val_manifest(
        manifest, data_dir
    )
    if len(train_files) != 27 or len(val_files) != 6 or len(test_names) != 6:
        raise ValueError("strict C1 manifest must remain 27 train + 6 val + 6 sealed names")
    selected_data_dir = data_dir.resolve()
    for path in (*train_files, *val_files):
        if path.parent != selected_data_dir or not path.is_file():
            raise ValueError(f"permitted manifest path drift: {path}")
    return (
        {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "train_file_count": len(train_files),
            "validation_file_count": len(val_files),
            "permitted_file_count": len(train_files) + len(val_files),
            "train_filenames": [path.name for path in train_files],
            "validation_filenames": [path.name for path in val_files],
            "formal_test_identifier_count": len(test_names),
            "formal_test_paths_resolved": False,
        },
        train_files,
        val_files,
    )


def _make_ts4_datamodule(
    *,
    data_dir: Path,
    manifest: Path,
    cache_dir: Path,
    signal_view: str,
    seed: int,
) -> Dandi688MultiSessionDataModule:
    return Dandi688MultiSessionDataModule(
        data_dir=str(data_dir),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=32,
        window_size=50,
        calibration_n_trials=10,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=0,
        random_calibration=False,
        seed=seed,
        max_units_exclusive=100,
        cache_dir=str(cache_dir),
        signal_view=signal_view,
        side_feature_group="ts4",
        side_feature_pool_size=50,
        side_permutation_seed=seed,
        train_val_manifest_path=str(manifest),
    )


def _assert_33_only(
    dm: Dandi688MultiSessionDataModule,
    *,
    expected_train: Sequence[Path],
    expected_val: Sequence[Path],
) -> None:
    if dm.session_files.get("test") != []:
        raise RuntimeError("TS4 microfit must not materialize a test-file path")
    if dm.session_files.get("train") != list(expected_train):
        raise RuntimeError("TS4 microfit train-file manifest drift")
    if dm.session_files.get("val") != list(expected_val):
        raise RuntimeError("TS4 microfit validation-file manifest drift")
    if len(dm.session_splits.get("test", [])) != 6:
        raise RuntimeError("TS4 microfit must retain six sealed test identifiers only")


def _normalizer_receipt(dm: Dandi688MultiSessionDataModule) -> dict[str, Any]:
    """Prove TS4 uses this view's ordinary T4 train-only normalizer."""
    ts4_stats = dm._get_side_feature_stats()
    if ts4_stats is None:
        raise RuntimeError("TS4 datamodule did not initialize a side-feature normalizer")
    ts4_mean, ts4_std = ts4_stats
    raw_group = base_feature_group("ts4")
    if raw_group != "t4":
        raise RuntimeError("TS4 must resolve to the ordinary T4 normalizer substrate")
    t4_mean, t4_std = fit_side_feature_stats(
        dm.session_files["train"],
        feature_group=raw_group,
        pool_size=50,
        cache_dir=dm.cache_dir,
        bin_size_ms=20,
        window_size=50,
        trial_result_filter="R",
        signal_view=dm.signal_view,
    )
    ts4_sha = side_feature_stats_sha256(ts4_mean, ts4_std)
    t4_sha = side_feature_stats_sha256(t4_mean, t4_std)
    if ts4_sha != t4_sha or not np.array_equal(ts4_mean, t4_mean) or not np.array_equal(ts4_std, t4_std):
        raise RuntimeError(f"{dm.signal_view}: T4/TS4 normalizer mismatch")
    return {
        "signal_view": dm.signal_view,
        "resolved_raw_group": raw_group,
        "t4_normalizer_sha256": t4_sha,
        "ts4_normalizer_sha256": ts4_sha,
        "same_view_t4_ts4_equal": True,
        "normalization_scope": "source_train_27_only",
    }


def _descriptor_shape_receipt(dm: Dandi688MultiSessionDataModule) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, dataset in (("train", dm.train_dataset), ("validation", dm.val_dataset)):
        if dataset is None:
            raise RuntimeError(f"{dm.signal_view} {split} dataset is missing")
        sessions: dict[str, Any] = {}
        for name, record in dataset.sessions.items():
            side = record.side_features
            expected_shape = (record.neural.shape[1], 4)
            if side is None or side.shape != expected_shape or not np.isfinite(side).all():
                raise RuntimeError(
                    f"{dm.signal_view}/{split}/{name}: TS4 must be finite with shape {expected_shape}"
                )
            sessions[name] = {
                "channel_count": int(record.neural.shape[1]),
                "descriptor_shape": [int(side.shape[0]), int(side.shape[1])],
                "finite": True,
            }
        result[split] = {"session_count": len(sessions), "sessions": sessions}
    if result["train"]["session_count"] != 27 or result["validation"]["session_count"] != 6:
        raise RuntimeError(f"{dm.signal_view}: descriptor receipt is not 27+6")
    return result


def _row_permutation_receipt(
    descriptor_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Audit all fixed seeds from channel counts only; do not reopen any NWB."""
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        sessions: dict[str, Any] = {}
        for session_name, row in descriptor_receipt[split]["sessions"].items():
            channel_count = int(row["channel_count"])
            if channel_count < 2:
                raise RuntimeError(
                    f"{split}/{session_name}: TS4 requires at least two rows for a nonidentity control"
                )
            seed_rows: dict[str, Any] = {}
            identity = np.arange(channel_count, dtype=np.int64)
            probe = identity.reshape(-1, 1)
            for seed in EXPECTED_SEEDS:
                order = permute_side_feature_rows(
                    probe, permutation_seed=seed
                ).reshape(-1).astype(np.int64, copy=False)
                if not np.array_equal(np.sort(order), identity):
                    raise RuntimeError(
                        f"{split}/{session_name}/s{seed}: TS4 order is not a full row permutation"
                    )
                if np.array_equal(order, identity):
                    raise RuntimeError(
                        f"{split}/{session_name}/s{seed}: TS4 order is identity"
                    )
                seed_rows[str(seed)] = {
                    "row_count": channel_count,
                    "nonidentity": True,
                    "complete_row_multiset_preserved": True,
                    "permutation_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
                }
            sessions[session_name] = seed_rows
        result[split] = sessions
    return result


def _axis_contract_receipt(paired: PairedViewC1DataModule) -> dict[str, Any]:
    exposure = paired.exposure_receipt()
    result: dict[str, Any] = {}
    for split in ("train", "validation"):
        axis = exposure[split]["axis_alignment"]
        if not (
            axis.get("behavior_targets_bitwise_equal") is True
            and axis.get("valid_time_indices_bitwise_equal") is True
            and axis.get("count_conservation_checked") is True
        ):
            raise RuntimeError(f"paired {split} axis/target/count contract failed")
        result[split] = {
            "sample_count": int(exposure[split]["sample_count"]),
            "batch_count": int(exposure[split]["batch_count"]),
            "session_count": int(axis["session_count"]),
            "behavior_targets_bitwise_equal": True,
            "valid_time_indices_bitwise_equal": True,
            "count_conservation_checked": True,
        }
    return result


def _move_batch_to_device(
    batch: Sequence[Any], device: torch.device
) -> tuple[Any, ...]:
    return tuple(
        value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for value in batch
    )


def _batch_shape_receipt(
    sua_batch: Sequence[Any], pseudo_batch: Sequence[Any]
) -> dict[str, Any]:
    validate_pair_batch(sua_batch, pseudo_batch)
    if len(sua_batch) != 5 or len(pseudo_batch) != 5:
        raise RuntimeError("ordinary C1 TS4 microfit requires five-item descriptor batches")
    sua_neural, sua_target, sua_calib, sua_sessions, sua_side = sua_batch
    pseudo_neural, pseudo_target, pseudo_calib, pseudo_sessions, pseudo_side = pseudo_batch
    if tuple(sua_sessions) != tuple(pseudo_sessions) or not torch.equal(sua_target, pseudo_target):
        raise RuntimeError("paired microfit batch lost matched session/target exposure")
    return {
        "sua": {
            "neural_shape": list(sua_neural.shape),
            "target_shape": list(sua_target.shape),
            "calibration_shape": list(sua_calib.shape),
            "side_feature_shape": list(sua_side.shape),
        },
        "pseudo_mua": {
            "neural_shape": list(pseudo_neural.shape),
            "target_shape": list(pseudo_target.shape),
            "calibration_shape": list(pseudo_calib.shape),
            "side_feature_shape": list(pseudo_side.shape),
        },
        "matched_session_names": list(sua_sessions),
        "matched_targets_bitwise_equal": True,
    }


def _make_model(teacher: Path) -> PairedViewC1LitModule:
    optimizer = functools.partial(torch.optim.Adam, lr=1e-4, weight_decay=0.0)
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
    if model.student is None:
        raise RuntimeError("paired TS4 microfit did not initialize a student")
    return model


def _checkpoint_round_trip(
    *,
    model: PairedViewC1LitModule,
    teacher: Path,
    receipt_parent: Path,
) -> dict[str, Any]:
    """Save, formal-load, count, hash, and delete one temporary checkpoint."""
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".c1_v3_ts4_microfit_ckpt_", dir=receipt_parent)
    )
    checkpoint_path = temporary_dir / "paired_ts4_microfit.ckpt"
    checkpoint_sha256: str | None = None
    checkpoint_bytes: int | None = None
    cleanup_ok = False
    try:
        torch.save(
            {
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__,
            },
            checkpoint_path,
        )
        checkpoint_sha256 = sha256_file(checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
        reloaded = load_frozen_model(
            checkpoint_path,
            teacher,
            "B3S",
            torch.device("cpu"),
        )
        if reloaded.student is None:
            raise RuntimeError("formal frozen-model loader returned no student")
        reloaded_parameter_count = int(
            sum(parameter.numel() for parameter in reloaded.student.parameters())
        )
        if reloaded_parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                "formal frozen-model loader parameter count drift: "
                f"expected {EXPECTED_PARAMETER_COUNT}, got {reloaded_parameter_count}"
            )
        del reloaded
    finally:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if temporary_dir.exists():
            try:
                temporary_dir.rmdir()
            except OSError:
                shutil.rmtree(temporary_dir)
        cleanup_ok = not checkpoint_path.exists() and not temporary_dir.exists()
    if checkpoint_sha256 is None or checkpoint_bytes is None or not cleanup_ok:
        raise RuntimeError("temporary checkpoint was not safely hashed and deleted")
    return {
        "temporary_checkpoint_path": str(checkpoint_path),
        "sha256_before_deletion": checkpoint_sha256,
        "bytes_before_deletion": checkpoint_bytes,
        "formal_loader": "select_gradient_free_protocol_dandi688.load_frozen_model",
        "reloaded_student_parameter_count": EXPECTED_PARAMETER_COUNT,
        "deleted_after_success": True,
        "path_absent_after_cleanup": True,
    }


def _run_audit(args: argparse.Namespace, *, receipt_parent: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("C1 v3 TS4 microfit requires a CUDA GPU")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"requested GPU {args.gpu} is unavailable")
    data_dir = args.data_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    teacher = args.teacher.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve()
    if not data_dir.is_dir() or not manifest.is_file() or not teacher.is_file():
        raise FileNotFoundError("data directory, strict manifest, or teacher is missing")
    sua_cache = cache_root / "sua"
    pseudo_cache = cache_root / "pseudo_mua"
    if not sua_cache.is_dir() or not pseudo_cache.is_dir() or sua_cache == pseudo_cache:
        raise ValueError("microfit requires the existing distinct SUA/pseudo-MUA cache namespaces")

    manifest_receipt, train_files, val_files = _permitted_manifest_receipt(
        manifest, data_dir
    )
    pl.seed_everything(args.seed, workers=True)
    sua_dm = _make_ts4_datamodule(
        data_dir=data_dir,
        manifest=manifest,
        cache_dir=sua_cache,
        signal_view="sua",
        seed=args.seed,
    )
    pseudo_dm = _make_ts4_datamodule(
        data_dir=data_dir,
        manifest=manifest,
        cache_dir=pseudo_cache,
        signal_view="pseudo_mua",
        seed=args.seed,
    )
    paired = PairedViewC1DataModule(
        sua_dm,
        pseudo_dm,
        paired_loss_weights=(0.5, 0.5),
        lambda_consistency=0.0,
    )
    paired.setup()
    _assert_33_only(sua_dm, expected_train=train_files, expected_val=val_files)
    _assert_33_only(pseudo_dm, expected_train=train_files, expected_val=val_files)

    sua_normalizer = _normalizer_receipt(sua_dm)
    pseudo_normalizer = _normalizer_receipt(pseudo_dm)
    if not normalizer_hashes_are_distinct(
        sua_normalizer["ts4_normalizer_sha256"],
        pseudo_normalizer["ts4_normalizer_sha256"],
    ):
        raise RuntimeError("SUA/pseudo-MUA TS4 normalizers must remain distinct")
    sua_descriptors = _descriptor_shape_receipt(sua_dm)
    pseudo_descriptors = _descriptor_shape_receipt(pseudo_dm)
    axis_contract = _axis_contract_receipt(paired)
    permutation_contract = {
        "sua": _row_permutation_receipt(sua_descriptors),
        "pseudo_mua": _row_permutation_receipt(pseudo_descriptors),
        "seeds_checked": list(EXPECTED_SEEDS),
        "derived_without_reopening_data": True,
    }

    # The paired loader is consumed exactly once.  There is no validation/test
    # loader or scorer invocation anywhere in this audit.
    paired_loader = paired.train_dataloader()
    sua_batch_cpu, pseudo_batch_cpu = next(iter(paired_loader))
    batch_shapes = _batch_shape_receipt(sua_batch_cpu, pseudo_batch_cpu)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    model = _make_model(teacher).to(device)
    model.train()
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config["optimizer"]
    sua_batch = _move_batch_to_device(sua_batch_cpu, device)
    pseudo_batch = _move_batch_to_device(pseudo_batch_cpu, device)
    validate_pair_batch(sua_batch, pseudo_batch)
    optimizer.zero_grad(set_to_none=True)
    sua_loss = model.model_step(sua_batch)["loss"]
    if not torch.isfinite(sua_loss):
        raise RuntimeError("SUA task loss is not finite during the TS4 microfit")
    (0.5 * sua_loss).backward()
    pseudo_loss = model.model_step(pseudo_batch)["loss"]
    if not torch.isfinite(pseudo_loss):
        raise RuntimeError("pseudo-MUA task loss is not finite during the TS4 microfit")
    (0.5 * pseudo_loss).backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    objective = model.c1_objective_receipt()
    if (
        objective.get("sua_task_loss_weight"),
        objective.get("pseudo_mua_task_loss_weight"),
        objective.get("lambda_consistency"),
        objective.get("backward_execution"),
    ) != (
        0.5,
        0.5,
        0.0,
        "sequential_half_weight_backward_then_single_optimizer_step",
    ):
        raise RuntimeError("microfit C1 objective drift")
    checkpoint_receipt = _checkpoint_round_trip(
        model=model, teacher=teacher, receipt_parent=receipt_parent
    )
    del model
    torch.cuda.empty_cache()

    return {
        "status": "passed",
        "seed": args.seed,
        "gpu": args.gpu,
        "data_access": {
            "strict_train_val_manifest": manifest_receipt,
            "formal_sua_files_opened": False,
            "formal_sua_paths_resolved": False,
            "held_out_test_evaluated": False,
        },
        "ts4_datamodules": {
            "sua": {
                "cache_dir": str(sua_cache),
                "normalizer": sua_normalizer,
                "descriptor_contract": sua_descriptors,
            },
            "pseudo_mua": {
                "cache_dir": str(pseudo_cache),
                "normalizer": pseudo_normalizer,
                "descriptor_contract": pseudo_descriptors,
            },
            "cross_view_normalizers_distinct": True,
        },
        "permutation_contract": permutation_contract,
        "paired_axis_target_count_contract": axis_contract,
        "one_gpu_paired_microfit": {
            "paired_batches_consumed": 1,
            "optimizer_steps": 1,
            "objective": objective,
            "batch_shapes": batch_shapes,
            "evaluation_or_scoring_invoked": False,
        },
        "temporary_checkpoint_round_trip": checkpoint_receipt,
        "formal_sua_files_opened": False,
        "formal_sua_paths_resolved": False,
        "held_out_test_evaluated": False,
    }


def main() -> int:
    args = parse_args()
    if os.environ.get("T4_PAIRED_VIEW_C1_AUTHORIZED") != "YES":
        raise RuntimeError("set T4_PAIRED_VIEW_C1_AUTHORIZED=YES after root authorization")
    out = (
        args.out.expanduser().resolve()
        if args.out is not None
        else (DEFAULT_RECEIPT_DIR / f"ts4_microfit_s{args.seed}.json").resolve()
    )
    if out.exists():
        raise FileExistsError(f"write-once microfit receipt already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    shared = {
        "schema_version": 1,
        "audit": "t4_paired_view_c1_v3_ts4_real_data_microfit",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "source_hashes": source_hashes(),
        "runtime_environment": runtime_environment(args.gpu),
        "formal_sua_files_opened": False,
        "formal_sua_paths_resolved": False,
        "no_r2_or_scorer_output_read": True,
    }
    try:
        result = _run_audit(args, receipt_parent=out.parent)
    except Exception as exc:
        failure = {
            **shared,
            "status": "failed",
            "failed_at": dt.datetime.now().astimezone().isoformat(),
            "error": repr(exc),
            "formal_sua_files_opened": False,
            "formal_sua_paths_resolved": False,
        }
        write_json_exclusive(out, failure)
        print(json.dumps({"status": "failed", "receipt": str(out)}, sort_keys=True))
        return 2
    receipt = {
        **shared,
        **result,
        "completed_at": dt.datetime.now().astimezone().isoformat(),
    }
    write_json_exclusive(out, receipt)
    print(json.dumps({"status": "passed", "receipt": str(out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
