#!/usr/bin/env python3
"""Train one source-only shared C1 direct-standardized Z4 seed.

This is an append-only control program.  It intentionally does not reuse the
historical generic ``z4`` feature token: that token computes T4 before masking
it.  Here both data modules have ``side_feature_group=None`` and the new
``SharedZero4PairedDataModule`` writes bitwise float32 zeros directly in the
already-standardized B3S side coordinate.

The entry point trains exactly twelve source-only epochs and emits the fixed
terminal ``epoch_011.ckpt``.  It never wires a validation loader into
``Trainer.fit`` and never runs a scorer; a future development score, if
authorized separately, must use the terminal checkpoint for every seed rather
than selecting a first-cell winner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration"))
sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))

from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
from mc_maze.paired_view_c1_shared_zero4 import (
    DIRECT_ZERO_CONSTRUCTION,
    SIDE_DIM,
    STANDARDIZED_COORDINATE_NAME,
    SharedZero4PairedDataModule,
)
from src.models.paired_view_c1_module import PairedViewC1LitModule


EXPECTED_SEEDS = (42, 43, 44)
TOTAL_EPOCHS = 12
CHECKPOINT_EPOCH_INDEX = 11
PROTOCOL_EPOCH_NUMBER = 12
TERMINAL_CHECKPOINT_FILENAME = "epoch_011.ckpt"
FIXED_TERMINAL_SELECTION = "fixed_terminal_epoch_011_no_selection"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EXPECTED_STRICT_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
SEALED_FORMAL_TEST_NAMES = (
    "sub-C_ses-CO-20151113",
    "sub-C_ses-CO-20151116",
    "sub-C_ses-CO-20151117",
    "sub-C_ses-CO-20151119",
    "sub-C_ses-CO-20151120",
    "sub-C_ses-CO-20151201",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialized)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def runtime_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        payload["nvidia_smi_gpus"] = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        payload["nvidia_smi_error"] = str(exc)
    return payload


def verify_single_visible_physical_gpu(expected_uuid: str) -> dict[str, Any]:
    """Assert the trainer sees exactly the signed physical GPU as logical zero."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("shared_zero4 CUDA_VISIBLE_DEVICES drifted from signed UUID")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("shared_zero4 requires exactly one visible CUDA device")
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    available = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    if expected_uuid not in available:
        raise RuntimeError("shared_zero4 signed GPU UUID disappeared or changed")
    device_uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
    observed_torch_uuid = str(device_uuid) if device_uuid is not None else None
    # Some torch builds omit UUID; the nvidia-smi + one-visible-device checks
    # remain authoritative in that case.
    if observed_torch_uuid is not None and expected_uuid not in observed_torch_uuid:
        raise RuntimeError("shared_zero4 torch logical device UUID mismatch")
    return {
        "physical_gpu_uuid": expected_uuid,
        "cuda_visible_devices": expected_uuid,
        "logical_cuda_device_index": 0,
        "torch_visible_device_count": int(torch.cuda.device_count()),
        "torch_logical_uuid": observed_torch_uuid,
        "nvidia_smi_uuid_present": True,
    }


def state_dict_digest(state_dict: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Hash every student tensor with key/dtype/shape boundaries preserved."""

    digest = hashlib.sha256()
    tensor_count = 0
    total_bytes = 0
    tensor_rows: list[dict[str, Any]] = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"student state has non-tensor item {name!r}")
        cpu = tensor.detach().cpu().contiguous()
        raw = cpu.numpy().tobytes(order="C")
        descriptor = {
            "name": name,
            "dtype": str(cpu.dtype),
            "shape": list(cpu.shape),
            "bytes": len(raw),
        }
        encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(raw)
        tensor_rows.append(descriptor)
        tensor_count += 1
        total_bytes += len(raw)
    return {
        "algorithm": "sha256_key_dtype_shape_and_contiguous_cpu_bytes_v1",
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "tensor_bytes": total_bytes,
        "tensors": tensor_rows,
    }


def build_shared_zero4_model(*, teacher: Path) -> PairedViewC1LitModule:
    """Build the same B3S shared objective as shared-T4, with width-matched Z4 input."""

    optimizer = partial(torch.optim.Adam, lr=LEARNING_RATE, weight_decay=0.0)
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
        side_dim=SIDE_DIM,
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
        raise RuntimeError("shared_zero4 model initialization produced no student")
    return model


def build_initial_state_for_seed(*, teacher: Path, seed: int) -> tuple[PairedViewC1LitModule, dict[str, Any]]:
    """Initialize a new Z4 model under an explicit seed-reset boundary."""

    if seed not in EXPECTED_SEEDS:
        raise ValueError(f"seed must be one of {EXPECTED_SEEDS}")
    # Re-seeding directly before model construction makes this program's own
    # initial-state evidence independent of any deterministic loader audit.
    # Historical T4/TS4 metadata lacks an analogous digest, so this does not
    # imply three-arm initial bit-exactness.
    pl.seed_everything(seed, workers=True)
    model = build_shared_zero4_model(teacher=teacher)
    assert model.student is not None
    digest = state_dict_digest(model.student.state_dict())
    digest.update(
        {
            "seed": seed,
            "variant": "B3S",
            "side_dim": SIDE_DIM,
            "coordinate": STANDARDIZED_COORDINATE_NAME,
            "initialization_scope": "new_shared_zero4_program_only",
            "old_t4_ts4_initial_digest_available": False,
        }
    )
    return model, digest


def static_model_cost_receipt(*, teacher: Path, seed: int = 42) -> dict[str, Any]:
    """CPU-only architecture/cost summary used by the write-once prelaunch."""

    model, initial = build_initial_state_for_seed(teacher=teacher, seed=seed)
    assert model.student is not None
    encoder = model.student.id_encoder.cost_profile(
        num_neurons=64, trial_length=100, num_trials=10
    )
    decoder = model.student.decoder_cost_comparison_receipt(batch_size=1, num_neurons=64)
    parameter_count = int(sum(parameter.numel() for parameter in model.student.parameters()))
    trainable_count = int(
        sum(parameter.numel() for parameter in model.student.parameters() if parameter.requires_grad)
    )
    return {
        "variant": "B3S",
        "side_dim": SIDE_DIM,
        "model_parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "fp32_weight_bytes": parameter_count * 4,
        "descriptor_persistent_bytes_per_channel": SIDE_DIM * 4,
        "encoder": asdict(encoder),
        "decoder": decoder,
        "initial_state_seed": seed,
        "initial_state_digest": initial,
        "deployment_weight_copies": 1,
        "deployment_extra_state_vs_separate": 0,
        "shared_training_view_forwards_per_optimizer_step": 2,
    }


def _program_contract(receipt: Mapping[str, Any], *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = receipt.get("frozen_protocol")
    coordinate = receipt.get("t4_normalizer_coordinate_reference")
    digests = receipt.get("initial_state_digests")
    if not isinstance(protocol, dict) or not isinstance(coordinate, dict) or not isinstance(digests, dict):
        raise ValueError("shared_zero4 program receipt lacks required contract fields")
    expected_protocol = (
        protocol.get("source_training_activity_calibration_n"),
        protocol.get("total_epochs"),
        protocol.get("checkpoint_epoch_index"),
        protocol.get("protocol_epoch_number"),
        protocol.get("checkpoint_filename"),
        protocol.get("checkpoint_selection"),
        protocol.get("batch_size"),
        protocol.get("learning_rate"),
        protocol.get("shared_objective"),
        protocol.get("lambda_consistency"),
        protocol.get("early_stopping"),
        protocol.get("side_dim"),
    )
    if expected_protocol != (
        10,
        TOTAL_EPOCHS,
        CHECKPOINT_EPOCH_INDEX,
        PROTOCOL_EPOCH_NUMBER,
        TERMINAL_CHECKPOINT_FILENAME,
        FIXED_TERMINAL_SELECTION,
        BATCH_SIZE,
        LEARNING_RATE,
        "0.5*L_task(SUA)+0.5*L_task(pseudo_MUA)",
        0.0,
        "forbidden",
        SIDE_DIM,
    ):
        raise ValueError("shared_zero4 frozen protocol drift")
    if "terminal_epoch" in protocol:
        raise ValueError("shared_zero4 ambiguous terminal_epoch key is forbidden")
    expected_digest = digests.get(str(seed))
    if not isinstance(expected_digest, dict) or not isinstance(expected_digest.get("sha256"), str):
        raise ValueError(f"shared_zero4 initial-state digest missing for seed {seed}")
    if (
        coordinate.get("coordinate") != STANDARDIZED_COORDINATE_NAME
        or coordinate.get("construction") != DIRECT_ZERO_CONSTRUCTION
        or coordinate.get("target_direction_label_reads_for_descriptor") != 0
        or coordinate.get("t4_trial_rate_reads_for_descriptor") != 0
        or coordinate.get("label_access_scope") != "descriptor_only"
        or coordinate.get("target_t4_rate_fit_calls") != 0
        or coordinate.get("raw_t4_constructed") is not False
        or coordinate.get("source_t4_normalizer_arithmetic_performed") is not False
    ):
        raise ValueError("shared_zero4 coordinate contract drift")
    return dict(coordinate), dict(expected_digest)


def validate_materialized_split_scope(
    *,
    sua_dm: Dandi688MultiSessionDataModule,
    pseudo_dm: Dandi688MultiSessionDataModule,
    strict_manifest: Mapping[str, Any],
    data_dir: Path,
) -> dict[str, Any]:
    """Fail closed before model/GPU construction if the 27/6/6 boundary drifts."""

    if sha256_file(Path(__file__).resolve().parents[1] / "configs/subc_co_27_6_strict_train_val_manifest.json") != EXPECTED_STRICT_MANIFEST_SHA256:
        raise ValueError("shared_zero4 canonical strict manifest hard pin drift")
    expected_splits = strict_manifest.get("session_splits")
    if not isinstance(expected_splits, dict) or any(
        not isinstance(expected_splits.get(split), list) for split in ("train", "val", "test")
    ):
        raise ValueError("shared_zero4 strict manifest has malformed session_splits")
    if [len(expected_splits[split]) for split in ("train", "val", "test")] != [27, 6, 6]:
        raise ValueError("shared_zero4 strict manifest is not 27/6/6")
    if expected_splits.get("test") != list(SEALED_FORMAL_TEST_NAMES):
        raise ValueError("shared_zero4 sealed formal test-name list drift")
    normalized_data_dir = data_dir.resolve()
    result: dict[str, Any] = {}
    for label, datamodule in (("sua", sua_dm), ("pseudo_mua", pseudo_dm)):
        if datamodule.session_files.get("test") != []:
            raise RuntimeError(f"shared_zero4 {label} resolved test file paths")
        if datamodule.session_splits != {
            split: list(expected_splits[split]) for split in ("train", "val", "test")
        }:
            raise RuntimeError(f"shared_zero4 {label} session split drift from strict manifest")
        paths_by_split: dict[str, list[str]] = {}
        for split in ("train", "val"):
            paths = list(datamodule.session_files.get(split, []))
            names = [path.stem.removesuffix("_behavior+ecephys") for path in paths]
            if names != list(expected_splits[split]):
                raise RuntimeError(f"shared_zero4 {label}/{split} path order drift")
            for path in paths:
                resolved = path.resolve()
                if resolved.parent != normalized_data_dir or "sub-M" in str(resolved):
                    raise RuntimeError(f"shared_zero4 {label}/{split} path escaped sub-C scope: {resolved}")
            paths_by_split[split] = [str(path) for path in paths]
        result[label] = {
            "test_files": [],
            "session_splits_exact": True,
            "paths": paths_by_split,
            "subm_paths_present": False,
        }
    if result["sua"]["paths"] != result["pseudo_mua"]["paths"]:
        raise RuntimeError("shared_zero4 SUA/pseudo-MUA train/validation path lists differ")
    return {
        "strict_manifest_27_6_6": True,
        "views": result,
        "formal_paths_resolved": False,
        "formal_files_opened": False,
        "subm_nwb_files_opened": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-ckpt", required=True)
    parser.add_argument("--out-name", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-val-manifest", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--program-receipt", required=True)
    parser.add_argument("--sua-cache-dir", required=True)
    parser.add_argument("--pseudo-mua-cache-dir", required=True)
    parser.add_argument("--seed", type=int, choices=EXPECTED_SEEDS, required=True)
    parser.add_argument("--max-epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--expected-physical-gpu-uuid", required=True)
    parser.add_argument("--accelerator", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--disable-progress-bar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_epochs != TOTAL_EPOCHS:
        raise ValueError(f"shared_zero4 freezes max_epochs={TOTAL_EPOCHS}")
    if args.batch_size != BATCH_SIZE:
        raise ValueError(f"shared_zero4 freezes batch_size={BATCH_SIZE}")
    if args.num_workers != 4:
        raise ValueError("shared_zero4 freezes num_workers=4")
    if args.learning_rate != LEARNING_RATE:
        raise ValueError(f"shared_zero4 freezes learning_rate={LEARNING_RATE}")
    if args.accelerator != "gpu" or not args.require_gpu:
        raise ValueError("shared_zero4 direct trainer requires --accelerator gpu and --require-gpu")
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("--require-gpu set but CUDA is unavailable")
    if not args.expected_physical_gpu_uuid.startswith("GPU-"):
        raise ValueError("shared_zero4 expected physical GPU UUID is malformed")
    initial_gpu_binding = verify_single_visible_physical_gpu(args.expected_physical_gpu_uuid)

    teacher = Path(args.teacher_ckpt).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    train_val_manifest = Path(args.train_val_manifest).expanduser().resolve()
    data_manifest = Path(args.data_manifest).expanduser().resolve()
    program_receipt_path = Path(args.program_receipt).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_dir = checkpoint_root / args.out_name
    sua_cache = Path(args.sua_cache_dir).expanduser().resolve()
    pseudo_cache = Path(args.pseudo_mua_cache_dir).expanduser().resolve()
    for label, path, kind in (
        ("teacher", teacher, "file"),
        ("data_dir", data_dir, "dir"),
        ("train_val_manifest", train_val_manifest, "file"),
        ("data_manifest", data_manifest, "file"),
        ("program_receipt", program_receipt_path, "file"),
    ):
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            raise FileNotFoundError(f"shared_zero4 {label} is missing: {path}")
    if output_dir.exists():
        raise FileExistsError(f"shared_zero4 refuses to reuse output directory: {output_dir}")
    if sua_cache == pseudo_cache:
        raise ValueError("shared_zero4 requires separate SUA/pseudo-MUA cache namespaces")

    receipt = load_json(program_receipt_path)
    coordinate_reference, expected_initial = _program_contract(receipt, seed=args.seed)
    if receipt.get("teacher", {}).get("sha256") != sha256_file(teacher):
        raise ValueError("shared_zero4 teacher SHA drift from program receipt")
    if receipt.get("strict_loader_manifest", {}).get("sha256") != sha256_file(train_val_manifest):
        raise ValueError("shared_zero4 strict manifest SHA drift from program receipt")
    if sha256_file(train_val_manifest) != EXPECTED_STRICT_MANIFEST_SHA256:
        raise ValueError("shared_zero4 strict manifest hard pin drift")
    if receipt.get("portable_data_manifest", {}).get("sha256") != sha256_file(data_manifest):
        raise ValueError("shared_zero4 portable data manifest SHA drift from program receipt")

    # This seed is used only for deterministic data-loader ordering; the model
    # has a second explicit seed boundary below, which produces the pinned new
    # Z4 initial-state digest.
    pl.seed_everything(args.seed, workers=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = output_dir / "run_metadata.json"

    common_dm_kwargs = dict(
        data_dir=str(data_dir),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=args.batch_size,
        window_size=50,
        calibration_n_trials=10,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=args.num_workers,
        random_calibration=False,
        seed=args.seed,
        max_units_exclusive=100,
        # The central safety boundary: generic side features never initialize.
        side_feature_group=None,
        train_val_manifest_path=str(train_val_manifest),
    )
    sua_dm = Dandi688MultiSessionDataModule(
        **common_dm_kwargs, cache_dir=str(sua_cache), signal_view="sua"
    )
    pseudo_dm = Dandi688MultiSessionDataModule(
        **common_dm_kwargs, cache_dir=str(pseudo_cache), signal_view="pseudo_mua"
    )
    paired_dm = SharedZero4PairedDataModule(sua_dm, pseudo_dm)
    paired_dm.setup()
    strict_manifest_payload = load_json(train_val_manifest)
    materialized_scope = validate_materialized_split_scope(
        sua_dm=sua_dm,
        pseudo_dm=pseudo_dm,
        strict_manifest=strict_manifest_payload,
        data_dir=data_dir,
    )
    zero4_exposure = paired_dm.exposure_receipt()
    # This all-source-batch bit audit runs before model/optimizer construction.
    # It is not a score, a selection step, or a gradient computation.
    all_batch_zero_audit = paired_dm.audit_all_train_batches()
    if sua_dm._side_feature_stats is not None or pseudo_dm._side_feature_stats is not None:
        raise RuntimeError("shared_zero4 unexpectedly constructed generic side-feature statistics")

    model, observed_initial = build_initial_state_for_seed(teacher=teacher, seed=args.seed)
    if observed_initial["sha256"] != expected_initial["sha256"]:
        raise RuntimeError(
            "shared_zero4 initial-state digest drift: "
            f"expected {expected_initial['sha256']}, got {observed_initial['sha256']}"
        )
    assert model.student is not None
    encoder_cost = model.student.id_encoder.cost_profile(
        num_neurons=64, trial_length=100, num_trials=10
    )
    decoder_cost = model.student.decoder_cost_comparison_receipt(batch_size=1, num_neurons=64)
    initial_path = output_dir / "initial_state_digest.json"
    write_json_exclusive(initial_path, observed_initial)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "initialized",
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "paired_view_c1_shared_zero4_source",
        "training_kind": "shared_paired_view_direct_standardized_zero4",
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
        "program_receipt": str(program_receipt_path),
        "program_receipt_sha256": sha256_file(program_receipt_path),
        "split_counts": [27, 6, 6],
        "max_units_exclusive": 100,
        "output_dir": str(output_dir),
        "held_out_test_evaluated": False,
        "formal_sua_files_opened": False,
        "subm_nwb_files_opened": False,
        "no_heldout_backprop_contract": {
            "source_train_sessions": 27,
            "development_heldout_sessions": 6,
            "formal_test_sessions": 6,
            "optimizer_and_backward_scope": "source_train_27_only",
            "development_enters_train_dataloader": False,
            "development_enters_loss": False,
            "development_enters_optimizer": False,
            "development_uses_backward_gradients": False,
            "development_scoring_invoked_by_this_run": False,
            "formal_paths_resolved": False,
            "formal_files_opened": False,
        },
        "session_splits": paired_dm.session_splits,
        "materialized_split_scope": materialized_scope,
        "session_files": {
            "train": [str(path) for path in sua_dm.session_files["train"]],
            "val": [str(path) for path in sua_dm.session_files["val"]],
            "test": [],
        },
        "view_configs": {
            view: {
                "signal_view": view,
                "cache_dir": str(cache),
                "side_features": {
                    "group": "shared_zero4_direct_standardized",
                    "side_dim": SIDE_DIM,
                    "coordinate": STANDARDIZED_COORDINATE_NAME,
                    "construction": DIRECT_ZERO_CONSTRUCTION,
                    "source_t4_normalizer_reference": coordinate_reference["views"][view],
                    "target_direction_label_reads_for_descriptor": 0,
                    "t4_trial_rate_reads_for_descriptor": 0,
                    "label_access_scope": "descriptor_only",
                    "target_t4_rate_fit_calls": 0,
                    "raw_t4_constructed": False,
                    "source_t4_normalizer_arithmetic_performed": False,
                },
            }
            for view, cache in (("sua", sua_cache), ("pseudo_mua", pseudo_cache))
        },
        "zero4_exposure": zero4_exposure,
        "all_batch_zero_audit": all_batch_zero_audit,
        "initial_state": {
            "path": str(initial_path),
            "sha256": sha256_file(initial_path),
            "digest": observed_initial,
            "matches_prelaunch": True,
            "old_shared_t4_ts4_initial_digests_available": False,
            "three_arm_initial_bit_exact_claim": "not_authorized",
            "model_seed_reset_after_all_batch_audit": True,
        },
        "paired_objective": model.c1_objective_receipt(),
        "training": {
            "max_epochs": TOTAL_EPOCHS,
            "no_early_stopping": True,
            "checkpoint_every_epoch": True,
            "epoch_checkpoints_dir": str(output_dir / "epoch_ckpts"),
            "terminal_checkpoint": TERMINAL_CHECKPOINT_FILENAME,
            "checkpoint_selection": FIXED_TERMINAL_SELECTION,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "source_activity_calibration_n_trials": 10,
            "future_development_activity_calibration_n_trials": 30,
            "future_development_query_start_trial": 50,
            "future_development_trials_30_49_enter_zero4_identity_or_descriptor": False,
            "loss_mode": "task_only",
            "freeze_decoder": False,
            "shared_optimizer_steps": True,
            "random_calibration": False,
            "development_score_invoked": False,
            "development_score_artifact_paths": [],
        },
        "cost_receipt_reference": {
            "model_parameter_count": sum(parameter.numel() for parameter in model.student.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.student.parameters() if parameter.requires_grad
            ),
            "fp32_weight_bytes": 4 * sum(parameter.numel() for parameter in model.student.parameters()),
            "descriptor_persistent_bytes_per_channel": SIDE_DIM * 4,
            "encoder": asdict(encoder_cost),
            "decoder": decoder_cost,
            "shared_training_view_forwards_per_optimizer_step": 2,
            "deployment_weight_copies": 1,
            # Preserve the historical shared-T4 canonical cost schema exactly;
            # Z4-specific interpretation lives in the separate supplement.
            "deployment_extra_state_vs_separate": 0,
        },
        "zero4_cost_supplement": {
            "deployment_extra_state_vs_shared_t4": 0,
            "side_dim_parity_with_shared_t4": True,
            "descriptor_persistent_bytes_per_channel_parity": True,
        },
        "runtime_environment_at_initialization": runtime_environment(),
        "runtime_gpu_binding": {
            "initial": initial_gpu_binding,
            "authorization_binding_checked_before_trainer": True,
        },
    }
    write_json_exclusive(metadata_path, metadata)

    epoch_ckpt_dir = output_dir / "epoch_ckpts"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(epoch_ckpt_dir),
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        every_n_epochs=1,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=TOTAL_EPOCHS,
        accelerator=args.accelerator,
        devices=1,
        callbacks=[checkpoint_callback],
        logger=False,
        enable_checkpointing=True,
        deterministic=True,
        enable_progress_bar=not args.disable_progress_bar,
        log_every_n_steps=50,
        default_root_dir=str(output_dir),
        num_sanity_val_steps=0,
    )
    using_gpu = trainer.accelerator.__class__.__name__.lower().startswith("cuda")
    if args.require_gpu and not using_gpu:
        raise RuntimeError("shared_zero4 requires a CUDA trainer")
    if using_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model, train_dataloaders=paired_dm.train_dataloader())
    if using_gpu:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    completion_gpu_binding = verify_single_visible_physical_gpu(args.expected_physical_gpu_uuid)
    expected_checkpoints = [epoch_ckpt_dir / f"epoch_{epoch:03d}.ckpt" for epoch in range(TOTAL_EPOCHS)]
    missing = [str(path) for path in expected_checkpoints if not path.is_file()]
    if missing:
        raise RuntimeError(f"shared_zero4 failed to write every fixed epoch checkpoint: {missing}")
    terminal_checkpoint = epoch_ckpt_dir / TERMINAL_CHECKPOINT_FILENAME
    if not terminal_checkpoint.is_file():
        raise RuntimeError("shared_zero4 fixed terminal epoch_011 checkpoint is missing")

    metadata.update(
        {
            "status": "completed",
            "completed_at": datetime.now().astimezone().isoformat(),
            "epoch_checkpoints": [str(path) for path in expected_checkpoints],
            "terminal_checkpoint": str(terminal_checkpoint),
            "terminal_checkpoint_sha256": sha256_file(terminal_checkpoint),
            "held_out_test_evaluated": False,
            "formal_sua_files_opened": False,
            "subm_nwb_files_opened": False,
            "runtime_environment_at_completion": runtime_environment(),
            "runtime_gpu_binding": {
                "initial": initial_gpu_binding,
                "completion": completion_gpu_binding,
                "authorization_binding_checked_before_trainer": True,
            },
        }
    )
    # The initialized metadata path is deliberately rewritten only inside this
    # fresh output directory before its completed state is sealed by the runner.
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cost_path = output_dir / "post_run_cost_receipt.json"
    if cost_path.exists():
        raise FileExistsError(f"shared_zero4 post-run cost path exists: {cost_path}")
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
        "terminal_checkpoint": str(terminal_checkpoint),
        "terminal_checkpoint_sha256": sha256_file(terminal_checkpoint),
        "formal_sua_files_opened": False,
        "subm_nwb_files_opened": False,
        "development_score_invoked": False,
    }
    write_json_exclusive(cost_path, cost)
    print(json.dumps({"run_metadata": str(metadata_path), "cost": str(cost_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
