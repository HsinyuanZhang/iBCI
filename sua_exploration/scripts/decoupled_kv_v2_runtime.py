"""Strict checkpoint reconstruction for the isolated decoupled K/V v2 path.

This module is intentionally imported only by the v2-only train/evaluation
entrypoints.  The active v1 runner and its generic evaluator never import it.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

_SCE_ROOT = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_SCE_ROOT))

from src.models.decoupled_kv_v2_module import (  # noqa: E402
    TeacherReadinDecoupledLitModule,
)

_RECEIPT_KEY = "teacher_readin_decoupled_v2_receipt"
_KEY_MODES = {"e_t4", "e_ts4", "e_only", "x_only"}
BEHAVIOR_SCALING_FACTOR = 5.0
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
ID_HIDDEN_DIM = 128
HIDDEN_DIM = 64
PAD_VALUE = -1.0


def checkpoint_architecture_kwargs(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Whitelist the common substrate fields required before v2 replacement."""
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise ValueError("v2 checkpoint is missing hyper_parameters")
    required = {
        "fixed_slot_count",
        "fixed_slot_dim",
        "fixed_slot_mode",
        "fixed_slot_fusion",
        "fixed_slot_temperature",
        "decoder_mode",
        "decoupled_key_mode",
        "decoupled_key_dim",
        "decoupled_value_dim",
        "decoupled_num_heads",
        "decoupled_key_permutation_seed",
        "side_dim",
        "electrode_embed_dim",
        "num_electrodes",
    }
    missing = sorted(required.difference(hparams))
    if missing:
        raise ValueError(
            f"v2 checkpoint is missing architecture hparams: {missing}"
        )
    return {
        "fixed_slot_count": int(hparams["fixed_slot_count"]),
        "fixed_slot_dim": int(hparams["fixed_slot_dim"]),
        "fixed_slot_mode": str(hparams["fixed_slot_mode"]),
        "fixed_slot_fusion": str(hparams["fixed_slot_fusion"]),
        "fixed_slot_temperature": float(
            hparams["fixed_slot_temperature"]
        ),
        "decoder_mode": str(hparams["decoder_mode"]),
        "decoupled_key_mode": str(hparams["decoupled_key_mode"]),
        "decoupled_key_dim": int(hparams["decoupled_key_dim"]),
        "decoupled_value_dim": int(hparams["decoupled_value_dim"]),
        "decoupled_num_heads": int(hparams["decoupled_num_heads"]),
        "decoupled_key_permutation_seed": hparams[
            "decoupled_key_permutation_seed"
        ],
        "side_dim": int(hparams["side_dim"]),
        "electrode_embed_dim": int(hparams["electrode_embed_dim"]),
        "num_electrodes": int(hparams["num_electrodes"]),
    }


def checkpoint_v2_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate the topology fields that define a v2 checkpoint."""
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise ValueError("v2 checkpoint is missing hyper_parameters")
    receipt = checkpoint.get(_RECEIPT_KEY)
    if not isinstance(receipt, dict):
        raise ValueError("v2 checkpoint is missing its active-factor receipt")
    expected_hparams = {
        "task": "mc_maze",
        "variant": "B3S",
        "window_size": 50,
        "trial_length": 100,
        "id_hidden_dim": 128,
        "hidden_dim": 64,
        "num_emas": 4,
        "num_filters": 4,
        "kernel_size": 5,
        "learnable_ema_alpha": False,
        "sparsity_k": 16,
        "pad_value": -1.0,
        "freeze_decoder": False,
        "freeze_encoder_base": False,
        "tune_encoder_fusion": False,
        "fusion_mean_lr_scale": 1.0,
        "loss_mode": "task_only",
        "lambda_y": 1.0,
        "lambda_E": 0.1,
        "decode_last_timestep_only": True,
        "predict_scaled_behavior": True,
        "behavior_scaling_factor": 5.0,
        "neuron_dropout_mode": "none",
        "neuron_dropout_p_low": 0.0,
        "neuron_dropout_p_high": 0.3,
        "neuron_dropout_block_size": 4,
        "neuron_dropout_warmup_epochs": 10,
        "support_prediction_consistency_weight": 0.0,
        "side_dim": 4,
        "electrode_embed_dim": 0,
        "num_electrodes": 0,
        "identity_mode": "calibrated",
        "decoder_mode": "coupled",
        "fixed_slot_count": 0,
        "fixed_slot_dim": 32,
        "fixed_slot_mode": "soft",
        "fixed_slot_fusion": "film",
        "fixed_slot_temperature": 1.0,
        "encoder_warmstart_path": None,
        "compile": False,
    }
    for name, expected in expected_hparams.items():
        if hparams.get(name) != expected:
            raise ValueError(
                f"v2 checkpoint hparam {name}={hparams.get(name)!r}, "
                f"expected {expected!r}"
            )
    if receipt.get("module") != "TeacherReadinDecoupledLitModule":
        raise ValueError("v2 checkpoint receipt has the wrong module family")

    key_mode = hparams.get("v2_key_mode")
    if key_mode not in _KEY_MODES:
        raise ValueError(f"invalid v2_key_mode in checkpoint: {key_mode!r}")
    config = {
        "v2_key_mode": key_mode,
        "v2_key_dim": int(hparams.get("v2_key_dim", 0)),
        "v2_value_dim": int(hparams.get("v2_value_dim", 0)),
        "v2_key_permutation_seed": hparams.get(
            "v2_key_permutation_seed"
        ),
    }
    if config["v2_key_dim"] <= 0 or config["v2_value_dim"] <= 0:
        raise ValueError("v2 checkpoint has non-positive key/value dimensions")
    if (config["v2_key_dim"], config["v2_value_dim"]) != (48, 64):
        raise ValueError("v2 experiment checkpoint must use Dk=48 and Dv=64")
    if (
        key_mode == "e_ts4"
        and config["v2_key_permutation_seed"] is None
    ):
        raise ValueError("v2 e_ts4 checkpoint is missing its permutation seed")
    for name, value in config.items():
        if receipt.get(name) != value:
            raise ValueError(
                f"v2 checkpoint receipt {name}={receipt.get(name)!r} "
                f"does not match hyper_parameters {value!r}"
            )
    active_hash = receipt.get("active_factor_sha256")
    if not isinstance(active_hash, str) or len(active_hash) != 64:
        raise ValueError("v2 checkpoint receipt has no valid active factor SHA256")
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_v2_model(
    ckpt_path: Path,
    teacher_ckpt: Path,
    variant: str,
    device: torch.device,
    identity_mode: str = "calibrated",
) -> TeacherReadinDecoupledLitModule:
    """Strictly restore one v2 checkpoint and verify its active factor hash."""
    checkpoint = torch.load(
        str(ckpt_path), map_location="cpu", weights_only=False
    )
    config = checkpoint_v2_config(checkpoint)
    checkpoint_receipt = checkpoint[_RECEIPT_KEY]
    if _sha256_file(teacher_ckpt) != checkpoint_receipt.get(
        "teacher_checkpoint_sha256"
    ):
        raise ValueError(
            "provided teacher checkpoint SHA256 does not match the v2 receipt"
        )
    architecture = checkpoint_architecture_kwargs(checkpoint)
    if architecture["decoder_mode"] != "coupled":
        raise ValueError(
            "v2 checkpoints must record decoder_mode='coupled' for the "
            "common substrate before selector replacement"
        )
    if architecture["side_dim"] != 4:
        raise ValueError("v2 checkpoint must record four-dimensional T4")
    if variant != "B3S":
        raise ValueError("v2 frozen loader requires variant='B3S'")
    if identity_mode != "calibrated":
        raise ValueError("v2 frozen loader requires calibrated identity")

    model = TeacherReadinDecoupledLitModule(
        task="mc_maze",
        variant=variant,
        teacher_ckpt_path=str(teacher_ckpt),
        window_size=WINDOW_SIZE,
        trial_length=TRIAL_LENGTH,
        id_hidden_dim=ID_HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_value=PAD_VALUE,
        freeze_decoder=False,
        loss_mode="task_only",
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=BEHAVIOR_SCALING_FACTOR,
        identity_mode=identity_mode,
        optimizer=None,
        scheduler=None,
        compile=False,
        **architecture,
        **config,
    )
    model.setup("fit")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.validate_loaded_v2_checkpoint_receipt()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()
    return model
