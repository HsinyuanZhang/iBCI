"""Strict checkpoint reconstruction for the exact-head K/V oracle."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

_SCE_ROOT = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_SCE_ROOT))

from src.models.head_oracle_module import (  # noqa: E402
    TeacherHeadOracleLitModule,
)


_RECEIPT_KEY = "teacher_head_oracle_receipt"
BEHAVIOR_SCALING_FACTOR = 5.0
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
ID_HIDDEN_DIM = 128
HIDDEN_DIM = 64
PAD_VALUE = -1.0


def checkpoint_architecture_kwargs(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Whitelist the common coupled substrate fields."""
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise ValueError("head-oracle checkpoint is missing hyper_parameters")
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
            f"head-oracle checkpoint is missing architecture hparams: {missing}"
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


def _sha256_hex(label: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex string") from exc
    return value


def checkpoint_oracle_config(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Validate and extract every topology field defining an oracle state."""
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise ValueError("head-oracle checkpoint is missing hyper_parameters")
    receipt = checkpoint.get(_RECEIPT_KEY)
    if not isinstance(receipt, dict):
        raise ValueError("head-oracle checkpoint is missing its receipt")
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
                f"head-oracle hparam {name}={hparams.get(name)!r}, "
                f"expected {expected!r}"
            )
    if receipt.get("module") != "TeacherHeadOracleLitModule":
        raise ValueError("head-oracle receipt has the wrong module family")
    mode = hparams.get("oracle_key_mode")
    if mode not in {"e_t4", "e_ts4"}:
        raise ValueError(f"invalid oracle_key_mode: {mode!r}")
    seed = hparams.get("oracle_key_permutation_seed")
    if mode == "e_ts4" and seed is None:
        raise ValueError("oracle e_ts4 is missing its permutation seed")
    if mode == "e_t4" and seed is not None:
        raise ValueError("oracle e_t4 must not have a permutation seed")
    config = {
        "oracle_key_mode": mode,
        "oracle_key_permutation_seed": seed,
    }
    for name, value in config.items():
        if receipt.get(name) != value:
            raise ValueError(
                f"head-oracle receipt {name}={receipt.get(name)!r} "
                f"does not match hyper_parameters {value!r}"
            )
    _sha256_hex(
        "head-oracle active factor",
        receipt.get("active_factor_sha256"),
    )
    _sha256_hex(
        "head-oracle initial factor",
        receipt.get("initial_factor_sha256"),
    )
    if receipt.get("initialization_strategy") != (
        "exact_teacher_head_projection_copy"
    ):
        raise ValueError("head-oracle initialization strategy drifted")
    if receipt.get("teacher_head_count") != 64:
        raise ValueError("production head oracle must preserve 64 heads")
    if receipt.get("teacher_headwise_softmax_preserved") is not True:
        raise ValueError("head-oracle receipt lost head-wise softmax")
    return config


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_oracle_model(
    ckpt_path: Path,
    teacher_ckpt: Path,
    variant: str,
    device: torch.device,
    identity_mode: str = "calibrated",
) -> TeacherHeadOracleLitModule:
    """Strict restore with teacher and active-factor verification."""
    checkpoint = torch.load(
        str(ckpt_path), map_location="cpu", weights_only=False
    )
    config = checkpoint_oracle_config(checkpoint)
    receipt = checkpoint[_RECEIPT_KEY]
    if _sha256_file(teacher_ckpt) != receipt.get(
        "teacher_checkpoint_sha256"
    ):
        raise ValueError(
            "provided teacher checkpoint SHA256 differs from oracle receipt"
        )
    architecture = checkpoint_architecture_kwargs(checkpoint)
    if architecture["decoder_mode"] != "coupled":
        raise ValueError("head-oracle base decoder mode must be coupled")
    if architecture["side_dim"] != 4:
        raise ValueError("head-oracle checkpoint must record T4 width four")
    if variant != "B3S":
        raise ValueError("head-oracle loader requires variant='B3S'")
    if identity_mode != "calibrated":
        raise ValueError("head-oracle loader requires calibrated identity")

    model = TeacherHeadOracleLitModule(
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
    model.validate_loaded_oracle_checkpoint_receipt()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()
    return model
