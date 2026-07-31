"""Strict checkpoint reconstruction for the coupled T4 key residual."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

_SCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "streaming_calibration_exp"
)
sys.path.insert(0, str(_SCE_ROOT))

from src.models.t4_key_residual_module import (  # noqa: E402
    T4KeyResidualLitModule,
)


_RECEIPT_KEY = "t4_key_residual_receipt"
BEHAVIOR_SCALING_FACTOR = 5.0
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
ID_HIDDEN_DIM = 128
HIDDEN_DIM = 64
PAD_VALUE = -1.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1 << 20), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_hex(label: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(
            f"{label} must be a SHA-256 hex string"
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be a SHA-256 hex string"
        ) from exc
    return value


def checkpoint_architecture_kwargs(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Whitelist the common selected-T4 substrate fields."""
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise ValueError(
            "key-residual checkpoint is missing hyper_parameters"
        )
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
            "key-residual checkpoint is missing architecture "
            f"hparams: {missing}"
        )
    return {
        "fixed_slot_count": int(
            hparams["fixed_slot_count"]
        ),
        "fixed_slot_dim": int(
            hparams["fixed_slot_dim"]
        ),
        "fixed_slot_mode": str(
            hparams["fixed_slot_mode"]
        ),
        "fixed_slot_fusion": str(
            hparams["fixed_slot_fusion"]
        ),
        "fixed_slot_temperature": float(
            hparams["fixed_slot_temperature"]
        ),
        "decoder_mode": str(hparams["decoder_mode"]),
        "decoupled_key_mode": str(
            hparams["decoupled_key_mode"]
        ),
        "decoupled_key_dim": int(
            hparams["decoupled_key_dim"]
        ),
        "decoupled_value_dim": int(
            hparams["decoupled_value_dim"]
        ),
        "decoupled_num_heads": int(
            hparams["decoupled_num_heads"]
        ),
        "decoupled_key_permutation_seed": hparams[
            "decoupled_key_permutation_seed"
        ],
        "side_dim": int(hparams["side_dim"]),
        "electrode_embed_dim": int(
            hparams["electrode_embed_dim"]
        ),
        "num_electrodes": int(
            hparams["num_electrodes"]
        ),
    }


def checkpoint_key_residual_config(
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Validate and extract all topology/policy fields."""
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise ValueError(
            "key-residual checkpoint is missing hyper_parameters"
        )
    receipt = checkpoint.get(_RECEIPT_KEY)
    if not isinstance(receipt, dict):
        raise ValueError(
            "key-residual checkpoint is missing its receipt"
        )
    expected_hparams = {
        "task": "mc_maze",
        "variant": "B3S",
        "window_size": 50,
        "trial_length": 100,
        "id_hidden_dim": 128,
        "hidden_dim": 64,
        "pad_value": -1.0,
        "freeze_decoder": False,
        "freeze_encoder_base": False,
        "loss_mode": "task_only",
        "decode_last_timestep_only": True,
        "predict_scaled_behavior": True,
        "behavior_scaling_factor": 5.0,
        "identity_mode": "calibrated",
        "side_dim": 4,
        "electrode_embed_dim": 0,
        "num_electrodes": 0,
        "decoder_mode": "coupled",
        "fixed_slot_count": 0,
        "compile": False,
    }
    for name, expected in expected_hparams.items():
        if hparams.get(name) != expected:
            raise ValueError(
                f"key-residual hparam {name}="
                f"{hparams.get(name)!r}, expected "
                f"{expected!r}"
            )
    if receipt.get("module") != "T4KeyResidualLitModule":
        raise ValueError(
            "key-residual receipt has the wrong module family"
        )
    mode = hparams.get("residual_mode")
    if mode not in {"aligned", "shuffled"}:
        raise ValueError(
            f"invalid residual_mode: {mode!r}"
        )
    rank = hparams.get("residual_rank")
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank <= 0
    ):
        raise ValueError("residual_rank must be positive")
    seed = hparams.get("residual_permutation_seed")
    if mode == "shuffled" and seed is None:
        raise ValueError(
            "shuffled key residual is missing its seed"
        )
    if mode == "aligned" and seed is not None:
        raise ValueError(
            "aligned key residual must not have a seed"
        )
    policy = hparams.get("residual_training_policy")
    if policy not in {
        "residual_only",
        "residual_plus_attention_out",
    }:
        raise ValueError(
            "invalid residual_training_policy"
        )
    anchor_path = hparams.get("encoder_warmstart_path")
    if not isinstance(anchor_path, str) or not anchor_path:
        raise ValueError(
            "key residual requires selected-T4 anchor path"
        )
    config = {
        "residual_mode": mode,
        "residual_rank": rank,
        "residual_permutation_seed": seed,
        "residual_training_policy": policy,
        "encoder_warmstart_path": anchor_path,
    }
    for name in (
        "residual_mode",
        "residual_rank",
        "residual_permutation_seed",
        "residual_training_policy",
    ):
        if receipt.get(name) != config[name]:
            raise ValueError(
                f"key-residual receipt {name}="
                f"{receipt.get(name)!r} does not match "
                f"hyper_parameters {config[name]!r}"
            )
    _sha256_hex(
        "key-residual active factor",
        receipt.get("active_factor_sha256"),
    )
    _sha256_hex(
        "key-residual initial factor",
        receipt.get("initial_factor_sha256"),
    )
    _sha256_hex(
        "selected-T4 anchor",
        receipt.get("selected_t4_anchor_sha256"),
    )
    if receipt.get("zero_initialized") is not True:
        raise ValueError(
            "key-residual zero-initialization receipt drifted"
        )
    if receipt.get("backbone_frozen") is not True:
        raise ValueError(
            "key-residual backbone-freeze receipt drifted"
        )
    recorded_anchor = receipt.get(
        "selected_t4_anchor_path"
    )
    if (
        not isinstance(recorded_anchor, str)
        or Path(recorded_anchor).expanduser().resolve()
        != Path(anchor_path).expanduser().resolve()
    ):
        raise ValueError(
            "selected-T4 anchor path differs between "
            "hparams and receipt"
        )
    return config


def load_frozen_key_residual_model(
    ckpt_path: Path,
    teacher_ckpt: Path,
    variant: str,
    device: torch.device,
    identity_mode: str = "calibrated",
) -> T4KeyResidualLitModule:
    """Strict restore with teacher, anchor and factor verification."""
    checkpoint = torch.load(
        str(ckpt_path),
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint_key_residual_config(checkpoint)
    receipt = checkpoint[_RECEIPT_KEY]
    if _sha256_file(teacher_ckpt) != receipt.get(
        "teacher_checkpoint_sha256"
    ):
        raise ValueError(
            "provided teacher SHA differs from key-residual receipt"
        )
    anchor_path = Path(
        config.pop("encoder_warmstart_path")
    ).expanduser().resolve()
    if not anchor_path.is_file():
        raise FileNotFoundError(
            f"selected-T4 anchor is missing: {anchor_path}"
        )
    if _sha256_file(anchor_path) != receipt.get(
        "selected_t4_anchor_sha256"
    ):
        raise ValueError(
            "selected-T4 anchor SHA differs from receipt"
        )
    architecture = checkpoint_architecture_kwargs(
        checkpoint
    )
    if architecture["decoder_mode"] != "coupled":
        raise ValueError(
            "key-residual base decoder must be coupled"
        )
    if architecture["side_dim"] != 4:
        raise ValueError(
            "key-residual checkpoint must record T4 width four"
        )
    if variant != "B3S" or identity_mode != "calibrated":
        raise ValueError(
            "key-residual loader requires calibrated B3S"
        )

    model = T4KeyResidualLitModule(
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
        behavior_scaling_factor=(
            BEHAVIOR_SCALING_FACTOR
        ),
        identity_mode=identity_mode,
        encoder_warmstart_path=str(anchor_path),
        optimizer=None,
        scheduler=None,
        compile=False,
        **architecture,
        **config,
    )
    model.setup("fit")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(
        checkpoint["state_dict"], strict=True
    )
    (
        model
        .validate_loaded_key_residual_checkpoint_receipt()
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device).eval()
    return model
