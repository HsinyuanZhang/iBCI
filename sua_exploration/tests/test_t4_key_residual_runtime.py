"""Fail-closed contracts for T4 key-residual checkpoint reconstruction."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "scripts"
)
sys.path.insert(0, str(_SCRIPT_DIR))

from t4_key_residual_runtime import (
    checkpoint_architecture_kwargs,
    checkpoint_key_residual_config,
)


def _checkpoint() -> dict:
    anchor = str(
        Path("/tmp/selected_t4_anchor.ckpt").resolve()
    )
    hparams = {
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
        "encoder_warmstart_path": anchor,
        "compile": False,
        "fixed_slot_count": 0,
        "fixed_slot_dim": 32,
        "fixed_slot_mode": "soft",
        "fixed_slot_fusion": "film",
        "fixed_slot_temperature": 1.0,
        "decoder_mode": "coupled",
        "decoupled_key_mode": "e_t4",
        "decoupled_key_dim": 32,
        "decoupled_value_dim": 32,
        "decoupled_num_heads": 2,
        "decoupled_key_permutation_seed": None,
        "side_dim": 4,
        "electrode_embed_dim": 0,
        "num_electrodes": 0,
        "residual_mode": "aligned",
        "residual_rank": 8,
        "residual_permutation_seed": None,
        "residual_training_policy": "residual_only",
    }
    receipt = {
        "module": "T4KeyResidualLitModule",
        "residual_mode": "aligned",
        "residual_rank": 8,
        "residual_permutation_seed": None,
        "residual_training_policy": "residual_only",
        "selected_t4_anchor_path": anchor,
        "selected_t4_anchor_sha256": "1" * 64,
        "initial_factor_sha256": "2" * 64,
        "active_factor_sha256": "3" * 64,
        "zero_initialized": True,
        "backbone_frozen": True,
        "teacher_checkpoint_sha256": "4" * 64,
    }
    return {
        "hyper_parameters": hparams,
        "t4_key_residual_receipt": receipt,
    }


def test_config_and_architecture_round_trip():
    checkpoint = _checkpoint()
    config = checkpoint_key_residual_config(
        checkpoint
    )
    assert config == {
        "residual_mode": "aligned",
        "residual_rank": 8,
        "residual_permutation_seed": None,
        "residual_training_policy": "residual_only",
        "encoder_warmstart_path": str(
            Path(
                "/tmp/selected_t4_anchor.ckpt"
            ).resolve()
        ),
    }
    architecture = checkpoint_architecture_kwargs(
        checkpoint
    )
    assert architecture["decoder_mode"] == "coupled"
    assert architecture["side_dim"] == 4
    assert architecture["fixed_slot_count"] == 0


@pytest.mark.parametrize(
    ("location", "key", "value", "match"),
    [
        (
            "hyper_parameters",
            "variant",
            "B3",
            "hparam variant",
        ),
        (
            "hyper_parameters",
            "residual_mode",
            "bad",
            "invalid residual_mode",
        ),
        (
            "hyper_parameters",
            "residual_rank",
            0,
            "residual_rank",
        ),
        (
            "t4_key_residual_receipt",
            "residual_rank",
            7,
            "receipt residual_rank",
        ),
        (
            "t4_key_residual_receipt",
            "active_factor_sha256",
            "bad",
            "SHA-256",
        ),
        (
            "t4_key_residual_receipt",
            "zero_initialized",
            False,
            "zero-initialization",
        ),
        (
            "t4_key_residual_receipt",
            "backbone_frozen",
            False,
            "backbone-freeze",
        ),
    ],
)
def test_config_tamper_guards(
    location, key, value, match
):
    checkpoint = copy.deepcopy(_checkpoint())
    checkpoint[location][key] = value
    with pytest.raises(ValueError, match=match):
        checkpoint_key_residual_config(checkpoint)


def test_shuffled_config_requires_matching_seed():
    checkpoint = _checkpoint()
    checkpoint["hyper_parameters"].update({
        "residual_mode": "shuffled",
        "residual_permutation_seed": 42,
    })
    checkpoint["t4_key_residual_receipt"].update({
        "residual_mode": "shuffled",
        "residual_permutation_seed": 42,
    })
    config = checkpoint_key_residual_config(
        checkpoint
    )
    assert config["residual_mode"] == "shuffled"
    assert config["residual_permutation_seed"] == 42

    checkpoint[
        "t4_key_residual_receipt"
    ]["residual_permutation_seed"] = 43
    with pytest.raises(
        ValueError,
        match="receipt residual_permutation_seed",
    ):
        checkpoint_key_residual_config(checkpoint)
