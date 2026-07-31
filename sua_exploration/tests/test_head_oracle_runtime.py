"""CPU fail-closed contracts for exact-head checkpoint reconstruction."""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pytest
import torch

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

from head_oracle_runtime import (
    checkpoint_oracle_config,
    load_frozen_oracle_model,
)
from src.models.head_oracle_module import TeacherHeadOracleLitModule


def _checkpoint(mode: str = "e_t4", seed=None) -> dict:
    hparams = {
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
        "decoupled_key_mode": "e_t4",
        "decoupled_key_dim": 32,
        "decoupled_value_dim": 32,
        "decoupled_num_heads": 2,
        "decoupled_key_permutation_seed": None,
        "oracle_key_mode": mode,
        "oracle_key_permutation_seed": seed,
    }
    receipt = {
        "module": "TeacherHeadOracleLitModule",
        "oracle_key_mode": mode,
        "oracle_key_permutation_seed": seed,
        "active_factor_sha256": "a" * 64,
        "initial_factor_sha256": "b" * 64,
        "teacher_checkpoint_sha256": "c" * 64,
        "initialization_strategy": "exact_teacher_head_projection_copy",
        "teacher_head_count": 64,
        "teacher_headwise_softmax_preserved": True,
    }
    return {
        "hyper_parameters": hparams,
        "teacher_head_oracle_receipt": receipt,
    }


def test_checkpoint_config_binds_mode_seed_and_exact_head_receipt():
    assert checkpoint_oracle_config(_checkpoint()) == {
        "oracle_key_mode": "e_t4",
        "oracle_key_permutation_seed": None,
    }
    assert checkpoint_oracle_config(
        _checkpoint("e_ts4", 42)
    )["oracle_key_permutation_seed"] == 42
    bad = _checkpoint()
    bad["teacher_head_oracle_receipt"]["teacher_head_count"] = 1
    with pytest.raises(ValueError, match="64 heads"):
        checkpoint_oracle_config(bad)
    with pytest.raises(ValueError, match="permutation seed"):
        checkpoint_oracle_config(_checkpoint("e_ts4", None))


def test_production_wrapper_hparams_satisfy_runtime_whitelist():
    module = TeacherHeadOracleLitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path="/not/opened.ckpt",
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
        optimizer=partial(torch.optim.Adam, lr=1.0e-4),
        scheduler=None,
        compile=False,
        oracle_key_mode="e_t4",
        oracle_key_permutation_seed=None,
    )
    checkpoint = _checkpoint()
    checkpoint["hyper_parameters"] = dict(module.hparams)
    assert checkpoint_oracle_config(checkpoint)[
        "oracle_key_mode"
    ] == "e_t4"


def test_runtime_loader_orders_receipt_and_strict_restore(monkeypatch):
    import head_oracle_runtime as runtime

    checkpoint = _checkpoint()
    checkpoint["state_dict"] = {}
    events = []

    class FakeModule:
        def __init__(self, **kwargs):
            events.append("construct")

        def setup(self, stage):
            assert stage == "fit"
            events.append("setup")

        def on_load_checkpoint(self, loaded):
            assert loaded is checkpoint
            events.append("on_load")

        def load_state_dict(self, state, strict):
            assert state == {} and strict is True
            events.append("strict_load")

        def validate_loaded_oracle_checkpoint_receipt(self):
            events.append("validate_active_hash")

        def parameters(self):
            return []

        def to(self, device):
            events.append("to")
            return self

        def eval(self):
            events.append("eval")
            return self

    monkeypatch.setattr(
        runtime.torch, "load", lambda *args, **kwargs: checkpoint
    )
    monkeypatch.setattr(
        runtime, "_sha256_file", lambda path: "c" * 64
    )
    monkeypatch.setattr(runtime, "TeacherHeadOracleLitModule", FakeModule)
    load_frozen_oracle_model(
        Path("/not/opened.ckpt"),
        Path("/teacher/not/opened.ckpt"),
        "B3S",
        torch.device("cpu"),
    )
    assert events[:5] == [
        "construct",
        "setup",
        "on_load",
        "strict_load",
        "validate_active_hash",
    ]
