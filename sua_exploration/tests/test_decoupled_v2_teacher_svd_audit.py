"""Artifact contracts for the checkpoint-only decoupled v2 SVD audit."""
from __future__ import annotations

from functools import partial

import torch

from scripts.audit_decoupled_v2_teacher_svd import audit
from src.models.components.spint import SpintModel


def test_audit_loads_only_checkpoint_and_reports_svd_and_costs(tmp_path):
    teacher = SpintModel(
        model_dim=16,
        num_covariates=2,
        window_size=6,
        num_heads=4,
        num_layers=1,
        num_id_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.0,
    )
    teacher.fc_id_in(torch.zeros(1, 1, 1, 6))
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save({
        "hyper_parameters": {
            "task": "fixture",
            "net": teacher,
            "optimizer": partial(torch.optim.Adam, lr=1.0e-4),
        },
        "state_dict": {
            f"net.{name}": tensor
            for name, tensor in teacher.state_dict().items()
        },
    }, checkpoint)

    result = audit(
        checkpoint,
        key_dim=6,
        value_dim=8,
        direct_feature_dim=4,
        reference_num_units=5,
    )
    assert result["data_access_receipt"] == {
        "neural_datasets_opened": 0,
        "train_caches_opened": 0,
        "train_sessions_opened": 0,
        "validation_sessions_opened": 0,
        "formal_sessions_opened": 0,
    }
    assert result["teacher_shape"] == {
        "model_dim": 16,
        "num_covariates": 2,
        "window_size": 6,
        "feedforward_dim": 2048,
        "head_count": 4,
        "head_dim": 4,
        "num_layers": 1,
    }
    initialization = result["initialization_receipt"]
    assert initialization["key_rank"] == 6
    assert initialization["value_rank"] == 8
    assert initialization["teacher_headwise_softmax_preserved"] is False
    assert initialization["exact_teacher_mha_equivalence"] is False
    static = result["cost_reference"]["static_e_t4"]
    dynamic = result["cost_reference"]["dynamic_x_only"]
    assert static["persistent_state"]["bytes"] == 5 * 6 * 4
    assert dynamic["persistent_state"]["bytes"] == 0
    assert (
        dynamic["online_macs_per_frame"]["total"]
        > static["online_macs_per_frame"]["total"]
    )
