"""Checkpoint-only contracts for the decoupled K/V rank audit."""
from __future__ import annotations

import torch

from scripts.audit_decoupled_teacher_low_rank import audit


def test_rank_audit_uses_only_checkpoint_tensors_and_reports_costs(tmp_path):
    model_dim = 8
    identity = torch.eye(model_dim)
    checkpoint = tmp_path / "teacher.ckpt"
    torch.save(
        {
            "state_dict": {
                "net.transformer.layers.0.cross_attn.in_proj_weight": torch.cat(
                    [identity, 2.0 * identity, 3.0 * identity],
                    dim=0,
                ),
                "net.transformer.layers.0.cross_attn.out_proj.weight": identity,
                "net.rep": torch.zeros(1, 2, 5),
                "net.transformer.layers.0.ffn.0.weight": torch.zeros(
                    16, model_dim
                ),
            }
        },
        checkpoint,
    )

    result = audit(
        checkpoint,
        ranks=(2, 4, 8),
        candidates=((4, 4), (5, 6)),
        reference_num_units=7,
    )

    assert result["data_access_receipt"] == {
        "neural_datasets_opened": 0,
        "train_sessions_opened": 0,
        "validation_sessions_opened": 0,
        "formal_sessions_opened": 0,
    }
    assert result["teacher_shape"] == {
        "model_dim": 8,
        "num_covariates": 2,
        "window_size": 5,
        "feedforward_dim": 16,
    }
    assert result["spectral_energy"]["Wq"]["cumulative_energy_fraction"]["4"] == 0.5
    assert (
        result["spectral_energy"]["Wo_at_Wv"]["cumulative_energy_fraction"]["4"]
        == 0.5
    )
    costs = result["configured_cost_reference_n64"]
    assert [entry["key_dim"] for entry in costs] == [4, 5]
    assert all(
        entry["online_mac_reduction_fraction_vs_coupled"] > 0
        for entry in costs
    )
    assert costs[0]["persistent_state_nonincreasing_vs_E50"] is True
    assert costs[1]["persistent_state_nonincreasing_vs_E50"] is True
