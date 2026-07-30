"""CPU-only guards for reconstructing/evaluating decoupled K/V checkpoints.

These tests exercise helpers only.  They do not discover or open any NWB file,
so the sealed formal-test split remains untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_adaptation_dandi688 import (  # noqa: E402
    checkpoint_architecture_kwargs,
    eval_r2,
    eval_r2_with_zero_identity,
)


class _SyntheticEvalDataset(Dataset):
    def __init__(self) -> None:
        self.neural = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)
        self.behavior = self.neural[:, :, :2] / 5.0
        self.calibration = torch.zeros(2, 3, 5)
        self.side = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)

    def __len__(self) -> int:
        return self.neural.shape[0]

    def __getitem__(self, index: int):
        return (
            self.neural[index],
            self.behavior[index],
            self.calibration,
            "synthetic_session",
            self.side,
        )


class _SyntheticStudent:
    decoder_mode = "decoupled"
    decoupled_key_mode = "e_t4"
    decoupled_direct_feature_dim = 4

    def __init__(self) -> None:
        self.forward_key: torch.Tensor | None = None
        self.zero_key: torch.Tensor | None = None

    def __call__(
        self,
        neural,
        *,
        calib_trials,
        side_features,
        decoder_key_features,
        electrode_ids,
    ):
        del calib_trials, side_features, electrode_ids
        self.forward_key = decoder_key_features.detach().clone()
        identity = neural.new_zeros(neural.shape[0], neural.shape[2], neural.shape[1])
        return neural[:, :, :2], identity

    def decode_with_decoupled_identity(
        self,
        neural,
        identity,
        *,
        decoder_key_features,
    ):
        del identity
        self.zero_key = decoder_key_features.detach().clone()
        return neural[:, :, :2]


class _SyntheticModel:
    def __init__(self) -> None:
        self.student = _SyntheticStudent()

    def eval(self):
        return self

    @staticmethod
    def decoder_key_features(side_features):
        return side_features + 7.0


def test_checkpoint_architecture_kwargs_restores_decoupled_topology():
    checkpoint = {
        "hyper_parameters": {
            "fixed_slot_count": 0,
            "decoder_mode": "decoupled",
            "decoupled_key_mode": "e_ts4",
            "decoupled_key_dim": 32,
            "decoupled_value_dim": 32,
            "decoupled_num_heads": 2,
            "decoupled_key_permutation_seed": 43,
            "side_dim": 4,
            "electrode_embed_dim": 0,
            "num_electrodes": 0,
        }
    }
    kwargs = checkpoint_architecture_kwargs(checkpoint)
    assert kwargs["decoder_mode"] == "decoupled"
    assert kwargs["decoupled_key_mode"] == "e_ts4"
    assert kwargs["decoupled_key_dim"] == 32
    assert kwargs["decoupled_value_dim"] == 32
    assert kwargs["decoupled_num_heads"] == 2
    assert kwargs["decoupled_key_permutation_seed"] == 43
    assert kwargs["side_dim"] == 4


def test_checkpoint_architecture_kwargs_is_legacy_default_safe():
    kwargs = checkpoint_architecture_kwargs({"hyper_parameters": {}})
    assert kwargs["decoder_mode"] == "coupled"
    assert kwargs["fixed_slot_count"] == 0
    assert kwargs["side_dim"] == 0


def test_eval_routes_direct_key_features_to_decoupled_student():
    model = _SyntheticModel()
    score = eval_r2(model, _SyntheticEvalDataset(), torch.device("cpu"))
    assert torch.isfinite(torch.tensor(score))
    assert model.student.forward_key is not None
    expected = _SyntheticEvalDataset().side.unsqueeze(0).expand(4, -1, -1) + 7.0
    torch.testing.assert_close(model.student.forward_key, expected)


def test_zero_identity_control_uses_decoupled_path_and_zero_direct_key():
    model = _SyntheticModel()
    score = eval_r2_with_zero_identity(
        model, _SyntheticEvalDataset(), torch.device("cpu")
    )
    assert torch.isfinite(torch.tensor(score))
    assert model.student.zero_key is not None
    assert model.student.zero_key.shape == (4, 5, 4)
    assert torch.count_nonzero(model.student.zero_key).item() == 0
