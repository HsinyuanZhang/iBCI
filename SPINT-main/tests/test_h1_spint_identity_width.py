"""Focused static contracts for the additive H1 SPINT width sweep."""
from __future__ import annotations

from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn.parameter import UninitializedParameter

from src.data.h1_spint_width_fold0 import width_accounting_manifest
from src.models.components.spint import SpintModel
from src.models.components.spint_identity_width import SpintIdentityWidthModel, identity_dense_macs, identity_parameter_count
from src.models.h1_spint_width_module import WIDTH_ARMS


def _seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def _kwargs() -> dict:
    return {
        "model_dim": 1024,
        "num_covariates": 7,
        "window_size": 700,
        "num_heads": 64,
        "num_layers": 1,
        "num_id_layers": 3,
        "use_learnable_id": True,
        "learnable_id_type": "mlp",
        "learnable_rep": True,
        "dropout_rate": 0.0,
        "dynamic_dropout": True,
        "dynamic_dropout_low": 0.0,
        "dynamic_dropout_high": 1.0,
        "tf_drop_rate": 0.1,
        "readin_layer_type": "mlp",
    }


def _materialize(model, identity: torch.Tensor) -> None:
    lazy = model.fc_id_in[0]
    if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
        lazy.initialize_parameters(identity.permute(0, 1, 3, 2))


def test_hs1024_is_bit_identical_to_standard_spint_after_lazy_materialization() -> None:
    identity = torch.randn(1, 4, 1024, 3)
    neural = torch.randn(1, 700, 3)
    _seed()
    standard = SpintModel(**_kwargs())
    _materialize(standard, identity)
    _seed()
    scaled = SpintIdentityWidthModel(identity_width=1024, **_kwargs())
    _materialize(scaled, identity)
    for key, value in standard.state_dict().items():
        assert torch.equal(value, scaled.state_dict()[key]), key
    standard.eval()
    scaled.eval()
    with torch.no_grad():
        assert torch.equal(
            standard(neural, calib_trialized_neural_features=identity),
            scaled(neural, calib_trialized_neural_features=identity),
        )


def test_predeclared_widths_have_expected_parameter_and_mac_compression() -> None:
    table = width_accounting_manifest((1024, 224, 32))
    assert table["1024"] == {"identity_width": 1024, "identity_parameters": 5_965_500, "identity_dense_macs_m4_n176": 2_709_848_064}
    assert table["224"]["identity_parameters"] == 588_700
    assert table["32"]["identity_parameters"] == 60_124
    assert 10.0 < table["1024"]["identity_parameters"] / table["224"]["identity_parameters"] < 11.0
    assert 99.0 < table["1024"]["identity_parameters"] / table["32"]["identity_parameters"] < 100.0
    assert identity_parameter_count(224) == 588_700
    assert identity_dense_macs(32) == table["32"]["identity_dense_macs_m4_n176"]


def test_compact_width_changes_only_id_parameter_shapes_and_forward_has_no_carrier() -> None:
    identity = torch.randn(1, 4, 1024, 3)
    _seed()
    full = SpintIdentityWidthModel(identity_width=1024, **_kwargs())
    _materialize(full, identity)
    _seed()
    compact = SpintIdentityWidthModel(identity_width=32, **_kwargs())
    _materialize(compact, identity)
    for key, value in full.state_dict().items():
        if key.startswith(("fc_id_in.", "fc_id_out.")):
            continue
        assert torch.equal(value, compact.state_dict()[key]), key
    assert "carrier" not in str(SpintIdentityWidthModel.forward)
    assert compact.identity_parameter_count() == 60_124


def test_configs_freeze_three_arms_and_activity_only_module() -> None:
    expected = {
        "h1_spint_width_fold0.yaml": ("H-S-1024", 1024),
        "h1_spint_width_fold0_w224.yaml": ("H-S-W224", 224),
        "h1_spint_width_fold0_w32.yaml": ("H-S-W32", 32),
    }
    for filename, (arm, width) in expected.items():
        config = OmegaConf.load(ROOT / "configs/experiment" / filename)
        assert config.width_sweep.arm == arm
        assert config.width_sweep.identity_width == width
    base = OmegaConf.load(ROOT / "configs/experiment/h1_spint_width_fold0.yaml")
    assert tuple(base.width_sweep.predeclared_arms) == tuple(WIDTH_ARMS)
    assert tuple(base.width_sweep.predeclared_widths) == (1024, 224, 32)
    assert base.width_sweep.noninferiority_margin_r2 == 0.03
    model = (ROOT / "configs/model/falcon_h1_spint_width.yaml").read_text(encoding="utf-8")
    data = (ROOT / "configs/data/falcon_h1_spint_width_fold0.yaml").read_text(encoding="utf-8")
    assert "SpintIdentityWidthModel" in model and "carrier" not in model.lower()
    assert "H1SpintWidthFold0DataModule" in data and "calibration_n_trials: 4" in data
