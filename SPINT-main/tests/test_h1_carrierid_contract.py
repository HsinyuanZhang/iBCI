from __future__ import annotations

import hashlib
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.nn.parameter import UninitializedParameter

from src.models.components.h1_carrierid_spint import (
    H1_CARRIERID_PARAMETERS,
    H1_CARRIERID_WHOLE_MODEL_PARAMETERS,
    H1_SPINT_ID_PARAMETERS,
    H1CarrierIdSpint,
)
from src.models.components.h1_m4_eb_normalized_v2_residual_spint import H1M4EBNormalizedV2ResidualSpint


ROOT = Path(__file__).resolve().parents[1]


def _model(*, zero_carrier: bool) -> H1CarrierIdSpint:
    return H1CarrierIdSpint(
        carrier_hidden_dim=32,
        carrier_dim=4,
        carrier_trial_length=1024,
        zero_carrier=zero_carrier,
        model_dim=1024,
        num_covariates=7,
        window_size=700,
        num_heads=64,
        num_layers=1,
        num_id_layers=3,
        use_learnable_id=True,
        learnable_id_type="mlp",
        learnable_rep=True,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.1,
        readin_layer_type="mlp",
    )


def test_h32_parameter_accounting_and_literal_carrier_zero_columns():
    model = _model(zero_carrier=False)
    assert model.carrier_parameter_count() == H1_CARRIERID_PARAMETERS == 58_140
    assert sum(parameter.numel() for parameter in model.parameters()) == H1_CARRIERID_WHOLE_MODEL_PARAMETERS == 10_947_836
    assert H1_SPINT_ID_PARAMETERS / H1_CARRIERID_PARAMETERS == pytest.approx(102.605780874)
    assert not hasattr(model, "fc_id_in") and not hasattr(model, "fc_id_out")
    assert torch.count_nonzero(model.carrier_post_pool[0].weight[:, 32:]).item() == 0


def test_full_and_model_boundary_zero_are_identical_at_initialization():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    full = _model(zero_carrier=False).eval()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    zero = _model(zero_carrier=True).eval()
    assert full.state_dict().keys() == zero.state_dict().keys()
    assert all(torch.equal(full.state_dict()[name], zero.state_dict()[name]) for name in full.state_dict())
    neural = torch.randn(1, 700, 2)
    identity = torch.randn(1, 4, 1024, 2)
    carrier = torch.randn(1, 2, 4)
    with torch.no_grad():
        assert torch.equal(full(neural, identity, carrier), zero(neural, identity, carrier))


def test_shared_downstream_is_tensorwise_equal_to_seed42_matched_v2_base():
    """CarrierID may change only the former identity branch, not the decoder."""
    common = dict(
        model_dim=1024, num_covariates=7, window_size=700, num_heads=64, num_layers=1,
        num_id_layers=3, use_learnable_id=True, learnable_id_type="mlp", learnable_rep=True,
        dropout_rate=0.0, dynamic_dropout=True, dynamic_dropout_low=0.0, dynamic_dropout_high=1.0,
        tf_drop_rate=0.1, readin_layer_type="mlp",
    )
    torch.manual_seed(42)
    v2 = H1M4EBNormalizedV2ResidualSpint(train_residual=False, **common)
    lazy = v2.fc_id_in[0]
    materializer = torch.zeros(1, 4, 2, 1024)
    if any(isinstance(parameter, UninitializedParameter) for parameter in lazy.parameters()):
        lazy.initialize_parameters(materializer)
    torch.manual_seed(42)
    carrierid = _model(zero_carrier=False)
    prefixes = ("fc_in.", "fc_out.", "rep", "transformer.")
    expected = {name: tensor for name, tensor in v2.state_dict().items() if name.startswith(prefixes)}
    observed = {name: tensor for name, tensor in carrierid.state_dict().items() if name.startswith(prefixes)}
    assert expected.keys() == observed.keys()
    assert all(torch.equal(expected[name], observed[name]) for name in expected)


def test_one_source_side_step_can_open_full_carrier_columns_but_never_zero_input_columns():
    """H-C0 receives a literal zero at the model boundary, not merely zero data."""
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    full = _model(zero_carrier=False)
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    zero = _model(zero_carrier=True)
    identity = torch.randn(1, 4, 1024, 2)
    carrier = torch.randn(1, 2, 4)
    for model in (full, zero):
        model.zero_grad(set_to_none=True)
        token = model.carrierid_identity_projection(identity, carrier)
        token.square().mean().backward()
    full_columns = full.carrier_post_pool[0].weight[:, 32:]
    zero_columns = zero.carrier_post_pool[0].weight[:, 32:]
    assert torch.count_nonzero(full.carrier_post_pool[0].weight.grad[:, 32:]).item() > 0
    assert torch.count_nonzero(zero.carrier_post_pool[0].weight.grad[:, 32:]).item() == 0
    torch.optim.Adam(full.parameters(), lr=5e-5).step()
    torch.optim.Adam(zero.parameters(), lr=5e-5).step()
    assert torch.count_nonzero(full_columns.detach()).item() > 0
    assert torch.count_nonzero(zero_columns.detach()).item() == 0


def test_invalid_shape_and_hidden_width_fail_closed():
    with pytest.raises(ValueError, match="hidden-width sweeps"):
        _model(zero_carrier=False).__class__(
            carrier_hidden_dim=64, carrier_dim=4, carrier_trial_length=1024, zero_carrier=False,
            model_dim=1024, num_covariates=7, window_size=700,
        )
    model = _model(zero_carrier=False)
    with pytest.raises(ValueError, match="requires T=1024"):
        model(torch.zeros(1, 700, 2), torch.zeros(1, 4, 100, 2), torch.zeros(1, 2, 4))
    with pytest.raises(ValueError, match="carrier must"):
        model(torch.zeros(1, 700, 2), torch.zeros(1, 4, 1024, 2), torch.zeros(1, 2, 3))


def test_required_new_configs_and_source_only_preflight_guard_are_present():
    required = [
        ROOT / "configs/model/falcon_h1_carrierid.yaml",
        ROOT / "configs/experiment/h1_carrierid_full.yaml",
        ROOT / "configs/experiment/h1_carrierid_zero.yaml",
        ROOT / "configs/callbacks/h1_carrierid_terminal.yaml",
        ROOT / "scripts/h1_carrierid_preflight.py",
        ROOT / "scripts/h1_carrierid_paired_launcher.py",
        ROOT / "scripts/h1_carrierid_evaluate.py",
    ]
    assert all(path.is_file() for path in required)
    preflight = (ROOT / "scripts/h1_carrierid_preflight.py").read_text(encoding="utf-8")
    assert "load_target_records" not in preflight
    assert "target_recordings_opened\": 0" in preflight


def test_formal_loader_is_inaccessible_in_reused_source_datamodule_without_setup():
    from src.data.h1_m4_eb_normalized_v2 import H1M4EBNormalizedV2DataModule

    source_only = H1M4EBNormalizedV2DataModule(
        task="h1", data_dir=str(ROOT / "data/000954"), raw_receipt_path="raw.json", eb_receipt_path="eb.json",
        cache_dir=str(ROOT / "pilot_artifacts" / "h1_carrierid" / "test_noformal_cache"),
    )
    with pytest.raises(RuntimeError, match="formal test loader is forbidden"):
        source_only.test_dataloader()


def test_launcher_source_closure_fails_on_byte_drift(tmp_path):
    from scripts.h1_carrierid_paired_launcher import _verify_source_closure

    source = tmp_path / "source.py"
    source.write_text("carrierid\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert _verify_source_closure(tmp_path, {"source.py": digest})["verified"]
    source.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 drift"):
        _verify_source_closure(tmp_path, {"source.py": digest})
