import copy
from types import SimpleNamespace

import pytest
import torch

from tfpd_exploration.src.m2_family_v1 import cold_compare as phase


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = torch.nn.Linear(96, 2)
    def trainable_parameters(self):
        return dict(self.named_parameters())
    def forward_last(self, x, bank, dropout_keep=None):
        return self.readout((x * dropout_keep[:, None]).mean(1))


def test_four_cells_share_data_and_masks_with_only_prefix_treatment_different():
    torch.set_num_threads(1)
    torch.manual_seed(98)
    template = Tiny()
    models = {key: copy.deepcopy(template) for key in phase.CELLS}
    optimizers = {key: torch.optim.AdamW(model.parameters(), lr=1e-5) for key, model in models.items()}
    emas = {key: phase.DecoderEMA(model) for key, model in models.items()}
    x = torch.rand(8, 50, 96)
    target = torch.rand(8, 2)
    batch = SimpleNamespace(X=x, last_target=target, unit_mask=torch.ones(96, dtype=torch.bool), bank=None)
    x_before, y_before = x.clone(), target.clone()
    global_before = torch.get_rng_state().clone()
    row = phase.step_cells(batch, models, optimizers, emas, epoch=1, batch_id=0)
    assert torch.equal(global_before, torch.get_rng_state())
    assert torch.equal(x, x_before) and torch.equal(target, y_before)
    assert 0 < row["cold_rows"] < row["rows"]
    assert row["loss"]["FLAT_CONTROL"] == row["loss"]["ROUTE_CONTROL"]
    assert row["loss"]["FLAT_PREFIX"] == row["loss"]["ROUTE_PREFIX"]
    assert row["loss"]["FLAT_CONTROL"] != row["loss"]["FLAT_PREFIX"]
    assert all(ema.n_updates == 1 for ema in emas.values())


def test_actual_checkpoint_serialization_contract_and_refusals(tmp_path):
    torch.set_num_threads(1)
    model = Tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ema = phase.DecoderEMA(model)
    ema.update_after_step(model)
    ema.n_updates = 6330  # Tiny helper fixture, not a training claim.
    bound = {"manifest_sha256": "fixture"}
    payload = phase._checkpoint("FLAT_CONTROL", model, optimizer, ema, 2, bound)
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    phase.validate_phase_checkpoint(restored, model, "FLAT_CONTROL", 2, bound)
    for field, bad in (("cell", "ROUTE_CONTROL"), ("global_step", 1), ("batch_id", 0), ("seed", 43)):
        changed = {**restored, field: bad}
        with pytest.raises(RuntimeError, match="identity"):
            phase.validate_phase_checkpoint(changed, model, "FLAT_CONTROL", 2, bound)
    restored["ema"]["n_updates"] = 2
    with pytest.raises(RuntimeError, match="EMA"):
        phase.validate_phase_checkpoint(restored, model, "FLAT_CONTROL", 2, bound)
