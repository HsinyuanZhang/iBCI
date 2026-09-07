"""Production-path CPU proof summary for the A1 preflight."""
from __future__ import annotations

import copy
import sys
from functools import partial
from io import BytesIO
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
STREAMING = REPO / "streaming_calibration_exp"
if str(STREAMING) not in sys.path:
    sys.path.insert(0, str(STREAMING))

from src.models.a1_hidden_carrier_module import A1HiddenCarrierLitModule
from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.streaming_spint_hidden_carrier_adapter import (
    HiddenSpaceCarrierStreamingSpint,
    ZeroInitializedHiddenCarrierMap,
)
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.window_size = 6

    def forward_batch(self, calib_trials, side_features=None, electrode_ids=None):
        del side_features, electrode_ids
        return calib_trials.mean(dim=1).permute(0, 2, 1) * self.scale


def _decoder() -> SpintModel:
    saved = torch.random.get_rng_state()
    torch.manual_seed(1181)
    decoder = SpintModel(
        model_dim=16, num_covariates=2, window_size=6, num_heads=4,
        num_layers=1, num_id_layers=1, dropout_rate=0.0,
        dynamic_dropout=False, tf_drop_rate=0.0,
    )
    decoder.fc_id_in(torch.zeros(1, 1, 1, 6))
    torch.random.set_rng_state(saved)
    return decoder


def _pair():
    decoder, encoder = _decoder(), _Encoder()
    w = StreamingSpintModel(decoder=copy.deepcopy(decoder), id_encoder=copy.deepcopy(encoder), decoder_mode="coupled")
    h = HiddenSpaceCarrierStreamingSpint(decoder=copy.deepcopy(decoder), id_encoder=copy.deepcopy(encoder), add_site="hidden")
    return w, h


def _batch(seed: int):
    gen = torch.Generator().manual_seed(seed)
    neural = torch.randn(2, 6, 7, generator=gen)
    calib = torch.randn(2, 3, 6, 7, generator=gen)
    carrier = torch.randn(2, 7, 4, generator=gen)
    target = torch.randn(2, 6, 2, generator=gen)
    return neural, calib, carrier, target


def _fake_parent_setup(self: StreamingCalibrationLitModule, stage: str) -> None:
    del stage
    if self.student is not None:
        return
    self.teacher = _decoder()
    self.student = StreamingSpintModel(decoder=_decoder(), id_encoder=_Encoder(), decoder_mode="coupled")
    if self._freeze_decoder:
        self.student.freeze_decoder()


def _lit() -> A1HiddenCarrierLitModule:
    module = A1HiddenCarrierLitModule(
        task="mc_maze", variant="B3S", teacher_ckpt_path="/not-opened.ckpt",
        window_size=6, trial_length=6, id_hidden_dim=8, hidden_dim=8,
        freeze_decoder=False, loss_mode="task_only", lambda_y=0.0,
        lambda_E=0.0, identity_mode="calibrated", side_dim=4,
        optimizer=partial(torch.optim.Adam, lr=1e-4), scheduler=None,
        compile=False, decoder_mode="coupled", a1_attachment_mode="aligned",
    )
    original = StreamingCalibrationLitModule.setup
    try:
        StreamingCalibrationLitModule.setup = _fake_parent_setup
        module.setup("fit")
    finally:
        StreamingCalibrationLitModule.setup = original
    return module


def _optimizer_parameter_state_equal(
    left: torch.optim.Optimizer,
    left_parameter: nn.Parameter,
    right: torch.optim.Optimizer,
    right_parameter: nn.Parameter,
) -> bool:
    left_state = left.state.get(left_parameter, {})
    right_state = right.state.get(right_parameter, {})
    if left_state.keys() != right_state.keys():
        return False
    for key in left_state:
        left_value = left_state[key]
        right_value = right_state[key]
        if torch.is_tensor(left_value):
            if not torch.is_tensor(right_value) or not torch.equal(left_value, right_value):
                return False
        elif left_value != right_value:
            return False
    return True


def run_cpu_proofs() -> dict[str, object]:
    proofs: dict[str, bool] = {}
    saved = torch.random.get_rng_state()
    torch.manual_seed(20260813)
    before = torch.random.get_rng_state().clone()
    projection = ZeroInitializedHiddenCarrierMap(carrier_dim=4, hidden_dim=512)
    proofs["direct_zero_projection_does_not_advance_rng"] = torch.equal(before, torch.random.get_rng_state())
    torch.random.set_rng_state(saved)
    proofs["p_shape_bias_and_zero"] = (
        tuple(projection.weight.shape) == (512, 4)
        and projection.bias is None
        and torch.equal(projection.weight, torch.zeros_like(projection.weight))
    )

    w, h = _pair()
    neural, calib, t4, target = _batch(4100)
    z4 = torch.zeros_like(t4)
    w_pred, w_id = w(neural, calib_trials=calib, side_features=z4)
    h_pred, h_id = h(neural, calib_trials=calib, side_features=z4)
    proofs["h_z4_forward_and_identity_bit_equal"] = torch.equal(w_pred, h_pred) and torch.equal(w_id, h_id)
    w_opt = torch.optim.AdamW(w.parameters(), lr=1e-3, foreach=False)
    h_opt = torch.optim.AdamW(h.parameters(), lr=1e-3, foreach=False)
    w_opt.zero_grad(set_to_none=True); h_opt.zero_grad(set_to_none=True)
    lw = torch.nn.functional.mse_loss(w_pred, target); lh = torch.nn.functional.mse_loss(h_pred, target)
    lw.backward(); lh.backward()
    w_shared = dict(w.named_parameters())
    h_shared = {name: p for name, p in h.named_parameters() if not name.startswith("hidden_carrier_map.")}
    proofs["h_z4_loss_shared_grad_and_p_grad_bit_equal"] = (
        torch.equal(lw, lh)
        and all(torch.equal(w_shared[name].grad, h_shared[name].grad) for name in w_shared if w_shared[name].grad is not None)
        and h.hidden_carrier_map.weight.grad is not None
        and torch.equal(h.hidden_carrier_map.weight.grad, torch.zeros_like(h.hidden_carrier_map.weight.grad))
    )
    w_opt.step(); h_opt.step()
    proofs["h_z4_shared_optimizer_step_bit_equal"] = all(torch.equal(w_shared[name], h_shared[name]) for name in w_shared)
    proofs["h_z4_shared_optimizer_state_bit_equal"] = all(
        _optimizer_parameter_state_equal(
            w_opt, w_shared[name], h_opt, h_shared[name]
        )
        for name in w_shared
    )

    w_checkpoint = copy.deepcopy(w.state_dict())
    h_checkpoint = copy.deepcopy(h.state_dict())
    checkpoint_buffer = BytesIO()
    torch.save({"w": w_checkpoint, "h": h_checkpoint}, checkpoint_buffer)
    checkpoint_buffer.seek(0)
    restored_pair = torch.load(checkpoint_buffer, map_location="cpu", weights_only=True)
    restored_w, restored_h = _pair()
    w_result = restored_w.load_state_dict(restored_pair["w"], strict=True)
    h_result = restored_h.load_state_dict(restored_pair["h"], strict=True)
    restored_h_shared = {
        name: value
        for name, value in restored_h.state_dict().items()
        if not name.startswith("hidden_carrier_map.")
    }
    proofs["h_z4_checkpoint_roundtrip_shared_parity"] = (
        not w_result.missing_keys
        and not w_result.unexpected_keys
        and not h_result.missing_keys
        and not h_result.unexpected_keys
        and restored_w.state_dict().keys() == restored_h_shared.keys()
        and all(
            torch.equal(value, restored_h_shared[name])
            for name, value in restored_w.state_dict().items()
        )
        and torch.equal(
            restored_h.hidden_carrier_map.weight,
            torch.zeros_like(restored_h.hidden_carrier_map.weight),
        )
    )

    _, active = _pair()
    active_opt = torch.optim.AdamW(active.parameters(), lr=1e-3, foreach=False)
    active_pred, _ = active(neural, calib_trials=calib, side_features=t4)
    active_opt.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(active_pred, target).backward()
    proofs["nonzero_t4_reaches_p"] = torch.count_nonzero(active.hidden_carrier_map.weight.grad).item() > 0

    module = _lit()
    configured = module.configure_optimizers()["optimizer"]
    observed = [p for group in configured.param_groups for p in group["params"]]
    expected = [p for p in module.student.parameters() if p.requires_grad]
    proofs["production_optimizer_covers_p_and_all_trainables_once"] = (
        len(observed) == len(set(map(id, observed)))
        and set(map(id, observed)) == set(map(id, expected))
        and sum(p is module.student.hidden_carrier_map.weight for p in observed) == 1
    )
    state = copy.deepcopy(module.state_dict())
    buffer = BytesIO(); torch.save(state, buffer); buffer.seek(0)
    restored_state = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = _lit()
    result = restored.load_state_dict(restored_state, strict=True)
    proofs["production_strict_checkpoint_roundtrip"] = (
        not result.missing_keys and not result.unexpected_keys
        and "student.hidden_carrier_map.weight" in state
        and all(torch.equal(state[name], restored.state_dict()[name]) for name in state)
    )
    proofs["cpu_only_no_data_or_training"] = all(parameter.device.type == "cpu" for parameter in module.parameters())
    return {
        "schema_version": 2,
        "family": "a1_hidden_space_carrier_production_cpu_proofs",
        "gpu_used": False, "training_started": False,
        "nwb_opened": False, "formal_test_opened": False,
        "proofs": proofs, "all_passed": all(proofs.values()),
    }
