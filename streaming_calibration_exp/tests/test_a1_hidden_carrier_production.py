"""Production-path contracts for A1 hidden-space carrier integration."""
from __future__ import annotations

import copy
from functools import partial
from io import BytesIO
from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.a1_hidden_carrier_module import A1HiddenCarrierLitModule
from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.streaming_spint_hidden_carrier_adapter import (
    HiddenSpaceCarrierStreamingSpint,
    ZeroInitializedHiddenCarrierMap,
)
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


class _ActivityIdentity(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.window_size = window_size
        self.seen_side: list[torch.Tensor | None] = []

    def forward_batch(self, calib_trials, side_features=None, electrode_ids=None):
        del electrode_ids
        self.seen_side.append(side_features)
        return calib_trials.mean(dim=1).permute(0, 2, 1) * self.scale


def _decoder(*, seed: int = 1181, window_size: int = 6) -> SpintModel:
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    decoder = SpintModel(
        model_dim=16,
        num_covariates=2,
        window_size=window_size,
        num_heads=4,
        num_layers=1,
        num_id_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.0,
    )
    decoder.fc_id_in(torch.zeros(1, 1, 1, window_size))
    torch.random.set_rng_state(state)
    return decoder


def _paired_models() -> tuple[StreamingSpintModel, HiddenSpaceCarrierStreamingSpint]:
    decoder = _decoder()
    encoder = _ActivityIdentity(6)
    w = StreamingSpintModel(
        decoder=copy.deepcopy(decoder),
        id_encoder=copy.deepcopy(encoder),
        decoder_mode="coupled",
    )
    h = HiddenSpaceCarrierStreamingSpint(
        decoder=copy.deepcopy(decoder),
        id_encoder=copy.deepcopy(encoder),
        add_site="hidden",
    )
    return w, h


def _shared_named(model: nn.Module) -> dict[str, nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("hidden_carrier_map.")
    }


def _optimizer_state_by_name(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, dict[str, object]]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    result: dict[str, dict[str, object]] = {}
    for parameter, state in optimizer.state.items():
        name = names[id(parameter)]
        result[name] = state
    return result


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    else:
        assert left == right


def _batch(seed: int, *, batch_size: int = 2, num_units: int = 7):
    generator = torch.Generator().manual_seed(seed)
    neural = torch.randn(batch_size, 6, num_units, generator=generator)
    calib = torch.randn(batch_size, 3, 6, num_units, generator=generator)
    carrier = torch.randn(batch_size, num_units, 4, generator=generator)
    target = torch.randn(batch_size, 6, 2, generator=generator)
    return neural, calib, carrier, target


def test_direct_zero_projection_does_not_advance_rng_and_has_no_bias() -> None:
    torch.manual_seed(20260813)
    before = torch.random.get_rng_state().clone()
    projection = ZeroInitializedHiddenCarrierMap(carrier_dim=4, hidden_dim=512)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert tuple(projection.weight.shape) == (512, 4)
    assert projection.bias is None
    assert list(dict(projection.named_parameters())) == ["weight"]
    assert list(projection.state_dict()) == ["weight"]
    assert torch.equal(projection.weight, torch.zeros_like(projection.weight))


def test_h_wrapper_construction_does_not_advance_substrate_rng() -> None:
    decoder = _decoder()
    encoder = _ActivityIdentity(6)
    torch.manual_seed(9431)
    before = torch.random.get_rng_state().clone()
    HiddenSpaceCarrierStreamingSpint(
        decoder=decoder,
        id_encoder=encoder,
        add_site="hidden",
    )
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("num_units", (1, 7, 64))
def test_actual_decoder_path_is_bit_exact_at_zero_initialized_p(
    batch_size: int, num_units: int
) -> None:
    w, h = _paired_models()
    w.eval()
    h.eval()
    neural, calib, t4, _ = _batch(
        3200 + 100 * batch_size + num_units,
        batch_size=batch_size,
        num_units=num_units,
    )
    z4 = torch.zeros_like(t4)
    w_prediction, w_identity = w(neural, calib_trials=calib, side_features=z4)
    h_prediction, h_identity = h(neural, calib_trials=calib, side_features=t4)
    assert torch.equal(w_identity, h_identity)
    assert torch.equal(w_prediction, h_prediction)
    assert h.hidden_carrier_map is not None
    assert torch.equal(
        h.hidden_carrier_map(t4),
        torch.zeros(batch_size, num_units, h.decoder.model_dim),
    )


def test_three_task_only_steps_keep_h_z4_shared_state_bit_exact() -> None:
    w, h = _paired_models()
    w.train()
    h.train()
    w_optimizer = torch.optim.AdamW(
        w.parameters(), lr=1.0e-3, weight_decay=1.0e-2, foreach=False
    )
    h_optimizer = torch.optim.AdamW(
        h.parameters(), lr=1.0e-3, weight_decay=1.0e-2, foreach=False
    )
    for step in range(3):
        neural, calib, _, target = _batch(4100 + step)
        z4 = torch.zeros(neural.shape[0], neural.shape[-1], 4)
        w_optimizer.zero_grad(set_to_none=True)
        h_optimizer.zero_grad(set_to_none=True)
        w_prediction, _ = w(neural, calib_trials=calib, side_features=z4)
        h_prediction, _ = h(neural, calib_trials=calib, side_features=z4)
        assert torch.equal(w_prediction, h_prediction)
        w_loss = torch.nn.functional.mse_loss(w_prediction, target)
        h_loss = torch.nn.functional.mse_loss(h_prediction, target)
        assert torch.equal(w_loss, h_loss)
        w_loss.backward()
        h_loss.backward()

        w_shared = _shared_named(w)
        h_shared = _shared_named(h)
        assert set(w_shared) == set(h_shared)
        for name in w_shared:
            assert (w_shared[name].grad is None) == (h_shared[name].grad is None)
            if w_shared[name].grad is not None:
                assert torch.equal(w_shared[name].grad, h_shared[name].grad)
        assert h.hidden_carrier_map is not None
        assert h.hidden_carrier_map.weight.grad is not None
        assert torch.equal(
            h.hidden_carrier_map.weight.grad,
            torch.zeros_like(h.hidden_carrier_map.weight.grad),
        )

        w_optimizer.step()
        h_optimizer.step()
        for name in w_shared:
            assert torch.equal(w_shared[name], h_shared[name])
        w_state = _optimizer_state_by_name(w, w_optimizer)
        h_state = _optimizer_state_by_name(h, h_optimizer)
        for name in w_shared:
            assert (name in w_state) == (name in h_state)
            if name in w_state:
                _assert_nested_equal(w_state[name], h_state[name])


def test_nonzero_t4_reaches_p_and_one_step_changes_prediction() -> None:
    _, h = _paired_models()
    h.train()
    optimizer = torch.optim.AdamW(h.parameters(), lr=1.0e-3, foreach=False)
    neural, calib, t4, target = _batch(5201)
    prediction_before, _ = h(neural, calib_trials=calib, side_features=t4)
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(prediction_before, target).backward()
    assert h.hidden_carrier_map is not None
    gradient = h.hidden_carrier_map.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0
    optimizer.step()
    prediction_after, _ = h(neural, calib_trials=calib, side_features=t4)
    assert torch.count_nonzero(h.hidden_carrier_map.weight).item() > 0
    assert not torch.equal(prediction_before.detach(), prediction_after.detach())
    assert torch.equal(
        h.hidden_carrier_map(torch.zeros_like(t4)),
        torch.zeros(neural.shape[0], neural.shape[-1], h.decoder.model_dim),
    )


def test_ts4_is_same_checkpoint_evaluation_only_attachment() -> None:
    _, model = _paired_models()
    neural, calib, t4, _ = _batch(5701)
    assert model.hidden_carrier_map is not None
    with torch.no_grad():
        model.hidden_carrier_map.weight.fill_(0.125)
    state_before = copy.deepcopy(model.state_dict())
    model.eval()
    aligned, aligned_identity = model(
        neural, calib_trials=calib, side_features=t4
    )
    model.set_attachment_control_for_evaluation(
        attachment_mode="shuffled", permutation_seed=42
    )
    shuffled, shuffled_identity = model(
        neural, calib_trials=calib, side_features=t4
    )
    assert torch.equal(aligned_identity, shuffled_identity)
    assert not torch.equal(aligned, shuffled)
    for name, tensor in state_before.items():
        assert torch.equal(tensor, model.state_dict()[name])
    model.train()
    with pytest.raises(RuntimeError, match="evaluation-only"):
        model.set_attachment_control_for_evaluation(
            attachment_mode="aligned"
        )


def test_production_model_and_optimizer_strict_checkpoint_roundtrip() -> None:
    _, model = _paired_models()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, foreach=False)
    neural, calib, t4, target = _batch(6101)
    optimizer.zero_grad(set_to_none=True)
    prediction, _ = model(neural, calib_trials=calib, side_features=t4)
    torch.nn.functional.mse_loss(prediction, target).backward()
    optimizer.step()
    expected, _ = model(neural, calib_trials=calib, side_features=t4)

    buffer = BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        buffer,
    )
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)
    _, restored = _paired_models()
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(), lr=1.0e-3, foreach=False
    )
    restored.load_state_dict(payload["model"], strict=True)
    restored_optimizer.load_state_dict(payload["optimizer"])
    actual, _ = restored(neural, calib_trials=calib, side_features=t4)
    assert torch.equal(expected, actual)
    assert set(model.state_dict()) == set(restored.state_dict())
    for name in model.state_dict():
        assert torch.equal(model.state_dict()[name], restored.state_dict()[name])
    _assert_nested_equal(optimizer.state_dict(), restored_optimizer.state_dict())


def _fake_parent_setup(self: StreamingCalibrationLitModule, stage: str) -> None:
    del stage
    if self.student is not None:
        return
    self.teacher = _decoder()
    self.student = StreamingSpintModel(
        decoder=_decoder(),
        id_encoder=_ActivityIdentity(6),
        decoder_mode="coupled",
    )
    if self._freeze_decoder:
        self.student.freeze_decoder()


def _lit_module(*, freeze_decoder: bool) -> A1HiddenCarrierLitModule:
    return A1HiddenCarrierLitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path="/not-opened.ckpt",
        window_size=6,
        trial_length=6,
        id_hidden_dim=8,
        hidden_dim=8,
        freeze_decoder=freeze_decoder,
        loss_mode="task_only",
        lambda_y=0.0,
        lambda_E=0.0,
        identity_mode="calibrated",
        side_dim=4,
        optimizer=partial(torch.optim.Adam, lr=1.0e-4),
        scheduler=None,
        compile=False,
        decoder_mode="coupled",
        a1_attachment_mode="aligned",
    )


@pytest.mark.parametrize("freeze_decoder", (False, True))
def test_lightning_optimizer_includes_p_exactly_once(
    monkeypatch, freeze_decoder: bool
) -> None:
    monkeypatch.setattr(
        StreamingCalibrationLitModule,
        "setup",
        _fake_parent_setup,
    )
    module = _lit_module(freeze_decoder=freeze_decoder)
    module.setup("fit")
    assert isinstance(module.student, HiddenSpaceCarrierStreamingSpint)
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    observed = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert module.student.hidden_carrier_map is not None
    projection = module.student.hidden_carrier_map.weight
    assert sum(parameter is projection for parameter in observed) == 1
    expected = [
        parameter
        for parameter in module.student.parameters()
        if parameter.requires_grad
    ]
    assert {id(parameter) for parameter in observed} == {
        id(parameter) for parameter in expected
    }
    decoder_ids = {id(parameter) for parameter in module.student.decoder.parameters()}
    if freeze_decoder:
        assert decoder_ids.isdisjoint({id(parameter) for parameter in observed})
    else:
        assert decoder_ids <= {id(parameter) for parameter in observed}


def test_lightning_state_dict_strict_roundtrip(monkeypatch) -> None:
    monkeypatch.setattr(
        StreamingCalibrationLitModule,
        "setup",
        _fake_parent_setup,
    )
    original = _lit_module(freeze_decoder=False)
    original.setup("fit")
    assert original.student is not None
    state = copy.deepcopy(original.state_dict())
    assert "student.hidden_carrier_map.weight" in state

    restored = _lit_module(freeze_decoder=False)
    restored.setup("fit")
    restored.load_state_dict(state, strict=True)
    assert set(state) == set(restored.state_dict())
    for name, tensor in state.items():
        assert torch.equal(tensor, restored.state_dict()[name])


def test_h_z4_is_declared_alias_not_a_second_configured_family() -> None:
    config = (
        Path(__file__).resolve().parents[1]
        / "configs/model/a1_hidden_carrier_b3s.yaml"
    ).read_text(encoding="utf-8")
    assert "A1HiddenCarrierLitModule" in config
    assert "freeze_decoder: false" in config
    assert "loss_mode: task_only" in config
    assert "a1_attachment_mode: aligned" in config
    assert "z4" not in config.lower()
