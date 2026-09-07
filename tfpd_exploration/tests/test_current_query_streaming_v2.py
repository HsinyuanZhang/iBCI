"""Model-level streaming: biased startup, gaps, reset and invalidation."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import CurrentQueryStream, ExactFullWindowStream, QueryTemporalStack


@dataclass
class Bank:
    E0: torch.Tensor
    T: torch.Tensor
    unit_mask: torch.Tensor


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(window=11)
        self.conv = nn.Conv1d(1, 3, 5, bias=True)
        self.proj = nn.Linear(3, 16)
        self.temporal = QueryTemporalStack(width=16, heads=4, layers=2, ffn=24, window=11)
        self.final_norm = nn.LayerNorm(16)
        self.readout = nn.Linear(16, 2)
        self.activity_scale = 2.0

    def _z(self, x, bank):
        b, t, n = x.shape
        features = self.conv(F.pad((x * self.activity_scale).transpose(1, 2).reshape(b*n, 1, t), (4, 0)))
        features = features.reshape(b, n, 3, t).permute(0, 3, 1, 2)
        keep = bank.unit_mask.to(x).view(1, 1, n, 1)
        static = (bank.E0[:, :3] + bank.T[:, :3]).to(x)[None, None]
        return self.proj(((features + static) * keep).sum(2) / keep.sum(2))

    def _fuse(self, x, bank, _mask):
        return self._z(x, bank)

    def forward_last(self, x, bank):
        return self.readout(self.final_norm(self.temporal(self._z(x, bank))))[:, -1]


def make_stream(task="h1", batch=1):
    torch.manual_seed(19)
    model = ToyDecoder().eval()
    bank = Bank(torch.randn(5, 3), torch.randn(5, 3), torch.ones(5, dtype=torch.bool))
    stream = CurrentQueryStream(model, bank, task=task, session_id="session-a", unit_ids=tuple(range(5)), batch_size=batch)
    return model, bank, stream


@pytest.mark.parametrize("task", ["h1", "m1"])
def test_startup_rollover_observe_gaps_and_batch_isolation(task):
    model, bank, stream = make_stream(task, batch=2)
    divisor = 20 if task == "h1" else 1
    raw = torch.zeros(2, 11, 5)
    for step in range(28):
        x = torch.randn(2, 5)
        raw = torch.cat((raw[:, 1:], x[:, None]), 1)
        if step % 3:
            pred = torch.from_numpy(stream.predict(x))
        else:
            stream.observe(x)
            pred = stream.current_prediction()
        torch.testing.assert_close(pred, model.forward_last(raw, bank) / divisor, atol=1e-5, rtol=1e-5)
        if step == 8:
            stream.on_done()
            assert stream.n_observations == 9
    assert stream.state_bytes > 0
    stream.on_done(reset_session=True)
    torch.testing.assert_close(stream.current_prediction(), model.forward_last(torch.zeros_like(raw), bank) / divisor)


def test_rebuild_on_weights_bank_mask_scalar_dtype():
    model, bank, stream = make_stream()
    history = torch.randn(1, 11, 5)
    stream.reset(history=history)
    with torch.no_grad():
        model.conv.weight.add_(.2)
    torch.testing.assert_close(stream.current_prediction(), model.forward_last(history, bank) / 20)
    bank.T.add_(.3)
    bank.unit_mask[0] = False
    model.activity_scale = 3.
    torch.testing.assert_close(stream.current_prediction(), model.forward_last(history, bank) / 20)
    model.double()
    torch.testing.assert_close(stream.current_prediction(), model.forward_last(history.double(), bank) / 20)
    assert stream.last_rebuild_reason == "model_bank_or_mask_changed"


def test_explicit_roster_permutation_and_session_reset():
    model, bank, stream = make_stream()
    history = torch.randn(1, 11, 5)
    stream.reset(history=history)
    expected = stream.current_prediction()
    perm = torch.tensor([3, 0, 4, 1, 2])
    changed = Bank(bank.E0[perm], bank.T[perm], bank.unit_mask[perm])
    stream.reset(bank=changed, unit_ids=perm.tolist(), session_id="b", history=history[:, :, perm])
    torch.testing.assert_close(stream.current_prediction(), expected)
    with pytest.raises(ValueError, match="session changed"):
        stream.predict(np.zeros((1, 5)), session_id="a")
    stream.reset(session_id="c")
    assert stream.n_observations == 0
    with pytest.raises(ValueError, match="unique"):
        stream.reset(unit_ids=[0] * 5)


def test_strict_shape_units_and_training_rejection():
    model, bank, stream = make_stream()
    with pytest.raises(ValueError, match="divisor"):
        CurrentQueryStream(model, bank, task="h1", session_id="a", unit_ids=range(5), prediction_divisor=1)
    with pytest.raises(ValueError, match="observation"):
        stream.observe(torch.zeros(5))
    with pytest.raises(ValueError, match="nonfinite"):
        stream.observe(torch.full((1, 5), float("nan")))
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        stream.current_prediction()


def test_inference_mode_bank_versions_are_explicitly_checked():
    model, bank, _ = make_stream()
    with torch.inference_mode():
        inference_bank = Bank(bank.E0.clone(), bank.T.clone(), bank.unit_mask.clone())
        stream = CurrentQueryStream(model, inference_bank, task="h1", session_id="a", unit_ids=range(5))
        stream.observe(torch.randn(1, 5))
        inference_bank.T.add_(.5)
        inference_bank.unit_mask[0] = False
        predicted = stream.current_prediction()
        torch.testing.assert_close(predicted, model.forward_last(stream.front.raw, inference_bank) / 20)
        assert stream.last_rebuild_reason == "model_bank_or_mask_changed"


def test_public_frontend_hook_does_not_require_legacy_private_z():
    class PublicHookOnly(ToyDecoder):
        encode_frontend = ToyDecoder._z

        @property
        def _z(self):
            raise AttributeError("this model exposes only encode_frontend")

    model = PublicHookOnly().eval()
    bank = Bank(torch.randn(5, 3), torch.randn(5, 3), torch.ones(5, dtype=torch.bool))
    stream = CurrentQueryStream(model, bank, task="h1", session_id="a", unit_ids=range(5))
    with torch.no_grad():
        value = stream.predict(torch.randn(1, 5))
        reference = model.readout(model.final_norm(model.temporal(model.encode_frontend(stream.front.raw, bank))))[:, -1] / 20
    torch.testing.assert_close(torch.from_numpy(value), reference)


@pytest.mark.parametrize("task", ["h1", "m1"])
def test_exact_full_window_control_matches_full_recomputation(task):
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1TemporalConfig
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import CausalTransformerStack
    model, bank, _ = make_stream(task)
    model.temporal = CausalTransformerStack(H1TemporalConfig(temporal_width=16, heads=4, layers=3,
                                                            ffn=24, window=11, pe_max_len=11))
    # The toy full-window shell must return its LAST readout, just like production.
    model.forward_last = lambda x, bank: model.readout(model.final_norm(model.temporal(model._z(x, bank))))[:, -1]
    model.eval()
    stream = ExactFullWindowStream(model, bank, task=task, session_id="a", unit_ids=range(5))
    divisor = 20 if task == "h1" else 1
    raw = torch.zeros(1, 11, 5)
    for step in range(28):
        x = torch.randn(1, 5)
        raw = torch.cat((raw[:, 1:], x[:, None]), 1)
        p = torch.from_numpy(stream.predict(x))
        torch.testing.assert_close(p, model.forward_last(raw, bank) / divisor, atol=1e-5, rtol=1e-5)
    assert stream.memory is None
    with torch.no_grad():
        model.conv.bias.add_(.3)
    bank.unit_mask[1] = False
    torch.testing.assert_close(stream.current_prediction(), model.forward_last(raw, bank) / divisor, atol=1e-5, rtol=1e-5)
