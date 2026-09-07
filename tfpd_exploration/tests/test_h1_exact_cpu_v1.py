"""CPU-only exactness tests for the isolated H1 exact CPU backend."""
import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tfpd_exploration.src.h1_exact_cpu_v1 import (
    CompiledExactFullWindowStream,
    CompiledStaticCarrierQOnlyExactFullWindowStream,
    ExactTemporalLastRow,
    QOnlyTemporalLastRow,
    StaticCarrierExactFullWindowStream,
    StaticCarrierQOnlyExactFullWindowStream,
)
from tfpd_exploration.src.h1_optimized_v4.model import SignedCarrierFrontend
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1TemporalConfig
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import CausalTransformerStack
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import ExactFullWindowStream


class Bank:
    def __init__(self, e0, temporal, mask):
        self.E0, self.T, self.unit_mask = e0, temporal, mask


class TinyFull(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(window=11, conv_kernel=5)
        self.mix = nn.Linear(1, 16)
        self.temporal = CausalTransformerStack(
            H1TemporalConfig(temporal_width=16, heads=4, layers=3, ffn=24, window=11, pe_max_len=11)
        )
        self.final_norm, self.readout = nn.LayerNorm(16), nn.Linear(16, 2)

    def encode_frontend(self, x, bank):
        # Causal frontend is unnecessary to test the temporal override, but
        # retains a public stream-compatible [B,W,D] output.
        return self.mix(x.mean(-1, keepdim=True))

    def forward_last(self, x, bank):
        return self.readout(self.final_norm(self.temporal(self.encode_frontend(x, bank))))[:, -1]


class TinySignedFull(nn.Module):
    """Small V4-form frontend used to test static-carrier exactness cheaply."""
    def __init__(self):
        super().__init__()
        self.cfg = H1TemporalConfig(e0_dim=3, hc_dim=2, temporal_width=16, heads=4,
                                    layers=3, ffn=24, window=11, pe_max_len=11,
                                    readout_hidden=12, out_dim=2)
        self.frontend = SignedCarrierFrontend(self.cfg)
        self.temporal = CausalTransformerStack(self.cfg)
        self.final_norm, self.readout = nn.LayerNorm(16), nn.Linear(16, 2)

    def encode_frontend(self, x, bank):
        return self.frontend(x, bank)

    def forward_last(self, x, bank):
        return self.readout(self.final_norm(self.temporal(self.encode_frontend(x, bank))))[:, -1]


def _items():
    torch.manual_seed(91)
    model = TinyFull().eval()
    bank = Bank(torch.randn(5, 2), torch.randn(5, 2), torch.ones(5, dtype=torch.bool))
    return model, bank


def test_exact_temporal_last_row_matches_full_stack_last_row():
    model, _ = _items()
    z = torch.randn(2, 11, 16)
    with torch.inference_mode():
        expected = model.temporal(z)[:, -1:]
        actual = ExactTemporalLastRow(model.temporal)(z)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_qonly_temporal_last_row_matches_full_stack_last_row():
    model, _ = _items()
    z = torch.randn(2, 11, 16)
    with torch.inference_mode():
        expected = model.temporal(z)[:, -1:]
        actual = QOnlyTemporalLastRow(model.temporal)(z)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_isolated_stream_matches_reference_native_api_through_startup_and_rollover():
    model, bank = _items()
    reference = ExactFullWindowStream(model, bank, task="h1", session_id="s", unit_ids=range(5))
    candidate = CompiledExactFullWindowStream(
        model, bank, task="h1", session_id="s", unit_ids=range(5), compile_backend=False
    )
    for _ in range(29):
        observation = torch.randn(1, 5)
        got = candidate.predict(observation)
        expected = reference.predict(observation)
        np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)


def test_isolated_backend_retains_parent_mutation_guard():
    model, bank = _items()
    stream = CompiledExactFullWindowStream(
        model, bank, task="h1", session_id="s", unit_ids=range(5), compile_backend=False
    )
    stream.predict(torch.randn(1, 5))
    with torch.no_grad():
        bank.T.add_(0.25)
    actual = stream.current_prediction()
    expected = model.forward_last(stream.front.raw, bank) / 20.0
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert stream.last_rebuild_reason == "model_bank_or_mask_changed"


def test_static_carrier_matches_reference_through_biased_startup_and_rollover():
    torch.manual_seed(777)
    model = TinySignedFull().eval()
    bank = Bank(torch.randn(5, 3), torch.randn(5, 2), torch.ones(5, dtype=torch.bool))
    reference = ExactFullWindowStream(model, bank, task="h1", session_id="s", unit_ids=range(5))
    candidate = StaticCarrierExactFullWindowStream(model, bank, task="h1", session_id="s", unit_ids=range(5))
    assert candidate.static_carrier_bytes == sum(t.numel() * t.element_size()
                                               for t in (candidate._static_weight, candidate._static_mask))
    assert candidate.state_bytes == reference.state_bytes + candidate.static_carrier_bytes
    for _ in range(31):
        observation = torch.randn(1, 5)
        np.testing.assert_allclose(candidate.predict(observation), reference.predict(observation), atol=1e-5, rtol=1e-5)


def test_static_carrier_rebuilds_on_bank_mask_parameter_and_dtype_changes():
    torch.manual_seed(778)
    model = TinySignedFull().eval()
    bank = Bank(torch.randn(5, 3), torch.randn(5, 2), torch.ones(5, dtype=torch.bool))
    stream = StaticCarrierExactFullWindowStream(model, bank, task="h1", session_id="s", unit_ids=range(5))
    stream.predict(torch.randn(1, 5)); initial = stream.static_rebuild_count
    with torch.no_grad():
        bank.T.add_(0.2)
    stream.current_prediction(); assert stream.static_rebuild_count == initial + 1
    with torch.no_grad():
        bank.unit_mask[0] = False
    stream.current_prediction(); assert stream.static_rebuild_count == initial + 2
    with torch.no_grad():
        model.frontend.phi.weight.add_(0.1)
    stream.current_prediction(); assert stream.static_rebuild_count == initial + 3
    model.double()
    # Parent token detection requests the virtual rebuild before static use.
    stream.current_prediction(); assert stream.static_rebuild_count == initial + 4
    expected = model.forward_last(stream.front.raw.double(), bank) / 20.0
    torch.testing.assert_close(stream.current_prediction(), expected, atol=1e-5, rtol=1e-5)


def test_static_carrier_qonly_matches_static_e_through_startup_rollover_and_mutation():
    torch.manual_seed(779)
    model = TinySignedFull().eval()
    bank = Bank(torch.randn(5, 3), torch.randn(5, 2), torch.ones(5, dtype=torch.bool))
    exact = StaticCarrierExactFullWindowStream(model, bank, task="h1", session_id="s", unit_ids=range(5))
    qonly = StaticCarrierQOnlyExactFullWindowStream(model, bank, task="h1", session_id="s", unit_ids=range(5))
    for step in range(27):
        observation = torch.randn(1, 5)
        np.testing.assert_allclose(qonly.predict(observation), exact.predict(observation), atol=1e-5, rtol=1e-5)
        if step == 12:
            with torch.no_grad():
                bank.T.add_(0.125)
    with torch.no_grad():
        model.temporal.blocks[-1].attn.qkv.weight.add_(0.01)
    torch.testing.assert_close(qonly.current_prediction(), exact.current_prediction(), atol=1e-5, rtol=1e-5)


def test_compiled_static_qonly_eager_backend_matches_static_e():
    torch.manual_seed(780)
    model = TinySignedFull().eval()
    bank = Bank(torch.randn(5, 3), torch.randn(5, 2), torch.ones(5, dtype=torch.bool))
    exact = StaticCarrierExactFullWindowStream(model, bank, task="h1", session_id="s", unit_ids=range(5))
    candidate = CompiledStaticCarrierQOnlyExactFullWindowStream(
        model, bank, task="h1", session_id="s", unit_ids=range(5), compile_backend=False
    )
    for _ in range(27):
        observation = torch.randn(1, 5)
        np.testing.assert_allclose(candidate.predict(observation), exact.predict(observation), atol=1e-5, rtol=1e-5)


def test_compiled_static_qonly_refreshes_after_temporal_replacement_and_dtype_change():
    torch.manual_seed(781)
    model = TinySignedFull().eval()
    bank = Bank(torch.randn(5, 3), torch.randn(5, 2), torch.ones(5, dtype=torch.bool))
    candidate = CompiledStaticCarrierQOnlyExactFullWindowStream(
        model, bank, task="h1", session_id="s", unit_ids=range(5), compile_backend=False
    )
    candidate.predict(torch.randn(1, 5))
    original_temporal = candidate.exact_temporal.temporal
    model.temporal = copy.deepcopy(model.temporal).eval()
    candidate.current_prediction()
    assert candidate.exact_temporal.temporal is model.temporal
    assert candidate.exact_temporal.temporal is not original_temporal
    model.double()
    candidate.current_prediction()
    assert next(candidate.exact_temporal.parameters()).dtype == torch.float64
