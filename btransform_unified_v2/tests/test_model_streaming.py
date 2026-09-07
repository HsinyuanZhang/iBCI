from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from btransform_unified_v1 import make_synthetic_bank
from btransform_unified_v2 import RiftDecoder, RiftStreamDecoder


def tiny() -> RiftDecoder:
    # M2 has the smallest real frontend geometry; R8 makes this CPU oracle fast.
    return RiftDecoder("m2", context_bins=8, seed=29).eval()


def sha_parameters(model: RiftDecoder) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_stable_frontend_batch_matches_explicit_f_last_and_zero_raw_is_not_zero_token() -> None:
    torch.manual_seed(5)
    model, bank = tiny(), make_synthetic_bank("m2", seed=2)
    x = torch.randn(2, 12, 96)
    full = model.frontend_tokens(x, bank)
    keep = model._frontend_owner._expand_keep(bank, None, x.shape[0])
    v1_frontend = model._frontend_owner._frontend(x, bank, keep, None)
    torch.testing.assert_close(full, v1_frontend, rtol=1e-5, atol=1e-5)
    tokens = []
    for time in range(x.shape[1]):
        local = x[:, max(0, time - 4): time + 1]
        local = torch.cat([torch.zeros(2, 5 - local.shape[1], 96), local], dim=1)
        tokens.append(model.frontend_tokens(local, bank)[:, -1])
    torch.testing.assert_close(full, torch.stack(tokens, 1), rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(model.frontend_tokens(torch.zeros(1, 1, 96), bank)) > 0


def test_frontend_last_executes_only_one_set_token_per_stream_row() -> None:
    torch.manual_seed(6)
    model, bank = tiny(), make_synthetic_bank("m2", seed=8)
    raw5 = torch.randn(3, 5, 96)
    seen: dict[str, tuple[int, ...]] = {}
    def note_token(_module: object, args: tuple[torch.Tensor, ...], _result: torch.Tensor) -> None:
        seen["token_mlp"] = tuple(args[0].shape)
    def note_attention(_module: object, args: tuple[torch.Tensor, ...], _result: object) -> None:
        seen["mha_q"] = tuple(args[0].shape)
    first = model.frontend.token_mlp[0].register_forward_hook(note_token)
    attention = model.frontend.mha.register_forward_hook(note_attention)
    try:
        actual = model.frontend_last(raw5, bank)
    finally:
        first.remove()
        attention.remove()
    assert seen == {"token_mlp": (3, 1, 96, 20), "mha_q": (3, 8, 256)}
    reference = model.frontend_tokens(raw5, bank)[:, -1]
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_decoder_batch_oracle_equals_stream_b1_and_b8() -> None:
    torch.manual_seed(8)
    for batch in (1, 8):
        model, bank = tiny(), make_synthetic_bank("m2", seed=3)
        x = torch.randn(batch, 15, 96)
        oracle = model.forward_scores(x, bank)
        stream = RiftStreamDecoder(model)
        actual = []
        ids = [f"s{row}" for row in range(batch)]
        for time in range(x.shape[1]):
            actual.append(stream.stream_step(x[:, time], bank, ids))
        torch.testing.assert_close(torch.stack(actual, 1), oracle, rtol=1e-5, atol=1e-5)


def test_left_padding_cold_start_is_not_kv_evidence_and_last_semantics() -> None:
    torch.manual_seed(11)
    model, bank = tiny(), make_synthetic_bank("m2", seed=4)
    real = torch.randn(2, 5, 96)
    padded = torch.cat([torch.randn(2, 3, 96), real], 1)
    valid = torch.tensor([[False] * 3 + [True] * 5] * 2)
    scores = model.forward_scores(padded, bank, input_valid_mask=valid)
    expected = model.forward_scores(real, bank)
    torch.testing.assert_close(scores[:, :3], model.readout(model.final_norm(torch.zeros(2, 3, 256))), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scores[:, 3:], expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(model(padded, bank, input_valid_mask=valid), scores[:, -1])


def test_stream_reorder_independent_reset_repeat_predict_and_bank_invalidation() -> None:
    torch.manual_seed(15)
    model, bank = tiny(), make_synthetic_bank("m2", seed=6)
    x = torch.randn(2, 5, 96)
    stream = RiftStreamDecoder(model)
    stream.stream_step(x[:, 0], bank, ["a", "b"])
    stream.stream_step(x.flip(0)[:, 1], bank, ["b", "a"])
    before = stream.predict("a").clone()
    torch.testing.assert_close(stream.predict("a"), before)
    stream.reset("a")
    stream.stream_step(x[0:1, 2], bank, ["a"])
    assert not torch.equal(stream.predict("a"), before)
    other = stream.predict("b").clone()
    changed_mask = torch.from_numpy(np.logical_not(bank.unit_mask.copy()))
    changed_mask[0] = True
    with pytest.raises(RuntimeError, match="changed"):
        stream.stream_step(x[1:2, 2], bank, ["b"], changed_mask)
    torch.testing.assert_close(stream.predict("b"), other)


def test_r300_boundary_causality_and_recency_flat_shared_initialization() -> None:
    torch.manual_seed(17)
    # Small frontend geometry, but the actual R300 temporal allocation.
    geometry = {"task": "tiny", "units": 8, "e0_dim": 16, "carrier_dim": 4, "out_dim": 3}
    recency = RiftDecoder(geometry, context_bins=300, bias_mode="recency", seed=99).eval()
    flat = RiftDecoder(geometry, context_bins=300, bias_mode="flat", seed=99).eval()
    assert recency.temporal_config.windows == (75, 75, 75, 74)
    assert sha_parameters(recency) == sha_parameters(flat)
    # The temporal response must be causal at the R300 boundary.  This needs no
    # TaskBank because it directly checks the new temporal dependency graph.
    z = torch.randn(1, 302, 256)
    base = recency.temporal(z)
    future = z.clone(); future[:, 301] += 10
    torch.testing.assert_close(recency.temporal(future)[:, :301], base[:, :301], rtol=1e-5, atol=1e-5)


def test_h1_primary_geometry_constructs_without_running_training() -> None:
    model = RiftDecoder("h1", context_bins=300).eval()
    assert (model.units, model.geometry["e0_dim"], model.out_dim) == (176, 700, 7)
    assert model.temporal_config.windows == (75, 75, 75, 74)


def test_task_default_contexts_and_batched_multibank_reordered_streams() -> None:
    assert RiftDecoder("m1").context_bins == 100
    assert RiftDecoder("m2").context_bins == 50
    torch.manual_seed(23)
    model = tiny()
    banks = [make_synthetic_bank("m2", seed=31), make_synthetic_bank("m2", seed=32)]
    x = torch.randn(2, 9, 96)
    masks = torch.stack([torch.from_numpy(bank.unit_mask) for bank in banks])
    batched = model.forward_scores(x, banks, masks)
    oracle = torch.cat([model.forward_scores(x[row:row + 1], banks[row], masks[row]) for row in range(2)])
    torch.testing.assert_close(batched, oracle, rtol=1e-5, atol=1e-5)
    stream = RiftStreamDecoder(model)
    out = []
    for time in range(x.shape[1]):
        order = [1, 0] if time % 2 else [0, 1]
        selected_x = x[order, time]
        selected_banks = [banks[row] for row in order]
        selected_masks = masks[order]
        got = stream.stream_step(selected_x, selected_banks, [f"s{row}" for row in order], selected_masks)
        restored = got[[order.index(0), order.index(1)]]
        out.append(restored)
    torch.testing.assert_close(torch.stack(out, 1), oracle, rtol=1e-5, atol=1e-5)
