"""CPU-only behavioral contracts for the additive H1 12→24 continuation."""
from __future__ import annotations

import copy
import hashlib
import random

import numpy as np
import pytest
import torch

from tfpd_exploration.src.h1_queryage_family_v1 import continue_prefix_train as c
from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_train as p
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA


def _manual_keep(n: int, epoch: int, batch_index: int, mask: torch.Tensor) -> torch.Tensor:
    """Independent statement of the published SHA-256 keep-stream law."""
    seed = int.from_bytes(
        hashlib.sha256(f"{c.SEED}|keep|{epoch}|{batch_index}".encode()).digest()[:8],
        "little",
    )
    return (torch.rand((n, mask.numel()), generator=torch.Generator().manual_seed(seed)) >= p.DROP_P) & mask.view(1, -1)


@pytest.mark.parametrize("epoch,batch_index", ((1, 0), (12, 730), (13, 0), (24, 730)))
def test_absolute_stateless_keep_and_prefix_laws_cover_parent_and_continuation(epoch, batch_index):
    """Epochs 1..12 stay parent-identical; 13..24 use that same absolute law."""
    mask = torch.tensor(([True] * 175) + [False])
    actual = c.dropout_keep(n=3, epoch=epoch, batch_index=batch_index, bank_mask=mask, device=torch.device("cpu"))
    assert torch.equal(actual, _manual_keep(3, epoch, batch_index, mask))
    if epoch <= 12:
        parent_keep = p.dropout_keep(n=3, epoch=epoch, batch_index=batch_index, bank_mask=mask, device=torch.device("cpu"))
        assert torch.equal(actual, parent_keep)

    x = torch.arange(3 * 700 * 176, dtype=torch.float32).reshape(3, 700, 176)
    prefixed, lengths, keep = c.prefix_and_keep(x, epoch=epoch, batch_index=batch_index, bank_mask=mask, device=torch.device("cpu"))
    assert torch.equal(keep, actual)
    assert torch.equal(lengths, p.prefix_lengths(3, epoch=epoch, batch_index=batch_index))
    assert torch.equal(prefixed[:, -1], x[:, -1])
    if epoch <= 12:
        parent_prefixed, parent_lengths, parent_keep = p.prefix_and_keep(x, epoch=epoch, batch_index=batch_index, bank_mask=mask, device=torch.device("cpu"))
        assert torch.equal(prefixed, parent_prefixed)
        assert torch.equal(lengths, parent_lengths)
        assert torch.equal(keep, parent_keep)


def test_continuation_geometry_lr_and_24_epoch_earliest_selection():
    assert c.warmup_lr(epoch=13, global_step=12 * c.EPOCH_UPDATES + 1) == c.LR
    assert c.warmup_lr(epoch=24, global_step=24 * c.EPOCH_UPDATES) == c.LR
    with pytest.raises(ValueError):
        c.warmup_lr(epoch=12, global_step=12 * c.EPOCH_UPDATES)
    scores = [(epoch, float(epoch)) for epoch in range(1, 25)]
    scores[23] = (24, 23.0)
    assert c.earliest_argmax_24(scores) == (23, 23.0)


def _new_toy():
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.LR, weight_decay=.01)
    return model, optimizer, DecoderEMA(model, decay=c.EMA)


def _toy_step(model, optimizer, ema):
    """One real AdamW/EMA update which also consumes every saved CPU RNG stream."""
    scale = float(np.random.uniform(.5, 1.5)) + random.random()
    x = torch.randn(4, 3) * scale + torch.rand(4, 3)
    y = torch.randn(4, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    ema.update_after_step(model)
    return float(loss.detach())


def _run(model, optimizer, ema, updates):
    for _ in range(updates):
        _toy_step(model, optimizer, ema)


def _assert_restore_rejected(payload, *, identities, shared, bindings, parent_sha, match=None):
    model, optimizer, ema = _new_toy()
    with pytest.raises(RuntimeError, match=match):
        c.strict_restore(payload=payload, model=model, optimizer=optimizer, ema=ema, epoch=13, identities=identities, arm="flat", shared_init_sha256=shared, bindings=bindings, parent_epoch12_sha256=parent_sha)


def test_real_adamw_decoderema_split_resume_matches_uninterrupted_and_is_strict(monkeypatch):
    """Reduced test geometry, but real 12→13 work, AdamW moments, EMA, and RNG."""
    monkeypatch.setattr(p, "EPOCH_UPDATES", 2)
    monkeypatch.setattr(c, "EPOCH_UPDATES", 2)
    torch.manual_seed(91)
    np.random.seed(91)
    random.seed(91)
    identities12 = {"sampler_sha256": "a" * 64, "keep_sha256": "b" * 64, "prefix_sha256": "c" * 64}
    identities13 = {"sampler_sha256": "d" * 64, "keep_sha256": "e" * 64, "prefix_sha256": "f" * 64}
    parent_bindings, continuation_bindings = {"fixture": "parent"}, {"fixture": "continuation"}
    shared, arm = "1" * 64, "flat"

    # An authentic parent state: AdamW moments and DecoderEMA history are made
    # by updates, never fabricated in the checkpoint payload.
    source_model, source_opt, source_ema = _new_toy()
    _run(source_model, source_opt, source_ema, 12 * p.EPOCH_UPDATES)
    parent_payload = p.checkpoint_payload(model=source_model, optimizer=source_opt, ema=source_ema, epoch=12, identities=identities12, arm=arm, shared_init_sha256=shared, bindings=parent_bindings)
    parent_sha = c.tree_digest(parent_payload)
    assert parent_payload["optimizer"]["state"] and source_ema.n_updates == 24

    # Both paths execute parent strict restore.  Only the split path performs
    # the epoch-13 continuation checkpoint/resume, then takes three real steps.
    whole_model, whole_opt, whole_ema = _new_toy()
    c.strict_parent_restore(payload=copy.deepcopy(parent_payload), model=whole_model, optimizer=whole_opt, ema=whole_ema, identities=identities12, arm=arm, shared_init_sha256=shared, bindings=parent_bindings)
    _run(whole_model, whole_opt, whole_ema, c.EPOCH_UPDATES + 3)
    whole_rng = copy.deepcopy(c.rng_state())

    split_model, split_opt, split_ema = _new_toy()
    c.strict_parent_restore(payload=copy.deepcopy(parent_payload), model=split_model, optimizer=split_opt, ema=split_ema, identities=identities12, arm=arm, shared_init_sha256=shared, bindings=parent_bindings)
    _run(split_model, split_opt, split_ema, c.EPOCH_UPDATES)
    continuation_payload = c.checkpoint_payload(model=split_model, optimizer=split_opt, ema=split_ema, epoch=13, identities=identities13, arm=arm, shared_init_sha256=shared, bindings=continuation_bindings, parent_checkpoint_sha256=parent_sha)
    assert continuation_payload["optimizer"]["state"] and continuation_payload["ema"]["n_updates"] == 26

    resumed_model, resumed_opt, resumed_ema = _new_toy()
    c.strict_restore(payload=copy.deepcopy(continuation_payload), model=resumed_model, optimizer=resumed_opt, ema=resumed_ema, epoch=13, identities=identities13, arm=arm, shared_init_sha256=shared, bindings=continuation_bindings, parent_epoch12_sha256=parent_sha)
    _run(resumed_model, resumed_opt, resumed_ema, 3)

    assert c.recursive_same(whole_model.state_dict(), resumed_model.state_dict())
    assert c.recursive_same(whole_opt.state_dict(), resumed_opt.state_dict())
    assert c.recursive_same(whole_ema.checkpoint_state(), resumed_ema.checkpoint_state())
    assert c.recursive_same(whole_rng, c.rng_state())

    wrong_arm = copy.deepcopy(continuation_payload)
    wrong_arm["arm"] = "route"
    _assert_restore_rejected(wrong_arm, identities=identities13, shared=shared, bindings=continuation_bindings, parent_sha=parent_sha, match="identity")
    _assert_restore_rejected(continuation_payload, identities=identities13, shared=shared, bindings=continuation_bindings, parent_sha="0" * 64, match="identity")
    _assert_restore_rejected(continuation_payload, identities={**identities13, "keep_sha256": "0" * 64}, shared=shared, bindings=continuation_bindings, parent_sha=parent_sha, match="identity")
    raw_tamper = copy.deepcopy(continuation_payload)
    raw_tamper["models"][next(iter(raw_tamper["models"]))].add_(1)
    _assert_restore_rejected(raw_tamper, identities=identities13, shared=shared, bindings=continuation_bindings, parent_sha=parent_sha, match="state")
    rng_tamper = copy.deepcopy(continuation_payload)
    rng_tamper["rng"]["torch"] = torch.empty(0, dtype=torch.uint8)
    _assert_restore_rejected(rng_tamper, identities=identities13, shared=shared, bindings=continuation_bindings, parent_sha=parent_sha)
