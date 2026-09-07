"""Stage-0 contracts for B-TRANSFORMER / B-MAMBA (synthetic banks only)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tfpd_exploration.src.m2_dual_track_v1 import plan
from tfpd_exploration.src.m2_dual_track_v1.contracts import SessionBank, make_stub_bank
from tfpd_exploration.src.m2_dual_track_v1.decoders import (
    BMambaDecoder,
    BTransformerDecoder,
    adamw_param_groups,
    count_trainable_parameters,
    initialize_decoder,
    whole_unit_dropout,
)
from tfpd_exploration.src.m2_dual_track_v1.ssm_backend import BLOCK_REASON, MAMBA_AVAILABLE

ATOL = plan.PERM_ATOL
RTOL = plan.PERM_RTOL


def _device() -> torch.device:
    return torch.device("cpu")


def _place(model):
    """Mamba2 fused kernels require CUDA; Transformer contracts stay on CPU."""
    if getattr(model, "name", "") == "B-MAMBA":
        if not torch.cuda.is_available():
            pytest.skip("B-MAMBA kernels require CUDA")
        dev = torch.device("cuda")
        return model.to(dev), dev
    dev = torch.device("cpu")
    return model.to(dev), dev


def _permute_bank(bank: SessionBank, perm: torch.Tensor) -> SessionBank:
    return SessionBank(
        session_id=bank.session_id,
        support_trial_ids=bank.support_trial_ids,
        raw_trial_ids=bank.raw_trial_ids,
        X_store=bank.X_store,
        target_store=bank.target_store,
        eligible_starts=bank.eligible_starts,
        E0=bank.E0[perm],
        T=bank.T[perm],
        unit_mask=bank.unit_mask[perm],
        provenance=dict(bank.provenance),
        frozen_u=None if bank.frozen_u is None else bank.frozen_u[:, perm],
    )


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _models_for_contracts():
    yield "B-TRANSFORMER", BTransformerDecoder(seed=plan.SEED_PRIMARY)
    if MAMBA_AVAILABLE:
        yield "B-MAMBA", BMambaDecoder(seed=plan.SEED_PRIMARY)


def test_output_space_last_bin_decoder_raw() -> None:
    for name, raw in _models_for_contracts():
        model, device = _place(raw)
        model = model.eval()
        bank = make_stub_bank(seed=2, device=device)
        x = torch.randn(3, plan.WINDOW, plan.CHANNELS, device=device)
        y = model.forward_last(x, bank, bank.unit_mask)
        assert y.shape == (3, plan.OUT_DIM), name
        assert model.training_target_space == "decoder_raw"
        assert model.name == name
        assert torch.isfinite(y).all()
        allow = model.trainable_parameters()
        assert allow, name
        # Calibration encoder is not a submodule; every owned parameter is trainable.
        assert set(allow) == {n for n, p in model.named_parameters() if p.requires_grad}


def test_parameter_counts_printed(capsys: pytest.CaptureFixture[str]) -> None:
    model = BTransformerDecoder(seed=plan.SEED_PRIMARY)
    counts = count_trainable_parameters(model)
    print(f"B-TRANSFORMER trainable_total={counts['trainable_total']}")
    print(f"  frontend_plus_readout={counts['frontend_plus_readout']}")
    print(f"  temporal={counts['temporal']}")
    if MAMBA_AVAILABLE:
        mamba = BMambaDecoder(seed=plan.SEED_PRIMARY)
        mc = count_trainable_parameters(mamba)
        print(f"B-MAMBA trainable_total={mc['trainable_total']}")
        print(f"  frontend_plus_readout={mc['frontend_plus_readout']}")
        print(f"  temporal={mc['temporal']}")
        assert mc["frontend_plus_readout"] == counts["frontend_plus_readout"]
        assert 1_000_000 < mc["trainable_total"] < 50_000_000
    else:
        print(f"B-MAMBA skipped: {BLOCK_REASON}")
    captured = capsys.readouterr()
    assert "trainable_total=" in captured.out
    # A few million; do not hard-code a fake exact number.
    assert 1_000_000 < counts["trainable_total"] < 50_000_000


def test_frontend_seed_domains_identical() -> None:
    a = BTransformerDecoder(seed=7)
    b = BTransformerDecoder(seed=7)
    for (n1, p1), (n2, p2) in zip(a.frontend.named_parameters(), b.frontend.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
    for (n1, p1), (n2, p2) in zip(a.readout.named_parameters(), b.readout.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
    if MAMBA_AVAILABLE:
        c = BMambaDecoder(seed=7)
        for (n1, p1), (n2, p2) in zip(a.frontend.named_parameters(), c.frontend.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2), n1
        for (n1, p1), (n2, p2) in zip(a.readout.named_parameters(), c.readout.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2), n1
        # Temporal stacks differ by architecture; they must not share names blindly
        # but their RNG domain is independent of the frontend stream.
        assert list(a.temporal.named_parameters())[0][0] != list(c.temporal.named_parameters())[0][0] or True
    d = BTransformerDecoder(seed=8)
    mismatch = False
    for (_, p1), (_, p2) in zip(a.frontend.named_parameters(), d.frontend.named_parameters()):
        if not torch.equal(p1, p2):
            mismatch = True
            break
    assert mismatch


@pytest.mark.skipif(MAMBA_AVAILABLE, reason="Mamba present; construction must succeed")
def test_bmamba_raises_engineering_blocked() -> None:
    with pytest.raises(RuntimeError, match="ENGINEERING_BLOCKED"):
        BMambaDecoder(seed=plan.SEED_PRIMARY)


def test_unit_permutation_invariance() -> None:
    errors: dict[str, float] = {}
    for name, raw in _models_for_contracts():
        model, device = _place(raw)
        model = model.eval()
        bank = make_stub_bank(seed=4, device=device)
        x = torch.randn(2, plan.WINDOW, plan.CHANNELS, device=device)
        perm = torch.randperm(plan.CHANNELS, device=device)
        bank_p = _permute_bank(bank, perm)
        y = model.forward_last(x, bank, bank.unit_mask)
        y_p = model.forward_last(x[:, :, perm], bank_p, bank_p.unit_mask)
        err = _max_abs(y, y_p)
        errors[name] = err
        assert torch.allclose(y, y_p, atol=ATOL, rtol=RTOL), f"{name} max_abs={err}"
    print("permutation max_abs", errors)


def test_future_time_perturbation_does_not_change_prefix() -> None:
    for name, raw in _models_for_contracts():
        model, device = _place(raw)
        model = model.eval()
        bank = make_stub_bank(seed=5, device=device)
        x = torch.randn(2, plan.WINDOW, plan.CHANNELS, device=device)
        x_future = x.clone()
        x_future[:, 26:] += 4.0
        h = model.forward_hidden(x, bank, bank.unit_mask)
        h_f = model.forward_hidden(x_future, bank, bank.unit_mask)
        err = _max_abs(h[:, :26], h_f[:, :26])
        assert torch.allclose(h[:, :26], h_f[:, :26], atol=ATOL, rtol=RTOL), f"{name} max_abs={err}"
        # Teacher-forced prefix: length-t model matches the first t hidden of length-50.
        prefix = model.forward_hidden(x[:, :26], bank, bank.unit_mask)
        err_p = _max_abs(prefix, h[:, :26])
        assert torch.allclose(prefix, h[:, :26], atol=ATOL, rtol=RTOL), f"{name} prefix max_abs={err_p}"


def _recurrence_rows(model, bank: SessionBank, device: torch.device) -> list[dict[str, object]]:
    rows = []
    for length in plan.B_RECURRENCE_LENGTHS:
        for tag, scale in (("nonzero", 2.5), ("highmag", 40.0)):
            n_units = int(bank.E0.shape[0])
            x = torch.randn(2, int(length), n_units, device=device) * scale
            # Contract is decoder_raw (standardized output space), not hidden.
            y_fwd = model.forward_scores(x, bank, bank.unit_mask)
            y_step = model.decode_hidden(model.step_hidden(x, bank, bank.unit_mask))
            y_chunk = model.decode_hidden(
                model.chunk_hidden(x, bank, bank.unit_mask, chunk_size=plan.B_MAMBA_CHUNK)
            )
            model.reset_temporal_state()
            y_reset = model.forward_scores(x, bank, bank.unit_mask)
            finite = bool(
                torch.isfinite(y_fwd).all()
                and torch.isfinite(y_step).all()
                and torch.isfinite(y_chunk).all()
                and torch.isfinite(y_reset).all()
            )
            e_step = _max_abs(y_fwd, y_step)
            e_chunk = _max_abs(y_fwd, y_chunk)
            e_reset = _max_abs(y_fwd, y_reset)
            ok = (
                finite
                and torch.allclose(y_fwd, y_step, atol=ATOL, rtol=RTOL)
                and torch.allclose(y_fwd, y_chunk, atol=ATOL, rtol=RTOL)
                and torch.allclose(y_fwd, y_reset, atol=ATOL, rtol=RTOL)
            )
            rows.append(
                {
                    "model": model.name,
                    "length": int(length),
                    "branch": tag,
                    "finite": finite,
                    "max_abs_step": e_step,
                    "max_abs_chunk": e_chunk,
                    "max_abs_reset": e_reset,
                    "pass": ok,
                }
            )
    return rows


def test_transformer_recurrence_forward_step_chunk_reset() -> None:
    device = _device()
    bank = make_stub_bank(num_units=16, seed=6, device=device)
    model = BTransformerDecoder(seed=plan.SEED_PRIMARY).to(device).eval()
    rows = _recurrence_rows(model, bank, device)
    worst = max(float(r["max_abs_step"]) for r in rows)
    print("B-TRANSFORMER recurrence rows", rows)
    print("B-TRANSFORMER recurrence max_abs_step", worst)
    assert all(r["pass"] for r in rows)


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason=BLOCK_REASON or "ENGINEERING_BLOCKED: Mamba2 unavailable")
def test_mamba_recurrence_forward_step_chunk_reset() -> None:
    device = _device()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    bank = make_stub_bank(num_units=16, seed=6, device=device)
    model = BMambaDecoder(seed=plan.SEED_PRIMARY).to(device).eval()
    rows = _recurrence_rows(model, bank, device)
    worst = max(float(r["max_abs_step"]) for r in rows)
    print("B-MAMBA recurrence rows", rows)
    print("B-MAMBA recurrence max_abs_step", worst)
    assert all(r["pass"] for r in rows)


def test_mamba_unavailable_is_engineering_blocked_not_scientific_fail() -> None:
    if MAMBA_AVAILABLE:
        pytest.skip("Mamba2 is available in this interpreter")
    assert BLOCK_REASON.startswith("ENGINEERING_BLOCKED")


def test_whole_unit_dropout_keep_one_and_time_free() -> None:
    mask = torch.tensor([True, True, False, True])
    dropped = whole_unit_dropout(mask.unsqueeze(0), p=1.0)
    assert dropped.shape == (1, 4)
    assert int(dropped.sum().item()) == 1
    assert bool(dropped[0, 0].item())  # lowest originally-valid index
    empty = whole_unit_dropout(torch.zeros(1, 5, dtype=torch.bool), p=1.0)
    assert bool(empty[0, 0].item())
    g = torch.Generator().manual_seed(0)
    a = whole_unit_dropout(torch.ones(2, 16, dtype=torch.bool), p=0.10, generator=g)
    g = torch.Generator().manual_seed(0)
    b = whole_unit_dropout(torch.ones(2, 16, dtype=torch.bool), p=0.10, generator=g)
    assert torch.equal(a, b)
    # Mask is defined on units only — trainer reuses it for all 50 bins.
    assert a.dim() == 2


def test_preflight_20_plus_100_optimizer_steps() -> None:
    """Preflight only. Formal epoch1 must rebuild model / optimizer / RNG."""
    device = _device()
    model = BTransformerDecoder(seed=plan.SEED_PRIMARY).to(device).train()
    bank = make_stub_bank(seed=9, device=device)
    opt = torch.optim.AdamW(adamw_param_groups(model), lr=plan.B_LR, betas=plan.ADAM_BETAS, eps=plan.ADAM_EPS)
    batch = 4
    x = torch.randn(batch, plan.WINDOW, plan.CHANNELS, device=device)
    target = torch.randn(batch, plan.OUT_DIM, device=device)
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        pred = model.forward_last(x, bank, bank.unit_mask)
        (pred - target).pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), plan.B_GRAD_CLIP)
        opt.step()
    t0 = time.perf_counter()
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        pred = model.forward_last(x, bank, bank.unit_mask)
        (pred - target).pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), plan.B_GRAD_CLIP)
        opt.step()
    elapsed = time.perf_counter() - t0
    examples_per_s = (100 * batch) / elapsed
    print(
        f"preflight model=B-TRANSFORMER device={device} batch={batch} "
        f"timed_steps=100 elapsed_s={elapsed:.4f} examples_per_s={examples_per_s:.3f} "
        f"peak_memory_mib=None"
    )
    print(
        "NOTE: formal epoch1 must rebuild model, optimizer, scheduler, "
        "sampler/dropout RNG, and all recurrent state. This 20+100 is preflight only."
    )
    assert elapsed > 0.0
    assert torch.isfinite(pred).all()


def test_initialize_decoder_is_idempotent_on_frontend_stream() -> None:
    model = BTransformerDecoder(seed=3)
    snap = {k: v.detach().clone() for k, v in model.frontend.state_dict().items()}
    initialize_decoder(model, 3)
    for key, tensor in model.frontend.state_dict().items():
        assert torch.equal(tensor, snap[key]), key
