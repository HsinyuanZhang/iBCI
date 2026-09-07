"""Test suite for the AM / IM activity-vs-identity mask cells (pre-launch gate).

Covers HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md §3 (cells AM and IM) and
§5 (receipt requirements):

- initial-state / graph / head-count / parameter-count parity vs canonical
  Arm A, strict load with state-SHA equality;
- the exact site (the two components are handed to the hook still separate;
  the mask multiplies exactly ONE of them before the sum, before `fc_in`);
- the mask law: one shared `p ~ U(0,1)` per training forward from the shared
  (never duplicated) PStream, AM/IM `p_sequence_sha256` equality, and the
  full 1,628,400-draw stream SHA equal to the R / S2 / T / G value
  `e62fc92f...` (verified in full, not as a prefix);
- the mask itself: the EXACT D call form (`torch.nn.functional.dropout` on a
  2-D all-ones [B, N] tensor with `training=True`), drawn from the GLOBAL
  torch RNG exactly as D's own code path draws it, [B, N] Bernoulli whole-unit
  law with survivor fraction ~ 1-p, survivor values bitwise equal to
  1/(1-p) (the gain INHERITED from the kernel, never reimplemented), p == 1
  returning exact zeros with no NaN, and no min_keep / clamp / floor;
- evaluation is bitwise equal to the unsparsified parent path for both cells
  (no mask object, no dropout call at all at the site), and enabling then
  disabling the perturbation never leaks;
- the launch proofs `prove_AM_site` / `prove_IM_site`: with a mask dropping
  unit u, changing the MASKED component at u is bitwise invisible while
  changing the UNMASKED component at u is visible, and a surviving unit's
  masked component still matters (the inherited gain is really applied);
- the recorder's realized per-epoch statistics and its bitwise mask-value-law
  violation check;
- scoring does no gradient/update; a training-mode step stays finite;
- the 33,925 x 48 budget guard and the runner CLI / receipt conventions
  (unknown cell, non-fresh output directory, overwrite refusal, failure
  receipts).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import activity_identity_cells as ai
from src.tfpd_lane import sparsification as sp

CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"
SEALED_R_TERMINAL = ROOT / "results/sparsification_v1/cellR_elementwise/terminal_receipt.json"
CELLS = ("AM", "IM")


def _canonical_state():
    payload = torch.load(CANONICAL, map_location="cpu", weights_only=False)
    return payload["state_dict"], payload["state_sha256"]


def _tiny_inputs(n_units=10, batch=3):
    torch.manual_seed(0)
    return (
        torch.rand(batch, 50, n_units),
        torch.rand(batch, 30, 100, n_units),
        torch.randn(batch, n_units, 4),
    )


def _built(cell):
    canonical, _ = _canonical_state()
    model = ai.build_activity_identity_model(seed=42, cell=cell)
    model.load_state_dict(canonical, strict=True)
    model.eval()
    return model


def _components(model, neural, calib, side):
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
    return neural.permute(0, 2, 1), identity


def _masked_forward(model, neural, identity, p, seed=1234):
    """A perturbation-enabled forward with a replayable (seeded) mask draw.

    Eval mode keeps every other dropout inert, so the site's single
    `F.dropout(training=True)` call is the only RNG consumer and can be
    reproduced bit-for-bit by re-seeding and calling the same D form.
    """
    torch.manual_seed(seed)
    model.current_p = float(p)
    model.perturbation_enabled = True
    try:
        with torch.no_grad():
            out, _ = model(neural, identity=identity)
    finally:
        model.perturbation_enabled = False
        model.current_p = None
    torch.manual_seed(seed)
    ref = neural.permute(0, 2, 1) if model.cell == "AM" else identity
    mask = ai.draw_site_mask(p, neural.shape[0], neural.shape[2], ref)
    return out, mask


def _spy_transformer(model):
    """Capture the kwargs reaching MultiLayerCrossAttention.forward."""
    captured = {}
    transformer = model.decoder.transformer
    original = transformer.forward

    def spy(query, key_value, attn_mask=None, key_padding_mask=None):
        captured["attn_mask"] = attn_mask
        captured["key_padding_mask"] = key_padding_mask
        return original(query, key_value, attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask)

    transformer.forward = spy
    return captured, lambda: transformer.__dict__.pop("forward", None)


def _count_site_dropout_calls(fn):
    """Run `fn` while counting F.dropout calls that look like D's unit mask."""
    calls = []
    original = torch.nn.functional.dropout

    def recording(input, p=0.5, training=True, inplace=False):
        calls.append({
            "dim": int(input.dim()), "shape": list(input.shape),
            "training": bool(training), "p": float(p),
            "all_ones": bool(input.numel() and (input == 1).all().item()),
        })
        return original(input, p=p, training=training, inplace=inplace)

    torch.nn.functional.dropout = recording
    try:
        fn()
    finally:
        torch.nn.functional.dropout = original
    return calls


# ---- graph / initial-state parity ------------------------------------------
def test_initial_state_graph_head_and_param_parity():
    canonical, canonical_sha = _canonical_state()
    from src.tfpd_lane import arm_common
    from torch.nn.parameter import UninitializedParameter

    params = set()
    for cell in CELLS:
        model = ai.build_activity_identity_model(seed=42, cell=cell)
        model.load_state_dict(canonical, strict=True)
        assert arm_common.state_sha256(model) == canonical_sha  # bitwise parity
        assert model.decoder.transformer.layers[0].cross_attn.num_heads == 2
        shapes = {
            k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter)
                else tuple(v.shape))
            for k, v in model.state_dict().items()
        }
        canonical_shapes = {
            k: (("uninitialized-lazy",) if isinstance(v, UninitializedParameter)
                else tuple(v.shape))
            for k, v in canonical.items()
        }
        assert sorted(shapes) == sorted(canonical_shapes) and shapes == canonical_shapes
        params.add(sum(
            p.numel() for p in model.parameters()
            if p.requires_grad and not isinstance(p, UninitializedParameter)
        ))
        assert model.decoder.dynamic_dropout is False  # route perturbation only
        assert model.decoder.tf_drop_rate == 0.1  # parent dropout structure
        assert model.cell == cell and ai.CELLS[cell]["num_heads"] == 2
    assert len(params) == 1  # parameter-count parity across AM/IM/ArmA graph


def test_exact_site_component_split_and_D_call_form():
    source = (ROOT / "src/tfpd_lane/activity_identity_cells.py").read_text()
    decode_body = source[source.index("    def decode_with_identity"):]
    site_pos = decode_body.index("self.apply_perturbation(activity, identity)")
    fc_in_pos = decode_body.index("self.decoder.fc_in(src)")
    assert 0 < site_pos < fc_in_pos
    perturb_body = source[
        source.index("    def apply_perturbation"):source.index("    def decode_with_identity")
    ]
    # the evaluation path is the literal parent sum; the training path masks
    # exactly one component through the shared site helper
    assert "return activity + identity" in perturb_body
    assert "apply_mask_at_site(self.cell, activity, identity, mask)" in perturb_body
    assert perturb_body.index("draw_site_mask(") < perturb_body.index("apply_mask_at_site")
    site_body = source[source.index("def apply_mask_at_site"):source.index("class AIMaskRecorder")]
    assert "activity * mask.unsqueeze(-1) + identity" in site_body  # AM
    assert "activity + identity * mask.unsqueeze(-1)" in site_body  # IM
    # the mask is drawn by the exact D call form on the global torch RNG
    mask_body = source[source.index("def draw_site_mask"):source.index("def apply_mask_at_site")]
    assert "torch.nn.functional.dropout(ones, p=p, training=True)" in mask_body
    assert "torch.ones(int(batch_size), int(n_units)).to(ref)" in mask_body
    assert "generator" not in mask_body  # no route-owned generator namespace
    # and D's own file still carries that call form
    d_source = (ROOT.parent / ai.D_SITE_RELATIVE).read_text()
    assert ai.D_MASK_CALL_MARKER in d_source
    # no min_keep / clamp / floor anywhere in the module
    for token in ("min_keep=", "clamp(", "torch.clamp", "floor"):
        assert token not in site_body, token


# ---- the shared p stream ----------------------------------------------------
def test_pstream_imported_never_duplicated_and_am_im_sha_equality():
    assert ai.P_STREAM_SEED == sp.P_STREAM_SEED
    stream_am, stream_im = ai.pstream(), ai.pstream()
    assert isinstance(stream_am, sp.PStream)  # the very same class object
    n = 500
    draws_am = [stream_am.next() for _ in range(n)]
    draws_im = [stream_im.next() for _ in range(n)]
    assert draws_am == draws_im
    assert stream_am.sha256() == stream_im.sha256()  # AM and IM share the stream
    sibling = sp.PStream()
    [sibling.next() for _ in range(n)]
    assert stream_am.sha256() == sibling.sha256()  # and identical to R / S2 / T / G
    assert all(0.0 <= v < 1.0 for v in draws_am)


def test_full_budget_p_sequence_sha256_equals_the_shared_stream_value():
    """The full 48 x 33,925-draw sequence, verified in full (0.4 s)."""
    stream = ai.pstream()
    for _ in range(ai.FULL_BUDGET_P_DRAWS):
        stream.next()
    assert ai.FULL_BUDGET_P_DRAWS == 1_628_400 == 48 * 33_925
    assert stream.stats()["n"] == ai.FULL_BUDGET_P_DRAWS
    assert stream.sha256() == ai.EXPECTED_P_SEQUENCE_SHA256
    assert stream.sha256().startswith("e62fc92f")
    # bound to the sealed R / S2 receipts when the artifact is present
    if SEALED_R_TERMINAL.is_file():
        payload = json.loads(SEALED_R_TERMINAL.read_text())
        assert payload["p_sequence_sha256"] == ai.EXPECTED_P_SEQUENCE_SHA256


def test_one_shared_p_per_forward():
    for cell in CELLS:
        model = _built(cell)
        neural, calib, side = _tiny_inputs()
        stream = ai.pstream()
        with torch.no_grad():
            for _ in range(5):
                p = stream.next()
                model.current_p = p
                model.perturbation_enabled = True
                model(neural, calib_trials=calib, side_features=side)
                model.perturbation_enabled = False
        assert stream.stats()["n"] == 5  # exactly one p draw per forward
        assert model.recorder.forwards == 5
        assert len(set(model.recorder.p_values)) == 5  # no per-row p
        assert stream.stats()["n"] == 5  # the mask draw never touches the p stream


# ---- the mask law -----------------------------------------------------------
def test_mask_law_bernoulli_survivor_fraction_and_whole_unit_shape():
    torch.manual_seed(3)
    ref = torch.zeros(64, 100, 50)
    p = 0.3
    mask = ai.draw_site_mask(p, 64, 100, ref)
    assert mask.shape == (64, 100) and mask.dim() == 2  # no window axis
    assert mask.dtype == ref.dtype and mask.device == ref.device
    assert float((mask != 0).float().mean().item()) == pytest.approx(1 - p, abs=0.02)
    survivor_counts = (mask != 0).sum(dim=1)
    assert len(set(survivor_counts.tolist())) > 1  # per-row variation
    # independent rows: a second draw disagrees (not one shared row pattern)
    torch.manual_seed(4)
    other = ai.draw_site_mask(p, 64, 100, ref)
    assert not torch.equal(mask, other)


def test_mask_law_gain_inherited_bitwise_and_edge_cases():
    ref = torch.zeros(64, 4000, 50)
    for p in (0.05, 0.25, 0.5, 0.75, 0.95, 0.999, 0.9999):
        torch.manual_seed(11)
        mask = ai.draw_site_mask(p, 64, 4000, ref)
        zeros = mask == 0
        assert bool((mask[zeros] == 0).all())  # dropped units are exact zeros
        survivors = mask[~zeros]
        assert survivors.numel() > 0
        # the 1/(1-p) gain is INHERITED from the kernel: every survivor is
        # bitwise the value F.dropout itself applies for this p, and that value
        # is the correctly-rounded 1/(1-p) to within one ULP
        expected = ai.inherited_gain_value(p, mask.dtype)
        assert torch.equal(survivors, torch.full_like(survivors, expected))
        assert float(survivors.min().item()) == float(survivors.max().item()) == expected
        rounded = torch.full_like(survivors, 1.0 / (1.0 - p))
        ulp = float(np.spacing(np.float32(1.0 / (1.0 - p))))
        assert torch.equal(survivors, rounded) or bool(
            ((survivors - rounded).abs() <= ulp).all()
        )
        # the characterized kernel rule really is a float32 division
        np_rule = float(np.float32(np.float32(1.0) / np.float32(1.0 - p)))
        assert np_rule == expected
    # p == 1: exact zeros, finite, no NaN, no clamp, no floor
    torch.manual_seed(11)
    all_masked = ai.draw_site_mask(1.0, 64, 4000, ref)
    assert bool((all_masked == 0).all()) and bool(torch.isfinite(all_masked).all())
    # p == 0: nothing is dropped (F.dropout's own identity branch)
    torch.manual_seed(11)
    none_masked = ai.draw_site_mask(0.0, 64, 4000, ref)
    assert bool((none_masked == 1).all())
    with pytest.raises(ValueError):
        ai.draw_site_mask(1.5, 4, 8, ref)


def test_mask_law_uses_the_global_torch_rng_like_D():
    ref = torch.zeros(8, 60, 50)
    torch.manual_seed(77)
    a = ai.draw_site_mask(0.4, 8, 60, ref)
    torch.manual_seed(77)
    b = ai.draw_site_mask(0.4, 8, 60, ref)
    assert torch.equal(a, b)  # seeded determinism through the global RNG
    torch.manual_seed(78)
    c = ai.draw_site_mask(0.4, 8, 60, ref)
    assert not torch.equal(a, c)  # the seed really drives the draw
    torch.manual_seed(79)
    state = torch.get_rng_state()
    ai.draw_site_mask(0.4, 8, 60, ref)
    assert not torch.equal(state, torch.get_rng_state())  # D's own RNG channel
    # and it is bit-for-bit the D call form, replayed independently
    torch.manual_seed(101)
    mine = ai.draw_site_mask(0.4, 8, 60, ref)
    torch.manual_seed(101)
    d_form = torch.nn.functional.dropout(torch.ones(8, 60).to(ref), p=0.4, training=True)
    assert torch.equal(mine, d_form)


def test_probe_mask_code_path_matches_the_D_call_form():
    for cell in CELLS:
        model = _built(cell)
        neural, calib, side = _tiny_inputs(n_units=12, batch=4)
        with torch.no_grad():
            identity = model.compute_identity(calib, side_features=side)
        proof = ai.probe_mask_code_path(model, neural, identity, p=0.5)
        call = proof["site_call"]
        assert proof["exactly_one_site_call"]
        assert proof["n_dropout_calls_total"] > 1  # the transformer's own
        # (inert, training=False) dropout calls are observed and filtered out
        assert call["input_dim"] == 2 and call["input_shape"] == [4, 12]
        assert call["input_all_ones"] and call["training"] and call["p"] == 0.5
        assert proof["input_is_2d_all_ones"] and proof["input_shape_is_BxN"]
        assert proof["training_flag_true"] and proof["p_equals_shared_current_p"]
        assert proof["matches_D_call_form"]
        assert model.perturbation_enabled is False
        # the probe records exactly one realized forward
        assert model.recorder.forwards == 1
        model.recorder.reset_epoch()
        assert model.recorder.summary()["n_forwards"] == 0


# ---- the AM / IM math on a fixed mask ---------------------------------------
@pytest.mark.parametrize("cell", CELLS)
def test_cell_math_on_a_fixed_mask_is_the_parent_decode_of_the_masked_component(cell):
    model = _built(cell)
    neural, calib, side = _tiny_inputs(n_units=12, batch=4)
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
    activity = neural.permute(0, 2, 1)
    out, mask = _masked_forward(model, neural, identity, p=0.4)
    # the site is exactly the one-line variant, applied pre-sum
    expected = ai.parent_decode_from_src(
        model, ai.apply_mask_at_site(cell, activity, identity, mask)
    )
    assert torch.equal(out, expected)
    # whole-unit structure: the [B, N] mask is broadcast over W, so each unit's
    # whole window is scaled by the same value (0 or 1/(1-p))
    scaled = activity if cell == "AM" else identity
    unmasked = identity if cell == "AM" else activity
    assembled = scaled * mask.unsqueeze(-1) + unmasked
    assert torch.equal(
        ai.apply_mask_at_site(cell, activity, identity, mask), assembled
    )
    # whole-unit structure: the mask is rank-2 [B, N] with no window axis
    assert mask.dim() == 2 and tuple(mask.shape) == (4, 12)
    # at a dropped (row, unit) the token is exactly the unmasked component
    dropped = mask == 0
    assert bool(dropped.any())
    assert torch.equal(assembled[dropped], unmasked[dropped])
    # at a surviving (row, unit) the masked component carries the inherited
    # 1/(1-p) gain (the residual-sum identity is checked to float32 rounding:
    # (a*g + u) - u is not bitwise a*g in IEEE arithmetic)
    kept = ~dropped
    gain = ai.inherited_gain_value(0.4, activity.dtype)
    assert torch.allclose(
        assembled[kept], unmasked[kept] + scaled[kept] * gain, rtol=1e-6, atol=1e-6
    )
    # a different p (hence a different mask) really changes the output
    other, _ = _masked_forward(model, neural, identity, p=0.8, seed=1234)
    assert not torch.equal(out, other)


@pytest.mark.parametrize("cell", CELLS)
def test_prove_site_bitwise_component_isolation(cell):
    model = _built(cell)
    neural, calib, side = _tiny_inputs(n_units=10, batch=2)
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
    activity = neural.permute(0, 2, 1)
    proof = (ai.prove_AM_site if cell == "AM" else ai.prove_IM_site)(
        model, activity, identity, dropped_unit=0
    )
    assert proof["cell"] == cell
    assert proof["masked_component"] == ai.CELLS[cell]["masked_component"]
    assert proof["unmasked_component"] == ai.CELLS[cell]["unmasked_component"]
    # the mask hit ONLY the intended component, bitwise
    assert proof["masked_component_change_bitwise_invisible"]
    assert proof["unmasked_component_change_visible"]
    assert proof["unmasked_component_change_max_abs"] > 0.0
    # a survivor's masked component still matters: the inherited 1/(1-p) gain
    assert proof["survivor_masked_component_change_visible"]
    assert proof["survivor_masked_component_change_max_abs"] > 0.0
    assert proof["output_finite"]
    assert proof["probe_survivor_gain"] == pytest.approx(2.0)
    assert model.perturbation_enabled is False  # the proof never leaks


@pytest.mark.parametrize("cell", CELLS)
def test_p_equals_one_edge_all_silent_or_all_anonymous_no_nan(cell):
    model = _built(cell)
    neural, calib, side = _tiny_inputs(n_units=10, batch=4)
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        model.current_p = 1.0
        model.perturbation_enabled = True
        out, _ = model(neural, identity=identity)
        model.perturbation_enabled = False
    assert bool(torch.isfinite(out).all())  # no NaN trap, no floor needed
    summary = model.recorder.summary()
    assert summary["all_zero_mask_rows"] == 4
    assert summary["all_zero_mask_forwards"] == 1
    assert summary["mask_value_law_violations"] == 0
    assert summary["surviving_units_min"] == 0 and summary["surviving_units_max"] == 0


# ---- evaluation parity and no-leak -----------------------------------------
@pytest.mark.parametrize("cell", CELLS)
def test_eval_bitwise_parent_parity_and_no_leak(cell):
    model = _built(cell)
    neural, calib, side = _tiny_inputs(n_units=12, batch=4)
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        # a perturbed (training-law) forward first, then the disabled eval pass
        model.current_p = 0.7
        model.perturbation_enabled = True
        model(neural, identity=identity)
        model.perturbation_enabled = False
        assert model.perturbation_enabled is False
        mine, _ = model(neural, calib_trials=calib, side_features=side)
        # the unsparsified parent path through the same modules, same order
        src = neural.permute(0, 2, 1)
        src = src + identity
        src = model.decoder.fc_in(src)
        rep = model.decoder.fc_in(model.decoder.rep).to(src)
        out, _ = model.decoder.transformer(rep.repeat(src.size(0), 1, 1), src)
        parent = model.decoder.fc_out(out).permute(0, 2, 1)
    assert torch.equal(mine, parent)
    # the disabled eval pass makes NO dropout call at the site at all
    def _eval_pass():
        with torch.no_grad():
            model(neural, calib_trials=calib, side_features=side)

    calls = _count_site_dropout_calls(_eval_pass)
    site_calls = [c for c in calls if c["training"] and c["dim"] == 2]
    assert site_calls == []
    captured, restore = _spy_transformer(model)
    try:
        with torch.no_grad():
            model(neural, calib_trials=calib, side_features=side)
    finally:
        restore()
    assert captured["key_padding_mask"] is None and captured["attn_mask"] is None
    assert model.perturbation_enabled is False
    # enabling and disabling again keeps reproducing the parent bitwise
    with torch.no_grad():
        model.current_p = 0.99
        model.perturbation_enabled = True
        model(neural, identity=identity)
        model.perturbation_enabled = False
        again, _ = model(neural, identity=identity)
    assert torch.equal(again, parent)
    # only the two perturbed passes above reached the recorder; the disabled
    # eval passes recorded nothing
    assert model.recorder.forwards == 2


@pytest.mark.parametrize("cell", CELLS)
def test_scoring_no_gradient_no_update(cell):
    model = _built(cell)
    neural, calib, side = _tiny_inputs()
    with torch.no_grad():
        model(neural, calib_trials=calib, side_features=side)
    assert all(
        p.grad is None
        for p in model.parameters()
        if hasattr(p, "grad")
    )
    assert model.perturbation_enabled is False


@pytest.mark.parametrize("cell", CELLS)
def test_train_mode_step_finite_with_mask_and_parent_dropout(cell):
    model = _built(cell)
    neural, calib, side = _tiny_inputs(n_units=10, batch=4)
    target = torch.rand(4, 50, 2)
    model.train()
    stream = ai.pstream()
    p = stream.next()
    model.current_p = p
    model.perturbation_enabled = True
    prediction, _identity = model(neural, calib_trials=calib, side_features=side)
    model.perturbation_enabled = False
    loss = ((prediction - target) ** 2).mean()
    assert bool(torch.isfinite(loss).item())
    loss.backward()
    grads = [q.grad for q in model.parameters() if q.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all().item()) for g in grads)
    model.eval()
    assert model.recorder.summary()["n_forwards"] == 1


# ---- realized per-epoch statistics -----------------------------------------
def test_recorder_realized_stats_and_violation_detection():
    recorder = ai.AIMaskRecorder()
    ref = torch.zeros(4, 20, 50)
    for p in (0.2, 0.5, 0.9):
        torch.manual_seed(5)
        recorder.record(p, ai.draw_site_mask(p, 4, 20, ref))
    summary = recorder.summary()
    assert summary["n_forwards"] == 3 and summary["rows_total"] == 12
    assert summary["mask_value_law_violations"] == 0
    assert summary["p_min"] == pytest.approx(0.2)
    assert summary["p_max"] == pytest.approx(0.9)
    assert summary["p_median"] == pytest.approx(0.5)
    # the cell-G-compatible aliases carry the same realized values
    assert summary["p_raw_min"] == summary["p_min"]
    assert summary["p_raw_median"] == summary["p_median"]
    assert summary["p_raw_max"] == summary["p_max"]
    assert summary["surviving_units_min"] <= summary["surviving_units_median"]
    assert summary["surviving_units_median"] <= summary["surviving_units_max"]
    assert sum(int(v) for v in summary["surviving_units_histogram"].values()) == 12
    assert 0.0 < summary["kept_fraction_mean"] < 1.0
    assert summary["min_keep"] == "none"
    assert "inherited" in summary["gain_rule"]
    # a corrupted mask (a reimplemented gain) is caught bitwise
    torch.manual_seed(5)
    good = ai.draw_site_mask(0.5, 4, 20, ref)
    bad = good.clone()
    bad[0, 0] = 1.9999999  # not exactly 1/(1-0.5) = 2
    recorder.record(0.5, bad)
    assert recorder.summary()["mask_value_law_violations"] == 1
    # the all-zero edge is legal and counted
    recorder.reset_epoch()
    recorder.record(1.0, torch.zeros(4, 20))
    summary = recorder.summary()
    assert summary["all_zero_mask_rows"] == 4
    assert summary["all_zero_mask_forwards"] == 1
    assert summary["mask_value_law_violations"] == 0
    assert summary["surviving_units_histogram"] == {"0": 4}
    recorder.reset_epoch()
    empty = recorder.summary()
    assert empty["n_forwards"] == 0 and empty["surviving_units_histogram"] == {}


def test_summarize_perturbation_binds_the_cell_and_code_path():
    for cell in CELLS:
        model = _built(cell)
        neural, calib, side = _tiny_inputs(n_units=8, batch=2)
        with torch.no_grad():
            model.current_p = 0.6
            model.perturbation_enabled = True
            model(neural, calib_trials=calib, side_features=side)
            model.perturbation_enabled = False
        summary = ai.summarize_perturbation(model)
        assert summary["cell"] == cell
        assert summary["masked_component"] == ai.CELLS[cell]["masked_component"]
        assert summary["n_forwards"] == 1
        assert summary["mask_code_path_matches_D"] is True
        assert summary["mask_value_law_violations"] == 0


# ---- budget guard and runner surface ----------------------------------------
def test_budget_guard():
    ai.assert_full_budget(33_925, 48, smoke=False)  # the frozen budget
    for steps, epochs in ((33_924, 48), (33_925, 47), (33_926, 48), (0, 48)):
        with pytest.raises(SystemExit):
            ai.assert_full_budget(steps, epochs, smoke=False)
    ai.assert_full_budget(30, 5, smoke=True)  # smoke budgets are exempt
    source = (ROOT / "scripts/run_activity_identity_cell.py").read_text()
    assert "aimask.assert_full_budget(steps_per_epoch, epochs, smoke=args.smoke)" in source
    assert "33,925" in source and "48 epochs" in source


def test_d_reference_site_fingerprint():
    site = ai.d_reference_site()
    assert site["file"] == ai.D_SITE_RELATIVE and site["marker_present"]
    assert len(site["source_sha256"]) == 64
    assert "torch.nn.functional.dropout" in site["call"]


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_activity_identity_cell_under_test",
        ROOT / "scripts/run_activity_identity_cell.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_cli_rejects_unknown_cell_and_existing_root(tmp_path):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    unknown = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_activity_identity_cell.py"),
         "--cell", "X"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert unknown.returncode == 2
    (tmp_path / "cellAM_activity_mask").mkdir()
    stale = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_activity_identity_cell.py"),
         "--cell", "AM", "--device", "cpu", "--output-root", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert stale.returncode == 2 and "fresh cell output directory" in stale.stderr
    source = (ROOT / "scripts/run_activity_identity_cell.py").read_text()
    for token in ("p_sequence_sha256", "mask_code_path_matches_D", "33,925",
                  "EXPECTED_P_SEQUENCE_SHA256", "all_zero_mask_forwards"):
        assert token in source, token
    assert "results/aimask_v1" in source
    assert "cellAM_activity_mask" in source and "cellIM_identity_mask" in source
    assert "src/tfpd_lane/activity_identity_cells.py" in source
    assert "src/tfpd_lane/subpop_cells.py" not in source  # sibling, never loaded
    integrity_fields = ("num_heads", "mask_structure", "p_law", "gain_rule",
                        "min_keep", "eval_mask_policy", "application_site",
                        "supersedes", "behavior_scaling_convention",
                        "generator_namespaces", "d_reference_site")
    for field in integrity_fields:
        assert field in source, field


def test_receipt_overwrite_refusal_and_failure_receipt(tmp_path, monkeypatch):
    pytest.importorskip("src.tfpd_lane.receipt")
    from src.tfpd_lane import receipt

    target = tmp_path / "probe.json"
    receipt.write_receipt_transactionally(target, {"probe": True})
    assert target.is_file()
    with pytest.raises(SystemExit) as exc:
        receipt.write_receipt_transactionally(target, {"probe": False})
    assert exc.value.code == 2  # immutable roots: no overwrite path

    if not sys.flags.no_user_site:
        pytest.skip("pytest must run with PYTHONNOUSERSITE=1 for the runner gate")
    runner = _runner_module()
    out_root = tmp_path / "aimask_fail"
    argv = [ "run_activity_identity_cell.py", "--cell", "AM", "--device", "cpu",
             "--output-root", str(out_root) ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(runner, "_run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    code = runner.main()
    assert code == 1
    receipt_path = out_root / "cellAM_activity_mask" / "terminal_receipt.json"
    payload = json.loads(receipt_path.read_text())
    assert payload["schema"] == "tfpd_aimask_cell_v1"
    assert payload["status"] == "CELL_FAILED"
    assert payload["cell"] == "AM"
    assert payload["failure"]["kind"] == "RuntimeError"
    assert "boom" in payload["failure"]["detail"]
    assert "Traceback" in payload["failure"]["traceback"]
    assert payload["smoke"] is False
    sidecar = Path(str(receipt_path) + ".sha256")
    assert sidecar.is_file()
    assert sidecar.read_text().split()[0] == receipt.sha256_file(receipt_path)
    # the failure receipt is itself immutable
    with pytest.raises(SystemExit):
        receipt.write_receipt_transactionally(receipt_path, {"again": True})


def test_cell_contract_tables_match_frozen_constants():
    assert ai.CELLS["AM"]["mask_structure"] == "whole_unit_F_dropout_on_activity"
    assert ai.CELLS["IM"]["mask_structure"] == "whole_unit_F_dropout_on_identity"
    assert ai.CELLS["AM"]["masked_component"] == "activity"
    assert ai.CELLS["IM"]["masked_component"] == "identity"
    for cell in CELLS:
        assert ai.CELLS[cell]["num_heads"] == 2
        assert ai.CELLS[cell]["p_clamp"] is None
        assert "none" in ai.CELLS[cell]["min_keep"].lower()
        assert "inherited" in ai.CELLS[cell]["gain_rule"]
    assert ai.CELLS["AM"]["application_site"] == "pre-sum: activity * mask + identity"
    assert ai.CELLS["IM"]["application_site"] == "pre-sum: activity + identity * mask"
    assert "present and identifiable, but silent" == ai.CELLS["AM"]["dropped_unit_becomes"]
    assert "audible but anonymous" == ai.CELLS["IM"]["dropped_unit_becomes"]
    assert ai.P_STREAM_SEED == 42
    for value in (ai.CELLS, ai.CELLS["AM"], ai.EVAL_POLICY, ai.MIN_KEEP_POLICY):
        json.dumps(value)  # receipt-serializable
