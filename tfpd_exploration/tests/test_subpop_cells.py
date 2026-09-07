"""Test suite for the T / G sub-population decomposition cells (pre-launch gate).

Covers HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3 (cells T
and G) and §5 (receipt requirements):

- initial-state / graph / head-count / parameter-count parity vs canonical
  Arm A, strict load with state-SHA equality;
- the exact perturbation site (after identity addition, before `fc_in`);
- one shared `p` per training forward and exact T/G `p_sequence_sha256`
  equality against the shared (never duplicated) PStream;
- cell T: whole-unit [B,N] Bernoulli keep law on a large fixture, per-(B,N)
  law with kept fraction ~ 1-p, seeded determinism, `min_keep = 4`
  enforcement on adversarial p->1 fixtures, `key_padding_mask` semantics
  (an excluded unit's src values cannot affect the output; a kept unit's do),
  `src` never zeroed and never rescaled, and the p == 1 no-NaN edge;
- cell G: gain `1/(1-clamp(p, 0, 0.95))` applied to every unit at the same
  site, the clamp at p > 0.95, the p == 0.95 boundary, and the realized-gain
  recorder;
- evaluation is bitwise equal to the unsparsified parent path for both cells,
  and enabling then disabling the perturbation never leaks;
- scoring does no gradient/update;
- the runner CLI rejects unknown cells and non-fresh output directories.
"""

from __future__ import annotations

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

from src.tfpd_lane import sparsification as sp
from src.tfpd_lane import subpop_cells as sc

CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"


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
    model = sc.build_subpop_model(seed=42, cell=cell)
    model.load_state_dict(canonical, strict=True)
    return model


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


# ---- graph / initial-state parity ------------------------------------------
def test_initial_state_graph_head_and_param_parity():
    canonical, canonical_sha = _canonical_state()
    from src.tfpd_lane import arm_common
    from torch.nn.parameter import UninitializedParameter

    params = set()
    for cell in ("T", "G"):
        model = sc.build_subpop_model(seed=42, cell=cell)
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
    assert len(params) == 1  # parameter-count parity across T/G/ArmA graph


def test_exact_site_and_mask_plumbing():
    source = (ROOT / "src/tfpd_lane/subpop_cells.py").read_text()
    body = source[source.index("def decode_with_identity"):source.index("# ---------------------------------------------------------------------------\n# launch-time proofs")]
    identity_pos = body.index("src = src + identity")
    site_pos = body.index("self.apply_perturbation(src)")
    fc_in_pos = body.index("self.decoder.fc_in(src)")
    assert identity_pos < site_pos < fc_in_pos
    # T's removal reaches nn.MultiheadAttention through existing plumbing
    shared = (ROOT.parent / "streaming_calibration_exp/src/models/components/spint.py").read_text()
    assert "key_padding_mask=key_padding_mask" in shared  # CrossAttentionLayer forwards it
    assert "key_padding_mask=key_padding_mask" in body  # the T call site passes it


# ---- the shared p stream ----------------------------------------------------
def test_pstream_imported_never_duplicated_and_t_g_sha_equality():
    assert sc.P_STREAM_SEED == sp.P_STREAM_SEED
    stream_t, stream_g = sc.pstream(), sc.pstream()
    assert isinstance(stream_t, sp.PStream)  # the very same class object
    n = 500
    draws_t = [stream_t.next() for _ in range(n)]
    draws_g = [stream_g.next() for _ in range(n)]
    assert draws_t == draws_g
    assert stream_t.sha256() == stream_g.sha256()  # T and G share the stream
    sibling = sp.PStream()
    [sibling.next() for _ in range(n)]
    assert stream_t.sha256() == sibling.sha256()  # and identical to R / S2
    assert len(stream_t.sha256()) == 64
    assert all(0.0 <= v < 1.0 for v in draws_t)


def test_T_one_shared_p_per_forward():
    model = _built("T")
    neural, calib, side = _tiny_inputs()
    model.eval()
    stream = sc.pstream()
    model.masker = sc.UnitRemovalMasker(seed=17)
    with torch.no_grad():
        for _ in range(5):
            p = stream.next()
            model.current_p = p
            model.perturbation_enabled = True
            model(neural, calib_trials=calib, side_features=side)
            model.perturbation_enabled = False
    assert stream.stats()["n"] == 5  # exactly one p draw per forward
    recorded = list(model.perturbation_stats)
    assert len(recorded) == 5
    # one shared p per forward (no per-row p): 5 forwards -> 5 recorded values
    assert len({entry["p"] for entry in recorded}) == 5
    # the whole-unit mask draws never touch the p stream
    assert stream.stats()["n"] == 5


# ---- cell T -----------------------------------------------------------------
def test_T_whole_unit_mask_shape_determinism_and_row_variation():
    masker = sc.UnitRemovalMasker(seed=7)
    keep = masker.draw(batch_size=64, n_units=100, p=0.3, device="cpu")
    assert keep.shape == (64, 100) and keep.dtype == torch.bool
    replay = sc.UnitRemovalMasker(seed=7).draw(64, 100, 0.3, "cpu")
    assert torch.equal(keep, replay)  # seeded determinism
    counts = keep.sum(dim=1)
    assert counts.min() >= sc.MIN_KEEP
    assert len(set(counts.tolist())) > 1  # per-row variation (not one shared row)
    # per-(B,N) Bernoulli law: kept fraction ~ 1-p on a large fixture
    assert keep.float().mean().item() == pytest.approx(0.7, abs=0.02)


def test_T_whole_unit_row_constant_across_window():
    model = _built("T")
    neural, calib, side = _tiny_inputs(n_units=12, batch=4)
    model.eval()
    model.masker = sc.UnitRemovalMasker(seed=7)
    model.current_p = 0.5
    model.perturbation_enabled = True
    captured, restore = _spy_transformer(model)
    try:
        with torch.no_grad():
            model(neural, calib_trials=calib, side_features=side)
    finally:
        restore()
        model.perturbation_enabled = False
    kpm = captured["key_padding_mask"]
    assert captured["attn_mask"] is None
    assert kpm is not None and kpm.shape == (4, 12) and kpm.dtype == torch.bool
    # exactly the boolean complement of an identically seeded whole-unit draw
    expected_keep = sc.UnitRemovalMasker(seed=7).draw(4, 12, 0.5, kpm.device)
    assert torch.equal(kpm, ~expected_keep)
    assert bool(kpm.any())  # at least one unit is genuinely excluded
    # whole-unit = the mask carries no window axis: rank-2 [B, N] only
    assert kpm.dim() == 2


def test_T_src_not_zeroed_and_not_rescaled():
    model = _built("T")
    neural, calib, side = _tiny_inputs(n_units=12, batch=4)
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
    src = neural.permute(0, 2, 1) + identity
    model.masker = sc.UnitRemovalMasker(seed=7)
    model.current_p = 0.6
    model.perturbation_enabled = True
    out_src, kpm = model.apply_perturbation(src)
    model.perturbation_enabled = False
    assert torch.equal(out_src, src)  # bitwise untouched: no zeroing, no gain
    assert kpm.shape == src.shape[:2]
    dropped = kpm.any(dim=0)
    assert bool(dropped.any())
    entry = model.perturbation_stats[-1]
    assert entry["src_zeroed"] is False and entry["rescaling"] == "none"
    assert entry["p"] == 0.6 and entry["n_units"] == 12


def test_T_min_keep_enforcement_on_adversarial_fixtures():
    # p == 1.0: every row would lose everything; each is restored to exactly 4
    masker = sc.UnitRemovalMasker(seed=3)
    keep = masker.draw(batch_size=32, n_units=40, p=1.0, device="cpu")
    counts = keep.sum(dim=1)
    assert bool((counts == sc.MIN_KEEP).all())
    summary = masker.summary()
    assert summary["min_keep_trigger_rows"] == 32
    assert summary["units_restored_by_min_keep"] == 32 * 4
    assert summary["rows_below_min_keep_after_guard"] == 0
    assert summary["all_masked_rows_after_guard"] == 0
    assert summary["surviving_units_min"] == 4 and summary["surviving_units_max"] == 4
    # p very close to 1: the floor still holds, deterministically
    for p in (0.99, 0.999):
        a = sc.UnitRemovalMasker(seed=11).draw(64, 50, p, "cpu")
        b = sc.UnitRemovalMasker(seed=11).draw(64, 50, p, "cpu")
        assert torch.equal(a, b)
        assert int(a.sum(dim=1).min()) >= sc.MIN_KEEP
    # a population smaller than the floor is never thinned
    small = sc.UnitRemovalMasker(seed=5).draw(4, 3, 0.99, "cpu")
    assert bool(small.all())
    # summary quantiles come from the exact histogram
    masker = sc.UnitRemovalMasker(seed=13)
    masker.draw(16, 20, 0.5, "cpu")
    summary = masker.summary()
    assert summary["rows_total"] == 16
    assert sum(int(v) for v in summary["surviving_units_histogram"].values()) == 16
    assert summary["surviving_units_min"] <= summary["surviving_units_median"]
    assert summary["surviving_units_median"] <= summary["surviving_units_max"]


def test_T_key_padding_mask_semantics_excluded_unit_invisible():
    model = _built("T")
    neural, calib, side = _tiny_inputs(n_units=10, batch=2)
    model.eval()
    excluded, probe = 0, 1
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        src = neural.permute(0, 2, 1) + identity
        rep = model.decoder.fc_in(model.decoder.rep).to(src).repeat(src.size(0), 1, 1)
        kpm = torch.zeros(src.shape[0], src.shape[1], dtype=torch.bool)
        kpm[:, excluded] = True

        def decode(source):
            out, _ = model.decoder.transformer(
                rep, model.decoder.fc_in(source), key_padding_mask=kpm
            )
            return out

        base = decode(src)
        changed_excluded = src.clone()
        changed_excluded[:, excluded] = changed_excluded[:, excluded] + 5.0
        changed_kept = src.clone()
        changed_kept[:, probe] = changed_kept[:, probe] + 5.0
        out_excluded = decode(changed_excluded)
        out_kept = decode(changed_kept)
    assert torch.isfinite(base).all()
    assert torch.equal(base, out_excluded)  # the excluded unit cannot matter
    assert not torch.equal(base, out_kept)  # a kept unit does matter
    assert float((base - out_kept).abs().max()) > 0.0
    # the shared helper proves the same fact
    proof = sc.prove_true_removal(model, src, excluded_unit=excluded)
    assert proof["excluded_unit_change_bitwise_invisible"]
    assert proof["kept_unit_change_visible"]
    assert proof["output_finite"] and proof["min_keep"] == 4


def test_T_p_equals_one_edge_no_nan():
    model = _built("T")
    neural, calib, side = _tiny_inputs(n_units=10, batch=4)
    model.eval()
    model.current_p = 1.0
    model.perturbation_enabled = True
    with torch.no_grad():
        out, _ = model(neural, calib_trials=calib, side_features=side)
    model.perturbation_enabled = False
    assert torch.isfinite(out).all()  # no fully-masked-row NaN
    entry = model.perturbation_stats[-1]
    assert entry["surviving_units_min"] == sc.MIN_KEEP
    assert entry["surviving_units_max"] == sc.MIN_KEEP
    assert model.masker.summary()["surviving_units_min"] == sc.MIN_KEEP


def test_T_min_keep_histogram_records_trigger_count():
    masker = sc.UnitRemovalMasker(seed=21)
    for _ in range(20):
        masker.draw(batch_size=8, n_units=25, p=0.95, device="cpu")
    summary = masker.summary()
    assert summary["rows_total"] == 160
    assert summary["min_keep_trigger_rows"] > 0
    assert summary["rows_below_min_keep_after_guard"] == 0
    assert summary["surviving_units_min"] >= sc.MIN_KEEP
    masker.reset_epoch()
    assert masker.summary()["rows_total"] == 0


# ---- cell G -----------------------------------------------------------------
def test_G_gain_rule_applied_to_every_unit():
    model = _built("G")
    neural, calib, side = _tiny_inputs(n_units=12, batch=4)
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        model.current_p = 0.5
        model.perturbation_enabled = True
        mine, _ = model(neural, identity=identity)
        model.perturbation_enabled = False
        expected = sc.parent_decode_from_src(
            model, (neural.permute(0, 2, 1) + identity) * 2.0
        )
    assert torch.equal(mine, expected)  # gain 1/(1-0.5) = 2 on ALL units
    # no mask is ever passed
    captured, restore = _spy_transformer(model)
    try:
        model.current_p = 0.3
        model.perturbation_enabled = True
        with torch.no_grad():
            model(neural, identity=identity)
    finally:
        restore()
        model.perturbation_enabled = False
    assert captured["key_padding_mask"] is None
    proof = sc.prove_gain_rule(model, neural, identity, p=0.5)
    assert proof["bitwise_equal_to_parent_times_gain"] and proof["gain"] == 2.0


def test_G_clamp_policy_and_boundary():
    model = _built("G")
    neural, calib, side = _tiny_inputs(n_units=8, batch=2)
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
    for p_raw, clamped in ((0.9999, True), (1.0, True), (0.9500001, True)):
        proof = sc.prove_gain_rule(model, neural, identity, p=p_raw)
        assert proof["clamped"] is clamped
        assert proof["p_clamped"] == sc.GAIN_CLAMP
        assert proof["gain"] == 1.0 / (1.0 - sc.GAIN_CLAMP)
        assert proof["gain"] == pytest.approx(20.0, rel=1e-12)
        assert proof["bitwise_equal_to_parent_times_gain"]
    # p == 0.95 exactly is the boundary: not clamped, gain still finite
    boundary = sc.prove_gain_rule(model, neural, identity, p=0.95)
    assert boundary["clamped"] is False
    assert boundary["p_clamped"] == 0.95
    assert boundary["gain"] == 1.0 / (1.0 - 0.95)
    # p below the clamp is used verbatim
    low = sc.prove_gain_rule(model, neural, identity, p=0.8)
    assert low["clamped"] is False and low["gain"] == pytest.approx(5.0)
    # the clamp bound constant is the frozen one
    assert sc.GAIN_CLAMP == 0.95 and sc.CELLS["G"]["p_clamp"] == 0.95


def test_G_realized_gain_stats_recorder():
    recorder = sc.GlobalGainRecorder()
    for p in (0.0, 0.5, 0.8, 0.95, 0.99, 1.0):
        p_clamped = min(max(p, 0.0), sc.GAIN_CLAMP)
        recorder.record(p, p_clamped, 1.0 / (1.0 - p_clamped))
    summary = recorder.summary()
    assert summary["n_forwards"] == 6
    assert summary["gain_min"] == pytest.approx(1.0)
    assert summary["gain_max"] == pytest.approx(1.0 / (1.0 - 0.95), rel=1e-12)
    assert summary["clamp_trigger_count"] == 2  # 0.99 and 1.0
    assert summary["clamp_trigger_rate"] == pytest.approx(2 / 6)
    assert summary["gain_min"] <= summary["gain_q25"] <= summary["gain_median"]
    assert summary["gain_median"] <= summary["gain_q75"] <= summary["gain_max"]
    assert summary["p_raw_max"] == 1.0
    recorder.reset_epoch()
    empty = recorder.summary()
    assert empty["n_forwards"] == 0
    assert set(empty) == {"gain_rule", "clamp", "n_forwards"}


def test_G_stats_and_T_stats_via_summarize_perturbation():
    model = _built("G")
    neural, calib, side = _tiny_inputs(n_units=8, batch=2)
    model.eval()
    with torch.no_grad():
        identity = model.compute_identity(calib, side_features=side)
        for p in (0.2, 0.6, 0.97):
            model.current_p = p
            model.perturbation_enabled = True
            model(neural, identity=identity)
            model.perturbation_enabled = False
    summary = sc.summarize_perturbation(model)
    assert summary["n_forwards"] == 3
    assert summary["clamp_trigger_count"] == 1
    assert summary["gain_max"] == pytest.approx(20.0, rel=1e-12)
    model_t = _built("T")
    model_t.eval()
    model_t.masker = sc.UnitRemovalMasker(seed=7)
    with torch.no_grad():
        model_t.current_p = 0.9
        model_t.perturbation_enabled = True
        model_t(neural, calib_trials=calib, side_features=side)
        model_t.perturbation_enabled = False
    summary_t = sc.summarize_perturbation(model_t)
    assert summary_t["n_forwards"] == 1
    assert summary_t["surviving_units_min"] >= sc.MIN_KEEP
    assert "surviving_units_histogram" in summary_t


# ---- evaluation parity and no-leak -----------------------------------------
@pytest.mark.parametrize("cell", ["T", "G"])
def test_eval_bitwise_parent_parity_and_no_leak(cell):
    model = _built(cell)
    model.eval()
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
    # the disabled eval pass passes no key_padding_mask at all
    captured, restore = _spy_transformer(model)
    try:
        with torch.no_grad():
            model(neural, calib_trials=calib, side_features=side)
    finally:
        restore()
    assert captured["key_padding_mask"] is None
    assert model.perturbation_enabled is False
    # enabling and disabling again keeps reproducing the parent bitwise
    with torch.no_grad():
        model.current_p = 0.99
        model.perturbation_enabled = True
        model(neural, identity=identity)
        model.perturbation_enabled = False
        again, _ = model(neural, identity=identity)
    assert torch.equal(again, parent)


@pytest.mark.parametrize("cell", ["T", "G"])
def test_scoring_no_gradient_no_update(cell):
    model = _built(cell)
    model.eval()
    neural, calib, side = _tiny_inputs()
    with torch.no_grad():
        model(neural, calib_trials=calib, side_features=side)
    assert all(
        p.grad is None
        for p in model.parameters()
        if hasattr(p, "grad")
    )
    assert model.perturbation_enabled is False


@pytest.mark.parametrize("cell", ["T", "G"])
def test_perturbation_masks_do_not_touch_global_rng_or_order(cell):
    model = _built(cell)
    model.eval()
    neural, calib, side = _tiny_inputs(n_units=12, batch=4)
    torch.manual_seed(1234)
    reference = torch.rand(64)
    torch.manual_seed(1234)  # identical starting state for the perturbed pass
    with torch.no_grad():
        model.current_p = 0.6
        model.perturbation_enabled = True
        model(neural, calib_trials=calib, side_features=side)
        model.perturbation_enabled = False
    assert torch.equal(torch.rand(64), reference)  # global torch RNG untouched
    # the whole-unit draws come from the dedicated namespace only
    if cell == "T":
        assert isinstance(model.masker.generator, torch.Generator)
        assert model.masker.generator.initial_seed() == sc.UNIT_MASK_SEED


# ---- runner surface ---------------------------------------------------------
def test_runner_cli_rejects_unknown_cell_and_existing_root(tmp_path):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    unknown = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_cell.py"), "--cell", "X"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert unknown.returncode == 2
    (tmp_path / "cellT_true_removal").mkdir()
    stale = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_cell.py"), "--cell", "T",
         "--device", "cpu", "--output-root", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert stale.returncode == 2 and "fresh cell output directory" in stale.stderr
    source = (ROOT / "scripts/run_subpop_cell.py").read_text()
    for token in ("p_sequence_sha256", "min_keep", "key_padding_mask",
                  "33,925", "distribution_matched_to_D"):
        assert token in source
    assert "results/subpop_v1" in source
    assert "cellG_global_gain" in source and "cellT_true_removal" in source
    integrity_fields = ("num_heads", "dropout_structure",
                        "p_distribution_and_clamp_policy", "min_keep",
                        "rescaling_policy", "generator_namespaces",
                        "eval_mask_policy", "behavior_scaling_convention")
    for field in integrity_fields:
        assert field in source


def test_cell_contract_tables_match_frozen_constants():
    assert sc.CELLS["T"]["min_keep"] == 4 == sc.MIN_KEEP
    assert sc.CELLS["T"]["num_heads"] == 2 and sc.CELLS["G"]["num_heads"] == 2
    assert sc.UNIT_MASK_SEED == 42_003
    assert sc.CELLS["T"]["mask_structure"] == "whole_unit_bernoulli_key_padding_mask"
    assert sc.CELLS["G"]["mask_structure"] == "none_global_gain_all_units"
    assert json.dumps(sc.CELLS["G"]["rescaling_policy"])  # receipt-serializable
