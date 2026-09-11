"""M2 vstate4 <-> 688 vstate correspondence assertions (alignment audit).

Authority: ``btransform_unified_v2/docs/CARRIER_M2_688_ALIGNMENT_MATRIX_20260909.md``
M2 reference implementation (transcribed below, literals pinned):
``btransform_unified_v2/scripts/carrier_v3_m2/build_m2_variant_cache.py``
(``_blocks`` + ``_vstate_raw`` + rms fitting).  The M2 builder module itself
imports the tfpd stack, so its core formulas are re-implemented here
verbatim (same ``carrier_profile_v3`` calls, same constants) and asserted
bitwise-equal to the 688 ``vstate`` estimator on identical synthetic inputs.

Run:
  cd /home/xinyuan/Work_host/SPINT && PYTHONNOUSERSITE=1 \
    /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest \
    btransform_unified_v2/dandi688_bench_v1/tests/test_vstate_alignment.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
for p in (PKG_ROOT / "src", PKG_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dandi688_bench_v1 import arms, plan, vstate  # noqa: E402
from btransform_unified_v2 import carrier_profile_v3 as v3  # noqa: E402

# --- M2 builder literals (scripts/carrier_v3_m2/build_m2_variant_cache.py) ---
M2_BIN_SECONDS = 0.02              # BIN_SECONDS
M2_BLOCK_BINS = 5                  # BLOCK_BINS
M2_BLOCK_SECONDS = 0.1             # BLOCK_SECONDS = 5 * 0.02
M2_N0_VSTATE = 10.0                # N0_VSTATE


def m2_vstate4_core_reference(
    rates: np.ndarray, vels: np.ndarray, rms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim transcription of the M2 builder's ``_vstate_raw`` (without its
    bundle I/O): poisson_standardize -> signed_state_weights ->
    conditional_response -> [a, c, hypot, mean_k R].  Returns the float64 core
    (the estimator math) and the ``v3.stack_t4`` float32 packing the M2
    builder writes (a pure cast of the same numbers)."""
    z, _, _ = v3.poisson_standardize(rates, M2_BLOCK_SECONDS)
    weights = v3.signed_state_weights(vels, rms)
    response = v3.conditional_response(z, weights, M2_N0_VSTATE)
    a = response[:, 0] - response[:, 1]
    c = response[:, 2] - response[:, 3]
    b = response.mean(axis=1)
    m = np.hypot(a, c)
    return np.column_stack((a, c, m, b)), v3.stack_t4(a, c, m, b)


def m2_rms_reference(block_vels: list[np.ndarray]) -> np.ndarray:
    """Verbatim transcription of the M2 builder's rms fit (held-in blocks)."""
    held = np.concatenate(block_vels, axis=0)
    return np.maximum(
        np.sqrt(np.mean(np.square(held), axis=0)), v3.STD_FLOOR
    )


def _synthetic_blocks(n_blocks: int = 80, n_units: int = 6, seed: int = 3):
    """Same synthetic pair for both estimators: block rates [blocks, units] in
    Hz and block velocities [blocks, 2] (positive-rms by construction)."""
    rng = np.random.default_rng(seed)
    rates = np.maximum(rng.normal(loc=20.0, scale=8.0, size=(n_blocks, n_units)), 0.5)
    vels = rng.normal(loc=0.0, scale=2.0, size=(n_blocks, 2))
    return rates, vels


# ---------------------------------------------------------------------------
# core estimator: 688 main variant vs M2 vstate4, bitwise on identical inputs
# ---------------------------------------------------------------------------
def test_constants_match_m2_builder_literals():
    """The shared-recipe constants of the 688 adaptation equal the M2 builder's
    pinned literals (blocks, shrinkage, Poisson delta, velocity sub-bins)."""
    assert plan.VSTATE_BLOCK_SECONDS == M2_BLOCK_SECONDS == 0.1
    assert plan.VSTATE_N0_BLOCKS == M2_N0_VSTATE == 10.0
    assert plan.VSTATE_VELOCITY_SUBBINS == M2_BLOCK_BINS == 5
    assert plan.VSTATE_VELOCITY_BIN_SECONDS == M2_BIN_SECONDS == 0.02
    assert M2_BLOCK_BINS * M2_BIN_SECONDS == M2_BLOCK_SECONDS
    assert plan.VSTATE_VELOCITY_SUBBINS * plan.VSTATE_VELOCITY_BIN_SECONDS \
        == plan.VSTATE_BLOCK_SECONDS


def test_main_variant_bitwise_identical_to_m2_vstate4():
    """Alignment matrix row 'readout' (main variant) + rows 'states'/
    'standardize'/'shrinkage': on identical block rates/velocities/rms the 688
    main carrier (b_mode='mean_k', H300 absent) is bitwise-equal to the M2
    vstate4 core, all four columns."""
    rates, vels = _synthetic_blocks()
    train_vels = [vels[:40], vels[40:]]  # two fake train sessions
    rms = m2_rms_reference(train_vels)
    core64, packed32 = m2_vstate4_core_reference(rates, vels, rms)
    got = vstate.vstate_carrier_from_blocks(rates, vels, None, None, rms)
    # same float64 composition -> bitwise equality of the estimator cores
    assert np.array_equal(got, core64)
    # and the float32 packing the M2 builder writes is the identical cast of
    # the same numbers (the 688 builder casts at the normalizer instead)
    assert np.array_equal(got.astype(np.float32), packed32)
    # passing H300 rows must NOT change the main variant (they are ignored)
    got_with_h300 = vstate.vstate_carrier_from_blocks(
        rates, vels, rates[:21], vels[:21], rms
    )
    assert np.array_equal(got_with_h300, got)


def test_b_hold_ablation_isolates_exactly_the_fourth_column():
    """Alignment matrix row 'readout' (ablation): b_mode='hold_diff' keeps
    a/c/m bitwise-equal to the main variant and subtracts exactly mean_k
    R_hold (the 688-local delta_b semantics, user-ruled switchable)."""
    rates, vels = _synthetic_blocks()
    rms = m2_rms_reference([vels])
    main = vstate.vstate_carrier_from_blocks(rates, vels, None, None, rms)
    hold = vstate.vstate_carrier_from_blocks(
        rates, vels, rates[10:], vels[10:], rms, b_mode="hold_diff"
    )
    assert np.array_equal(hold[:, :3], main[:, :3])
    # recompute mean_k R_hold exactly as the estimator does (M2 functions)
    _, rate_mean, noise_rate = v3.poisson_standardize(rates, M2_BLOCK_SECONDS)
    z_h = v3.apply_affine(rates[10:], rate_mean, noise_rate)
    w_h = v3.signed_state_weights(vels[10:], rms)
    r_hold = v3.conditional_response(z_h, w_h, M2_N0_VSTATE)
    expected_b = main[:, 3] - r_hold.mean(axis=1)
    assert np.array_equal(hold[:, 3], expected_b)
    assert not np.array_equal(hold[:, 3], main[:, 3])  # hold actually bites


def test_fit_velocity_rms_matches_m2_formula():
    """Alignment matrix row 'states': the 688 rms fit is the M2 formula
    (sqrt of mean square over the concatenated train blocks)."""
    rng = np.random.default_rng(11)
    parts = [rng.normal(size=(30, 2)) + 0.1, rng.normal(size=(50, 2)) - 0.2]
    got = vstate.fit_velocity_rms(parts)
    assert np.array_equal(got, m2_rms_reference(parts))
    # degenerate all-zero blocks: M2 floors at STD_FLOOR, 688 fails closed
    # (guard difference recorded in the alignment matrix; real rms ~ 5)
    with pytest.raises(RuntimeError):
        vstate.fit_velocity_rms([np.zeros((4, 2))])


# ---------------------------------------------------------------------------
# rate primitive + velocity semantics parity (rows 'rate primitive'/'velocity')
# ---------------------------------------------------------------------------
def test_rate_primitive_bin_sum_vs_searchsorted_parity():
    """M2 counts blocks by bin sums; 688 by half-open searchsorted counting.
    On the SAME edges the two primitives are exactly equal (audited on real
    sessions: 183,658 block-unit checks, max abs diff 0)."""
    rng = np.random.default_rng(5)
    edges = np.arange(0.0, 10.0 + 1e-9, M2_BIN_SECONDS)
    spikes = np.sort(rng.uniform(0.0, 10.0, size=5000))
    hist_bins = np.histogram(spikes, bins=edges)[0]
    ss_bins = (
        np.searchsorted(spikes, edges[1:], side="left")
        - np.searchsorted(spikes, edges[:-1], side="left")
    )
    assert np.array_equal(hist_bins.astype(np.int64), ss_bins.astype(np.int64))
    # block level: 5-bin sums vs searchsorted on the SAME block edges
    n_blocks = (len(edges) - 1) // M2_BLOCK_BINS
    for i in range(n_blocks):
        lo, hi = edges[M2_BLOCK_BINS * i], edges[M2_BLOCK_BINS * (i + 1)]
        bin_sum = int(hist_bins[M2_BLOCK_BINS * i:M2_BLOCK_BINS * (i + 1)].sum())
        sscnt = int(np.searchsorted(spikes, hi, side="left")
                    - np.searchsorted(spikes, lo, side="left"))
        assert bin_sum == sscnt
    # the 688 block_rate_matrix is exactly that half-open counting / duration
    # (block edges = every 5th grid edge, so the same boundary values)
    block_edges = edges[:: M2_BLOCK_BINS]
    rates = vstate.block_rate_matrix([spikes], block_edges)
    counts = ss_bins.reshape(n_blocks, M2_BLOCK_BINS).sum(axis=1)
    assert np.array_equal(rates[:, 0], counts / M2_BLOCK_SECONDS)


def test_block_velocity_mean_semantics_match_m2():
    """Alignment matrix row 'velocity source': both datasets take the block
    mean of bin-center velocity samples.  With cursor_vel sampled exactly at
    the 20 ms bin centers, the 688 block mean equals the M2-style mean of the
    5 per-bin covariate rows."""
    edges = vstate.block_edges(0.0, 0.5)
    centers = (
        edges[:-1, None]
        + (np.arange(5, dtype=np.float64)[None, :] + 0.5) * M2_BIN_SECONDS
    ).ravel()
    rng = np.random.default_rng(7)
    vel_values = rng.normal(size=(centers.size, 2))
    got = vstate.block_velocity_means(centers, vel_values, edges)
    m2_style = vel_values.reshape(len(edges) - 1, 5, 2).mean(axis=1)
    assert np.allclose(got, m2_style, rtol=0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# variant / arm wiring (rows 'support' and 'fusion' of the alignment matrix)
# ---------------------------------------------------------------------------
def test_vstate_family_variants_declared_without_building():
    """Alignment matrix rows 'support' (vstate_full) and 'readout' (b_hold):
    the cache builder declares the variant build parameters; nothing is built
    by this test (definition-only per the matrix).  (equiv_zero joined
    VARIANTS with the component-ablation clarification 2026-09-09.)"""
    import build_vstate_cache as builder

    assert builder.VARIANTS == (
        "vstate", "vstate_full", "vstate_b_hold", "f_labelfree", "equiv_zero",
    )
    spec_main = builder._variant_spec("vstate")
    spec_full = builder._variant_spec("vstate_full")
    spec_hold = builder._variant_spec("vstate_b_hold")
    # main: the SAME M10 trial set as the t4 arm, M2-identical fourth column
    assert spec_main["positions"] == plan.VSTATE_SUPPORT_POSITIONS == tuple(range(10))
    assert spec_main["namespace"] == "candidate"
    assert spec_main["b_mode"] == plan.VSTATE_B_MODE_MAIN == "mean_k"
    # full: all 30 activity-support trials, M2 all-support semantics
    assert spec_full["positions"] == tuple(range(30))
    assert spec_full["namespace"] == "reliability_audit"
    assert spec_full["label_budget"] == 30
    assert spec_full["b_mode"] == "mean_k"
    # b_hold: M10 support again (so a/c/m/rms match the main variant exactly),
    # only the fourth column switches to delta_b
    assert spec_hold["positions"] == spec_main["positions"]
    assert spec_hold["namespace"] == spec_main["namespace"]
    assert spec_hold["b_mode"] == "hold_diff"
    # b formulas are recorded verbatim for receipts
    assert "mean_k R_u" in builder.B_FORMULAS["mean_k"]
    assert "R_hold" in builder.B_FORMULAS["hold_diff"]
    # f_labelfree carries no labels at all
    assert builder._variant_spec("f_labelfree")["b_mode"] is None
    with pytest.raises(ValueError):
        builder._variant_spec("vstate_m33")
    # destination naming: vstate keeps the historical cache name, the new
    # variants append their own tag, f_labelfree is protocol-free
    root = Path("/tmp/unused")
    assert builder._variant_dest("vstate", root, "exp1_narrow").name == "cache_vstate_exp1_narrow"
    assert builder._variant_dest("vstate_full", root, "exp1_narrow").name == "cache_vstate_full_exp1_narrow"
    assert builder._variant_dest("vstate_b_hold", root, "exp2_full").name == "cache_vstate_b_hold_exp2_full"
    assert builder._variant_dest("f_labelfree", root, "exp1_narrow").name == "cache_f_labelfree"


def test_vstate_concat_arm_registered_and_ungated():
    """Alignment matrix row 'fusion': the vstate carrier has a concat-fusion
    arm (M2 mainline R50 D4 concat); readings only, never inside any gate."""
    assert "vstate_concat" in plan.ARMS and "vstate_concat" in arms.ARM_SPECS
    assert arms.arm_fusion("vstate_concat") == "concat"
    carrier = np.arange(4 * plan.CARRIER_DIM, dtype=np.float32).reshape(4, -1)
    mask = np.ones(4, bool)
    out = arms.apply_arm("vstate_concat", "s", carrier, mask)
    assert np.array_equal(out, carrier)  # fusion never touches carrier bytes
    rec = arms.verify_arm("vstate_concat", "s", carrier, mask, out)
    assert rec["verify"] == "PASSED" and rec["fusion"] == "concat"
    # gate reports stay gated on their own arms only
    three_arm = plan.gate_report(0.6, 0.5, 0.5)
    vseries = plan.vstate_gate_report(0.6, 0.55, 0.5, 0.4)
    assert "vstate_concat" not in plan.json_dumps(three_arm)
    assert "vstate_concat" not in plan.json_dumps(vseries)
    # family binding: vstate_concat accepts every vstate-family variant cache
    assert plan.ARM_REQUIRED_CACHE_VARIANT["vstate_concat"] == frozenset(plan.VSTATE_VARIANTS)


def test_existing_arm_semantics_untouched():
    """The correction does not disturb the pre-existing arms (t4/f0/ts4/
    t4_concat/z_srcbank/f_labelfree semantics preserved byte-for-byte).
    EXCEPTION (user clarification 2026-09-09, plan.COMPONENT_ABLATION): z0
    was redefined to the carrier-only cell (E0 zero, carrier identity) and
    its old both-zero semantics moved to the new floor arm -- an intentional
    component-ablation restructure, not a drift."""
    specs = arms.ARM_SPECS
    assert specs["t4"] == {"carrier": "identity", "fusion": "proj_add", "e0": "identity"}
    assert specs["f0"] == {"carrier": "zero", "fusion": "proj_add", "e0": "identity"}
    assert specs["ts4"] == {"carrier": "shuffle", "fusion": "proj_add", "e0": "identity"}
    assert specs["t4_concat"] == {"carrier": "identity", "fusion": "concat", "e0": "identity"}
    assert specs["z0"] == {"carrier": "identity", "fusion": "proj_add", "e0": "zero"}
    # the old z0 (both-zero / Z_NONE) semantics live on as the floor arm
    assert specs["floor"] == {"carrier": "zero", "fusion": "proj_add", "e0": "zero"}
    assert specs["z_vstate_srcbank"] == {"carrier": "identity", "fusion": "proj_add", "e0": "identity"}
    assert specs["f_labelfree"] == {"carrier": "zero", "fusion": "proj_add", "e0": "identity"}
