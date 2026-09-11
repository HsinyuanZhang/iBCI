"""pseudo-MUA (electrode-pooled) cache correctness tests (variant pmua_t4).

Covers the three acceptance families of scripts/build_pmua_cache.py:
  1. pooling correctness: per-time-bin count conservation, channel count
     <= unit count, >=1 unit per electrode, singleton identity, unit-order
     permutation invariance, leading-axis (calib geometry) pooling;
  2. channel-level T4: finiteness, shape, exact linearity identity against
     member units (and the non-linearity of column m -- why the channel
     refit is NOT a unit-T4 average);
  3. schema compatibility of the built cache: run_688_bench
     verify_prepared_cache + load_session digests, Nmax-91 padding law,
     byte-copied behavior/starts, finite carrier/e0.

A real-data test additionally pins the pooled-calib construction against the
frozen pseudo-MUA bridge datamodule cache (byte parity on one session).

Run:
  cd /home/xinyuan/Work_host/SPINT && PYTHONNOUSERSITE=1 \
    /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest \
    btransform_unified_v2/dandi688_bench_v1/tests/test_pmua_cache.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
for p in (PKG_ROOT / "src", PKG_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

WS = PKG_ROOT.parents[1]
for p in (WS / "btransform_unified_v2" / "src", WS / "btransform_unified_v1" / "src",
          WS / "sua_exploration", WS / "streaming_calibration_exp", WS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dandi688_bench_v1 import plan  # noqa: E402
from build_pmua_cache import (  # noqa: E402
    N_PAD,
    assert_pool_conservation,
    pool_count_array_by_electrode,
    sua_vs_pmua_column_correlation,
)

PMUA_DEST = PKG_ROOT / "results" / "cache_pmua_t4"
PMUA_F_LABELFREE_DEST = PKG_ROOT / "results" / "cache_pmua_f_labelfree"
BRIDGE_CACHE = WS / "sua_exploration/cache/dandi688_subc_co_pseudomua_t4_v1"
PARITY_SESSION = "sub-C_ses-CO-20131003"


# ---------------------------------------------------------------------------
# 1. pooling correctness (pure synthetic)
# ---------------------------------------------------------------------------
def _synthetic_counts(seed: int = 0, n_bins: int = 200, n_units: int = 9):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 5, size=(n_bins, n_units)).astype(np.float32)
    electrode_ids = np.asarray([3, 3, 7, 7, 7, 1, 1, 9, 9], dtype=np.int64)
    return counts, electrode_ids


def test_pool_conservation_channel_count_and_membership():
    counts, electrode_ids = _synthetic_counts()
    pooled, channel_ids, inverse = pool_count_array_by_electrode(counts, electrode_ids)
    # channel count <= unit count, ascending deterministic order
    assert channel_ids.tolist() == sorted(set(electrode_ids.tolist()))
    assert pooled.shape == (counts.shape[0], 4)
    assert pooled.shape[1] <= counts.shape[1]
    # every electrode holds >= 1 unit (bincount over the inverse mapping)
    assert np.bincount(inverse).min() >= 1
    # exact per-channel count conservation (half-open non-overlapping bins)
    assert_pool_conservation(pooled, counts, inverse, pooled.shape[1])
    # total conservation
    assert float(pooled.sum()) == float(counts.sum())


def test_pool_singleton_identity():
    """One unit per electrode: pooling is the identity up to channel order."""
    counts, _ = _synthetic_counts()
    electrode_ids = np.arange(counts.shape[1], dtype=np.int64)
    pooled, channel_ids, inverse = pool_count_array_by_electrode(counts, electrode_ids)
    assert np.array_equal(pooled, counts)
    assert channel_ids.tolist() == electrode_ids.tolist()
    assert np.array_equal(inverse, electrode_ids)


def test_pool_unit_order_permutation_invariance():
    counts, electrode_ids = _synthetic_counts()
    rng = np.random.default_rng(7)
    perm = rng.permutation(counts.shape[1])
    base, channel_ids, _ = pool_count_array_by_electrode(counts, electrode_ids)
    got, channel_ids_p, _ = pool_count_array_by_electrode(
        counts[:, perm], electrode_ids[perm]
    )
    assert np.array_equal(channel_ids, channel_ids_p)
    assert np.array_equal(base, got)


def test_pool_leading_axis_geometry_matches_flat_pooling():
    """[K, W, units] (calib-trial geometry) pools identically to flat views."""
    counts, electrode_ids = _synthetic_counts()
    calib_like = np.repeat(counts[:12][None, :, :], 5, axis=0)  # [5, 12, units]
    pooled3, channel_ids, inverse = pool_count_array_by_electrode(calib_like, electrode_ids)
    flat, flat_channels, _ = pool_count_array_by_electrode(
        calib_like.reshape(-1, calib_like.shape[-1]), electrode_ids
    )
    assert np.array_equal(pooled3.reshape(flat.shape), flat)
    assert np.array_equal(channel_ids, flat_channels)
    assert pooled3.shape == (5, 12, 4)
    assert_pool_conservation(pooled3, calib_like, inverse, pooled3.shape[-1])


# ---------------------------------------------------------------------------
# 2. channel-level T4 (frozen descriptor recipe on synthetic trials)
# ---------------------------------------------------------------------------
def _synthetic_direction_trials(n_trials: int = 10):
    """Ten legal phase trials cycling the 8 canonical directions."""
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
        _phase_trials,
        single_pool_interval_layout,
    )

    trials = []
    for k in range(n_trials):
        angle = -3.0 * math.pi / 4.0 + (k % 8) * (math.pi / 4.0)
        trials.append({
            "start_time": 0.0, "stop_time": 5.0,
            "target_on_time": 1.0, "go_cue_time": 2.0,
            "target_dir": angle,
        })
    selected, _ = _phase_trials(
        trials, support_positions=tuple(range(10)), namespace="candidate"
    )
    assert len(selected) == n_trials  # all synthetic trials are legal
    _intervals, slices = single_pool_interval_layout(trials, groups=("candidate",))
    return trials, slices


def _tuned_rates(n_units: int, n_intervals: int, seed: int = 11):
    """Rates whose per-unit direction tuning is a closed-form cosine."""
    rng = np.random.default_rng(seed)
    preferred = rng.uniform(-math.pi, math.pi, size=n_units)
    baseline = rng.uniform(5.0, 15.0, size=n_units)
    modulation = rng.uniform(2.0, 8.0, size=n_units)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=n_intervals)
    rates = baseline[:, None] + modulation[:, None] * np.cos(
        phases[None, :] - preferred[:, None]
    )
    return np.maximum(rates, 0.0), preferred, baseline, modulation


def test_channel_t4_finite_shape_and_linearity_identity():
    """The pooled channel T4 is the frozen cosine refit on SUMMED rates.

    Exact identity: columns a/c/b are linear in the rate rows, so the
    channel refit's a/c/b equal the sums of its member units' a/c/b; column
    m = hypot(a, c) is NOT additive -- refitting on pooled rates (the frozen
    protocol) is therefore not a unit-T4 row average."""
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
        refit_from_single_pool,
    )

    trials, slices = _synthetic_direction_trials()
    n_units, electrode_ids = 9, np.asarray([3, 3, 7, 7, 7, 1, 1, 9, 9], dtype=np.int64)
    rates, _pref, _base, _mod = _tuned_rates(n_units, 30)
    _w_sua, _p_sua, sua_raw = refit_from_single_pool(
        rates, trials, slices, group="candidate", signal_view="sua"
    )
    _w_pm, _p_pm, pmua_raw = refit_from_single_pool(
        rates, trials, slices, group="candidate",
        signal_view="pseudo_mua", electrode_ids=electrode_ids,
    )
    channel_ids, inverse = np.unique(electrode_ids, return_inverse=True)
    # finiteness + shape gates
    assert pmua_raw.shape == (channel_ids.size, 4)
    assert pmua_raw.shape[0] <= n_units
    assert np.isfinite(pmua_raw).all() and np.isfinite(sua_raw).all()
    # linearity identity on a/c/b (float32 lstsq rounding -> ~1e-6 tolerance)
    member_sums = np.stack([sua_raw[inverse == ch].sum(axis=0) for ch in range(4)])
    assert np.allclose(pmua_raw[:, [0, 1, 3]], member_sums[:, [0, 1, 3]], atol=1e-5)
    # m is the pooled hypot, NOT the sum of member m
    assert np.allclose(
        pmua_raw[:, 2], np.hypot(pmua_raw[:, 0], pmua_raw[:, 1]), atol=1e-6
    )
    multi = [ch for ch in range(4) if (inverse == ch).sum() > 1]
    assert multi and not np.allclose(
        pmua_raw[multi, 2], member_sums[multi, 2], atol=1e-3
    )


def test_channel_t4_equals_refit_on_manually_summed_rates():
    """Pooled channel T4 == the frozen refit applied to the manually summed
    rate matrix (channel formation happens BEFORE the fit, never after)."""
    from mc_maze.dandi688_sparse_event_t4_v1.descriptors import (
        refit_from_single_pool,
    )

    trials, slices = _synthetic_direction_trials()
    electrode_ids = np.asarray([3, 3, 7, 7, 7, 1, 1, 9, 9], dtype=np.int64)
    rates, _pref, _base, _mod = _tuned_rates(9, 30)
    channel_ids, inverse = np.unique(electrode_ids, return_inverse=True)
    manual = np.zeros((channel_ids.size, rates.shape[1]), dtype=np.float64)
    np.add.at(manual, inverse, rates)
    _w, _p, manual_raw = refit_from_single_pool(
        manual, trials, slices, group="candidate", signal_view="sua"
    )
    _w2, _p2, pmua_raw = refit_from_single_pool(
        rates, trials, slices, group="candidate",
        signal_view="pseudo_mua", electrode_ids=electrode_ids,
    )
    assert np.allclose(pmua_raw, manual_raw, atol=1e-9)


def test_column_correlation_helper_identity_case():
    """On a singleton mapping the PMUA-vs-SUA correlation is exactly 1/0."""
    rates, _pref, _base, _mod = _tuned_rates(6, 30)
    raw = np.column_stack((
        _mod * np.cos(_pref), _mod * np.sin(_pref),
        np.abs(_mod), _base,
    ))
    inverse = np.arange(6)
    corr = sua_vs_pmua_column_correlation(raw, raw, inverse)
    assert corr["pearson_r_col_a"] == pytest.approx(1.0)
    assert corr["pearson_r_col_c"] == pytest.approx(1.0)
    assert corr["mean_abs_angle_delta_deg"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. real-data parity against the frozen pseudo-MUA bridge datamodule cache
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not BRIDGE_CACHE.is_dir() or not list((BRIDGE_CACHE / "sessions").glob(f"{PARITY_SESSION}_*.npz")),
    reason="frozen pseudo-MUA bridge datamodule cache not present",
)
def test_pooled_calib_matches_bridge_datamodule_cache():
    """Our count-domain pooled calib (pool counts, then interpolate) must be
    byte-identical to the bridge datamodule's pseudo_mua calib_trials on the
    pinned session (M10 budget in that cache)."""
    from mc_maze.multisession_datamodule import _build_calib_trials
    from build_pmua_cache import _load_frozen, _session_profiles

    prof = _session_profiles(PARITY_SESSION)
    meta, rows = _load_frozen(plan.prepared_cache_path())
    row = rows[PARITY_SESSION]
    mask = np.asarray(row["mask"], dtype=bool)
    n_real = int(mask.sum())
    assert prof["n_units"] == n_real
    neural = np.asarray(row["neural"])[:, :n_real]
    pooled, _channel_ids, _inv = pool_count_array_by_electrode(neural, prof["electrode_ids"])
    first10 = [
        {"start": start, "stop": stop}
        for start, stop in prof["calib_trial_bins"][:10]
    ]
    calib10 = _build_calib_trials(
        pooled, first10, 10, 100, pooled.shape[1],
        pad_value=-1.0, interpolate_trials=True,
    )
    bridge_file = next((BRIDGE_CACHE / "sessions").glob(f"{PARITY_SESSION}_*.npz"))
    with np.load(bridge_file, allow_pickle=False) as z:
        bridge_calib = np.asarray(z["calib_trials"])
        bridge_neural = np.asarray(z["neural"])
    assert bridge_calib.shape == calib10.shape == (10, 100, pooled.shape[1])
    assert np.array_equal(calib10, bridge_calib)
    assert np.array_equal(pooled, bridge_neural)


# ---------------------------------------------------------------------------
# 4. built-cache schema compatibility (requires the built cache)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (PMUA_DEST / "prepared_contract.json").is_file(),
    reason="pmua_t4 cache not built yet (scripts/build_pmua_cache.py)",
)
class TestBuiltCacheSchema:
    def test_verify_prepared_cache_and_variant(self):
        sys.path.insert(0, str(PKG_ROOT / "scripts"))
        from run_688_bench import verify_prepared_cache

        meta = verify_prepared_cache(PMUA_DEST)
        assert meta["variant"] == "pmua_t4"
        assert meta["formal_test_used"] is False
        assert meta["estimator"]["signal_view"] == "pseudo_mua"
        assert meta["estimator"]["norm_protocol"] in plan.PROTOCOLS
        assert meta["estimator"]["sua_parity_gate"]["pass"] is True

    def test_row_geometry_padding_and_digests(self):
        from dandi688_bench_v1 import eval_local

        frozen = json.loads(
            (plan.prepared_cache_path() / "prepared_contract.json").read_text()
        )
        meta = json.loads((PMUA_DEST / "prepared_contract.json").read_text())
        for name in ("sub-C_ses-CO-20131003", "sub-C_ses-CO-20151103"):
            row = eval_local.load_session(PMUA_DEST, name)
            info = meta["identity_log"][name]
            n_elec = info["n_electrodes"]
            assert n_elec <= info["n_units"] <= N_PAD
            # Nmax-91 padding law: real channels first, padding all zero
            assert int(row["mask"].sum()) == n_elec
            assert bool(row["mask"][:n_elec].all()) and not bool(row["mask"][n_elec:].any())
            assert row["neural"].shape[1] == N_PAD
            assert bool((row["neural"][:, n_elec:] == 0).all())
            assert row["carrier"].shape == (N_PAD, plan.CARRIER_DIM)
            assert row["e0"].shape == (N_PAD, plan.E0_DIM)
            assert int((np.abs(row["carrier"]).sum(axis=1) > 0).sum()) == n_elec
            assert int((np.abs(row["e0"]).sum(axis=1) > 0).sum()) == n_elec
            assert np.isfinite(row["carrier"]).all() and np.isfinite(row["e0"]).all()
            # behavior/starts copied byte-identically from the frozen cache
            for key in ("behavior", "starts"):
                assert (meta["rows_hashes"][name][key]
                        == frozen["rows_hashes"][name][key])
            # conservation record
            assert info["conservation_exact"] is True

    def test_splits_and_sessions_cover_33(self):
        meta = json.loads((PMUA_DEST / "prepared_contract.json").read_text())
        assert len(meta["sessions"]) == 33
        assert sum(s["split"] == "train" for s in meta["sessions"].values()) == 27
        assert sum(s["split"] == "val" for s in meta["sessions"].values()) == 6
        assert not set(meta["sessions"]) & set(plan.FORMAL_TEST_SESSIONS)


# ---------------------------------------------------------------------------
# 5. f_labelfree variant cache (label-free ACTIVITY-ONLY rung on PMUA)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (PMUA_F_LABELFREE_DEST / "prepared_contract.json").is_file(),
    reason="pmua f_labelfree cache not built yet "
           "(scripts/build_pmua_cache.py --variant f_labelfree)",
)
class TestPmuaLabelFreeCache:
    def test_variant_binding_and_estimator_law(self):
        sys.path.insert(0, str(PKG_ROOT / "scripts"))
        from run_688_bench import assert_cache_variant_for_arm, verify_prepared_cache

        meta = verify_prepared_cache(PMUA_F_LABELFREE_DEST)
        assert meta["variant"] == "f_labelfree"
        binding = assert_cache_variant_for_arm(meta, "f_labelfree", "exp2015_full")
        assert binding == {"arm": "f_labelfree", "variant": "f_labelfree",
                           "allowed_variants": ["f_labelfree"]}
        assert meta["estimator"]["signal_view"] == "pseudo_mua"
        assert "zero side" in meta["estimator"]["e0_side"]
        assert meta["estimator"]["sua_parity_gate"]["pass"] is True

    def test_bytes_law_zero_carrier_nonzero_e0_neural_parity(self):
        from dandi688_bench_v1 import eval_local

        for name in ("sub-C_ses-CO-20150309", "sub-C_ses-CO-20151103"):
            row = eval_local.load_session(PMUA_F_LABELFREE_DEST, name)
            t4_row = eval_local.load_session(PMUA_DEST, name)
            # label-free law: carrier bytes exactly zero everywhere
            assert row["carrier"].shape == (N_PAD, plan.CARRIER_DIM)
            assert not bool(np.any(row["carrier"]))
            # the activity pathway survives: E0 nonzero on the real channels
            n_elec = int(row["mask"].sum())
            assert n_elec > 0 and bool(np.any(row["e0"][:n_elec]))
            assert not bool(np.any(row["e0"][n_elec:]))
            # the zero side melts a DIFFERENT identity than the carrier side
            assert not np.array_equal(row["e0"], t4_row["e0"])
            # pooling identical: neural/behavior/starts/mask byte-equal to the
            # pmua_t4 cache rows
            for key in ("neural", "behavior", "starts", "mask"):
                assert plan.array_digest(row[key]) == plan.array_digest(t4_row[key])

    def test_runner_loads_protocol_face(self):
        sys.path.insert(0, str(PKG_ROOT / "scripts"))
        from run_688_bench import load_rows, verify_prepared_cache

        meta = verify_prepared_cache(PMUA_F_LABELFREE_DEST)
        rows = load_rows(PMUA_F_LABELFREE_DEST, meta, "exp2015_full")
        assert len(rows) == 24
        assert sum(r["protocol_role"] == "train" for r in rows.values()) == 18
        assert sum(r["protocol_role"] == "exam" for r in rows.values()) == 6
        assert all(not np.any(r["carrier"]) for r in rows.values())
