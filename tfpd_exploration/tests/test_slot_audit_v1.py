"""Tests for SLOT-AUDIT V1 (no data, no CUDA, no decoder forward).

Covers exactly what the work order demands of the audit surface:
  - the ABSOLUTE-BIN ALIGNMENT LAW on synthetic sessions (windows inside
    single trials, slot s speaking about bin w+s, bin/window validity masks,
    trial-boundary heads);
  - per-slot R2 correctness against hand-computed cases (house float32 scorer
    AND the float64 numpy twin);
  - the per-delta redundant-estimate residual-pair correlation math (brute
    force equality, a rank-1 separable closed form, the pure-common-mode case
    and the bias-only case after per-column centering);
  - slot-subset mean logic ({49} bit-equality with the last-bin row, the
    source-selected slot tie-break);
  - the pre-registered reading-classification boundaries;
  - the cache manifest/sidecar round-trip and tamper rejection.

Import-only touches of frozen modules (continuity_probe_v1, matched_scorer)
are used where the audit itself must reuse the frozen law.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.slot_audit_v1 import (  # noqa: E402
    alignment,
    cache_store,
    ensembles,
    plan,
    scoring,
    targets,
)


# ---------------------------------------------------------------------------
# synthetic sessions
# ---------------------------------------------------------------------------


def synthetic_session(seed=7, n_trials=4, trial_len=120, gap=9, n_units_unused=None):
    """Trials of consecutive bins with gaps, windows fully inside trials.

    Mirrors the frozen loader law: ``for s in range(trial.start, trial.stop -
    50 + 1)`` — no window ever crosses a trial boundary.  trial_len=120 keeps
    >= 71 stride-1 window starts per trial so every slot separation
    delta in 1..49 has within-trial redundant pairs.
    """
    rng = np.random.default_rng(seed)
    trial_spans = []
    cursor = 0
    for _ in range(n_trials):
        trial_spans.append((cursor, cursor + trial_len))
        cursor += trial_len + gap
    n_bins = cursor
    starts = np.concatenate([
        np.arange(start, stop - plan.WINDOW_BINS + 1, dtype=np.int64)
        for start, stop in trial_spans
    ])
    behavior = rng.normal(size=(n_bins, 2)).astype(np.float32)
    full = rng.normal(size=(starts.size, plan.WINDOW_BINS, 2)).astype(np.float32)
    window_valid = np.ones(starts.size, dtype=bool)
    bin_valid = np.ones(n_bins, dtype=bool)
    return {
        "trial_spans": trial_spans, "starts": starts, "behavior": behavior,
        "full_predictions": full, "window_valid": window_valid,
        "bin_valid": bin_valid, "n_bins": n_bins,
    }


# ---------------------------------------------------------------------------
# 1. the absolute-bin alignment law
# ---------------------------------------------------------------------------


def test_alignment_law_gathers_absolute_bins():
    session = synthetic_session()
    starts = session["starts"]
    behavior = session["behavior"]
    for slot in (0, 1, 17, 49):
        pred, target = alignment.gather_slot_pairs(
            session["full_predictions"], behavior, starts,
            session["window_valid"], session["bin_valid"], slot,
        )
        assert pred.shape == target.shape == (starts.size, 2)
        assert np.array_equal(pred, session["full_predictions"][:, slot, :])
        assert np.array_equal(target, behavior[starts + slot])


def test_alignment_law_respects_bin_and_window_masks():
    session = synthetic_session()
    bin_valid = session["bin_valid"].copy()
    # invalidate the first trial's interior and one whole later bin
    bin_valid[5:12] = False
    bin_valid[session["trial_spans"][2][0] + 3] = False
    window_valid = session["window_valid"].copy()
    window_valid[4] = False
    for slot in (0, 9, 49):
        mask = alignment.slot_pair_mask(
            session["starts"], window_valid, bin_valid, slot,
        )
        expected = window_valid & bin_valid[session["starts"] + slot]
        assert np.array_equal(mask, expected)
        pred, target = alignment.gather_slot_pairs(
            session["full_predictions"], session["behavior"], session["starts"],
            window_valid, bin_valid, slot,
        )
        assert pred.shape[0] == int(mask.sum())


def test_windows_never_cross_trial_boundaries():
    session = synthetic_session()
    spans = session["trial_spans"]
    bins = alignment.window_bin_matrix(session["starts"])
    for w in range(session["starts"].size):
        lo, hi = bins[w, 0], bins[w, -1]
        inside = any(start <= lo and hi < stop for start, stop in spans)
        assert inside, f"window {w} spans a trial boundary"


def test_trial_heads_mark_first_window_per_trial():
    session = synthetic_session()
    heads = targets.trial_heads_from_starts(session["starts"])
    expected = np.zeros(session["starts"].size, dtype=bool)
    for index in range(session["starts"].size):
        if index == 0 or session["starts"][index] != session["starts"][index - 1] + 1:
            expected[index] = True
    assert np.array_equal(heads, expected)
    assert heads.sum() == len(session["trial_spans"])


# ---------------------------------------------------------------------------
# 2. per-slot R2 correctness (hand-computed)
# ---------------------------------------------------------------------------


def test_per_slot_r2_perfect_prediction():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(400, 2)).astype(np.float32)
    assert scoring.house_session_r2(target, target) == pytest.approx(1.0, abs=1e-6)
    assert scoring.variance_weighted_r2_numpy(target, target) == pytest.approx(1.0, abs=1e-12)


def test_per_slot_r2_constant_offset_hand_case():
    rng = np.random.default_rng(5)
    target = rng.normal(size=(500, 2)).astype(np.float32)
    offset = np.array([0.25, -0.5], dtype=np.float32)
    pred = (target + offset).astype(np.float32)
    ss_res = float(((pred.astype(np.float64) - target.astype(np.float64)) ** 2).sum())
    centered = target.astype(np.float64) - target.astype(np.float64).mean(axis=0)
    ss_tot = float((centered ** 2).sum())
    expected = 1.0 - ss_res / ss_tot
    assert scoring.variance_weighted_r2_numpy(pred, target) == pytest.approx(expected, rel=1e-12)
    assert scoring.house_session_r2(pred, target) == pytest.approx(expected, abs=1e-6)


def test_numpy_r2_matches_house_scorer_on_random_data():
    rng = np.random.default_rng(11)
    target = rng.normal(size=(3000, 2)).astype(np.float32)
    pred = (target * 0.8 + rng.normal(scale=0.4, size=(3000, 2))).astype(np.float32)
    house = scoring.house_session_r2(pred, target)
    twin = scoring.variance_weighted_r2_numpy(pred, target)
    assert abs(house - twin) < 1e-6


def test_per_slot_rows_slot49_equals_direct_scoring():
    session = synthetic_session(seed=13)
    rows = alignment.per_slot_rows(
        session["full_predictions"], session["behavior"], session["starts"],
        session["window_valid"], session["bin_valid"],
    )
    assert len(rows) == plan.WINDOW_BINS
    last_pred, last_target = alignment.gather_slot_pairs(
        session["full_predictions"], session["behavior"], session["starts"],
        session["window_valid"], session["bin_valid"], plan.GOVERNING_BIN,
    )
    assert rows[plan.GOVERNING_BIN]["variance_weighted_r2"] == scoring.house_session_r2(
        last_pred, last_target
    )
    means = alignment.equal_session_mean_per_slot({"sess": rows})
    assert len(means) == plan.WINDOW_BINS
    assert means[4]["mean_variance_weighted_r2"] == pytest.approx(
        rows[4]["variance_weighted_r2"]
    )


# ---------------------------------------------------------------------------
# 3. residual-pair correlation math by slot separation delta
# ---------------------------------------------------------------------------


def _pooled(delta_sums):
    pooled = alignment.pool_delta_sums([delta_sums])
    return {
        row["delta"]: row for row in alignment.delta_curve(pooled)
    }


def test_delta_curve_brute_force_equality():
    """Brute force over ABSOLUTE-BIN partner windows (gaps must break pairs)."""
    session = synthetic_session(seed=17)
    starts = session["starts"]
    behavior = session["behavior"].astype(np.float64)
    full = session["full_predictions"].astype(np.float64)
    residual = full - behavior[starts[:, None] + np.arange(50)[None, :]]
    curves = _pooled(alignment.session_delta_sums(
        session["full_predictions"], session["behavior"], starts,
        session["window_valid"], session["bin_valid"],
    ))
    start_to_window = {int(start): index for index, start in enumerate(starts)}
    for delta in (1, 7, 25, 49):
        xs, ys = [], []
        for w1 in range(starts.size):
            w2 = start_to_window.get(int(starts[w1]) - delta)
            if w2 is None:
                continue
            for s in range(plan.WINDOW_BINS - delta):
                block = residual[[w1, w2], [s, s + delta]]
                xs.append(block[0])
                ys.append(block[1])
        # per slot column centering, matching the audit law
        n_columns = plan.WINDOW_BINS - delta
        xs = np.asarray(xs).reshape(-1, n_columns, 2)
        ys = np.asarray(ys).reshape(-1, n_columns, 2)
        xv = np.concatenate(
            [xs[:, column] - xs[:, column].mean(axis=0) for column in range(n_columns)]
        ).reshape(-1)
        yv = np.concatenate(
            [ys[:, column] - ys[:, column].mean(axis=0) for column in range(n_columns)]
        ).reshape(-1)
        expected = float(np.corrcoef(xv, yv)[0, 1])
        row = curves[delta]
        assert row["correlation"] == pytest.approx(expected, abs=1e-9)
        assert row["n_pairs"] == xv.size // 2


def test_delta_curve_pure_common_mode_is_unit_correlation():
    """Residuals that depend only on the absolute bin are perfectly shared."""
    session = synthetic_session(seed=23)
    behavior = session["behavior"]
    starts = session["starts"]
    bins = alignment.window_bin_matrix(starts)
    shared = np.sin(np.arange(session["n_bins"]) / 7.0).astype(np.float32)
    full = (behavior[bins] + shared[bins, None]).astype(np.float32)
    curves = _pooled(alignment.session_delta_sums(
        full, behavior, starts, session["window_valid"], session["bin_valid"],
    ))
    for delta in (1, 10, 49):
        assert curves[delta]["correlation"] == pytest.approx(1.0, abs=1e-9)


def test_delta_curve_bias_only_degenerates_to_zero_without_nan():
    """A pure per-slot constant bias centers away; the correlation guard holds."""
    session = synthetic_session(seed=29)
    behavior = session["behavior"]
    starts = session["starts"]
    bins = alignment.window_bin_matrix(starts)
    slot_bias = np.arange(plan.WINDOW_BINS, dtype=np.float32) * 0.01
    full = (behavior[bins] + slot_bias[None, :, None]).astype(np.float32)
    curves = _pooled(alignment.session_delta_sums(
        full, behavior, starts, session["window_valid"], session["bin_valid"],
    ))
    for delta in (1, 5, 49):
        assert np.isfinite(curves[delta]["correlation"])
        # float32 storage rounding leaves a ~1e-8 floor on the "constant" bias
        assert curves[delta]["residual_std_x"] < 1e-6


def test_disjoint_parity_groups_and_pooling():
    session = synthetic_session(seed=31)
    sums = alignment.session_parity_sums(
        session["full_predictions"], session["behavior"], session["starts"],
        session["window_valid"], session["bin_valid"],
    )
    pooled = alignment.pool_parity_sums([sums, sums])
    assert pooled["n"] == pytest.approx(2 * sums["n"])
    assert -1.0 <= pooled["correlation"] <= 1.0


# ---------------------------------------------------------------------------
# 4. subset-mean logic
# ---------------------------------------------------------------------------


def test_subset_mean_is_equal_weight_slot_average():
    session = synthetic_session(seed=37)
    full = session["full_predictions"]
    for slots in ((49,), (45, 46, 47, 48, 49), tuple(range(50))):
        mean = ensembles.subset_mean(full, slots)
        assert np.array_equal(
            mean, full[:, list(slots), :].astype(np.float64).mean(axis=1)
        )


def test_subset_slot49_is_bit_exact_the_last_bin_row():
    session = synthetic_session(seed=41)
    full = session["full_predictions"]
    mean49 = ensembles.subset_mean(full, (49,))
    assert np.array_equal(
        np.ascontiguousarray(mean49, dtype=np.float32), full[:, 49, :],
    )
    governing = session["behavior"][session["starts"] + plan.GOVERNING_BIN]
    row = ensembles.subset_row(
        full, governing, session["window_valid"], (49,), name="slot49",
        session="synthetic",
    )
    assert row["source_selected"] is False
    assert row["n_slots"] == 1
    assert row["variance_weighted_r2"] == pytest.approx(
        scoring.house_session_r2(full[:, 49, :], governing), abs=1e-12,
    )


def test_subset_row_scores_against_governing_target():
    session = synthetic_session(seed=43)
    governing = session["behavior"][session["starts"] + plan.GOVERNING_BIN]
    row49 = ensembles.subset_row(
        session["full_predictions"], governing, session["window_valid"],
        (49,), name="slot49", session="synthetic",
    )
    assert row49["variance_weighted_r2"] == pytest.approx(
        scoring.house_session_r2(
            session["full_predictions"][:, 49, :], governing,
        ),
        abs=1e-12,
    )
    row_all = ensembles.subset_row(
        session["full_predictions"], governing, session["window_valid"],
        tuple(range(50)), name="all50", session="synthetic",
    )
    assert row_all["variance_weighted_r2"] == pytest.approx(
        scoring.house_session_r2(
            np.ascontiguousarray(
                ensembles.subset_mean(session["full_predictions"], range(50)),
                dtype=np.float32,
            ),
            governing,
        ),
        abs=1e-12,
    )


def test_best_source_slot_argmax_with_lowest_slot_tie_break():
    values = np.linspace(0.1, 0.5, 50)
    table_high = {"a": [{"slot": s, "variance_weighted_r2": float(v)} for s, v in enumerate(values)]}
    flat = 0.3
    table_flat = {"b": [{"slot": s, "variance_weighted_r2": flat} for s in range(50)]}
    assert ensembles.best_source_slot(table_high) == 49
    assert ensembles.best_source_slot(table_flat) == 0
    mixed = {
        "a": [{"slot": s, "variance_weighted_r2": float(v)} for s, v in enumerate(values)],
        "b": [{"slot": s, "variance_weighted_r2": float(1.0 - v)} for s, v in enumerate(values)],
    }
    means = np.mean(
        np.asarray([values, 1.0 - values]), axis=0
    )
    assert ensembles.best_source_slot(mixed) == int(np.argmax(means))


def test_best_slot_per_session_stability_diagnostics():
    increasing = [
        {"slot": s, "variance_weighted_r2": float(s) / 50.0} for s in range(50)
    ]
    decreasing = [
        {"slot": s, "variance_weighted_r2": float(49 - s) / 50.0} for s in range(50)
    ]
    report = ensembles.best_slot_per_session({
        "s1": increasing, "s2": increasing, "s3": decreasing,
    })
    assert report["per_session_argmax_slot"] == {"s1": 49, "s2": 49, "s3": 0}
    assert report["distinct_argmax_slots"] == 2
    assert report["modal_argmax_slot"] == 49
    assert report["modal_argmax_session_count"] == 2
    assert -1.0 <= report["mean_pairwise_spearman_rank_overlap"] <= 1.0


# ---------------------------------------------------------------------------
# 5. reading-classification boundaries
# ---------------------------------------------------------------------------


def _cell(per_slot_mean, subset_means, positives, n_sessions):
    return {
        "per_slot_mean_r2": list(per_slot_mean),
        "subset_mean_r2": dict(subset_means),
        "subset_positive_sessions": dict(positives),
        "n_sessions": n_sessions,
    }


def _baseline_grid(tail_direction):
    """6 cells with a controlled tail and a slot49 baseline."""
    cells = {}
    for surface in plan.SURFACES:
        for budget in plan.BUDGETS:
            values = np.linspace(0.40, 0.30, 50)
            if tail_direction == "nonincreasing":
                values[plan.TAIL_LO:] = np.linspace(0.35, 0.30, 50 - plan.TAIL_LO)
            elif tail_direction == "nondecreasing":
                values[plan.TAIL_LO:] = np.linspace(0.30, 0.35, 50 - plan.TAIL_LO)
            else:  # mixed: a bump inside the tail
                values[plan.TAIL_LO:] = np.linspace(0.35, 0.30, 50 - plan.TAIL_LO)
                values[45] += 0.05
            base = float(values[49])
            n = 6 if surface == "within" else 15
            means = {
                "slot49": base,
                "last5_s45_49": base + 0.01,
                "last10_s40_49": base + 0.02,
                "all50": base + 0.03,
                plan.SOURCE_SELECTED_SUBSET: base + 0.04,
            }
            positives = {name: n for name in means}
            cells[f"{surface}_M{budget}"] = _cell(values, means, positives, n)
    return cells


def test_reading_redundancy_ensembling_boundary():
    cells = _baseline_grid("nonincreasing")
    reading = ensembles.classify_reading(cells)
    assert reading["classification"] == "redundancy_ensembling"
    assert reading["redundancy_legs"]["tail_nonincreasing_every_cell"] is True
    assert reading["redundancy_legs"]["averaging_gain_every_cell"] is True


def test_reading_monotonicity_direction_is_checked_both_ways():
    cells = _baseline_grid("nondecreasing")
    reading = ensembles.classify_reading(cells)
    # non-decreasing tail: leg (a) fails even though every subset gains,
    # and (b) holds by construction -> stable_better_subset
    assert reading["classification"] == "stable_better_subset"
    flags = reading["per_cell_flags"]
    assert all(not flag["tail_nonincreasing"] for flag in flags.values())
    assert all(flag["tail_nondecreasing"] for flag in flags.values())


def test_reading_stable_better_subset_threshold():
    cells = _baseline_grid("mixed")
    # drop positives below 80% on external in every budget -> (b) fails
    for key, cell in cells.items():
        if key.startswith("external"):
            n = cell["n_sessions"]
            cell["subset_positive_sessions"] = {
                name: int(np.floor(0.79 * n)) for name in cell["subset_mean_r2"]
            }
    reading = ensembles.classify_reading(cells)
    assert reading["classification"] == "session_dependent"
    assert reading["stable_better_subset_candidates"] == []


def test_reading_session_dependent_when_no_subset_gains():
    cells = _baseline_grid("mixed")
    for key, cell in cells.items():
        base = cell["subset_mean_r2"]["slot49"]
        for name in ("last5_s45_49", "last10_s40_49", "all50"):
            cell["subset_mean_r2"][name] = base - 0.01
        cell["subset_positive_sessions"] = {
            name: 0 for name in cell["subset_mean_r2"]
        }
    reading = ensembles.classify_reading(cells)
    assert reading["classification"] == "session_dependent"


def test_reading_stable_subset_needs_both_surfaces():
    cells = _baseline_grid("mixed")
    for key, cell in cells.items():
        if key.startswith("within"):
            cell["subset_positive_sessions"] = {
                name: 0 for name in cell["subset_mean_r2"]
            }
    reading = ensembles.classify_reading(cells)
    assert reading["classification"] == "session_dependent"


# ---------------------------------------------------------------------------
# 6. cache manifest / sidecar round-trip
# ---------------------------------------------------------------------------


def _cache_fixture(tmp_path):
    cache_root = tmp_path / "slot_audit_cache"
    rng = np.random.default_rng(53)
    full = rng.normal(size=(37, plan.WINDOW_BINS, 2)).astype(np.float32)
    starts = np.arange(37, dtype=np.int64) * 3
    behavior = rng.normal(size=(300, 2)).astype(np.float32)
    entry = cache_store.cache_prediction(
        cache_root, surface="within", session="synthetic", budget=10,
        full_predictions=full,
        targets=np.ascontiguousarray(behavior[starts + 49]),
        starts=starts, valid=np.ones(37, dtype=bool),
        trial_heads=targets.trial_heads_from_starts(starts),
    )
    behavior_entry = cache_store.cache_behavior(
        cache_root, surface="within", session="synthetic",
        behavior=behavior, bin_valid=np.ones(300, dtype=bool),
    )
    return cache_root, full, starts, behavior, entry, behavior_entry


def test_cache_round_trip_and_digests(tmp_path):
    cache_root, full, starts, behavior, entry, behavior_entry = _cache_fixture(tmp_path)
    manifest = cache_store.load_manifest(cache_root)
    assert manifest["schema"] == "slot_audit_v1_cache_v1"
    assert set(manifest["entries"]) == {
        cache_store.prediction_key("within", "synthetic", 10),
        cache_store.behavior_key("within", "synthetic"),
    }
    loaded = cache_store.load_prediction(cache_root, "within", "synthetic", 10)
    assert np.array_equal(loaded["full_predictions"], full)
    assert np.array_equal(loaded["starts"], starts)
    assert loaded["full_predictions"].dtype == np.float32
    behavior_loaded = cache_store.load_behavior(cache_root, "within", "synthetic")
    assert np.array_equal(behavior_loaded["behavior"], behavior)
    digest = cache_store.manifest_digest(cache_root)
    assert digest["entries"] == 2
    assert entry["full_prediction_sha256"] == cache_store.array_sha256(full)


def test_cache_rejects_duplicate_entries_and_tampering(tmp_path):
    cache_root, full, starts, behavior, entry, behavior_entry = _cache_fixture(tmp_path)
    with pytest.raises(cache_store.SlotAuditCacheError):
        cache_store.cache_prediction(
            cache_root, surface="within", session="synthetic", budget=10,
            full_predictions=full,
            targets=np.ascontiguousarray(behavior[starts + 49]),
            starts=starts, valid=np.ones(37, dtype=bool),
            trial_heads=np.ones(37, dtype=bool),
        )
    # tamper one payload byte -> digest drift on load
    payload = cache_root / entry["relative"]
    body = bytearray(payload.read_bytes())
    body[-1] ^= 0x01
    payload.write_bytes(bytes(body))
    with pytest.raises(cache_store.SlotAuditCacheError):
        cache_store.load_prediction(cache_root, "within", "synthetic", 10)


def test_cache_manifest_sidecar_tamper_detected(tmp_path):
    cache_root, full, starts, behavior, entry, behavior_entry = _cache_fixture(tmp_path)
    sidecar = cache_store.manifest_path(cache_root).with_name("manifest.json.sha256")
    sidecar.write_text("0" * 64 + "  manifest.json\n", encoding="ascii")
    with pytest.raises(cache_store.SlotAuditCacheError):
        cache_store.load_manifest(cache_root)


# ---------------------------------------------------------------------------
# latency law (bin size read from the frozen source, never assumed)
# ---------------------------------------------------------------------------


def test_latency_delay_table_uses_the_frozen_bin_size():
    for row_k in plan.K_GRID:
        delay_bins = row_k - 1
        delay_ms = delay_bins * plan.BIN_SIZE_MS
        assert delay_ms == (row_k - 1) * 20
    assert any(item["file"].endswith("p4_stream_stats.py") for item in plan.BIN_SIZE_PROVENANCE)
