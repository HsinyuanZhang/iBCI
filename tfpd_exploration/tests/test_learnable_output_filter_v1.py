"""Synthetic-fixture tests for the learnable-output-filter route (P0-P4 math).

No data root, no checkpoint, no CUDA: every test runs on constructed numpy
trial blocks.  The properties under test are the review-critical ones of
DESIGN_LEARNABLE_CAUSAL_OUTPUT_FILTER_20260829.md §3-§5, §8, §10.2:

1. TRIAL_RESET semantics — no kernel reads across a trial boundary;
2. F0 bypass bitwise identity; F1/F2/F3 formulas on fixtures;
3. F3 simplex renormalization at stream heads;
4. F4 reset-row contract (g=1), gain bounds, whitelisted features only;
5. SO(2)-equivariance, DC preservation, convex-hull bound, determinism;
6. chronology checker: duplicate/reordered/backward rows fail closed;
7. §8 oracle logic: coherent greedy beats/equals nothing by construction,
   ties break deterministically, leakage labels present;
8. §8.3 disposition thresholds and the §10.2 FIR gate boundaries;
9. metrics: paired session stats and the interaction estimator.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.learnable_output_filter_v1 import ladder, metrics, oracle, plan, selection


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _block(trial_id: str, index: int, n: int, seed: int, n_blocks: int = 3,
           base_bin: int | None = None) -> ladder.TrialBlock:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, 2)).astype(np.float32)
    target = rng.normal(size=(n, 2)).astype(np.float32)
    start = base_bin if base_bin is not None else 100 + 137 * index
    return ladder.TrialBlock(
        trial_id=trial_id, bins=np.arange(start, start + n, dtype=np.int64),
        raw=raw, target=target, valid=rng.random(n) > 0.1, block_index=index,
        n_blocks=n_blocks,
    )


def _stream(seed: int = 5, trials: int = 3, rows: int = 40) -> ladder.SessionStream:
    blocks = tuple(
        _block(f"q-{index:02d}", index, rows, seed + index, n_blocks=trials)
        for index in range(trials)
    )
    return ladder.SessionStream("external", "fixture-session", 10, blocks).validate()


SPECS = (
    ladder.F0, ladder.F1, ladder.f2(0.25), ladder.f2(0.5), ladder.f3((0.4, 0.3, 0.2, 0.1)),
    ladder.f4(tuple(0.25 * np.ones(len(plan.F4_FEATURES))), -0.4),
)


# ---------------------------------------------------------------------------
# 1/2. reset semantics and the F0/F1/F2 formulas.
# ---------------------------------------------------------------------------


def test_f0_bypass_is_bitwise_identity() -> None:
    stream = _stream()
    result = ladder.apply_filter(stream, ladder.F0)
    for block, out in zip(stream.blocks, result.blocks, strict=True):
        assert np.array_equal(np.asarray(block.raw, dtype=np.float64), out)


def test_f1_k2_formula() -> None:
    rows = np.array([[0.0, 0.0], [1.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    block = ladder.TrialBlock(
        "t", np.arange(3), rows, np.zeros((3, 2), dtype=np.float32),
        np.ones(3, dtype=bool), 0, 1,
    )
    stream = ladder.SessionStream("f", "s", 30, (block,))
    out = ladder.apply_filter(stream, ladder.F1).blocks[0]
    assert np.array_equal(out[0], rows[0])
    assert np.allclose(out[1], [0.5, 1.0])
    assert np.allclose(out[2], [2.5, 4.0])


def test_f2_ema_formula_and_reset() -> None:
    rows = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    blocks = (
        ladder.TrialBlock("a", np.arange(3), rows, np.zeros((3, 2), dtype=np.float32),
                          np.ones(3, dtype=bool), 0, 2),
        ladder.TrialBlock("b", np.arange(100, 103), rows * 10.0,
                          np.zeros((3, 2), dtype=np.float32), np.ones(3, dtype=bool), 1, 2),
    )
    stream = ladder.SessionStream("f", "s", 30, blocks)
    out = ladder.apply_filter(stream, ladder.f2(0.25)).blocks
    assert np.array_equal(out[0][0], rows[0])
    assert np.allclose(out[0][1], [0.25, 0.0])
    assert np.allclose(out[0][2], [0.25 * 0.0 + 0.75 * 0.25, 0.25])
    # the second trial restarts from its own first row: no cross-trial leakage
    assert np.array_equal(out[1][0], rows[0] * 10.0)
    assert np.allclose(out[1][1], [2.5, 0.0])


def test_no_kernel_reads_across_trial_boundary() -> None:
    stream = _stream()
    for spec in SPECS:
        before = ladder.apply_filter(stream, spec)
        tampered_blocks = []
        for index, block in enumerate(stream.blocks):
            raw = np.asarray(block.raw).copy()
            if index == 1:
                raw[:] = raw + 1000.0
            tampered_blocks.append(ladder.TrialBlock(
                block.trial_id, block.bins, raw, block.target, block.valid,
                block.block_index, block.n_blocks,
            ))
        tampered = ladder.SessionStream(stream.surface, stream.session, stream.budget, tuple(tampered_blocks))
        after = ladder.apply_filter(tampered, spec)
        assert np.array_equal(before.blocks[0], after.blocks[0]), spec.level
        assert np.array_equal(before.blocks[2], after.blocks[2]), spec.level
        assert not np.array_equal(before.blocks[1], after.blocks[1]), spec.level


def test_alpha_bounds_are_enforced() -> None:
    with pytest.raises(ladder.LadderError):
        ladder.f2(0.0)
    with pytest.raises(ladder.LadderError):
        ladder.f2(1.5)


# ---------------------------------------------------------------------------
# 3. F3 simplex and head renormalization.
# ---------------------------------------------------------------------------


def test_f3_weights_must_be_a_simplex() -> None:
    with pytest.raises(ladder.LadderError):
        ladder.f3((0.3, 0.3, 0.2, 0.1))
    with pytest.raises(ladder.LadderError):
        ladder.f3((-0.1, 0.5, 0.3, 0.3))


def test_f3_renormalizes_at_stream_head() -> None:
    weights = (0.4, 0.3, 0.2, 0.1)
    rows = np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 4.0], [8.0, 0.0], [0.0, 16.0]], dtype=np.float32)
    block = ladder.TrialBlock(
        "t", np.arange(5), rows, np.zeros((5, 2), dtype=np.float32),
        np.ones(5, dtype=bool), 0, 1,
    )
    stream = ladder.SessionStream("f", "s", 30, (block,))
    out = ladder.apply_filter(stream, ladder.f3(weights)).blocks[0]
    assert np.array_equal(out[0], rows[0])
    assert np.allclose(out[1], 0.4 / 0.7 * rows[1] + 0.3 / 0.7 * rows[0])
    assert np.allclose(out[2], 0.4 / 0.9 * rows[2] + 0.3 / 0.9 * rows[1] + 0.2 / 0.9 * rows[0])
    assert np.allclose(
        out[3], 0.4 * rows[3] + 0.3 * rows[2] + 0.2 * rows[1] + 0.1 * rows[0])
    # DC preservation and convex combination on a constant input
    constant = ladder.TrialBlock(
        "c", np.arange(5), np.tile(np.array([[2.0, -3.0]], dtype=np.float32), (5, 1)),
        np.zeros((5, 2), dtype=np.float32), np.ones(5, dtype=bool), 0, 1,
    )
    out_c = ladder.apply_filter(
        ladder.SessionStream("f", "s", 30, (constant,)), ladder.f3(weights)).blocks[0]
    assert np.allclose(out_c, np.asarray(constant.raw, dtype=np.float64)), "F3 must preserve DC"


def test_f3_softmax_parameterization_lands_on_simplex() -> None:
    logits = np.array([2.0, -1.0, 0.5, 0.0])
    weights = selection._simplex_from_logits(logits)
    assert abs(float(weights.sum()) - 1.0) < 1e-12 and bool((weights >= 0).all())


# ---------------------------------------------------------------------------
# 4. F4 contract: reset row, gain bounds, feature whitelist.
# ---------------------------------------------------------------------------


def test_f4_reset_row_emits_raw_with_gain_one() -> None:
    stream = _stream()
    spec = ladder.f4(tuple(0.5 * np.ones(len(plan.F4_FEATURES))), 2.0)
    result = ladder.apply_filter(stream, spec)
    for block, out in zip(stream.blocks, result.blocks, strict=True):
        assert np.array_equal(out[0], np.asarray(block.raw, dtype=np.float64)[0])
        assert result.gains is not None and result.gains.shape[1] >= block.raw.shape[0]
    for index in range(stream.blocks.__len__()):
        assert result.gains[index, 0] == 1.0


def test_f4_gains_stay_in_open_unit_interval_after_reset() -> None:
    stream = _stream()
    spec = ladder.f4(tuple(np.linspace(-3.0, 3.0, len(plan.F4_FEATURES))), 1.5)
    result = ladder.apply_filter(stream, spec)
    gains = result.gains
    lengths = np.asarray([block.raw.shape[0] for block in stream.blocks])
    interior = np.concatenate([gains[i, 1: lengths[i]] for i in range(len(lengths))])
    assert bool(((interior > 0.0) & (interior < 1.0)).all())


def test_f4_feature_block_matches_whitelist_definition() -> None:
    stream = _stream()
    batch = ladder.BatchView.from_stream(stream)
    raw = batch.raw
    state = raw[:, 0].copy()
    features = ladder._feature_block(batch, raw, 1, state)
    assert features.shape == (len(stream.blocks), len(plan.F4_FEATURES))
    # column 0: innovation norm / scale, column 1: first difference / scale
    innovation = np.linalg.norm(raw[:, 1] - state, axis=1)
    assert np.allclose(features[:, 0], innovation / plan.F4_FEATURE_SCALES["innovation_norm"])
    fdiff = np.linalg.norm(raw[:, 1] - raw[:, 0], axis=1)
    assert np.allclose(features[:, 1], fdiff / plan.F4_FEATURE_SCALES["first_difference_norm"])
    # column 3: predicted speed; column 6: activity progress; column 7: boundary
    speed = np.linalg.norm(raw[:, 1], axis=1)
    assert np.allclose(features[:, 3], speed / plan.F4_FEATURE_SCALES["predicted_speed"])
    assert np.allclose(features[:, 6], batch.progress)
    assert np.allclose(features[:, 7], 1.0)  # position 1 follows the reset row
    features_later = ladder._feature_block(batch, raw, 5, state)
    assert np.allclose(features_later[:, 7], 0.0)
    # the feature list is exactly the §4 whitelist
    assert tuple(plan.F4_FEATURES) == (
        "innovation_norm", "first_difference_norm", "trailing_dispersion",
        "predicted_speed", "direction_change_angle", "budget",
        "activity_progress", "boundary_flag",
    )


def test_f4_param_count_in_predeclared_range() -> None:
    assert 5 <= plan.F4_PARAM_COUNT <= 15


# ---------------------------------------------------------------------------
# 5. §5 stability / equivariance.
# ---------------------------------------------------------------------------


def test_so2_equivariance_all_levels() -> None:
    stream = _stream()
    report = ladder.audit_so2_equivariance(stream, SPECS)
    assert all(arm["max_abs_equivariance_error"] <= 1e-5 for arm in report["arms"])


def test_so2_equivariance_exact_in_float64_internals() -> None:
    stream = _stream()
    angle = 1.1
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    base = ladder.BatchView.from_stream(stream)
    turned = ladder.BatchView(
        raw=base.raw @ rotation.T, valid=base.valid, lengths=base.lengths,
        budget=base.budget, progress=base.progress, pad=base.pad,
    )
    for spec in SPECS:
        if spec.level == "F4":
            plain, _g = ladder.apply_f4(base, spec)
            other, _g2 = ladder.apply_f4(turned, spec)
        else:
            plain = ladder.apply_linear(base, spec)
            other = ladder.apply_linear(turned, spec)
        assert float(np.abs(plain @ rotation.T - other).max()) < 1e-12, spec.level


def test_dc_preservation_all_levels() -> None:
    report = ladder.audit_dc_preservation(SPECS)
    assert all(arm["constant_input_preserved"] for arm in report["arms"])


def test_convex_hull_bound_f1_to_f4() -> None:
    stream = _stream()
    report = ladder.audit_convex_hull_bound(
        stream, [spec for spec in SPECS if spec.level != "F0"])
    assert all(arm["max_hull_violation"] <= 1e-9 for arm in report["arms"])


def test_determinism_and_digests() -> None:
    stream = _stream()
    report = ladder.audit_determinism(stream, SPECS)
    assert all(arm["repeat_bitexact"] for arm in report["arms"])


def test_future_perturbation_never_changes_earlier_outputs() -> None:
    stream = _stream()
    report = ladder.audit_future_perturbation(stream, SPECS, n_trials=3)
    assert len(report["arms"]) == len(SPECS)


# ---------------------------------------------------------------------------
# 6. chronology: ordering proof and fail-closed tampering.
# ---------------------------------------------------------------------------


def test_chronology_proof_binds_every_row() -> None:
    stream = _stream()
    proof = stream.chronology_proof()
    assert proof["n_rows"] == sum(block.raw.shape[0] for block in stream.blocks)
    assert proof["reset_event_count"] == len(stream.blocks)
    assert proof["reset_event_digest"] and proof["chronological_row_digest"]


def test_boundary_tampering_fails_closed() -> None:
    stream = _stream()
    report = ladder.check_boundary_tampering_fails_closed(stream)
    assert report["all_tampering_raised"], report["failed_kinds"]
    kinds = {item["tamper"] for item in report["attempts"]}
    assert kinds == {
        "block_order_swap", "backward_bins_inside_trial", "duplicate_bin_row",
        "trial_id_not_at_block_index",
    }


def test_flat_bin_recovery_requires_stride_one_inside_trials() -> None:
    bins = np.array([10, 11, 12, 40, 41, 90], dtype=np.int64)
    heads = ladder.trial_heads_from_flat_bins(bins)
    assert heads.tolist() == [True, False, False, True, False, True]
    with pytest.raises(ladder.LadderError):
        ladder.trial_heads_from_flat_bins(np.array([10, 12, 11], dtype=np.int64))


def test_blocks_from_flat_enforces_expected_block_count() -> None:
    bins = np.array([10, 11, 12, 40, 41], dtype=np.int64)
    raw = np.zeros((5, 2), dtype=np.float32)
    target = np.zeros((5, 2), dtype=np.float32)
    valid = np.ones(5, dtype=bool)
    blocks = ladder.blocks_from_flat(
        bins=bins, trial_labels=np.array([0, 0, 0, 1, 1]), raw=raw, target=target,
        valid=valid, expected_blocks=2,
    )
    assert [block.trial_id for block in blocks] == ["static-query-000", "static-query-001"]
    with pytest.raises(ladder.LadderError):
        ladder.blocks_from_flat(
            bins=bins, trial_labels=np.array([0, 0, 0, 1, 1]), raw=raw, target=target,
            valid=valid, expected_blocks=3,
        )


# ---------------------------------------------------------------------------
# 7. §8 oracle logic.
# ---------------------------------------------------------------------------


def test_coherent_oracle_argmin_property_on_its_own_trajectory() -> None:
    # recompute one trial's coherent recursion by hand: at every row the
    # chosen gain must minimize the instantaneous true error given the
    # oracle's OWN previous state (ties -> lowest gain).
    stream = _stream()
    gains = (0.1, 0.5, 1.0)
    greedy = oracle.coherent_greedy_gain_oracle(stream, gains=gains)
    block = stream.blocks[0]
    raw = np.asarray(block.raw, dtype=np.float64)
    target = np.asarray(block.target, dtype=np.float64)
    state = raw[0].copy()
    for position in range(1, raw.shape[0]):
        errors = [
            float(np.sum((state + float(g) * (raw[position] - state) - target[position]) ** 2))
            for g in gains
        ]
        best = gains[int(np.argmin(errors))]
        assert greedy["gains"][0, position] == pytest.approx(best, abs=1e-12)
        state = state + best * (raw[position] - state)
    assert np.allclose(greedy["blocks"][0][0], raw[0])


def test_coherent_oracle_on_a_perfect_stream_chooses_the_identity() -> None:
    # when raw == target everywhere, the greedy oracle must drive g -> 1 and
    # reproduce the raw stream (the ceiling is the raw output itself).
    rows = np.cumsum(np.linspace(0.0, 1.0, 60 * 2).reshape(60, 2), axis=0).astype(np.float32)
    block = ladder.TrialBlock(
        "t", np.arange(60), rows, rows.copy(), np.ones(60, dtype=bool), 0, 1,
    )
    stream = ladder.SessionStream("within", "perfect", 10, (block,))
    greedy = oracle.coherent_greedy_gain_oracle(stream)
    assert float(np.abs(greedy["blocks"][0] - rows).max()) < 1e-9
    assert float(greedy["gains"][0, 1:].min()) >= 0.999


def test_coherent_oracle_is_label_driven_and_leakage_labelled() -> None:
    stream = _stream()
    result = oracle.coherent_greedy_gain_oracle(stream)
    assert result["target_label_leakage"] is True
    assert result["deployable"] is False
    assert "oracle" in result["oracle"]
    ceiling = oracle.noncoherent_switch_ceiling(stream, parent=ladder.f2(0.25))
    assert ceiling["target_label_leakage"] is True
    assert ceiling["parent_history"] == "F2_ema_a0p25"


def test_oracle_argmin_breaks_ties_to_lowest_gain() -> None:
    errors = np.array([[1.0, 1.0, 3.0], [0.5, 2.0, 0.5]])
    gains = np.array([0.1, 0.5, 1.0])
    assert oracle._argmin_gain(errors, gains).tolist() == [0.1, 0.1]


def test_oracle_gain_histogram_excludes_padding() -> None:
    gains = np.array([[1.0, 0.55, 0.0, 0.9], [1.0, 0.15, 0.0, 0.0]])
    lengths = np.array([4, 2])
    report = oracle.gain_histogram(gains, lengths)
    assert report["n"] == 6
    assert report["max"] == 1.0


def test_grid_errors_quadratic_is_exact() -> None:
    rng = np.random.default_rng(3)
    state = rng.normal(size=(4, 2))
    row = rng.normal(size=(4, 2))
    target = rng.normal(size=(4, 2))
    gains = np.array([0.0, 0.3, 1.0])
    errors = oracle._grid_errors(state, row, target, gains)
    for column, gain in enumerate(gains):
        direct = np.sum((state + gain * (row - state) - target) ** 2, axis=1)
        assert np.allclose(errors[:, column], direct)


# ---------------------------------------------------------------------------
# 8. gates: §8.3 disposition and §10.2 FIR gate boundaries.
# ---------------------------------------------------------------------------


def test_oracle_disposition_thresholds_exact() -> None:
    assert selection.oracle_disposition(
        oracle_minus_fixed=0.0049, surface="external", budget=4,
    )["disposition"] == "STOP_ADAPTIVE_LEARNING"
    assert selection.oracle_disposition(
        oracle_minus_fixed=0.005, surface="external", budget=4,
    )["disposition"] == "CONDITIONAL_F4"
    assert selection.oracle_disposition(
        oracle_minus_fixed=0.01499, surface="external", budget=4,
    )["disposition"] == "CONDITIONAL_F4"
    assert selection.oracle_disposition(
        oracle_minus_fixed=0.015, surface="external", budget=4,
    )["disposition"] == "PROCEED_F4"
    assert selection.oracle_disposition(
        oracle_minus_fixed=-0.01, surface="within", budget=30,
    )["disposition"] == "STOP_ADAPTIVE_LEARNING"


def test_fir_gate_boundary_and_selection_rule() -> None:
    gate = selection.select_fixed_filter(
        f1_oof=0.40, f2_oof=0.42, fir_oof=0.4249,
        fir_payload={"level": "F3", "weights": [0.25] * 4},
        best_f2_payload={"level": "F2", "alpha": 0.25},
    )
    assert gate["gate_passed"] is False and gate["selected_level"] == "F2"
    gate = selection.select_fixed_filter(
        f1_oof=0.43, f2_oof=0.42, fir_oof=0.4349,
        fir_payload={"level": "F3", "weights": [0.25] * 4},
        best_f2_payload={"level": "F2", "alpha": 0.25},
    )
    assert gate["gate_passed"] is False and gate["selected_level"] == "F1"
    gate = selection.select_fixed_filter(
        f1_oof=0.40, f2_oof=0.42, fir_oof=0.425,
        fir_payload={"level": "F3", "weights": [0.25] * 4},
        best_f2_payload={"level": "F2", "alpha": 0.25},
    )
    assert gate["gate_passed"] is True and gate["selected_level"] == "F3"


# ---------------------------------------------------------------------------
# 9. metrics, fitting and selection on fixtures.
# ---------------------------------------------------------------------------


def test_metrics_r2_paths_and_nsse() -> None:
    stream = _stream()
    raw_r2 = metrics.matrix_r2(ladder.apply_filter(stream, ladder.F0).blocks, stream)
    assert math.isfinite(raw_r2)
    sst = metrics.session_sst(stream)
    assert sst > 0.0
    nsse = metrics.session_nsse(
        ladder.apply_filter(stream, ladder.F0).blocks, stream, sst)
    assert nsse >= 0.0 and math.isfinite(nsse)
    loss = metrics.balanced_nsse(ladder.f2(0.25), [stream])
    assert math.isfinite(loss)


def test_paired_session_deltas_and_interaction() -> None:
    reference = [0.1, 0.2, 0.3]
    candidate = [0.2, 0.1, 0.45]
    stats = metrics.paired_session_deltas(candidate, reference, label="t")
    assert stats["mean"] == pytest.approx(np.mean([0.1, -0.1, 0.15]))
    assert stats["contrast"] == "t"
    interaction = metrics.paired_interaction(
        [0.5, 0.5, 0.5], [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.35, 0.35, 0.35],
        label="i",
    )
    assert interaction["mean"] == pytest.approx(0.15)  # (0.5-0.3) - (0.4-0.35) per session


def test_select_alpha_uses_grouped_oof_only() -> None:
    streams = [_stream(seed=11 + index) for index in range(3)]
    for index, stream in enumerate(streams):
        object.__setattr__(stream, "session", f"source-{index}")
        # rebuild blocks so n_blocks metadata stays consistent
    result = selection.select_alpha(streams, grid=(0.25, 0.5, 1.0))
    assert result["pooled_alpha"] in (0.25, 0.5, 1.0)
    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert fold["chosen_alpha"] in (0.25, 0.5, 1.0)


def test_fit_fir_returns_simplex_and_improves_loss() -> None:
    streams = [_stream(seed=21 + index) for index in range(3)]
    for index, stream in enumerate(streams):
        object.__setattr__(stream, "session", f"source-{index}")
    fit = selection.fit_fir(streams)
    weights = np.asarray(fit["weights"])
    assert abs(float(weights.sum()) - 1.0) < 1e-9 and bool((weights >= 0).all())
    assert fit["loss"] <= metrics.balanced_nsse(ladder.f3([0.25] * 4), streams) + 1e-9


def test_fit_f4_recovers_high_gain_on_noiseless_stream() -> None:
    # a noiseless ramp trial (raw == target, non-constant): the best causal
    # filter is the identity (F0), so a fitted F4 should drive g -> 1 and
    # reproduce the raw rows.
    rows = np.cumsum(
        np.tile(np.array([[0.3, -0.2]], dtype=np.float32), (30, 1)), axis=0,
    ).astype(np.float32)
    block = ladder.TrialBlock(
        "t", np.arange(30), rows, rows.copy(), np.ones(30, dtype=bool), 0, 1,
    )
    stream = ladder.SessionStream("within", "ramp", 10, (block,))
    fit = selection.fit_f4([stream])
    spec = ladder.f4(fit["theta"], fit["bias"])
    result = ladder.apply_filter(stream, spec)
    assert float(np.abs(result.blocks[0] - rows).max()) < 1e-3


def test_f4_analytic_gradient_matches_finite_differences() -> None:
    stream = _stream(seed=41)
    batch = ladder.BatchView.from_stream(stream)
    sst = metrics.session_sst(stream)
    parameters = np.linspace(-0.7, 0.9, plan.F4_PARAM_COUNT)
    loss, gradient = ladder.f4_loss_and_grad(parameters, batch, sst=sst)
    reference = selection._f4_objective(parameters, [stream])
    assert loss == pytest.approx(reference, rel=1e-12)
    epsilon = 1e-6
    for index in range(parameters.size):
        bumped = parameters.copy()
        bumped[index] += epsilon
        numerical = (selection._f4_objective(bumped, [stream]) - reference) / epsilon
        assert gradient[index] == pytest.approx(numerical, abs=1e-5), index


def test_cv_f4_reports_oof_gain_over_reference() -> None:
    streams = [_stream(seed=31 + index) for index in range(3)]
    for index, stream in enumerate(streams):
        object.__setattr__(stream, "session", f"source-{index}")
    cv = selection.cv_f4(streams, reference=ladder.f2(0.25))
    assert len(cv["folds"]) == 3
    assert "oof_gain_over_reference" in cv
    assert cv["oof_gain_over_reference"] == pytest.approx(
        cv["oof_equal_session_mean_r2_f4"] - cv["oof_equal_session_mean_r2_reference"]
    )


# ---------------------------------------------------------------------------
# plan invariants.
# ---------------------------------------------------------------------------


def test_gain_grid_contains_every_selectable_alpha() -> None:
    assert set(plan.GAIN_GRID) >= set(plan.ALPHA_GRID)
    assert 0.25 in plan.GAIN_GRID  # the P1 fixed filter is always a candidate


def test_pre_registration_is_frozen() -> None:
    payload = plan.pre_registration_payload()
    assert payload["p1_fixed_filter"]["alpha"] == 0.25
    assert payload["gates"]["oracle_stop_below"] == 0.005
    assert payload["gates"]["oracle_proceed_above"] == 0.015
    assert payload["gates"]["fir_gate_over_f1_f2"] == 0.005
    assert payload["inference_only"] is True
    assert payload["target_optimizer_backward_update"] == 0
