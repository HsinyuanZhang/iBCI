"""Tests for P4 — label-free eval-stream activity statistics.

Covers the pre-registered machinery exactly as the work order demands:
  - causal-prefix audit (no future leakage; a tampered fixture MUST fail),
  - running-statistics math vs brute-force recomputation,
  - identity-refresh semantics through the REAL B3S encoder (warm start
    bit-exact, growing-pool mean, frozen finalize),
  - the reused ledger/C3 anchors (strict deployment-recipe mapping),
  - the pre-registered verdict thresholds and the TTA labelling discipline.

The only real-artifact dependency is the sealed receipt set loaded through
the SHA-verified ledger plus the pinned Z1 receipt (same convention as
test_calibration_gap_v1.py / test_calibration_gap_z1_z4.py).
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
STREAMING_ENCODERS = (
    REPO_ROOT / "streaming_calibration_exp/src/models/components/streaming_encoders.py"
)


def _load_pkg():
    spec = importlib.util.spec_from_file_location(
        "calibration_gap_v1_p", ROOT / "src/calibration_gap_v1/__init__.py",
        submodule_search_locations=[str(ROOT / "src/calibration_gap_v1")],
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["calibration_gap_v1_p"] = pkg
    spec.loader.exec_module(pkg)
    return pkg


def _load_p4():
    spec = importlib.util.spec_from_file_location(
        "calibration_gap_p4_stream_stats", ROOT / "src/calibration_gap_v1/p4_stream_stats.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["calibration_gap_p4_stream_stats"] = module
    spec.loader.exec_module(module)
    return module


def _load_streaming_encoders():
    spec = importlib.util.spec_from_file_location("p4_test_streaming_encoders", STREAMING_ENCODERS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["p4_test_streaming_encoders"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pkg():
    return _load_pkg()


@pytest.fixture(scope="module")
def p4():
    return _load_p4()


@pytest.fixture(scope="module")
def ledger(pkg):
    return pkg.load_ledger()


# ---------------------------------------------------------------------------
# causal-prefix audit (block schedule)
# ---------------------------------------------------------------------------


def test_block_schedule_partitions_and_prefixes_are_strictly_causal(p4):
    starts = np.asarray([100, 105, 2100, 2200, 4100, 4300], dtype=np.int64)
    blocks = p4.block_schedule(starts, calib_end_bin=100, stride=2000)
    assert [b["index"] for b in blocks] == [0, 1, 2]
    assert [b["window_lo"] for b in blocks] == [0, 2, 4]
    assert [b["window_hi"] for b in blocks] == [2, 4, 6]
    # block 0 has the EMPTY warm-start prefix; every prefix end is <= every
    # window start inside its own block
    assert blocks[0]["prefix_end_bin"] == 100
    for block in blocks:
        window_starts = starts[block["window_lo"]:block["window_hi"]]
        assert int(window_starts.min()) >= block["prefix_end_bin"]
        assert block["min_prefix_margin_bins"] >= 0
    # a window starting exactly at a refresh mark joins the LATER block with
    # margin 0: the prefix [e, mark) never contains that window's first bin
    assert blocks[1]["prefix_end_bin"] == 100 + 2000
    assert blocks[1]["first_window_start"] == 2100
    assert blocks[1]["min_prefix_margin_bins"] == 0
    assert blocks[2]["min_prefix_margin_bins"] == 0
    audit = p4.audit_block_causality(blocks, starts)
    assert audit["blocks"] == 3 and audit["min_prefix_margin_bins"] == 0


def test_block_schedule_fails_on_unsorted_or_pre_stream_windows(p4):
    with pytest.raises(p4.P4Error):
        p4.block_schedule(np.asarray([300, 200]), calib_end_bin=100, stride=2000)
    with pytest.raises(p4.P4Error):
        p4.block_schedule(np.asarray([50, 300]), calib_end_bin=100, stride=2000)


def test_causality_audit_rejects_tampered_prefix(p4):
    """The off-by-one audit: a prefix reaching into its own window MUST fail."""
    starts = np.asarray([100, 120, 2100, 2110], dtype=np.int64)
    blocks = p4.block_schedule(starts, calib_end_bin=100, stride=2000)
    p4.audit_block_causality(blocks, starts)  # untampered passes

    off_by_one = copy.deepcopy(blocks)
    off_by_one[1]["prefix_end_bin"] += 1  # now 2101 > first window start 2100
    with pytest.raises(p4.P4Error, match="future leakage"):
        p4.audit_block_causality(off_by_one, starts)

    into_next_block = copy.deepcopy(blocks)
    into_next_block[1]["prefix_end_bin"] = 2110  # reaches the second window
    with pytest.raises(p4.P4Error, match="future leakage"):
        p4.audit_block_causality(into_next_block, starts)

    reordered = copy.deepcopy(blocks)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(p4.P4Error):
        p4.audit_block_causality(reordered, starts)


def test_complete_pseudo_trial_count_is_floor_of_strict_prefix(p4):
    assert p4.complete_pseudo_trial_count(100, 100) == 0
    assert p4.complete_pseudo_trial_count(199, 100) == 0
    assert p4.complete_pseudo_trial_count(200, 100) == 1
    assert p4.complete_pseudo_trial_count(2100, 100) == 20
    with pytest.raises(p4.P4Error):
        p4.complete_pseudo_trial_count(50, 100)


# ---------------------------------------------------------------------------
# running-statistics math vs brute force
# ---------------------------------------------------------------------------


def test_stream_bin_running_sums_match_brute_force(p4):
    rng = np.random.default_rng(11)
    neural = rng.poisson(1.5, size=(5_000, 9)).astype(np.float32)
    e = 137
    limits = [e, e, e + 2_000, e + 2_001, e + 4_137, neural.shape[0]]
    snaps = p4.stream_bin_running_sums(neural, e, limits)
    for limit in set(limits):
        snap = snaps[limit]
        assert snap["count"] == limit - e
        brute = np.zeros(9, dtype=np.float64)
        for value in neural[e:limit].astype(np.float64):
            brute = brute + value
        assert np.allclose(snap["sum"], brute, rtol=0, atol=1e-9)
        if limit > e:
            cum = np.cumsum(neural[e:limit].astype(np.float64), axis=0)
            assert np.allclose(snap["sum"], cum[-1], rtol=1e-9, atol=1e-6)
    # duplicates collapse to the same snapshot and zero-width prefix is empty
    assert snaps[e]["count"] == 0 and float(np.abs(snaps[e]["sum"]).max()) == 0.0


def test_stream_running_sums_reject_tampered_limits(p4):
    rng = np.random.default_rng(12)
    neural = rng.normal(size=(600, 4)).astype(np.float32)
    with pytest.raises(p4.P4Error):
        p4.stream_bin_running_sums(neural, 100, [601])  # past the session end
    with pytest.raises(p4.P4Error):
        p4.stream_bin_running_sums(neural, 100, [99])   # before the stream starts


def test_stream_gain_math_guards_and_clip(p4):
    stream_sum = np.asarray([10.0, 0.0, 4.0, 90.0])
    calib_mean = np.asarray([5.0, 2.0, 1e-9, 10.0])
    gain, diag = p4.stream_normalization_gain(stream_sum, 10, calib_mean)
    assert gain[0] == pytest.approx(0.2)          # (10/10 bins) / 5.0
    assert gain[1] == pytest.approx(0.1)          # zero stream rate -> clipped low
    assert gain[2] == pytest.approx(1.0)          # calib rate <= eps -> guard
    assert gain[3] == pytest.approx(0.9)          # (90/10) / 10
    assert diag["n_units_calib_rate_guarded"] == 1
    assert diag["n_units_clipped_low"] == 1
    with pytest.raises(p4.P4Error):
        p4.stream_normalization_gain(stream_sum, 0, calib_mean)
    big = np.asarray([1.0, 1e5])
    small = np.asarray([1.0, 1.0])
    clipped, diag2 = p4.stream_normalization_gain(big, 1, small)
    assert clipped[1] == p4.STREAM_GAIN_CLIP[1]
    assert diag2["n_units_clipped_high"] == 1


# ---------------------------------------------------------------------------
# identity-refresh semantics through the REAL B3S encoder
# ---------------------------------------------------------------------------


def _small_encoder():
    torch = pytest.importorskip("torch")
    encoders = _load_streaming_encoders()
    torch.manual_seed(0)
    encoder = encoders.build_encoder(
        "B3S", window_size=50, trial_length=100, id_hidden_dim=8, hidden_dim=4,
        side_dim=4,
    )
    encoder.eval()
    return encoder


def test_calibration_pooled_sums_bit_exact_forward_batch(p4):
    torch = pytest.importorskip("torch")
    encoder = _small_encoder()
    rng = np.random.default_rng(13)
    calib = torch.from_numpy(rng.poisson(1.0, size=(10, 100, 6)).astype(np.float32))
    side = torch.from_numpy(rng.normal(size=(1, 6, 4)).astype(np.float32))
    state = p4._calibration_pooled_sums(encoder, calib[:4])
    assert state["trial_count"] == 4
    identity_state = encoder.finalize_identity({
        "sum_feat": state["sum_feat"], "trial_count": state["trial_count"],
        "side_features": side,
    })
    identity_ref = encoder.forward_batch(calib[:4].unsqueeze(0), side_features=side)
    # bit-exact: the P4 pooled path IS the frozen encoder's own accumulation
    assert torch.equal(identity_state, identity_ref)


def test_stream_pseudo_trial_sums_match_push_order_and_causality(p4):
    torch = pytest.importorskip("torch")
    encoder = _small_encoder()
    rng = np.random.default_rng(14)
    neural = rng.poisson(1.2, size=(1_200, 6)).astype(np.float32)
    e = 50
    max_count = 10
    snaps = p4.stream_pseudo_trial_running_sums(encoder, neural, e, max_count)
    assert len(snaps) == max_count + 1
    # snapshot k equals pushing exactly the first k pseudo-trials
    running = torch.zeros(1, 6, encoder.hidden_dim)
    with torch.no_grad():
        for k in range(max_count + 1):
            assert torch.equal(snaps[k], running)
            if k < max_count:
                trial = torch.from_numpy(
                    neural[e + k * 100:e + (k + 1) * 100]
                ).unsqueeze(0).permute(0, 2, 1)
                running = running + encoder.pre_pool(trial)
    # snapshots never depend on bins beyond the consumed prefix
    tampered = neural.copy()
    tampered[e + max_count * 100 + 5, 0] += 100.0  # strictly after the prefix
    snaps2 = p4.stream_pseudo_trial_running_sums(encoder, tampered, e, max_count)
    for k in range(max_count + 1):
        assert torch.equal(snaps[k], snaps2[k])
    # ReLU features are non-negative -> the running sums are monotone
    stacked = torch.stack(snaps)
    assert bool((stacked[1:] - stacked[:-1] >= -1e-7).all())


def test_p4b_pooling_semantics_warm_start_and_growth(p4):
    torch = pytest.importorskip("torch")
    encoder = _small_encoder()
    rng = np.random.default_rng(15)
    calib = torch.from_numpy(rng.poisson(1.0, size=(10, 100, 6)).astype(np.float32))
    side = torch.from_numpy(rng.normal(size=(1, 6, 4)).astype(np.float32))
    neural = rng.poisson(2.0, size=(700, 6)).astype(np.float32)
    e = 100
    state = p4._calibration_pooled_sums(encoder, calib[:4])
    calib_sum, n_calib = state["sum_feat"], state["trial_count"]
    snaps = p4.stream_pseudo_trial_running_sums(encoder, neural, e, 6)
    with torch.no_grad():
        warm = encoder.finalize_identity({
            "sum_feat": calib_sum + snaps[0], "trial_count": n_calib,
            "side_features": side,
        })
        strict = encoder.forward_batch(calib[:4].unsqueeze(0), side_features=side)
        assert torch.equal(warm, strict)  # bit-exact warm start (empty prefix)

        grown = encoder.finalize_identity({
            "sum_feat": calib_sum + snaps[6], "trial_count": n_calib + 6,
            "side_features": side,
        })
        # the pooled mean is over M calibration trials + 6 stream pseudo-trials
        trials = [calib[i] for i in range(4)] + [
            torch.from_numpy(neural[e + k * 100:e + (k + 1) * 100]) for k in range(6)
        ]
        feats = torch.stack([
            encoder.pre_pool(t.unsqueeze(0).permute(0, 2, 1)) for t in trials
        ])
        manual = encoder.post_pool(
            torch.cat([feats.mean(dim=0), side], dim=-1)
        )
        assert torch.allclose(grown, manual, atol=1e-5)
        assert not torch.allclose(grown, strict, atol=1e-4)


# ---------------------------------------------------------------------------
# anchors reused (ledger + pinned Z1 receipt)
# ---------------------------------------------------------------------------


def test_strict_receipt_cell_maps_the_frozen_deployment_recipe(p4, ledger):
    expected = {"external": {4: 0.1197, 10: 0.2955, 30: 0.4286},
                "within": {4: 0.3089, 10: 0.4677, 30: 0.5665}}
    for surface in ("external", "within"):
        for budget in (4, 10, 30):
            rel, cell = p4.strict_receipt_cell(ledger, surface, budget)
            value = float(cell["summary"]["equal_session_mean_r2"])
            assert value == pytest.approx(expected[surface][budget], abs=1e-3), (
                surface, budget, rel, value,
            )


def test_z1_receipt_loads_only_with_the_pinned_sha(p4):
    body = p4.load_z1_receipt(ROOT)
    assert set(body["ladders"]) == {
        f"{s}_M{b}" for s in ("external", "within") for b in (4, 10, 30)
    }
    assert body["ladders"]["external_M4"]["activity_cost_of_oracle_at_m_budget"] == (
        pytest.approx(0.2036, abs=1e-3)
    )


def _z1_anchor_body():
    return {
        "cells": [
            {"surface": "external", "budget": 4, "sessions": [
                {"session": "s1", "variance_weighted_r2": 0.5,
                 "calibration_m30_sha256": "c1",
                 "target_sha256": "t1",
                 "normalized_side_sha256": "side1", "prediction_sha256": "p1"},
            ]},
        ]
    }


def _z1_anchor_row(**override):
    row = {
        "surface": "external", "budget": 4, "session": "s1",
        "variant": "anchor_c3_z1_reproduction",
        "variance_weighted_r2": 0.5 + 1e-9,
        "neural_sha256": "n1", "calibration_m30_sha256": "c1",
        "target_sha256": "t1", "valid_mask_sha256": "m1",
        "normalized_side_sha256": "side1", "prediction_sha256": "p1",
    }
    row.update(override)
    return row


def test_check_z1_anchor_accepts_and_flags_bitexactness(p4):
    report = p4.check_z1_anchor([_z1_anchor_row()], _z1_anchor_body())
    assert report["sessions_checked"] == 1
    assert report["prediction_sha256_bitexact_matches"] == 1
    assert report["max_abs_delta_r2"] <= p4.Z1_ANCHOR_TOLERANCE


def test_check_z1_anchor_rejects_input_sha_and_r2_drift(p4):
    with pytest.raises(p4.P4Error):
        p4.check_z1_anchor([_z1_anchor_row(calibration_m30_sha256="tampered")], _z1_anchor_body())
    with pytest.raises(p4.P4Error):
        p4.check_z1_anchor([_z1_anchor_row(normalized_side_sha256="x")], _z1_anchor_body())
    with pytest.raises(p4.P4Error):
        p4.check_z1_anchor(
            [_z1_anchor_row(variance_weighted_r2=0.4)], _z1_anchor_body()
        )


class _FakeLedger:
    """Only ``cell``; enough for check_strict_anchor / readings."""

    def __init__(self, cell):
        self._cell = cell

    def cell(self, rel, **match):
        return self._cell


def _strict_cell():
    return {
        "summary": {"equal_session_mean_r2": 0.12},
        "sessions": [
            {"session": "s1", "r2": 0.12, "selected_indices_sha256": "sel1",
             "ridge_fit": {"raw_t4_sha256": "raw1"}},
        ],
    }


def _strict_row(**override):
    row = {
        "surface": "external", "budget": 4, "session": "s1",
        "variant": "strict_fss_baseline", "variance_weighted_r2": 0.1200001,
        "selected_indices_sha256": "sel1", "raw_t4_sha256": "raw1",
    }
    row.update(override)
    return row


def test_check_strict_anchor_verifies_selection_carrier_and_r2(p4):
    ledger = _FakeLedger(_strict_cell())
    report = p4.check_strict_anchor(
        [_strict_row()], ledger, surfaces=("external",), budgets=(4,),
    )
    assert report["sessions_checked"] == 1
    assert report["max_abs_delta_r2"] <= p4.STRICT_ANCHOR_TOLERANCE
    assert report["per_budget"]["external_M4"]["selection_sha_checked"] is True


def test_check_strict_anchor_accepts_rows_built_from_the_real_receipts(p4, ledger):
    """Full-roster acceptance: rows transcribed from the sealed cells pass.

    This exercises both receipt schemas (factorial ``ridge_fit`` at M4/M10,
    comparators ``fit`` at M30) across both surfaces and all 21 sessions.
    """
    rows = []
    for surface in ("external", "within"):
        for budget in (4, 10, 30):
            _rel, cell = p4.strict_receipt_cell(ledger, surface, budget)
            for anchor in cell["sessions"]:
                rows.append({
                    "surface": surface, "budget": budget,
                    "session": anchor["session"],
                    "variant": "strict_fss_baseline",
                    "variance_weighted_r2": float(anchor["r2"]),
                    "selected_indices_sha256": anchor.get("selected_indices_sha256"),
                    "raw_t4_sha256": p4._receipt_carrier_sha(anchor, budget),
                })
    report = p4.check_strict_anchor(rows, ledger)
    assert report["sessions_checked"] == (15 + 6) * 3  # sessions x budgets
    assert report["max_abs_delta_r2"] == 0.0
    assert all(entry["receipt_cell_aggregate_equal_session_mean_r2"] > 0
               for entry in report["per_budget"].values())


def test_check_strict_anchor_rejects_carrier_and_selection_drift(p4):
    ledger = _FakeLedger(_strict_cell())
    with pytest.raises(p4.P4Error):
        p4.check_strict_anchor(
            [_strict_row(raw_t4_sha256="tampered")], ledger,
            surfaces=("external",), budgets=(4,),
        )
    with pytest.raises(p4.P4Error):
        p4.check_strict_anchor(
            [_strict_row(selected_indices_sha256="tampered")], ledger,
            surfaces=("external",), budgets=(4,),
        )
    with pytest.raises(p4.P4Error):
        p4.check_strict_anchor(
            [_strict_row(variance_weighted_r2=0.05)], ledger,
            surfaces=("external",), budgets=(4,),
        )


# ---------------------------------------------------------------------------
# paired deltas, TTA labelling, verdict thresholds
# ---------------------------------------------------------------------------


def test_cell_matrix_labels_every_variant_tta_or_fss(p4):
    matrix = p4.cell_matrix()
    assert len(matrix) == 2 * 3 * 4
    for cell in matrix:
        if cell["variant"].startswith("p4"):
            assert cell["data_use_class"] == "TTA" and cell["transductive"] is True
        else:
            assert cell["data_use_class"] in ("FSS", "FSS_leakage_diagnostic")
            assert cell["transductive"] is False
    strict_cells = [c for c in matrix if c["variant"] == "strict_fss_baseline"]
    assert all("deployable" in c["carrier"] for c in strict_cells)
    assert all(c["activity"] == f"b3s_selected_{c['budget']}_calibration_trials"
                for c in strict_cells)


def test_paired_deltas_require_identical_roster_and_label_contrast(p4):
    cells = _cells_with_deltas(p4, {("external", 4): 0.12, ("external", 10): 0.05,
                                    ("external", 30): 0.0, ("within", 4): 0.1,
                                    ("within", 10): 0.03, ("within", 30): -0.001})
    pairs = p4.paired_deltas(cells)
    assert len(pairs) == 2 * 3 * 2
    for entry in pairs:
        assert entry["data_use_class"] == "TTA_minus_FSS_paired_within_harness"
        assert entry["n_total"] == 5 and len(entry["all_deltas"]) == 5
    tampered = copy.deepcopy(cells)
    tampered[0]["sessions"] = tampered[0]["sessions"][::-1]
    with pytest.raises(p4.P4Error):
        p4.paired_deltas(tampered)


def _cells_with_deltas(p4, delta_map):
    cells = []
    for surface in ("external", "within"):
        for budget in (4, 10, 30):
            base = [0.10 + 0.02 * i for i in range(5)]
            dd = float(delta_map[(surface, budget)])
            for variant, multiplier in (
                ("strict_fss_baseline", 0.0),
                ("p4a_stream_norm", 0.2),
                ("p4b_stream_identity", 1.0),
            ):
                rows = [
                    {"session": f"s{i}", "surface": surface, "budget": budget,
                     "variant": variant,
                     "variance_weighted_r2": base[i] + dd * multiplier + 1e-4 * i}
                    for i in range(5)
                ]
                cells.append({
                    "surface": surface, "budget": budget, "variant": variant,
                    "data_use_class": p4.DATA_USE_CLASS[variant],
                    "transductive": p4.TRANSDUCTIVE[variant],
                    "sessions": rows,
                    "summary": {"equal_session_mean_r2": float(np.mean(
                        [r["variance_weighted_r2"] for r in rows]))},
                })
    return cells


def test_readings_verdict_thresholds(p4, ledger):
    z1_body = p4.load_z1_receipt(ROOT)

    def payload_with(m4_delta, m10_delta, m30_delta):
        cells = _cells_with_deltas(p4, {
            ("external", 4): m4_delta, ("external", 10): m10_delta,
            ("external", 30): m30_delta, ("within", 4): m4_delta * 0.5,
            ("within", 10): m10_delta * 0.5, ("within", 30): m30_delta * 0.5,
        })
        return {"cells": cells, "paired_deltas": p4.paired_deltas(cells)}

    major = p4.readings(payload_with(0.15, 0.05, 0.001), ledger, z1_body)
    assert "substantial share" in major["verdicts"]["primary_p4b_external_M4"]
    assert "monotone decay" in major["verdicts"]["mechanism_monotonicity"]
    assert 0.0 < major["p4b_external_M4"]["recovered_share_of_z1_activity_ceiling"]
    partial = p4.readings(payload_with(0.05, 0.02, 0.0), ledger, z1_body)
    assert "partial share" in partial["verdicts"]["primary_p4b_external_M4"]
    none = p4.readings(payload_with(0.01, 0.005, 0.0), ledger, z1_body)
    assert "does not close" in none["verdicts"]["primary_p4b_external_M4"]
    negative = p4.readings(payload_with(-0.068, -0.125, -0.200), ledger, z1_body)
    assert "actively degrades" in negative["verdicts"]["primary_p4b_external_M4"]
    assert "monotone across budgets" in negative["verdicts"]["mechanism_monotonicity"]
    assert "NOT ~ 0" in negative["verdicts"]["mechanism_monotonicity"]
    assert "does not help" in negative["verdicts"]["p4a_vs_p4b_attribution"]
    violation = p4.readings(payload_with(0.02, 0.03, 0.09), ledger, z1_body)
    assert "MONOTONICITY VIOLATION" in violation["verdicts"]["mechanism_monotonicity"]


def test_readings_attributions_and_loaded_ceilings(p4, ledger):
    z1_body = p4.load_z1_receipt(ROOT)
    cells = _cells_with_deltas(p4, {
        ("external", 4): 0.10, ("external", 10): 0.04, ("external", 30): 0.002,
        ("within", 4): 0.08, ("within", 10): 0.03, ("within", 30): 0.0,
    })
    out = p4.readings({"cells": cells, "paired_deltas": p4.paired_deltas(cells)},
                      ledger, z1_body)
    assert "volume fix dominates" in out["verdicts"]["p4a_vs_p4b_attribution"]
    ceilings = out["ceilings_loaded"]
    assert ceilings["z1_matched_activity_ceiling_external_M4"] == pytest.approx(0.2036, abs=1e-3)
    assert ceilings["deployable_carrier_activity_ceiling_external_M4"] == pytest.approx(
        0.0705, abs=5e-3
    )
    assert out["strict_baselines_receipt_loaded"]["external_M4"] == pytest.approx(
        0.1197, abs=1e-3
    )


# ---------------------------------------------------------------------------
# module hygiene (binding conditions declared before results)
# ---------------------------------------------------------------------------


def test_p4_module_declares_binding_conditions_and_labels(p4):
    text = Path(p4.__file__).read_text()
    for needle in (
        "CAUSALITY", "CLASS DECLARATION", "transductive", "TTA",
        "Z6-Q1", "warm start", "REFRESH_BIN_STRIDE", "P4B_MAJOR_DELTA",
        "never overwritten", "leakage",
    ):
        assert needle in text, needle
