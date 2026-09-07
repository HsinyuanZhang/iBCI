"""Tests for the continuity probe v1 (output-level smoothing arms).

Covers exactly what the work order demands:
  - CAUSALITY AUDIT: a tampered fixture where FUTURE windows are perturbed
    must not change earlier outputs of any causal arm (bit-exact), while the
    trajectory-aligned family — declared non-causal — MUST react (positive
    control) and must stay invariant strictly before its declared support;
  - trajectory-alignment math on synthetic fixtures (brute-force equality,
    gap handling, K=1 identities, the naive whole-window-mean identity);
  - the anchor discipline: the P4 receipt loads only with its pinned SHA, the
    baseline anchor accepts receipt-faithful rows and rejects drift;
  - the pre-registered reading thresholds and verdict wording.

The only real-artifact dependency is the pinned P4 receipt (SHA-verified);
no session data, no model, no GPU.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "continuity_probe_v1_test", ROOT / "src/continuity_probe_v1.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["continuity_probe_v1_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe():
    return _load_probe()


def _fixture(n=64, seed=7, stride_pattern="unit"):
    """Synthetic [W, 50, 2] predictions + starts (with gaps in the gapped case)."""
    rng = np.random.default_rng(seed)
    if stride_pattern == "unit":
        starts = np.arange(n, dtype=np.int64)
    elif stride_pattern == "gapped":
        starts = np.cumsum(
            rng.choice([1, 1, 1, 1, 7, 13], size=n)
        ).astype(np.int64)
    else:
        raise ValueError(stride_pattern)
    pred = rng.normal(size=(n, 50, 2))
    return starts, pred


# ---------------------------------------------------------------------------
# causal window-index family
# ---------------------------------------------------------------------------


def test_causal_mean_matches_brute_force(probe):
    starts, pred = _fixture()
    last = pred[:, 49, :]
    for k in probe.K_GRID:
        weights = probe.plain_weights(k)
        out = probe.smooth_causal_window(last, weights)
        reference = np.empty_like(out)
        for t in range(last.shape[0]):
            lo = max(0, t - k + 1)
            reference[t] = last[lo:t + 1].mean(axis=0)
        assert np.allclose(out, reference, rtol=0, atol=1e-12)


def test_causal_exp_matches_brute_force_with_head_renormalization(probe):
    starts, pred = _fixture()
    last = pred[:, 49, :]
    for alpha in probe.ALPHA_GRID:
        support = probe.exponential_support(alpha)
        weights = probe.exponential_weights(alpha, support)
        assert abs(sum(weights) - 1.0) < 1e-12
        out = probe.smooth_causal_window(last, weights)
        reference = np.empty_like(out)
        for t in range(last.shape[0]):
            ages = np.arange(0, min(support, t + 1))
            w = np.asarray([alpha * (1 - alpha) ** d for d in ages])
            w = w / w.sum()
            rows = last[t - ages]
            reference[t] = (w[:, None] * rows).sum(axis=0)
        assert np.allclose(out, reference, rtol=0, atol=1e-12)


def test_causal_K1_is_identity(probe):
    starts, pred = _fixture()
    last = pred[:, 49, :]
    out = probe.smooth_causal_window(last, [1.0])
    assert np.array_equal(out, last)


def test_naive_whole_trajectory_mean_identity(probe):
    """bin-49 of a mean of whole trajectories == mean of the bin-49s (K grid)."""
    starts, pred = _fixture()
    last = pred[:, 49, :]
    for k in probe.K_GRID:
        traj_mean = probe.smooth_causal_window(
            pred.reshape(pred.shape[0], -1), probe.plain_weights(k)
        ).reshape(pred.shape[0], 50, 2)[:, 49, :]
        last_mean = probe.smooth_causal_window(last, probe.plain_weights(k))
        assert np.allclose(traj_mean, last_mean, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------------
# trajectory-aligned family
# ---------------------------------------------------------------------------


def test_trajalign_matches_brute_force_unit_stride(probe):
    """With stride 1, out[t] = sum_d w_d * pred[t+d, 49-d] / sum w_d."""
    starts, pred = _fixture()
    for k in probe.K_GRID:
        out, valid = probe.smooth_trajectory_aligned(
            pred, starts, probe.plain_weights(k)
        )
        assert bool(valid[: pred.shape[0] - k].all())
        weights = probe.plain_weights(k)
        reference = np.empty_like(out)
        for t in range(pred.shape[0]):
            leads = np.arange(0, min(k, pred.shape[0] - t))
            w = np.asarray(weights[: leads.size])
            w = w / w.sum()
            rows = np.stack(
                [pred[t + d, 49 - d] for d in leads]
            )
            reference[t] = (w[:, None] * rows).sum(axis=0)
        assert np.allclose(out, reference, rtol=0, atol=1e-12)


def test_trajalign_respects_gaps_by_exact_containment(probe):
    """A window that does not CONTAIN the scored bin must not contribute."""
    starts, pred = _fixture(stride_pattern="gapped")
    k = 4
    out, valid = probe.smooth_trajectory_aligned(
        pred, starts, probe.plain_weights(k)
    )
    bin_t = starts + 49
    for t in range(pred.shape[0] - k):
        contributors = []
        for d in range(k):
            tp = t + d
            if starts[tp] <= bin_t[t] < starts[tp] + 50:
                contributors.append(pred[tp, bin_t[t] - starts[tp]])
        assert valid[t, : len(contributors)].all()
        assert not valid[t, len(contributors):].any() or len(contributors) == k
        expected = np.mean(contributors, axis=0)
        assert np.allclose(out[t], expected, rtol=0, atol=1e-12)


def test_trajalign_K1_is_raw_last_bin(probe):
    starts, pred = _fixture()
    out, _ = probe.smooth_trajectory_aligned(pred, starts, [1.0])
    assert np.array_equal(out, pred[:, 49, :])


def test_trajalign_exp_support_capped_at_window(probe):
    for alpha in probe.ALPHA_GRID:
        support = probe.exponential_support(alpha)
        capped = min(support, probe.TRAJALIGN_MAX_SUPPORT)
        arm = next(
            a for a in probe.arm_matrix()
            if a["arm"] == f"trajalign_exp_a{alpha}"
        )
        assert arm["support"] == capped
        assert arm["support"] <= 49  # relative position 49-d must stay >= 0


# ---------------------------------------------------------------------------
# causality audit (the tampered-fixture proof)
# ---------------------------------------------------------------------------


def test_causality_future_perturbation_leaves_causal_outputs_bitexact(probe):
    starts, pred = _fixture(n=96, stride_pattern="gapped")
    last = pred[:, 49, :].copy()
    cut = 40
    tampered = pred.copy()
    tampered[cut + 1:] += 12345.0
    tampered_last = tampered[:, 49, :]
    for arm in probe.arm_matrix():
        if arm["family"] == "baseline":
            continue
        before = probe.apply_arm(arm, last, pred, starts)
        after = probe.apply_arm(arm, tampered_last, tampered, starts)
        if arm["causal"]:
            assert np.array_equal(before[: cut + 1], after[: cut + 1]), arm["arm"]
        else:
            support = int(arm["support"])
            safe = cut - support
            if safe >= 0:
                assert np.array_equal(before[: safe + 1], after[: safe + 1]), arm["arm"]
            window = before[max(cut - support + 1, 0): cut + 1]
            window_after = after[max(cut - support + 1, 0): cut + 1]
            assert not np.array_equal(window, window_after), arm["arm"]


def test_audit_causality_passes_and_fails_closed(probe):
    starts, pred = _fixture(n=96)
    last = pred[:, 49, :]
    arms = probe.arm_matrix()
    report = probe.audit_causality(last, pred, starts, arms)
    assert report["tamper_cut"] == 48
    causal = [a for a in report["arms"] if a["causal"]]
    traj = [a for a in report["arms"] if not a["causal"]]
    assert len(causal) == 6 and len(traj) == 6
    assert all(a["invariant_upto_cut_bitexact"] for a in causal)
    assert all(a["reacts_to_future_perturbation"] for a in traj)
    # fail-closed: a doctored causal arm (reads one window ahead) must FAIL
    rogue = copy.deepcopy(arms)
    rogue[1] = dict(rogue[1])
    rogue[1]["arm"] = "rogue_lookahead"
    rogue[1]["causal"] = True

    def lookahead_apply(arm, last_predictions, full_predictions, starts):
        shifted = np.vstack([last_predictions[1:], last_predictions[-1:]])
        return 0.5 * np.asarray(last_predictions) + 0.5 * shifted

    original_apply = probe.apply_arm
    probe.apply_arm = lookahead_apply
    try:
        with pytest.raises(probe.ContinuityProbeError, match="CAUSALITY AUDIT FAILED"):
            probe.audit_causality(last, pred, starts, rogue)
    finally:
        probe.apply_arm = original_apply


# ---------------------------------------------------------------------------
# arm matrix + receipts + anchors + readings
# ---------------------------------------------------------------------------


def test_arm_matrix_grid_and_labels(probe):
    arms = probe.arm_matrix()
    assert len(arms) == 13
    ids = {a["arm"] for a in arms}
    for family in ("causal_window", "trajalign"):
        for k in probe.K_GRID:
            arm = next(a for a in arms if a["arm"] == f"{family}_mean_K{k}")
            assert arm["deployment_legal"] == (family == "causal_window")
            assert arm["weighting"] == "mean" and arm["K"] == k
        for alpha in probe.ALPHA_GRID:
            arm = next(a for a in arms if a["arm"] == f"{family}_exp_a{alpha}")
            assert arm["alpha"] == alpha and arm["support"] >= 1
    assert "baseline" in ids


def test_p4_receipt_loads_only_with_pinned_sha(probe):
    body = probe.load_p4_receipt(ROOT)
    assert body["schema"] == "calibration_gap_p4_stream_stats_v1"
    probe.P4_RECEIPT_SHA256 = probe.P4_RECEIPT_SHA256 + "0"
    try:
        with pytest.raises(probe.ContinuityProbeError, match="SHA drift"):
            probe.load_p4_receipt(ROOT)
    finally:
        probe.P4_RECEIPT_SHA256 = probe.P4_RECEIPT_SHA256[:-1]


def _baseline_rows_from_receipt(probe, body):
    rows = []
    for cell in body["cells"]:
        for row in cell["sessions"]:
            if row["variant"] != "strict_fss_baseline":
                continue
            row = dict(row)
            row["arm"] = "baseline"
            rows.append(row)
    return rows


def test_baseline_anchor_accepts_receipt_faithful_rows(probe):
    body = probe.load_p4_receipt(ROOT)
    rows = _baseline_rows_from_receipt(probe, body)
    assert len(rows) == 63
    report = probe.check_p4_strict_anchor(rows, body)
    assert report["sessions_checked"] == 63
    assert report["max_abs_delta_r2"] == 0.0
    assert report["prediction_sha256_bitexact_matches"] == 63


def test_baseline_anchor_rejects_r2_and_sha_drift(probe):
    body = probe.load_p4_receipt(ROOT)
    rows = _baseline_rows_from_receipt(probe, body)
    drifted = copy.deepcopy(rows)
    drifted[0]["variance_weighted_r2"] += 1e-6
    with pytest.raises(probe.ContinuityProbeError, match="anchor drift"):
        probe.check_p4_strict_anchor(drifted, body)
    drifted = copy.deepcopy(rows)
    drifted[3]["raw_t4_sha256"] = "0" * 64
    with pytest.raises(probe.ContinuityProbeError, match="raw_t4_sha256"):
        probe.check_p4_strict_anchor(drifted, body)
    body_missing = copy.deepcopy(body)
    for cell in body_missing["cells"]:
        if any(r["variant"] == "strict_fss_baseline" for r in cell["sessions"]):
            cell["sessions"].pop()  # the reference loses its last strict row
            break
    with pytest.raises(probe.ContinuityProbeError, match="without a p4 anchor"):
        probe.check_p4_strict_anchor(rows, body_missing)


def test_readings_verdicts_apply_threshold_to_both_families(probe):
    arms = [a["arm"] for a in probe.arm_matrix()]
    base = 0.30
    pairs = []
    for surface in probe.SURFACES:
        for budget in probe.BUDGETS:
            for arm in arms:
                if arm == "baseline":
                    continue
                delta = 0.02 if (
                    surface == "external" and arm == "causal_window_mean_K8"
                ) else -0.001
                pairs.append({
                    "surface": surface, "budget": budget, "arm": arm,
                    "deployment_legal": arm.startswith("causal_window"),
                    "baseline_mean_r2": base,
                    "mean": delta, "median": delta,
                    "n_positive": 14 if delta > 0 else 1, "n_total": 15,
                    "min": delta, "max": delta,
                    "bootstrap_95_interval": [delta - 0.001, delta + 0.001],
                })
    payload = {"paired_deltas": pairs}
    out = probe.readings(payload)
    assert out["threshold"] == probe.READING_THRESHOLD == 0.01
    causal_ext = out["verdicts"]["causal_window_external"]
    assert causal_ext["any_arm_passes_threshold"] is True
    assert causal_ext["best_per_budget"]["M4"]["best_arm"] == "causal_window_mean_K8"
    traj_ext = out["verdicts"]["trajalign_external"]
    assert traj_ext["any_arm_passes_threshold"] is False
    assert "dead" in traj_ext["verdict"]
    within = out["verdicts"]["causal_window_within"]
    assert within["any_arm_passes_threshold"] is False
    assert "CEBRA" in within["verdict"]
    # the table carries every arm x budget x surface cell
    for surface in probe.SURFACES:
        for budget in probe.BUDGETS:
            cell = out["table"][f"{surface}_M{budget}"]
            assert cell["baseline_mean_r2"] == base
            assert len(cell["arms"]) == 12


def test_module_declares_the_alignment_decision_and_boundaries(probe):
    text = Path(ROOT / "src/continuity_probe_v1.py").read_text()
    for needle in (
        "PAST windows end",
        "trajectory-aligned pooling of b_t",
        "FUTURE windows",
        "renormalized",
        "CAUSALITY AUDIT FAILED",
        "cdm_followup_note",
    ):
        assert needle in text
