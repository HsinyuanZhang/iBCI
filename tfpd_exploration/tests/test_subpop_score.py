"""Tests for scripts/run_subpop_score.py (sub-population invariance scorer).

Covers HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md §3 gates and
§5 receipt requirements as implemented by the matched scorer:

- the pre-registered verdict logic on synthetic stats: the T trichotomy
  (|d| < 0.01 / >= +0.01 / <= -0.01), the G three-band reading
  (>= +0.08 / [+0.03, +0.08) / < +0.03) and the conjunctive C gate — including
  the case where the bootstrap interval looks good but the mean fails, where
  the gate must stay False (no bootstrap escape clause);
- references are LOADED from SHA-verified sealed receipts (A2 pooled +
  per-seed, Arm A / D governing bars) and the loader hard-fails on a SHA
  mismatch; no governing constant is hardcoded in the scorer source;
- the Step 0C baseline loader: missing receipt is non-fatal, a tampered
  sidecar or a wrong expected body SHA refuses;
- validate_cell_terminal: a faithful fixture passes; every tampered field
  refuses; a missing terminal receipt means "not landed" (None), never a
  silent score of a partial artifact; smoke / truncated / non-48-epoch runs
  are refused;
- the curve math: per-fraction cell-minus-D paired deltas with sign counts on
  fixtures, seed aggregation, drop profiles and the descriptive flattening
  comparison; roster mismatch refuses;
- paired stats reuse (matched_scorer.paired_session_stats) and the
  window-weighted / n_windows receipt conventions;
- the transactional receipt writer refuses to overwrite and seals 0444 +
  sidecar;
- the curve machinery is IMPORTED from run_subpop_step0c.py, never duplicated.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat as stat_mod
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "subpop_score_under_test", ROOT / "scripts/run_subpop_score.py"
)
score = importlib.util.module_from_spec(spec)
sys.modules["subpop_score_under_test"] = score
spec.loader.exec_module(score)

matched_scorer = score._load_module(
    "tfpd_lane_matched_scorer_subpop_score_test",
    ROOT / "src/tfpd_lane/matched_scorer.py",
)
receipt_mod = score._load_module(
    "tfpd_lane_receipt_subpop_score_test", ROOT / "src/tfpd_lane/receipt.py"
)


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _ArmCommonStub:
    sha256_file = staticmethod(_sha_file)


# ---------------------------------------------------------------------------
# pre-registered verdict logic
# ---------------------------------------------------------------------------
def test_t_trichotomy_thresholds():
    rule = score.T_TRICHOTOMY_RULE
    assert "|d| < 0.01 -> T_APPROX_EQUAL" in rule
    assert ">= +0.01 -> T_GREATER" in rule
    assert "<= -0.01 -> T_LESS" in rule
    assert score.t_trichotomy(0.00999) == "T_APPROX_EQUAL"
    assert score.t_trichotomy(-0.00999) == "T_APPROX_EQUAL"
    assert score.t_trichotomy(0.0) == "T_APPROX_EQUAL"
    assert score.t_trichotomy(0.01) == "T_GREATER"          # boundary inclusive
    assert score.t_trichotomy(0.5) == "T_GREATER"
    assert score.t_trichotomy(-0.01) == "T_LESS"            # boundary inclusive
    assert score.t_trichotomy(-0.25) == "T_LESS"
    assert score.t_trichotomy(None) is None
    # every verdict carries its handoff consequence
    for verdict in ("T_GREATER", "T_APPROX_EQUAL", "T_LESS"):
        assert score.T_CONSEQUENCES[verdict]


def test_g_band_thresholds():
    assert ">= +0.08" in score.G_BAND_RULE and "< +0.03" in score.G_BAND_RULE
    assert score.g_band(0.08) == "GAIN_EXPLAINS_MOST"       # boundary inclusive
    assert score.g_band(0.2) == "GAIN_EXPLAINS_MOST"
    assert score.g_band(0.07999) == "GAIN_SECONDARY_TERM"
    assert score.g_band(0.03) == "GAIN_SECONDARY_TERM"      # boundary inclusive
    assert score.g_band(0.02999) == "GAIN_NOT_THE_MECHANISM"
    assert score.g_band(-0.05) == "GAIN_NOT_THE_MECHANISM"
    assert score.g_band(None) is None
    for verdict in score.G_CONSEQUENCES:
        assert score.G_CONSEQUENCES[verdict]


def _stats(mean, n_positive, n_total=15, ci=(0.0, 1.0)):
    return {"mean": mean, "n_positive": n_positive, "n_total": n_total,
            "bootstrap_95_interval": list(ci)}


def test_c_gate_conjunctive_all_conditions_required():
    conditions, gate = score.c_gate_conditions(
        _stats(0.05, 12), _stats(0.01, 5, n_total=6)
    )
    assert gate is True
    assert conditions["external_mean_ge_plus_0.03"]
    assert conditions["external_positive_ge_10_of_15"]
    assert conditions["within_mean_ge_minus_0.03"]
    assert conditions["external_surface_is_15_sessions"]


def test_c_gate_stays_false_when_bootstrap_looks_good_but_mean_fails():
    # interval far from zero, 15/15 positive, within fine — but the external
    # MEAN is below +0.03: no bootstrap escape clause may rescue it
    ext = _stats(0.029, 15, ci=(0.025, 0.033))
    within = _stats(0.02, 5, n_total=6)
    conditions, gate = score.c_gate_conditions(ext, within)
    assert gate is False
    assert conditions["external_mean_ge_plus_0.03"] is False
    assert conditions["external_positive_ge_10_of_15"] is True
    assert conditions["within_mean_ge_minus_0.03"] is True


@pytest.mark.parametrize(
    "ext,within,failing",
    [
        (_stats(0.05, 9), _stats(0.01, 5, 6), "external_positive_ge_10_of_15"),
        (_stats(0.05, 12), _stats(-0.031, 1, 6), "within_mean_ge_minus_0.03"),
        (_stats(0.05, 12, n_total=14), _stats(0.01, 5, 6), "external_surface_is_15_sessions"),
    ],
)
def test_c_gate_fails_on_each_single_broken_condition(ext, within, failing):
    conditions, gate = score.c_gate_conditions(ext, within)
    assert gate is False and conditions[failing] is False


def test_c_gate_rule_string_recorded_and_no_escape():
    rule = score.C_GATE_RULE
    assert ">= +0.03" in rule and ">= 10 of 15" in rule and "-0.03" in rule
    assert "NO bootstrap escape clause" in rule
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert '"bootstrap_escape_clause": "none"' in source


def test_a2_screen_requires_both_absolute_bars():
    bars = {"external": 0.34, "within": 0.57}
    both = score.a2_development_screen(0.40, 0.58, bars)
    assert both["development_screen_pass"] is True
    ext_only = score.a2_development_screen(0.40, 0.50, bars)
    assert ext_only["development_screen_pass"] is False
    win_only = score.a2_development_screen(0.30, 0.58, bars)
    assert win_only["development_screen_pass"] is False
    assert "not a superiority claim" in ext_only["note"]
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert "seeds 43/44 remain MANDATORY" in source
    assert '"a2_superiority_claim_admissible": False' in source


# ---------------------------------------------------------------------------
# references: SHA-verified receipts only
# ---------------------------------------------------------------------------
def test_references_loaded_from_sha_verified_receipts_bit_exactly():
    refs = score.load_governing_references(_sha_file)
    assert refs["a2"]["pooled"]["external"] == pytest.approx(0.3460880616472827, abs=0)
    assert refs["a2"]["pooled"]["within"] == pytest.approx(0.5776186750994788, abs=0)
    assert refs["governing_bars"]["armA"]["external"] == 0.2603564786414305
    assert refs["governing_bars"]["armA"]["within"] == 0.5538396437962850
    assert refs["governing_bars"]["D"]["external"] == 0.4179362749059995
    assert refs["governing_bars"]["D"]["within"] == 0.5696851710478464
    assert refs["a2"]["receipt_sha256"] == score.A2_R1_RECEIPT_SHA
    assert refs["governing_bars"]["receipt_sha256"] == score.SPARSIFY_SCORE_RECEIPT_SHA
    # per-seed A2 references are loaded, not hardcoded
    assert set(refs["a2"]["per_seed"]) == {"42", "43", "44"}
    assert refs["a2"]["per_seed"]["44"]["external"] == pytest.approx(0.3837, abs=1e-4)
    assert refs["a2"]["per_seed_external_spread"] > 0.05
    # per-session sealed tables are available for cross-checking
    table = refs["governing_bars"]["per_session_governing_last_bin"]["D_swa"]["external"]
    assert len(table) == 15


def test_reference_loader_hard_fails_on_sha_mismatch(monkeypatch):
    frozen_a2_sha = score.A2_R1_RECEIPT_SHA
    frozen_score_sha = score.SPARSIFY_SCORE_RECEIPT_SHA
    monkeypatch.setattr(score, "A2_R1_RECEIPT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA mismatch"):
        score.load_governing_references(_sha_file)
    monkeypatch.setattr(score, "A2_R1_RECEIPT_SHA", frozen_a2_sha)
    monkeypatch.setattr(score, "SPARSIFY_SCORE_RECEIPT_SHA", "0" * 64)
    with pytest.raises(SystemExit, match="SHA mismatch"):
        score.load_governing_references(_sha_file)


def test_no_hardcoded_governing_constants_in_source():
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    for literal in (
        "0.3460880616472827", "0.5776186750994788", "0.2603564786414305",
        "0.5538396437962850", "0.4179362749059995", "0.5696851710478464",
        "0.3837",
    ):
        assert literal not in source, literal


# ---------------------------------------------------------------------------
# Step 0C baseline loader
# ---------------------------------------------------------------------------
def _step0c_fixture(tmp_path: Path, mode: str = "faithful"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "step0c_receipt.json"
    payload = {
        "status": "SUBPOP_STEP0C_SCORED",
        "plan": {"fractions": [0.0, 0.5], "n_mask_seeds": 3},
        "results": {
            "D": {
                "zero_nogain_f0.0": {
                    "governing_mean_r2": 0.40, "n_seeds": 3,
                    "per_session_mean_over_seeds": {"s1": 0.40, "s2": 0.40},
                },
                "zero_nogain_f0.5": {
                    "governing_mean_r2": 0.20, "n_seeds": 3,
                    "per_session_mean_over_seeds": {"s1": 0.21, "s2": 0.19},
                },
            },
            "armA": {"native": {"governing_mean_r2": 0.26, "n_seeds": 1,
                                "per_session_mean_over_seeds": {"s1": 0.26}}},
        },
    }
    body = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    path.write_text(body)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_text = _sha_file(path) + "  step0c_receipt.json\n"
    if mode == "sidecar":
        sidecar_text = "0" * 64 + "  step0c_receipt.json\n"
    sidecar.write_text(sidecar_text)
    if mode == "missing":
        path.unlink()
        sidecar.unlink(missing_ok=True)
    return path


def test_step0c_baseline_missing_receipt_is_not_fatal():
    baseline = score.load_step0c_baseline(_sha_file, path=Path("/nonexistent/x.json"))
    assert baseline["available"] is False
    assert "scored LIVE in this pass" in baseline["note"]


def test_step0c_baseline_faithful_and_tampered(tmp_path):
    path = _step0c_fixture(tmp_path)
    baseline = score.load_step0c_baseline(_sha_file, path=path)
    assert baseline["available"] is True
    assert baseline["receipt_sha256"] == _sha_file(path)
    assert baseline["curves"]["D"]["zero_nogain_f0.5"]["governing_mean_r2"] == 0.20
    assert baseline["curves"]["D"]["zero_nogain_f0.5"]["per_session_mean_over_seeds"]["s2"] == 0.19
    # a strict expected body SHA binds the sealed baseline
    with pytest.raises(SystemExit, match="Step 0C baseline receipt rejected"):
        score.load_step0c_baseline(_sha_file, path=path, expected_sha="0" * 64)
    # sidecar tampering refuses
    tampered = _step0c_fixture(tmp_path / "tampered", mode="sidecar")
    with pytest.raises(SystemExit, match="Step 0C baseline receipt rejected"):
        score.load_step0c_baseline(_sha_file, path=tampered)


# ---------------------------------------------------------------------------
# cell terminal validation
# ---------------------------------------------------------------------------
def _cell_integrity(cell: str) -> dict:
    base = {
        "num_heads": 2,
        "dropout_structure": {"dynamic_dropout": False, "tf_drop_rate": 0.1},
        "behavior_scaling_convention": "unscaled standardized behavior (exact Arm A replica)",
    }
    if cell == "T":
        base.update({
            "mask_structure": "whole_unit_bernoulli_key_padding_mask",
            "min_keep": 4,
            "rescaling_policy": ("none: src is not zeroed and no 1/(1-p) rescaling is "
                                 "applied; attention renormalizes over the surviving keys"),
            "key_padding_mask_semantics": "boolean complement of the whole-unit keep mask",
            "p_distribution_and_clamp_policy": {"clamp": "none", "seed": 42},
            "generator_namespaces": {"unit_mask": "torch.Generator CPU seeded 42003"},
            "eval_mask_policy": "perturbation disabled",
            "theta_authority": {"sha256": _sha_file(score.THETA_AUTHORITY_FILE),
                                "authority_sha256": "theta-authority-sha"},
        })
    elif cell == "G":
        base.update({
            "mask_structure": "none_global_gain_all_units",
            "min_keep": "n/a (no masking)",
            "rescaling_policy": "all unit windows multiplied by 1/(1-clamp(p, 0, 0.95))",
            "p_distribution_and_clamp_policy": {
                "clamp": "p clamped to [0, 0.95] before the gain", "seed": 42},
            "generator_namespaces": {"p_stream": "numpy PCG64(42)"},
            "eval_mask_policy": "perturbation disabled",
            "theta_authority": {"sha256": _sha_file(score.THETA_AUTHORITY_FILE),
                                "authority_sha256": "theta-authority-sha"},
        })
    else:
        base.update({
            "lambda": 0.1, "lambda_frozen": True, "lambda_sweep": "none",
            "consistency_site": "behaviour predictions (never a latent)",
            "skip_floor": 1, "skip_rule": "per-sample skip",
            "compute_option": "two-view same-batch",
            "preregistration": "compute-matched D control required",
            "eval_mask": "exact ones; single forward",
            "p_stream": {"draws_per_step": 2, "seed": 42},
        })
    return base


def _cell_fixture(tmp_path: Path, cell: str = "T", tamper: str | None = None):
    """Faithful landed-cell fixture: 0444 receipt + sidecar + sealed SWA."""
    directory = tmp_path / "results/subpop_v1" / score.CELL_SPECS[cell]["directory"]
    directory.mkdir(parents=True)
    swa = directory / "swa_final4.pt"
    swa.write_bytes(b"swa-bytes")
    swa_sha = _sha_file(swa)
    if tamper == "swa_bytes":
        swa.write_bytes(b"tampered")
    if tamper == "swa_missing":
        swa.unlink()
    diagnostics = [
        {"epoch": 46, "perturbation_summary": {"n_forwards": 10, "gain_max": 2.0,
         "surviving_units_min": 4, "min_keep_trigger_rate": 0.01,
         "clamp_trigger_rate": 0.02, "kept_fraction_mean": 0.5}},
        {"epoch": 47, "perturbation_summary": {"n_forwards": 10, "gain_max": 3.0,
         "surviving_units_min": 4, "min_keep_trigger_rate": 0.02,
         "clamp_trigger_rate": 0.04, "kept_fraction_mean": 0.48}},
    ]
    if cell == "C":
        diagnostics = [
            {"epoch": 47, "mask_summary": {"branch1": {"kept_fraction_mean": 0.5,
              "all_zero_branch_samples": 3},
              "branch2": {"kept_fraction_mean": 0.52, "all_zero_branch_samples": 1}},
             "jaccard_overlap": {"mean": 0.34, "n": 320},
             "behavior_loss_sum_mean_per_step": 0.5,
             "consistency_loss_mean_per_step": 0.2},
        ]
    payload = {
        "schema": "tfpd_subpop_cell_v1",
        "status": "CELL_TERMINAL",
        "cell": cell,
        "cell_name": score.CELL_SPECS[cell]["directory"],
        "smoke": False,
        "max_train_steps": None,
        "epochs_run": 48,
        "invariant_failures": [],
        "source_closure": {"launch_final_closure_equal": True},
        "initial_state": {
            "state_dict_sha256": "canonical-state-sha",
            "artifact_sha256": _sha_file(score.CANONICAL_INITIAL_STATE),
        },
        "data_contract": {
            "behavior_normalizer_semantic_sha256":
                "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391",
        },
        "integrity": _cell_integrity(cell),
        "p_sequence_sha256": "p" * 64,
        "p_stream_total_draws": 48 * 33925,
        "diagnostics_per_epoch": diagnostics,
        "swa": {"sha256": swa_sha, "window_epochs": [44, 45, 46, 47]},
    }
    if tamper == "status":
        payload["status"] = "CELL_FAILED"
    elif tamper == "smoke_status":
        payload["status"] = "CELL_SMOKE_COMPLETE__NON_AUTHORITATIVE"
    elif tamper == "smoke_flag":
        payload["smoke"] = True
    elif tamper == "truncated":
        payload["max_train_steps"] = 30
    elif tamper == "epochs":
        payload["epochs_run"] = 5
    elif tamper == "closure":
        payload["source_closure"]["launch_final_closure_equal"] = False
    elif tamper == "invariants":
        payload["invariant_failures"] = [17]
    elif tamper == "initial_state":
        payload["initial_state"]["state_dict_sha256"] = "0" * 64
    elif tamper == "normalizer":
        payload["data_contract"]["behavior_normalizer_semantic_sha256"] = "deadbeef"
    elif tamper == "swa_sha":
        payload["swa"]["sha256"] = "0" * 64
    elif tamper == "num_heads":
        payload["integrity"]["num_heads"] = 64
    elif tamper == "theta":
        payload["integrity"]["theta_authority"]["sha256"] = "0" * 64
    elif tamper == "min_keep":
        payload["integrity"]["min_keep"] = 1
    elif tamper == "clamp":
        payload["integrity"]["p_distribution_and_clamp_policy"]["clamp"] = "none"
    elif tamper == "lambda":
        payload["integrity"]["lambda"] = 0.5
    elif tamper == "consistency_site":
        payload["integrity"]["consistency_site"] = "latent"
    body = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    receipt = directory / "terminal_receipt.json"
    receipt.write_text(body)
    os.chmod(receipt, stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)
    sidecar = directory / "terminal_receipt.json.sha256"
    sidecar_text = _sha_file(receipt) + "  terminal_receipt.json\n"
    if tamper == "sidecar":
        sidecar_text = "0" * 64 + "  terminal_receipt.json\n"
    sidecar.write_text(sidecar_text)
    os.chmod(sidecar, stat_mod.S_IRUSR | stat_mod.S_IRGRP | stat_mod.S_IROTH)
    if tamper == "mode":
        os.chmod(receipt, 0o644)
    return receipt


_CANONICAL_STUB = {"state_sha256": "canonical-state-sha"}
_THETA_STUB = {"authority_sha256": "theta-authority-sha"}


def test_validate_cell_terminal_passes_on_faithful_fixture(tmp_path):
    for cell in ("T", "C", "G"):
        _cell_fixture(tmp_path, cell=cell)
        record = score.validate_cell_terminal(
            cell, _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
        )
        assert record["cell"] == cell
        assert record["mode_0444"] and record["non_symlink"]
        assert record["launch_final_closure_equal"]
        assert record["initial_state_reconciled"]
        assert record["normalizer_reconciled_f062506c"]
        assert record["epochs_run"] == 48 and record["num_heads"] == 2
        assert record["perturbation_law"]
        assert record["realized_perturbation_statistics"]["available"]
        assert record["realized_perturbation_statistics"]["n_epochs"] >= 1


def test_validate_cell_terminal_missing_receipt_means_not_landed(tmp_path):
    record = score.validate_cell_terminal(
        "T", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    )
    assert record is None  # the caller records this as not_landed and skips it


@pytest.mark.parametrize(
    "tamper",
    ["status", "smoke_status", "smoke_flag", "truncated", "epochs", "closure",
     "invariants", "initial_state", "normalizer", "sidecar", "mode", "swa_sha",
     "swa_bytes", "swa_missing", "num_heads", "theta", "min_keep"],
)
def test_validate_cell_terminal_refuses_each_tampering(tmp_path, tamper):
    _cell_fixture(tmp_path, cell="T", tamper=tamper)
    with pytest.raises(SystemExit):
        score.validate_cell_terminal(
            "T", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
        )


@pytest.mark.parametrize("tamper", ["lambda", "consistency_site"])
def test_validate_cell_terminal_refuses_cell_c_law_tampering(tmp_path, tamper):
    _cell_fixture(tmp_path, cell="C", tamper=tamper)
    with pytest.raises(SystemExit):
        score.validate_cell_terminal(
            "C", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
        )


def test_validate_cell_terminal_refuses_cell_g_clamp_tampering(tmp_path):
    _cell_fixture(tmp_path, cell="G", tamper="clamp")
    with pytest.raises(SystemExit):
        score.validate_cell_terminal(
            "G", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
        )


def test_cell_c_theta_binding_is_documented_not_assumed(tmp_path):
    _cell_fixture(tmp_path, cell="C")
    record = score.validate_cell_terminal(
        "C", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    )
    assert "no session authority" in str(record["theta_authority_reconciled"])


def test_realized_perturbation_statistics_series(tmp_path):
    _cell_fixture(tmp_path, cell="C")
    record = score.validate_cell_terminal(
        "C", _ArmCommonStub, _CANONICAL_STUB, _THETA_STUB, root=tmp_path
    )
    series = record["realized_perturbation_statistics"]["per_epoch_series"]
    assert series[0]["jaccard_overlap"]["mean"] == 0.34
    assert series[0]["branch1_kept_fraction_mean"] == 0.5
    assert series[0]["consistency_loss_mean_per_step"] == 0.2


# ---------------------------------------------------------------------------
# curve math
# ---------------------------------------------------------------------------
def _curve_block(per_session, mean=None):
    values = list(per_session.values())
    return {
        "governing_mean_r2": (float(sum(values) / len(values)) if mean is None else mean),
        "per_session_mean_over_seeds": dict(per_session),
    }


def test_curve_contrast_paired_deltas_and_sign_counts():
    cell = _curve_block({"s1": 0.30, "s2": 0.20, "s3": 0.10})
    d = _curve_block({"s1": 0.25, "s2": 0.25, "s3": 0.15})
    stats = score.curve_contrast(cell, d, matched_scorer)
    assert stats["all_deltas"] == pytest.approx([0.05, -0.05, -0.05], abs=1e-12)
    assert stats["mean"] == pytest.approx(-0.05 / 3.0, abs=1e-12)
    assert stats["n_positive"] == 1 and stats["n_total"] == 3
    assert stats["exact_sign_pattern"] == "+--"
    assert "bootstrap_95_interval" in stats  # descriptive only
    assert stats["contrast"].startswith("cell - D")


def test_curve_contrast_refuses_roster_mismatch():
    cell = _curve_block({"s1": 0.3, "s2": 0.2})
    d = _curve_block({"s1": 0.25, "s3": 0.15})
    with pytest.raises(SystemExit, match="roster mismatch"):
        score.curve_contrast(cell, d, matched_scorer)


def test_aggregate_seed_rows_matches_step0c_registration():
    rows = [
        {"mean_r2": 0.2, "per_session": [
            {"session": "s1", "r2": 0.2, "n_windows": 10},
            {"session": "s2", "r2": 0.4, "n_windows": 30}]},
        {"mean_r2": 0.4, "per_session": [
            {"session": "s1", "r2": 0.4, "n_windows": 10},
            {"session": "s2", "r2": 0.2, "n_windows": 30}]},
    ]
    block = score.aggregate_seed_rows("zero_nogain_f0.5", "zero_nogain", 0.5, rows)
    assert block["governing_mean_r2"] == pytest.approx(0.3)
    assert block["per_seed_mean_r2"] == [0.2, 0.4]
    assert block["per_session_mean_over_seeds"] == pytest.approx({"s1": 0.3, "s2": 0.3})
    assert block["n_windows_per_session"] == {"s1": 10, "s2": 30}
    assert block["n_seeds"] == 2 and block["intervention"] == "zero_nogain"


def test_curve_drop_profile_and_flattening():
    blocks = {
        "zero_nogain_f0.0": _curve_block({"s": 0.40}),
        "zero_nogain_f0.1": _curve_block({"s": 0.30}),
        "zero_nogain_f0.5": _curve_block({"s": 0.10}),
    }
    profile = score.curve_drop_profile(blocks, "zero_nogain", (0.0, 0.1, 0.5))
    assert profile["base_fraction_0_mean_r2"] == 0.40
    assert profile["drop_from_fraction_0"]["0.1"] == pytest.approx(-0.10)
    assert profile["drop_from_fraction_0"]["0.5"] == pytest.approx(-0.30)
    assert profile["mean_drop_over_nonzero_fractions"] == pytest.approx(-0.20)
    flatter = blocks
    steeper = {
        "zero_nogain_f0.0": _curve_block({"s": 0.40}),
        "zero_nogain_f0.1": _curve_block({"s": 0.10}),
        "zero_nogain_f0.5": _curve_block({"s": -0.30}),
    }
    comparison = score.flatter_than_d(profile, score.curve_drop_profile(
        steeper, "zero_nogain", (0.0, 0.1, 0.5)), (0.0, 0.1, 0.5))
    assert comparison["n_nonzero_fractions"] == 2
    assert comparison["n_fractions_flatter_than_d"] == 2
    assert comparison["per_fraction"]["0.5"]["cell_flatter"] is True
    assert "DESCRIPTIVE" in comparison["note"]
    assert score.curve_drop_profile({}, "zero_nogain", (0.0,))["available"] is False


# ---------------------------------------------------------------------------
# paired stats reuse + receipt conventions
# ---------------------------------------------------------------------------
def test_paired_session_stats_reused_for_every_contrast():
    stats = matched_scorer.paired_session_stats([0.05, -0.02, 0.01, 0.03], seed=42)
    assert stats["mean"] == pytest.approx(0.0175)
    assert stats["n_positive"] == 3 and stats["n_total"] == 4
    assert stats["exact_sign_pattern"] == "+-++"
    assert len(stats["bootstrap_95_interval"]) == 2
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert "matched_scorer.paired_session_stats" in source
    assert "DESCRIPTIVE ONLY" in source  # intervals never rescue a failed mean


def test_window_weighted_mean_and_n_windows_annotation():
    rows = [
        {"r2": 0.2, "n_windows_scored": 100},
        {"r2": 0.4, "n_windows_scored": 300},
    ]
    assert score.window_weighted_mean(rows) == (0.2 * 100 + 0.4 * 300) / 400
    assert score.window_weighted_mean([]) is None
    block = {"per_session": [{"session": "s", "r2": 0.5, "n_windows_scored": 7}]}
    score.annotate_n_windows(block)
    assert block["per_session"][0]["n_windows"] == 7


def test_receipt_writer_refuses_overwrite_and_seals_0444(tmp_path):
    target = tmp_path / "subpop_score_receipt.json"
    receipt_mod.write_receipt_transactionally(target, {"schema": "tfpd_subpop_score_v1"})
    mode = stat_mod.S_IMODE(target.stat().st_mode)
    assert mode == 0o444
    sidecar = target.with_suffix(target.suffix + ".sha256")
    assert sidecar.read_text().startswith(hashlib.sha256(target.read_bytes()).hexdigest())
    with pytest.raises(SystemExit) as excinfo:
        receipt_mod.write_receipt_transactionally(target, {"schema": "overwritten"})
    assert excinfo.value.code == 2
    assert json.loads(target.read_text())["schema"] == "tfpd_subpop_score_v1"


def test_fresh_output_root_enforced(tmp_path):
    existing = tmp_path / "already_there"
    existing.mkdir()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_score.py"),
         "--device", "cpu", "--output-root", str(existing),
         "--authorize-target", score.AUTH_VALUE],
        capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 2
    assert "fresh output root required" in result.stderr


def test_authorization_gate_for_the_external_surface():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_score.py"), "--device", "cpu"],
        capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 3
    assert score.AUTH_VALUE in result.stderr


def test_dry_run_needs_no_authorization_and_opens_no_nwb():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_subpop_score.py"),
         "--device", "cpu", "--dry-run"],
        capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRY_RUN__NO_NWB_OPENED"
    # cell W joined the default roster (native-only, exploratory) and the
    # AM/IM decomposition cells joined after it (handoff 2026-08-19); a bare
    # run picks each up as soon as its terminal receipt lands
    assert payload["cells_requested"] == ["T", "C", "G", "W", "AM", "IM"]
    assert set(payload["references"]["armA_governing"]) == {"external", "within"}
    assert payload["curve"]["seed_rule"].startswith("torch.Generator()")
    # W is documented in the plan as native-only (the 0C wrapper would drop
    # its residual), never silently curved
    assert set(payload["curve"]["native_only_cells"]) == {"W"}
    # every requested cell is either landed (with provenance) or explicitly
    # not_landed with a note — never silently scored from a partial artifact
    assert set(payload["cells_landed"]) | set(payload["cells_not_landed"]) == {
        "T", "C", "G", "W", "AM", "IM"}
    for cell in payload["cells_not_landed"]:
        assert "terminal receipt not present yet" in payload["not_landed"][cell]["note"]
    for cell in payload["cells_landed"]:
        assert payload["landed_cell_provenance"][cell]["swa_sha256"]


# ---------------------------------------------------------------------------
# the curve machinery is imported from Step 0C, never duplicated
# ---------------------------------------------------------------------------
def _step0c_module():
    return sys.modules.get("tfpd_subpop_step0c_machinery") or score._load_module(
        "tfpd_subpop_step0c_machinery", ROOT / "scripts/run_subpop_step0c.py"
    )


def test_step0c_machinery_is_imported_not_duplicated():
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert "class SubpopEvalModel" not in source
    assert "def curve_forward" not in source
    assert "def score_condition" not in source
    assert "def intervention_generator_seed" not in source
    for reused in (
        "step0c.SubpopEvalModel", "step0c.curve_forward", "step0c.score_condition",
        "step0c.native_forward", "step0c.verify_checkpoints", "step0c.SEED_RULE_DOC",
        "step0c.SPEC_FRACTIONS", "step0c.CURVE_KINDS", "step0c.MIN_KEEP",
        "step0c.date_block",
    ):
        assert reused in source, reused
    step0c = _step0c_module()
    assert step0c.CURVE_KINDS[0] == "zero_nogain"   # PRIMARY
    assert step0c.SPEC_FRACTIONS == (0.0, 0.1, 0.25, 0.5, 0.75)
    assert step0c.SPEC_N_MASK_SEEDS == 3
    assert step0c.date_block("sub-M_ses-CO-20140307") == "2014"
    assert step0c.date_block("sub-M_ses-CO-20150625") == "2015"
    assert step0c.date_block(score.SEPARATE_SESSION) == "2014"


def test_curve_condition_keys_match_step0c_verbatim():
    """The paired seed rule requires the IDENTICAL condition keys."""
    step0c = _step0c_module()
    for fraction in step0c.SPEC_FRACTIONS:
        for kind in step0c.CURVE_KINDS:
            label = f"{kind}_f{fraction}"
            assert step0c.intervention_generator_seed(label, 0) == (
                step0c.intervention_generator_seed(label, 0)
            )
    # the scorer builds labels with the same f-string, so its masks are the
    # sealed 0C draws
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert 'label = f"{kind}_f{fraction}"' in source
    assert 'label = f"padding_f{fraction}"' in source


def test_not_landed_cells_are_omitted_with_a_note():
    source = (ROOT / "scripts/run_subpop_score.py").read_text()
    assert '"not_landed": not_landed' in source
    assert "terminal receipt not present yet" in source


# ---------------------------------------------------------------------------
# synthetic-fixture integration: the ROUTE cell graphs + the 0C wrapper + the
# scorer call shapes (CPU, no data opened)
# ---------------------------------------------------------------------------
def _lane_module(name, relative):
    return sys.modules.get(name) or score._load_module(name, ROOT / relative)


def _synthetic_inputs(batch=2, units=12, window=50):
    torch.manual_seed(0)
    return (
        torch.randn(batch, window, units),
        torch.randn(batch, 4, 100, units),
        torch.randn(batch, units, 4),
    )


def _built_models():
    subpop = _lane_module("tfpd_lane_subpop_cells_score_test",
                          "src/tfpd_lane/subpop_cells.py")
    consistency = _lane_module("tfpd_lane_consistency_cell_score_test",
                               "src/tfpd_lane/consistency_cell.py")
    return subpop, consistency


@pytest.mark.parametrize("cell", ["T", "G", "C"])
def test_route_cell_graphs_strict_load_the_canonical_arm_a_state(cell):
    """The scorer's load path: strip 'model.' conditionally, then strict load."""
    if not score.CANONICAL_INITIAL_STATE.is_file():
        pytest.skip("canonical initial state not present in this checkout")
    subpop, consistency = _built_models()
    model = score.build_cell_model(subpop, consistency, cell)
    payload = torch.load(
        score.CANONICAL_INITIAL_STATE, map_location="cpu", weights_only=False
    )
    state = payload["state_dict"]
    state = {(k[len("model."):] if k.startswith("model.") else k): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    # the canonical initial state carries no 'model.' prefix and the strict
    # load already proves key parity; the strip must be a no-op here
    assert not any(k.startswith("model.") for k in payload["state_dict"])
    assert score.perturbation_disabled(model)


@pytest.mark.parametrize("cell", ["T", "G", "C"])
def test_step0c_wrapper_accepts_route_cell_graphs(cell):
    """The curve machinery runs on the cell graphs and the perturbation never leaks.

    With no intervention the 0C wrapper must reproduce the cell's own eval
    forward bitwise — that is what makes the curve's fraction-0 point an exact
    cross-check of the native governing external score.
    """
    step0c = _step0c_module()
    subpop, consistency = _built_models()
    model = score.build_cell_model(subpop, consistency, cell).eval()
    wrapper = step0c.SubpopEvalModel(model, min_keep=step0c.MIN_KEEP).eval()
    neural, calib, side = _synthetic_inputs()
    with torch.no_grad():
        mine, identity = model(neural, calib_trials=calib, side_features=side)
        wrapped, identity_w = wrapper(neural, calib_trials=calib, side_features=side)
        decode = model.decode_with_identity(neural, identity)  # score_track path
        forward = step0c.curve_forward(wrapper, "zero_nogain", 0.25, 0, "zero_nogain_f0.25")
        curve, _ = forward(neural, calib, side)
        padded = step0c.curve_forward(wrapper, "padding", 0.5, 0, "padding_f0.5")
        curve_pad, _ = padded(neural, calib, side)
    assert torch.equal(mine, wrapped)
    assert torch.equal(mine, decode)
    assert torch.equal(identity, identity_w)
    assert bool(torch.isfinite(curve).all()) and bool(torch.isfinite(curve_pad).all())
    assert not torch.equal(curve, mine)
    assert score.perturbation_disabled(model)
    # the exact 0C condition key drives the mask generator (paired masks)
    assert step0c.intervention_generator_seed("zero_nogain_f0.25", 0) == (
        step0c.intervention_generator_seed("zero_nogain_f0.25", 0)
    )
