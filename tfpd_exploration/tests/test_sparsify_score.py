"""Tests for the audited run_sparsify_score.py (user audit 2026-08-18, 5 gaps).

- (3) A2 bars and ArmA governing means are LOADED from the SHA-verified _r1
  receipt (no hardcoded bars remain in source); the loader is SHA-gated and
  reproduces the frozen values bit-exactly;
- (1) the window-weighted full-window aggregation exists and is exact, and is
  labelled diagnostic-only alongside the governing last-bin pair;
- (4) validate_cell_terminal enforces sidecar SHA, 0444 mode, non-symlink,
  closure equality, and the terminal-bound initial-state / theta-authority /
  normalizer SHA reconciliation — passing on a faithful fixture and raising on
  each tampered field;
- (5) the contrast matrix covers all §7 pairs (R-ArmA/R-D/S2-D/S2-ArmA) on
  both surfaces in BOTH granularities;
- (2) block statistics carry mean, paired delta, sign/count and window share.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat as stat_mod
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "sparsify_score_under_test", ROOT / "scripts/run_sparsify_score.py"
)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def _sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bars_loaded_from_sha_verified_receipt_bit_exactly():
    refs = score.load_governing_references(_sha_file)
    assert refs["a2_bars"]["external"] == pytest.approx(0.3460880616472827, abs=0, rel=0)
    assert refs["a2_bars"]["within"] == pytest.approx(0.5776186750994788, abs=0, rel=0)
    assert refs["armA_external"] == 0.2603564786414305
    assert refs["armA_within"] == 0.5538396437962850
    assert refs["receipt_sha256"] == score.A2_R1_RECEIPT_SHA
    assert refs["void_root_opened"] is False
    # SHA-gated: a wrong body hash refuses
    with pytest.raises(SystemExit, match="SHA mismatch"):
        score.load_governing_references(lambda _p: "0" * 64)
    # no hardcoded bar constants remain in source
    source = (ROOT / "scripts/run_sparsify_score.py").read_text()
    for literal in ("0.3460880616472827", "0.5776186750994788",
                    "0.2603564786414305", "0.5538396437962850"):
        assert literal not in source  # zero hardcoded governing constants


def test_window_weighted_mean_exact_and_labelled():
    rows = [
        {"r2": 0.2, "n_windows_scored": 100},
        {"r2": 0.4, "n_windows_scored": 300},
    ]
    assert score.window_weighted_mean(rows) == (0.2 * 100 + 0.4 * 300) / 400
    assert score.window_weighted_mean([]) is None
    assert score.window_weighted_mean([{"r2": 0.5, "n_windows_scored": 0}]) is None
    source = (ROOT / "scripts/run_sparsify_score.py").read_text()
    assert "DIAGNOSTIC ONLY, never governing" in source
    assert "window_weighted_full_window_diagnostic" in source


def test_contrast_matrix_covers_all_pairs_both_surfaces_both_granularities():
    source = (ROOT / "scripts/run_sparsify_score.py").read_text()
    assert '"governing_last_bin", "diagnostic_full_window"' in source
    for pair in (
        '("R_swa", "armA_swa")', '("R_swa", "D_swa")',
        '("S2_swa", "D_swa")', '("S2_swa", "armA_swa")',
    ):
        assert pair in source
    assert 'f"{model_a}_minus_{model_b}_{surface}_{granularity}"' in source


def _make_cell_fixture(tmp_path: Path, name: str, tamper: str | None = None):
    """Faithful cell fixture: terminal receipt + 0444 sidecar + bound SHAs."""
    directory = tmp_path / "results/sparsification_v1" / name
    directory.mkdir(parents=True)
    canonical = torch.load(
        ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
        map_location="cpu", weights_only=False,
    )
    theta = torch.load(
        ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt",
        map_location="cpu", weights_only=False,
    )
    canonical_sha = _sha_file(ROOT / "results/admission_arms_v1/canonical_initial_state.pt")
    theta_sha = _sha_file(
        ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt"
    )
    payload = {
        "status": "CELL_TERMINAL",
        "source_closure": {"launch_final_closure_equal": True},
        "initial_state": {
            "state_dict_sha256": canonical["state_sha256"],
            "artifact_sha256": canonical_sha,
        },
        "integrity": {"theta_authority": {
            "sha256": theta_sha, "authority_sha256": theta["authority_sha256"],
        }},
        "data_contract": {
            "behavior_normalizer_semantic_sha256": "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391",
        },
        "swa": {"sha256": "ab" * 32},
        "p_sequence_sha256": "cd" * 32,
        "epochs_run": 48,
    }
    if tamper == "status":
        payload["status"] = "CELL_FAILED"
    elif tamper == "closure":
        payload["source_closure"]["launch_final_closure_equal"] = False
    elif tamper == "initial_state":
        payload["initial_state"]["state_dict_sha256"] = "0" * 64
    elif tamper == "theta":
        payload["integrity"]["theta_authority"]["sha256"] = "0" * 64
    elif tamper == "normalizer":
        payload["data_contract"]["behavior_normalizer_semantic_sha256"] = "deadbeef"
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


class _ArmCommonStub:
    sha256_file = staticmethod(_sha_file)


def test_validate_cell_terminal_passes_on_faithful_fixture(tmp_path, monkeypatch):
    _make_cell_fixture(tmp_path, "cellR_elementwise")
    canonical = torch.load(
        ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
        map_location="cpu", weights_only=False,
    )
    theta = torch.load(
        ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt",
        map_location="cpu", weights_only=False,
    )
    monkeypatch.setattr(score, "ROOT", tmp_path, raising=False)
    # ROOT is captured at call time through the module global
    record = score.validate_cell_terminal(
        "cellR_elementwise", _ArmCommonStub, canonical, theta
    )
    assert record["mode_0444"] and record["non_symlink"]
    assert record["initial_state_reconciled"] and record["theta_authority_reconciled"]
    assert record["normalizer_reconciled_f062506c"]


@pytest.mark.parametrize(
    "tamper",
    ["status", "closure", "initial_state", "theta", "normalizer", "sidecar", "mode"],
)
def test_validate_cell_terminal_raises_on_each_tampered_field(tmp_path, monkeypatch, tamper):
    _make_cell_fixture(tmp_path, "cellS2_carrier_sector", tamper=tamper)
    canonical = torch.load(
        ROOT / "results/admission_arms_v1/canonical_initial_state.pt",
        map_location="cpu", weights_only=False,
    )
    theta = torch.load(
        ROOT / "results/sparsification_theta_authority_v1/theta_authority.pt",
        map_location="cpu", weights_only=False,
    )
    monkeypatch.setattr(score, "ROOT", tmp_path, raising=False)
    with pytest.raises(SystemExit):
        score.validate_cell_terminal("cellS2_carrier_sector", _ArmCommonStub, canonical, theta)


def test_block_statistics_fields_present():
    source = (ROOT / "scripts/run_sparsify_score.py").read_text()
    for field in (
        "block_mean", "paired_delta_vs_armA_mean",
        "paired_delta_vs_armA_n_positive", "window_share_of_external",
    ):
        assert field in source
    assert score.date_block("sub-M_ses-CO-20140307") == "2014"
    assert score.date_block("sub-M_ses-CO-20150625") == "2015"
    assert score.date_block("sub-M_ses-CO-20141203") == "2014"  # separated by name, not block
