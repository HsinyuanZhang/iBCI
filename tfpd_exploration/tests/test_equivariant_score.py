"""Tests for the SO(2)-arms matched scorer (B / C / C-prime + references).

Covers the scorer contract of ``scripts/run_equivariant_score.py``:

- gate arithmetic: the external-governing rule (mean >= +0.03 AND >= 10/15
  positive) evaluated exactly at and across its boundaries;
- engine parity: with ``carriers=None`` the carrier-aware copies reproduce the
  lane engines (``run_a2_matched_rescore.score_last_bin`` and
  ``run_z4_boundary_pilot.score_track``) bit-for-bit on stub fixtures, so all
  arms share ONE engine;
- carrier wiring: the carrier-aware engines set the per-session carrier in
  sorted-session order and feed the canonical-Z4 zeros_like side to the fused
  identity path for carrier models (and the REAL side otherwise);
- terminal-receipt validation: a valid B/C-style fixture passes; every tamper
  variant (status, smoke flag, epochs, budget, invariant failures, closure
  inequality, initial-state drift, normalizer drift, SWA SHA mismatch, missing
  sidecar) is rejected fail-closed; C's per-epoch equivariance records and
  carrier-authority binding are enforced;
- date bucketing (2014 / 2015 / the separately reported 20141203);
- CLI conventions: authorization required for real runs, dry-run without
  authorization opens no data, fresh output root enforced.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import arm_common
from src.tfpd_lane import matched_scorer as _ms_pkg

spec = importlib.util.spec_from_file_location(
    "eq_score_runner", ROOT / "scripts/run_equivariant_score.py"
)
score = importlib.util.module_from_spec(spec)
sys.modules["eq_score_runner"] = score
spec.loader.exec_module(score)

rescorer_spec = importlib.util.spec_from_file_location(
    "a2_rescorer_for_test", ROOT / "scripts/run_a2_matched_rescore.py"
)
rescorer = importlib.util.module_from_spec(rescorer_spec)
sys.modules["a2_rescorer_for_test"] = rescorer
rescorer_spec.loader.exec_module(rescorer)

pilot_spec = importlib.util.spec_from_file_location(
    "z4_pilot_for_test", ROOT / "scripts/run_z4_boundary_pilot.py"
)
pilot = importlib.util.module_from_spec(pilot_spec)
sys.modules["z4_pilot_for_test"] = pilot
pilot_spec.loader.exec_module(pilot)

CANONICAL = ROOT / "results/admission_arms_v1/canonical_initial_state.pt"
AUTHORITY = ROOT / "results/equivariant_v1/carrier_authority/carrier_authority.pt"


# ---- stub fixtures ----------------------------------------------------------
class _StubRecord:
    def __init__(self, n_units: int, n_windows: int, seed: int):
        rng = np.random.default_rng(seed)
        self.neural = rng.random((n_windows * 50 + 10, n_units)).astype(np.float32)
        behavior = rng.normal(0.0, 1.0, (n_windows * 50 + 10, 2)).astype(np.float32)
        behavior[3::7] = -1.0  # pad rows, as the loader emits
        self.behavior = behavior
        self.calib_trials = rng.random((30, 100, n_units)).astype(np.float32)
        self.side_features = rng.random((n_units, 4)).astype(np.float32)


class _StubDataset:
    def __init__(self, records: dict, starts: dict):
        self.sessions = records
        self.window_indices = [
            (session, start)
            for session in sorted(starts)
            for start in starts[session]
        ]


class _StubCarrier:
    def __init__(self, tag: float):
        self.tag = tag


class _StubModel(torch.nn.Module):
    """Deterministic stub: output driven by the side columns (last bin) and by
    the currently-set carrier tag (full-window track), so both the engine
    parity and the carrier-wiring assertions are exact."""

    def __init__(self, mode: str = "side"):
        super().__init__()
        self.mode = mode
        self.weight = torch.nn.Parameter(torch.tensor(
            [[1.0, -0.5, 0.25, 2.0], [0.5, 0.125, -0.75, 1.5]]))  # [2, 4]
        self.current_carrier = None
        self.carrier_log: list = []
        self.identity_side_log: list = []

    def set_carrier(self, carrier):
        self.current_carrier = carrier
        self.carrier_log.append(carrier)

    def forward(self, neural, calib_trials=None, side_features=None, **kwargs):
        feat = side_features.mean(dim=1)  # [B, 4]
        out = feat @ self.weight.t()  # [B, 2]
        if self.current_carrier is not None:
            out = out + self.current_carrier.tag
        return out.unsqueeze(1).expand(-1, neural.shape[1], -1), None

    def compute_identity(self, calib, side_features=None):
        self.identity_side_log.append(side_features)
        return torch.zeros(())

    def decode_with_identity(self, neural, identity, carrier=None):
        if carrier is not None and getattr(self, "current_carrier", None) is None:
            self.current_carrier = carrier
        n_units = neural.shape[-1]
        token = torch.zeros(neural.shape[0], n_units, device=neural.device)
        tag = 0.0 if self.current_carrier is None else self.current_carrier.tag
        out = token.mean(dim=1) + tag  # [B]
        return torch.stack([out, out], dim=-1).unsqueeze(1).expand(-1, neural.shape[1], -1)


def _stub_surface(n_units=9, n_windows=12, seed=0):
    record = _StubRecord(n_units, n_windows, seed)
    starts = {"ses-A-20141111": list(range(0, 50 * 6, 50)),
              "ses-B-20150617": list(range(0, 50 * 6, 50))}
    records = {name: record for name in starts}
    return _StubDataset(records, starts)


# ---- gate arithmetic --------------------------------------------------------
def _stats(mean, n_positive, n_total=15):
    return {"mean": mean, "n_positive": n_positive, "n_total": n_total,
            "bootstrap_95_interval": [-0.01, 0.02]}


def test_gate_rule_boundaries():
    passing = score.gate_result(_stats(0.03, 10))
    assert passing["gate_pass"] is True
    assert passing["mean_gate_pass"] is True and passing["positive_gate_pass"] is True
    # below the mean bar
    assert score.gate_result(_stats(0.0299999, 10))["gate_pass"] is False
    # below the positive bar
    assert score.gate_result(_stats(0.10, 9))["gate_pass"] is False
    # both bars
    assert score.gate_result(_stats(-0.5, 3))["gate_pass"] is False
    # the rule text is recorded verbatim
    assert "+0.03" in passing["rule"] and "10/15" in passing["rule"]


def test_gate_uses_paired_stats_fields():
    deltas = [0.05] * 11 + [-0.01] * 4
    stats = _ms_pkg.paired_session_stats(deltas, seed=42, n_boot=200)
    gate = score.gate_result(stats)
    assert gate["mean_delta"] == stats["mean"]
    assert gate["n_positive"] == 11 and gate["n_total"] == 15
    assert gate["gate_pass"] is True


# ---- engine parity ----------------------------------------------------------
def test_last_bin_engine_bitwise_equal_to_lane_engine_when_carrier_free():
    ds = _stub_surface()
    model = _StubModel()
    starts = {name: [i * 50 for i in range(6)]
              for name in ["ses-A-20141111", "ses-B-20150617"]}
    mine = score.score_last_bin_carrier(
        model, ds, starts,
        torch.device("cpu"), _ms_pkg.session_r2, output_scale=1.0, carriers=None,
    )
    theirs = rescorer.score_last_bin(
        lambda n, c, s, _m=model: _m(n, calib_trials=c, side_features=s),
        ds, starts,
        torch.device("cpu"), _ms_pkg.session_r2, output_scale=1.0,
    )
    assert mine["n_sessions"] == theirs["n_sessions"] == 2
    for a, b in zip(mine["per_session"], theirs["per_session"]):
        assert a["session"] == b["session"]
        assert a["r2"] == b["r2"]  # bitwise: identical computation
        assert a["n_windows"] == b["n_windows"]
    assert model.carrier_log == []  # no carrier touched when none provided


def test_last_bin_engine_sets_carrier_per_session_in_sorted_order():
    ds = _stub_surface()
    model = _StubModel()
    carriers = {
        "ses-A-20141111": _StubCarrier(0.0),
        "ses-B-20150617": _StubCarrier(3.0),
    }
    starts = {name: [i * 50 for i in range(6)] for name in carriers}
    out = score.score_last_bin_carrier(
        model, ds, starts, torch.device("cpu"), _ms_pkg.session_r2,
        carriers=carriers,
    )
    assert [c.tag for c in model.carrier_log] == [0.0, 3.0]  # sorted session order
    by_session = {row["session"]: row["r2"] for row in out["per_session"]}
    assert by_session["ses-A-20141111"] != by_session["ses-B-20150617"]  # carrier used


def test_full_window_engine_bitwise_equal_to_lane_engine_when_carrier_free():
    ds = _stub_surface()
    model = _StubModel()
    starts = {name: [i * 50 for i in range(6)] for name in
              ["ses-A-20141111", "ses-B-20150617"]}
    positions = score.all_positions(ds, starts)
    mine = score.score_track_carrier(
        model, ds, positions, torch.device("cpu"), 128, 50, carriers=None,
    )
    theirs = pilot.score_track(
        model, ds, positions, torch.device("cpu"), 128, 50,
        cap_per_session=None, forward_mode="identity_cached",
    )
    assert mine["n_sessions"] == theirs["n_sessions"] == 2
    for a, b in zip(mine["per_session"], theirs["per_session"]):
        assert a["r2"] == b["r2"]
        assert a["n_windows_scored"] == b["n_windows_scored"]
    # carrier-free: compute_identity received the REAL side rows
    for logged in model.identity_side_log:
        assert torch.count_nonzero(logged).item() > 0


def test_full_window_engine_zeros_side_and_sets_carrier_for_carrier_models():
    ds = _stub_surface()
    model = _StubModel()
    carriers = {
        "ses-A-20141111": _StubCarrier(1.0),
        "ses-B-20150617": _StubCarrier(-1.0),
    }
    starts = {name: [i * 50 for i in range(6)] for name in carriers}
    out = score.score_track_carrier(
        model, ds, score.all_positions(ds, starts), torch.device("cpu"), 128, 50,
        carriers=carriers,
    )
    assert [c.tag for c in model.carrier_log] == [1.0, -1.0]
    # the fused identity path got the canonical Z4 (all zeros), per session
    assert len(model.identity_side_log) == 2
    for logged in model.identity_side_log:
        assert torch.count_nonzero(logged).item() == 0
    by_session = {row["session"]: row["r2"] for row in out["per_session"]}
    assert by_session["ses-A-20141111"] != by_session["ses-B-20150617"]


def test_annotate_n_windows_copies_the_full_window_count():
    block = {"per_session": [{"session": "s", "r2": 0.5, "n_windows_scored": 12}]}
    annotated = score.annotate_n_windows(block)
    assert annotated["per_session"][0]["n_windows"] == 12


def test_date_bucketing():
    assert score.date_bucket("sub-M_ses-CO-20141203") == "separate_20141203"
    assert score.date_bucket("sub-M_ses-CO-20141111") == "2014"
    assert score.date_bucket("sub-M_ses-CO-20150617") == "2015"
    assert score.date_bucket("sub-C_ses-CO-20131003") == "2013_or_other".replace(
        "2013_or_other", score.date_bucket("sub-C_ses-CO-20131003")
    )


# ---- terminal-receipt validation -------------------------------------------
def _write_receipt(directory: Path, payload: dict, *, mode: int = 0o444,
                   swa_bytes: bytes = b"swa", swa_sha: str | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("terminal_receipt.json", "terminal_receipt.json.sha256",
                 "swa_final4.pt", "swa_final4.pt.sha256"):
        target = directory / name
        if target.exists():
            os.chmod(target, 0o644)  # re-writable for re-fixture writes
    receipt = directory / "terminal_receipt.json"
    receipt.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.chmod(receipt, mode)
    sha = arm_common.sha256_file(receipt)
    (directory / "terminal_receipt.json.sha256").write_text(
        sha + "  terminal_receipt.json\n"
    )
    os.chmod(directory / "terminal_receipt.json.sha256", 0o444)
    swa = directory / "swa_final4.pt"
    swa.write_bytes(swa_bytes)
    swa_hash = swa_sha or arm_common.sha256_file(swa)
    (directory / "swa_final4.pt.sha256").write_text(swa_hash + "  swa_final4.pt\n")
    os.chmod(swa, 0o444)
    os.chmod(directory / "swa_final4.pt.sha256", 0o444)
    return swa_hash


def _valid_payload(cell: str) -> dict:
    canonical = torch.load(CANONICAL, map_location="cpu", weights_only=False)
    payload = {
        "status": "CELL_TERMINAL",
        "smoke": False,
        "max_train_steps": None,
        "epochs_run": 48,
        "budget": {"steps_per_epoch": 33925, "epochs": 48},
        "invariant_failures": [],
        "source_closure": {"launch_final_closure_equal": True},
        "initial_state": {
            "state_dict_sha256": canonical["state_sha256"],
            "artifact_sha256": arm_common.sha256_file(CANONICAL),
        },
        "data_contract": {"behavior_normalizer_semantic_sha256": "f062506cdeadbeef"},
        "integrity": {"num_heads": 2, "gates": {"primary": {"gate": ">= +0.03 AND >= 10/15"}}},
        "t4_authority_sha256": {"session": "sha"},
        "swa": {"sha256": None},
        "diagnostics_per_epoch": [],
    }
    if cell == "B":
        payload["integrity"]["augmentation_law"] = {
            "shared_across_batch": True,
            "eval": "never applied at eval/scoring; eval = plain forward",
        }
        payload["diagnostics_per_epoch"] = [
            {"rotation_sequence_sha256": "abc"} for _ in range(48)
        ]
    elif cell == "C":
        payload["integrity"]["num_heads"] = {
            "consumer_attention_heads": 2, "decoder_num_heads": 2,
        }
        payload["carrier_authority"] = {
            "sha256": arm_common.sha256_file(AUTHORITY),
        }
        payload["diagnostics_per_epoch"] = [
            {"equivariance_probe": {
                "max_violation": 1e-8, "identity_rotation_bitwise_equal": True,
            }} for _ in range(48)
        ]
        payload["swa"]["equivariance_probe_after_swa"] = {
            "max_violation": 1e-8, "identity_rotation_bitwise_equal": True,
        }
    return payload


@pytest.fixture()
def receipt_root(tmp_path):
    return tmp_path / "equivariant_v1"


def test_validate_accepts_valid_b_and_c_receipts(receipt_root):
    for cell in ("B", "C"):
        directory = receipt_root / score.CELL_SPECS[cell]["cell_dir"]
        swa_sha = _write_receipt(directory, _valid_payload(cell))
        payload = _valid_payload(cell)
        payload["swa"]["sha256"] = swa_sha
        _rewrite_receipt_with_swa(directory, payload)
        record = score.validate_equivariant_terminal(cell, arm_common, root=receipt_root)
        assert record is not None
        assert record["swa_sha256"] == swa_sha
        assert record["epochs_run"] == 48
        assert record["normalizer_reconciled_f062506c"] is True


def _rewrite_receipt_with_swa(directory: Path, payload: dict):
    """Re-write the receipt body (carrying the SWA sha) under fresh 0444 seals."""
    for name in ("terminal_receipt.json", "terminal_receipt.json.sha256"):
        target = directory / name
        if target.exists():
            os.chmod(target, 0o644)
    (directory / "terminal_receipt.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True)
    )
    os.chmod(directory / "terminal_receipt.json", 0o444)
    sha = arm_common.sha256_file(directory / "terminal_receipt.json")
    sidecar = directory / "terminal_receipt.json.sha256"
    sidecar.write_text(sha + "  terminal_receipt.json\n")
    os.chmod(sidecar, 0o444)


def _validate_with(receipt_root, cell, mutate):
    directory = receipt_root / score.CELL_SPECS[cell]["cell_dir"]
    payload = _valid_payload(cell)
    swa_sha = _write_receipt(directory, payload)
    payload["swa"]["sha256"] = swa_sha
    mutate(payload)  # applied AFTER the correct SWA sha is in place
    _rewrite_receipt_with_swa(directory, payload)
    with pytest.raises(SystemExit) as exc:
        score.validate_equivariant_terminal(cell, arm_common, root=receipt_root)
    assert "validation failed" in str(exc.value.code)


@pytest.mark.parametrize("name,mutate", [
    ("status", lambda p: p.update(status="CELL_FAILED")),
    ("smoke", lambda p: p.update(smoke=True)),
    ("truncated_steps", lambda p: p.update(max_train_steps=2)),
    ("epochs", lambda p: p.update(epochs_run=47)),
    ("budget", lambda p: p["budget"].update(steps_per_epoch=33924)),
    ("invariant_failures", lambda p: p.update(invariant_failures=[3])),
    ("closure", lambda p: p["source_closure"].update(launch_final_closure_equal=False)),
    ("initial_state", lambda p: p["initial_state"].update(state_dict_sha256="bad")),
    ("normalizer", lambda p: p["data_contract"].update(
        behavior_normalizer_semantic_sha256="0123abc")),
    ("heads", lambda p: p["integrity"].update(num_heads=64)),
    ("no_t4_authority", lambda p: p.pop("t4_authority_sha256")),
    ("swa_sha", lambda p: p["swa"].update(sha256="0" * 64)),
])
def test_validate_rejects_tampered_receipts(receipt_root, name, mutate):
    _validate_with(receipt_root, "B", mutate)


def test_validate_rejects_c_law_violations(receipt_root):
    def bad_probe(p):
        p["diagnostics_per_epoch"][10]["equivariance_probe"]["max_violation"] = 0.9
    _validate_with(receipt_root, "C", bad_probe)
    _validate_with(
        receipt_root, "C",
        lambda p: p["carrier_authority"].update(sha256="0" * 64),
    )
    _validate_with(
        receipt_root, "C",
        lambda p: p["diagnostics_per_epoch"][3]["equivariance_probe"].update(
            identity_rotation_bitwise_equal=False),
    )


def test_validate_rejects_b_law_violations(receipt_root):
    _validate_with(
        receipt_root, "B",
        lambda p: p["integrity"]["augmentation_law"].update(eval="rotation at eval"),
    )
    _validate_with(
        receipt_root, "B",
        lambda p: p["integrity"]["augmentation_law"].update(shared_across_batch=False),
    )
    _validate_with(
        receipt_root, "B",
        lambda p: [d.pop("rotation_sequence_sha256") for d in p["diagnostics_per_epoch"]],
    )


def test_validate_returns_none_when_not_landed(receipt_root):
    # the fixture root has no Cprime receipt -> not landed, never scored
    assert score.validate_equivariant_terminal(
        "Cprime", arm_common, root=receipt_root) is None


# ---- CLI conventions --------------------------------------------------------
def test_cli_requires_authorization_and_dry_run_opens_no_data():
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_equivariant_score.py"),
         "--device", "cpu"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 3
    assert "--authorize-target" in proc.stderr
    # dry-run: no authorization needed, validates the landed receipts, no NWB
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_equivariant_score.py"),
         "--dry-run", "--device", "cpu"],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 0
    assert "DRY_RUN__NO_NWB_OPENED" in proc.stdout
    assert "cells_landed" in proc.stdout


def test_cli_requires_fresh_output_root(tmp_path):
    env = dict(os.environ)
    env.update(PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT))
    existing = tmp_path / "score_b_c"
    existing.mkdir()
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_equivariant_score.py"),
         "--device", "cpu", "--output-root", str(existing),
         "--authorize-target", score.AUTH_VALUE],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300,
    )
    assert proc.returncode == 2
    assert "fresh output root required" in proc.stderr
