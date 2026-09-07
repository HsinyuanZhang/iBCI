from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


SUA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUA_ROOT / "scripts"))

import aggregate_cebra_pseudosession_sua_terminal as subject  # noqa: E402


EXTERNAL = tuple(f"external_{index}" for index in range(15))
WITHIN = tuple(f"within_{index}" for index in range(6))


def _lattice(t4_external_delta: float, z4_external_delta: float,
             t4_within_delta: float = 0.0, z4_within_delta: float = 0.0):
    parent = {}
    mixed = {}
    for seed in subject.SEEDS:
        parent[seed] = {domain: {} for domain in subject.DOMAINS}
        mixed[seed] = {domain: {} for domain in subject.DOMAINS}
        for arm, external_delta, within_delta in (
            ("t4", t4_external_delta, t4_within_delta),
            ("z4", z4_external_delta, z4_within_delta),
        ):
            parent[seed]["external_subject_M"][arm] = {name: 0.2 for name in EXTERNAL}
            parent[seed]["within_subject"][arm] = {name: 0.5 for name in WITHIN}
            mixed[seed]["external_subject_M"][arm] = {
                name: 0.2 + external_delta for name in EXTERNAL
            }
            mixed[seed]["within_subject"][arm] = {
                name: 0.5 + within_delta for name in WITHIN
            }
    return parent, mixed


def test_terminal_carrier_specific_positive_gate() -> None:
    parent, mixed = _lattice(0.05, 0.00)
    result = subject.compute_terminal(parent, mixed)
    assert result["accuracy_effective"] is True
    assert result["carrier_specific"] is True
    assert result["verdict"] == "TERMINAL_CARRIER_SPECIFIC_PSEUDOSESSION_EFFECTIVE"
    assert result["summary"]["external_t4_positive_session_count"] == 15
    assert result["summary"]["external_t4_crossed_seed_session_bootstrap_95ci"] == pytest.approx([0.05, 0.05])


def test_terminal_generic_augmentation_is_not_carrier_specific() -> None:
    parent, mixed = _lattice(0.05, 0.05)
    result = subject.compute_terminal(parent, mixed)
    assert result["accuracy_effective"] is True
    assert result["carrier_specific"] is False
    assert result["verdict"] == "TERMINAL_GENERIC_PSEUDOSESSION_AUGMENTATION_EFFECTIVE"


def test_terminal_absolute_t4_gate_cannot_be_rescued_by_collapsing_z4() -> None:
    parent, mixed = _lattice(0.01, -0.20)
    result = subject.compute_terminal(parent, mixed)
    assert result["summary"]["mean_external_carrier_interaction"] > 0.20
    assert result["accuracy_effective"] is False
    assert result["carrier_specific"] is False
    assert result["verdict"] == "TERMINAL_PSEUDOSESSION_NEGATIVE__STOP_ROUTE"


def test_terminal_within_regression_stops_external_gain() -> None:
    parent, mixed = _lattice(0.05, 0.00, t4_within_delta=-0.031)
    result = subject.compute_terminal(parent, mixed)
    assert result["accuracy_gates"]["mean_external_t4_delta_at_least_0p03"] is True
    assert result["accuracy_gates"]["mean_within_t4_delta_at_least_minus_0p03"] is False
    assert result["accuracy_effective"] is False


def test_terminal_requires_all_three_positive_seed_deltas() -> None:
    parent, mixed = _lattice(0.05, 0.00)
    for session in EXTERNAL:
        mixed[44]["external_subject_M"]["t4"][session] = 0.19
    result = subject.compute_terminal(parent, mixed)
    assert result["summary"]["mean_external_t4_delta"] >= 0.03
    assert result["accuracy_gates"]["all_three_external_t4_seed_deltas_positive"] is False
    assert result["accuracy_effective"] is False


@pytest.mark.parametrize("sha256sum_style", [False, True])
def test_load_pair_accepts_both_lineage_sidecar_formats(tmp_path, sha256sum_style) -> None:
    path = tmp_path / "receipt.json"
    payload = {"receipt_kind": "test_kind", "value": 1}
    path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = tmp_path / "receipt.json.sha256"
    sidecar.write_text(
        f"{digest}  {path.name}\n" if sha256sum_style else f"{digest}\n",
        encoding="ascii",
    )
    loaded, actual = subject.load_pair(path, "test_kind")
    assert loaded == payload
    assert actual == digest


def test_load_pair_rejects_wrong_sha256sum_filename(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"receipt_kind": "test_kind"}), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "receipt.json.sha256").write_text(
        f"{digest}  other.json\n", encoding="ascii"
    )
    with pytest.raises(subject.TerminalAggregateError, match="sidecar drift"):
        subject.load_pair(path, "test_kind")
