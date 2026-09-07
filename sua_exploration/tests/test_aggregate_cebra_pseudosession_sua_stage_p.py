from __future__ import annotations

import hashlib
import json

import pytest

from scripts.aggregate_cebra_pseudosession_sua_stage_p import (
    AggregateError,
    compute_stage_p,
    load_pair,
)


def _lattice(external_t4: float, within_t4: float, external_z4: float = 0.0,
             within_z4: float = 0.0):
    sessions = {
        "within_subject": [f"c{i}" for i in range(6)],
        "external_subject_M": [f"m{i}" for i in range(15)],
    }
    parent = {domain: {arm: {s: 0.0 for s in names} for arm in ("t4", "z4")}
              for domain, names in sessions.items()}
    mixed = {
        "within_subject": {
            "t4": {s: within_t4 for s in sessions["within_subject"]},
            "z4": {s: within_z4 for s in sessions["within_subject"]},
        },
        "external_subject_M": {
            "t4": {s: external_t4 for s in sessions["external_subject_M"]},
            "z4": {s: external_z4 for s in sessions["external_subject_M"]},
        },
    }
    return parent, mixed


def test_synthetic_pass_gate_can_fire() -> None:
    parent, mixed = _lattice(0.04, -0.02, 0.0, 0.0)
    result = compute_stage_p(parent, mixed)
    assert result["passes_stage_p"]
    assert result["verdict"].startswith("STAGE_P_PASS")
    assert result["interpretation"] == "carrier_specific_candidate"


def test_synthetic_accuracy_fail_cannot_be_rescued_by_z4_collapse() -> None:
    parent, mixed = _lattice(0.0, 0.0, -0.20, -0.20)
    result = compute_stage_p(parent, mixed)
    assert result["deltas"]["external_carrier_specific_interaction"] == pytest.approx(0.20)
    assert not result["passes_stage_p"]
    assert result["verdict"].startswith("STAGE_P_STOP")


def test_synthetic_within_regression_stops_external_gain() -> None:
    parent, mixed = _lattice(0.10, -0.04)
    result = compute_stage_p(parent, mixed)
    assert not result["passes_stage_p"]


@pytest.mark.parametrize("sha256sum_style", [False, True])
def test_load_pair_accepts_both_lineage_sidecar_formats(tmp_path, sha256sum_style) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = tmp_path / "receipt.json.sha256"
    sidecar.write_text(
        f"{digest}  {path.name}\n" if sha256sum_style else f"{digest}\n",
        encoding="ascii",
    )
    payload, actual = load_pair(path)
    assert payload == {"value": 1}
    assert actual == digest


def test_load_pair_rejects_wrong_sha256sum_filename(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "receipt.json.sha256").write_text(
        f"{digest}  other.json\n", encoding="ascii"
    )
    with pytest.raises(AggregateError, match="sidecar drift"):
        load_pair(path)
