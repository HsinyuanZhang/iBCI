from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mc_maze import setkv_delta_forward_core as core


ROOT = Path(__file__).resolve().parents[2]


def _script(name: str):
    path = ROOT / "sua_exploration" / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(value: float) -> dict:
    return {"per_session_mean_r2": {"s1": value, "s2": value}}


def test_intervention_lattice_has_matched_t4_z4_and_two_controls() -> None:
    assert core.INTERVENTIONS == {
        "setkv_t4": {"source_arm": "source_t4", "decode_mode": "carrier", "carrier_mode": "aligned"},
        "setkv_z4": {"source_arm": "source_z4", "decode_mode": "carrier", "carrier_mode": "zero"},
        "setkv_rs4": {"source_arm": "source_t4", "decode_mode": "carrier", "carrier_mode": "row_shuffle"},
        "duplicate_activity_t4": {
            "source_arm": "source_t4",
            "decode_mode": "duplicate_activity",
            "carrier_mode": None,
        },
    }


def test_domain_summary_separates_carrier_interaction_attachment_and_set_size() -> None:
    aggregate = _script("aggregate_setkv_delta_forward.py")
    observed = aggregate.summarize_domain(
        {
            "setkv_t4": _row(0.60),
            "setkv_z4": _row(0.31),
            "setkv_rs4": _row(0.54),
            "duplicate_activity_t4": _row(0.56),
        },
        {"source_t4": _row(0.55), "source_z4": _row(0.30)},
    )
    assert observed["deltas_from_matched_a2_parent"]["setkv_t4"] == pytest.approx(0.05)
    assert observed["deltas_from_matched_a2_parent"]["setkv_z4"] == pytest.approx(0.01)
    assert observed["carrier_specific_interaction"] == pytest.approx(0.04)
    assert observed["attachment_contrast_setkv_t4_minus_rs4"] == pytest.approx(0.06)
    assert observed["specificity_vs_duplicate_delta"] == pytest.approx(0.04)


def test_score_refuses_body_or_sidecar_collision_before_execution(tmp_path: Path) -> None:
    scorer = _script("score_setkv_delta_forward.py")
    body = tmp_path / "cell.json"
    scorer._require_fresh_output(body)
    body.write_text("occupied", encoding="utf-8")
    with pytest.raises(scorer.SetKVForwardScoreError, match="already exists"):
        scorer._require_fresh_output(body)
    body.unlink()
    Path(str(body) + ".sha256").write_text("occupied", encoding="utf-8")
    with pytest.raises(scorer.SetKVForwardScoreError, match="sidecar already exists"):
        scorer._require_fresh_output(body)


def test_execution_outputs_are_canonical_and_cannot_create_extra_score_or_terminal_receipts(tmp_path: Path) -> None:
    preflight = _script("preflight_setkv_delta_forward.py")
    scorer = _script("score_setkv_delta_forward.py")
    aggregate = _script("aggregate_setkv_delta_forward.py")
    with pytest.raises(core.SetKVForwardContractError, match="canonical path"):
        preflight._require_canonical_output(tmp_path / "extra_preflight.json")
    with pytest.raises(scorer.SetKVForwardScoreError, match="canonical cell path"):
        scorer._require_canonical_output("setkv_t4", "within_subject", tmp_path / "extra.json")
    with pytest.raises(aggregate.SetKVAggregateError, match="canonical path"):
        aggregate._require_canonical_output(tmp_path / "extra_terminal.json")


def test_incomplete_domain_lattice_fails_closed() -> None:
    aggregate = _script("aggregate_setkv_delta_forward.py")
    with pytest.raises(aggregate.SetKVAggregateError, match="incomplete SetKV"):
        aggregate.summarize_domain(
            {"setkv_t4": _row(0.5)},
            {"source_t4": _row(0.4), "source_z4": _row(0.3)},
        )
