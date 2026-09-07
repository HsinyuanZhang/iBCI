from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SCRIPTS = ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mc_maze import decoder_error_structure as des
import aggregate_decoder_error_structure as aggregate


def _bundle_from_records(records: list[des.QueryWindowRecord]) -> dict:
    return {
        "windows": [
            {
                "session_name": record.session_name,
                "trial_index": record.trial_index,
                "bin_index": record.bin_index,
                "bins_in_trial": record.bins_in_trial,
                "target": record.target.tolist(),
                "pred_carrier": record.pred_carrier.tolist(),
                "pred_control": record.pred_control.tolist(),
                "unit_modulation_m": record.unit_modulation_m.tolist(),
                "unit_activity": record.unit_activity.tolist(),
                "session_n_directions": record.session_n_directions,
                "session_design_rank": record.session_design_rank,
                "session_design_condition": record.session_design_condition,
                "target_dir_rad": record.target_dir_rad,
            }
            for record in records
        ]
    }


def _receipt_from_records(
    records: list[des.QueryWindowRecord],
    *,
    carrier_sha: str = "a" * 64,
    control_sha: str = "b" * 64,
) -> dict:
    return des.decompose_records(
        records,
        carrier_checkpoint_sha256=carrier_sha,
        control_checkpoint_sha256=control_sha,
    )


def test_planted_effect_localizes_to_affected_directions() -> None:
    affected = (0, 3, 5)
    records = des.build_planted_direction_effect_fixture(seed=11, affected_directions=affected)
    receipt = _receipt_from_records(records)
    direction_axis = receipt["axes"]["target_direction"]["strata"]
    affected_deltas = [
        direction_axis[f"dir_{index}"]["paired_delta_primary"]
        for index in affected
    ]
    unaffected_deltas = [
        direction_axis[f"dir_{index}"]["paired_delta_primary"]
        for index in range(8)
        if index not in affected
    ]
    assert all(delta < -0.01 for delta in affected_deltas)
    assert all(abs(delta) < 1.0e-9 for delta in unaffected_deltas)
    assert receipt["axes"]["target_direction"]["uniform_gain_diagnostics"]["uniform_gain_flag"] is False

    multiplicity = receipt["axes"]["target_direction"]["multiplicity_correction"]["strata"]
    assert all(multiplicity[f"dir_{index}"]["discovery"] for index in affected)
    assert not any(
        multiplicity[f"dir_{index}"]["discovery"]
        for index in range(8)
        if index not in affected
    )


def test_uniform_effect_reports_uniform_direction_gain() -> None:
    records = des.build_uniform_effect_fixture(seed=17)
    receipt = _receipt_from_records(records)
    direction_axis = receipt["axes"]["target_direction"]["strata"]
    relative_improvements = []
    for row in direction_axis.values():
        if row["n_windows"] == 0:
            continue
        control_mse = float(row["control"]["residual"]["mse"])
        carrier_mse = float(row["carrier"]["residual"]["mse"])
        relative_improvements.append((control_mse - carrier_mse) / control_mse)
    assert len(relative_improvements) == 8
    assert float(np.std(relative_improvements)) < 1.0e-9
    assert receipt["axes"]["target_direction"]["uniform_gain_diagnostics"]["uniform_gain_flag"] is True


def test_stratum_edges_are_frozen_not_data_dependent() -> None:
    records_a = des.build_uniform_effect_fixture(seed=1)
    records_b = des.build_planted_direction_effect_fixture(seed=2, affected_directions=(1,))
    receipt_a = _receipt_from_records(records_a)
    receipt_b = _receipt_from_records(records_b)
    assert receipt_a["stratum_definitions"] == des.FROZEN_STRATUM_DEFINITIONS
    assert receipt_b["stratum_definitions"] == receipt_a["stratum_definitions"]
    assert des.SPEED_NORM_EDGES == (0.0, 0.35, 0.70, 1.05, float("inf"))


def test_circular_wrap_assigns_same_direction_bin() -> None:
    records = des.build_wrap_boundary_fixture()
    labels = [des.assign_target_direction_label(record.target, target_dir_rad=record.target_dir_rad) for record in records]
    assert labels[0] == labels[1]


def test_aggregator_rejects_mismatched_query_windows() -> None:
    records = des.build_uniform_effect_fixture(seed=3)
    left = _receipt_from_records(records)
    right = copy.deepcopy(left)
    right["query_window_identity_sha256"] = ["deadbeef" * 8] * len(
        right["query_window_identity_sha256"]
    )
    with pytest.raises(ValueError, match="mismatched query-window identities"):
        aggregate.aggregate_pair(left, right)


def test_sealed_session_guard_raises() -> None:
    records = des.build_uniform_effect_fixture(seed=4)
    records[0] = des.QueryWindowRecord(
        session_name="sub-C_ses-CO-20151113",
        trial_index=records[0].trial_index,
        bin_index=records[0].bin_index,
        bins_in_trial=records[0].bins_in_trial,
        target=records[0].target,
        pred_carrier=records[0].pred_carrier,
        pred_control=records[0].pred_control,
        unit_modulation_m=records[0].unit_modulation_m,
        unit_activity=records[0].unit_activity,
        session_n_directions=records[0].session_n_directions,
        session_design_rank=records[0].session_design_rank,
        session_design_condition=records[0].session_design_condition,
        target_dir_rad=records[0].target_dir_rad,
    )
    with pytest.raises(ValueError, match="refusing sealed formal-test sessions"):
        des.decompose_records(
            records,
            carrier_checkpoint_sha256="c" * 64,
            control_checkpoint_sha256="d" * 64,
        )


def test_determinism_under_fixed_seed() -> None:
    records = des.build_planted_direction_effect_fixture(seed=99, affected_directions=(2, 7))
    first = des.canonical_json_bytes(_receipt_from_records(records))
    second = des.canonical_json_bytes(_receipt_from_records(records))
    assert first == second


def test_decompose_script_round_trip(tmp_path: Path) -> None:
    import decompose_decoder_error as script

    records = des.build_uniform_effect_fixture(seed=5)
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle_from_records(records)), encoding="utf-8")
    carrier_ckpt = tmp_path / "carrier.ckpt"
    control_ckpt = tmp_path / "control.ckpt"
    carrier_ckpt.write_bytes(b"carrier")
    control_ckpt.write_bytes(b"control")
    output = tmp_path / "receipt.json"
    assert (
        script.main(
            [
                "--window-bundle",
                str(bundle_path),
                "--carrier-checkpoint",
                str(carrier_ckpt),
                "--control-checkpoint",
                str(control_ckpt),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["sealed_test_sessions_opened"] is False
    assert body["overall"]["n_windows"] == len(records)


def test_aggregator_accepts_matching_receipts(tmp_path: Path) -> None:
    records = des.build_uniform_effect_fixture(seed=6)
    receipt = _receipt_from_records(records)
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(receipt), encoding="utf-8")
    right_path.write_text(json.dumps(receipt), encoding="utf-8")
    left = aggregate._load_receipt(left_path)
    right = aggregate._load_receipt(right_path)
    result = aggregate.aggregate_pair(left, right)
    assert result["sealed_test_sessions_opened"] is False
    assert result["n_receipts"] == 2
    direction_family = result["multiplicity_correction"]["axes"]["target_direction"]
    assert direction_family["family_size_total"] == 8
    assert direction_family["family_size_tested"] == 8
    assert all(
        "bh_adjusted_p_value" in row
        for row in direction_family["strata"].values()
    )


def test_bh_keeps_planted_stratum_and_rejects_nulls_in_family() -> None:
    affected = (2,)
    records = des.build_planted_direction_effect_fixture(
        seed=21,
        affected_directions=affected,
        windows_per_direction=40,
    )
    receipt = _receipt_from_records(records)
    correction = receipt["axes"]["target_direction"]["multiplicity_correction"]
    assert correction["family_size_tested"] == 8
    assert correction["strata"]["dir_2"]["discovery"] is True
    assert sum(row["discovery"] for row in correction["strata"].values()) == 1


def test_bh_all_null_family_flags_nothing_under_fdr() -> None:
    records = des.build_all_null_direction_fixture(seed=123, windows_per_direction=40)
    receipt = _receipt_from_records(records)
    correction = receipt["axes"]["target_direction"]["multiplicity_correction"]
    assert correction["family_size_tested"] == 8
    assert not any(row["survives_fdr"] for row in correction["strata"].values())
    uncorrected_hits = [
        label
        for label, row in correction["strata"].items()
        if row["uncorrected_significant"]
    ]
    assert len(uncorrected_hits) <= 2


def test_bh_uniform_pvalues_demonstrates_correction_works() -> None:
    rng = np.random.default_rng(0)
    p_values = rng.uniform(0.0, 1.0, size=200)
    adjusted = des.benjamini_hochberg_adjusted(p_values)
    uncorrected = int(np.sum(p_values < des.NOMINAL_ALPHA_UNCORRECTED))
    survives = int(np.sum(adjusted <= des.FDR_Q_THRESHOLD))
    assert 4 <= uncorrected <= 20
    assert survives == 0


def test_empty_and_undersized_strata_excluded_from_family_size() -> None:
    records = des.build_uniform_effect_fixture(seed=30)
    undersized = des.QueryWindowRecord(
        session_name="undersized_session",
        trial_index=0,
        bin_index=0,
        bins_in_trial=6,
        target=np.array([0.2, 0.0]),
        pred_carrier=np.array([0.25, 0.0]),
        pred_control=np.array([0.30, 0.0]),
        unit_modulation_m=np.array([0.1]),
        unit_activity=np.array([1.0]),
        session_n_directions=6,
        session_design_rank=3,
        session_design_condition=8.0,
        target_dir_rad=float(des.CANONICAL_DIRECTIONS_RAD[0]),
    )
    records.append(undersized)
    receipt = _receipt_from_records(records)
    correction = receipt["axes"]["movement_speed"]["multiplicity_correction"]
    excluded = [
        row
        for row in correction["strata"].values()
        if not row["eligible_for_correction"]
    ]
    assert correction["excluded_counts"]["too_few_windows"] >= 1 or correction["excluded_counts"]["too_few_sessions"] >= 1
    assert any(row["exclusion_reason"] in {"too_few_windows", "too_few_sessions"} for row in excluded)
    assert correction["family_size_tested"] < correction["family_size_total"]


def test_uniformity_falsifier_independent_of_fdr_on_uniform_fixture() -> None:
    records = des.build_uniform_effect_fixture(seed=44)
    receipt = _receipt_from_records(records)
    direction_axis = receipt["axes"]["target_direction"]
    assert direction_axis["uniform_gain_diagnostics"]["uniform_gain_flag"] is True
    discoveries = [
        label
        for label, row in direction_axis["multiplicity_correction"]["strata"].items()
        if row["discovery"]
    ]
    assert len(discoveries) in {0, 8}
