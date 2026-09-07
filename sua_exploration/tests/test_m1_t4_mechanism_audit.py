from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_m1_t4_mechanism.py"
DATA = ROOT / "SPINT-main/data"
ARTIFACT = ROOT / "sua_exploration/results/m1_t4_mechanism_v1/audit.json"

requires_falcon_data = pytest.mark.skipif(
    not (DATA / "000941").is_dir() or not (DATA / "000953").is_dir(),
    reason="local FALCON M1/M2 dandisets are not present",
)


def load_module():
    spec = importlib.util.spec_from_file_location("m1_t4_mechanism_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> dict:
    if ARTIFACT.is_file():
        return json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return load_module().build_audit()


@requires_falcon_data
def test_m2_direction_explains_an_order_of_magnitude_more_than_m1_emg(audit: dict) -> None:
    decomp = audit["variance_decomposition"]
    m2 = decomp["m2_behavior"]["direction_variance_explained"]
    m1 = decomp["m1_behavior"]["direction_variance_explained"]
    assert m2 > 8.0 * m1
    assert m2 > 0.5
    assert m1 < 0.2


@requires_falcon_data
def test_m1_object_identity_beats_direction_for_behavior_and_neural(audit: dict) -> None:
    behavior = audit["variance_decomposition"]["m1_behavior"]
    neural = audit["variance_decomposition"]["m1_neural"]
    assert behavior["obj_id_variance_explained"] > behavior["direction_variance_explained"]
    assert neural["obj_id_variance_explained"] > neural["direction_variance_explained"]
    assert neural["direction_variance_explained"] == pytest.approx(0.14, abs=0.001)
    assert neural["obj_id_variance_explained"] == pytest.approx(0.369, abs=0.001)


@requires_falcon_data
def test_neural_qualitative_conclusions_are_invariant_across_rate_definitions(audit: dict) -> None:
    decomp = audit["variance_decomposition"]
    neural = decomp["m1_neural"]
    superseded = neural["superseded_binned_count_estimand"]
    invariance = decomp["qualitative_invariance"]
    for block in (neural, superseded):
        assert 0.12 < block["direction_variance_explained"] < 0.15
        assert 0.35 < block["obj_id_variance_explained"] < 0.38
        assert (
            block["obj_id_variance_explained"] / block["direction_variance_explained"]
            > 2.5
        )
    assert 2.5 < invariance["published_obj_over_direction_ratio"] < 3.0
    assert 2.5 < invariance["superseded_obj_over_direction_ratio"] < 3.0
    assert "does not depend on this resolution" in invariance["statement"]


@requires_falcon_data
def test_m1_directions_lie_in_a_half_plane_while_m2_does_not(audit: dict) -> None:
    coverage = audit["direction_coverage"]
    assert coverage["m1"]["all_sessions_half_plane"] is True
    assert coverage["m1"]["direction_range_deg"] == [0.0, 157.5]
    assert coverage["m2"]["covers_full_circle"] is True
    assert coverage["m2"]["direction_range_deg"][0] < 0.0


@requires_falcon_data
def test_m1_condition_number_is_invariant_while_sd_c_scales_like_inverse_sqrt_n(
    audit: dict,
) -> None:
    curve = audit["direction_geometry"]["m1_support_size_curve"]["values"]
    conditions = [row["design_condition_number"] for row in curve]
    sd_values = [row["sd_c_analytic"] for row in curve]
    assert max(conditions) - min(conditions) < 0.15
    assert conditions[0] > 4.0
    ratios = [sd_values[index] / sd_values[index + 1] for index in range(len(sd_values) - 1)]
    for index, ratio in enumerate(ratios):
        support_ratio = (
            curve[index + 1]["support_trials"] / curve[index]["support_trials"]
        ) ** 0.5
        assert abs(ratio - support_ratio) < 0.12


@requires_falcon_data
def test_all_four_obj_id_levels_appear_within_first_ten_held_out_calibration_trials(
    audit: dict,
) -> None:
    deploy = audit["label_deployability"]
    assert deploy["all_four_obj_id_levels_in_first_10"] is True
    assert len(deploy["held_out_calib_sessions"]) == 3
    for session in deploy["held_out_calib_sessions"]:
        assert len(session["obj_id_levels_in_first_10"]) == 4


@requires_falcon_data
def test_interpretation_boundaries_are_recorded(audit: dict) -> None:
    boundaries = audit["interpretation_boundaries"]
    assert "linear" in boundaries["linearity_scope"].lower()
    assert "coincidence" in boundaries["direction_ratio_coincidence"].lower()
    assert "interaction" in boundaries["condition_id_vs_joint"].lower()
    assert "lacks decodable information" in boundaries["no_information_loss_claim"]
    assert "spike_times" in boundaries["neural_estimand_resolution"].lower()


@requires_falcon_data
def test_neural_reference_mismatch_is_resolved(audit: dict) -> None:
    comparison = audit["reference_comparison"]
    assert comparison["all_match_within_tolerance"] is True
    assert comparison["mismatches"] == []
    assert len(comparison["resolved_mismatches"]) == 1
    resolved = comparison["resolved_mismatches"][0]
    assert resolved["status"] == "resolved"
    assert "binned" in resolved["definitional_difference"].lower()
    assert resolved["published_values"]["direction_variance_explained"] == 0.14
    assert resolved["published_values"]["obj_id_variance_explained"] == 0.369


@requires_falcon_data
def test_audit_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "audit"
    module.run(out)
    with pytest.raises(FileExistsError):
        module.run(out)


@requires_falcon_data
def test_written_artifact_matches_recorded_digest(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "audit"
    written = module.run(out)
    recorded = (out / "audit.sha256").read_text().split()[0]
    assert recorded == module.sha256(written)


@requires_falcon_data
def test_committed_artifact_matches_recorded_sha256() -> None:
    module = load_module()
    assert ARTIFACT.is_file()
    digest = module.sha256(ARTIFACT)
    recorded = (ARTIFACT.parent / "audit.sha256").read_text().split()[0]
    assert recorded == digest
