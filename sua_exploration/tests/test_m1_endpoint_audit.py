from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_m1_endpoint_disjointness.py"
DATA = ROOT / "SPINT-main/data"
V2_ARTIFACT = ROOT / "sua_exploration/results/m1_endpoint_infeasibility_v2/audit.json"
V2_ARTIFACT_SHA256 = (
    "d448e8015bf99e244e37b9c3fbf0329f49b74c41c2d2c7a78b60482e4b429a1f"
)

requires_falcon_data = pytest.mark.skipif(
    not (DATA / "000941").is_dir() or not (DATA / "000953").is_dir(),
    reason="local FALCON M1/M2 dandisets are not present",
)


def load_module():
    spec = importlib.util.spec_from_file_location("m1_endpoint_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> dict:
    return load_module().build_audit(include_datamodule_check=False)


@requires_falcon_data
def test_no_m1_heldout_session_has_a_query_after_the_frozen_support(audit: dict) -> None:
    endpoint = audit["m1"]["local_heldout_calib_endpoint"]
    assert endpoint["support_prefix_trials"] == 10
    assert endpoint["total_session_count"] == 3
    assert endpoint["eligible_session_count"] == 0
    for session in endpoint["sessions"].values():
        assert session["n_trials"] == 10
        assert session["query_trials_after_support"] == 0
        assert session["has_disjoint_query"] is False


@requires_falcon_data
def test_every_m1_heldin_minival_is_a_prefix_inside_its_own_support(audit: dict) -> None:
    endpoint = audit["m1"]["internal_loso_heldin_endpoint"]
    assert endpoint["total_session_count"] == 4
    assert endpoint["contaminated_session_count"] == 4
    for session in endpoint["sessions"].values():
        assert session["minival_is_exact_prefix_of_calib"] is True
        assert session["same_session_start_time"] is True
        assert session["scored_trials_inside_support_prefix"] is True
        assert session["minival"]["n_trials"] < session["support_prefix_trials"]
        assert session["minival_last_stop_time"] < session["calib_support_boundary_stop_time"]


@requires_falcon_data
def test_each_heldin_session_has_post_support_trials_after_frozen_support(audit: dict) -> None:
    constructible = audit["m1"]["constructible_endpoint"]
    assert constructible["frozen_support_budget_trials"] == 10
    assert constructible["total_session_count"] == 4
    for session in constructible["sessions"].values():
        assert session["total_calib_trials"] > 10
        assert session["post_support_trials"] == session["total_calib_trials"] - 10
        assert session["has_nonempty_post_support"] is True
        assert session["first_post_support_trial_start_time"] > session["support_boundary_stop_time"]
        assert session["session_end_stop_time"] > session["support_boundary_stop_time"]
    assert len(constructible["sessions_with_nonempty_post_support"]) == 4


@requires_falcon_data
def test_loso_left_out_session_is_absent_from_train_sessions(audit: dict) -> None:
    module = load_module()
    evidence = module.datamodule_evidence()
    isolation = evidence["loso_session_isolation"]
    assert len(isolation) == 4
    for fold in isolation.values():
        assert fold["left_out_session_absent_from_train"] is True
        left_out = fold["left_out_session"]
        assert left_out not in fold["train_session_names"]
        assert left_out in fold["val_heldin_session_names"]


@requires_falcon_data
def test_m2_shares_the_heldin_structure_but_keeps_a_valid_heldout_endpoint(audit: dict) -> None:
    cross = audit["m2_cross_check"]
    heldin = cross["internal_loso_heldin_endpoint"]
    assert heldin["contaminated_session_count"] == heldin["total_session_count"] > 0
    heldout = cross["local_heldout_calib_endpoint"]
    assert heldout["support_prefix_trials"] == 33
    # M2 retains genuine future queries; this is what made its correction possible.
    assert 0 < heldout["eligible_session_count"] <= heldout["total_session_count"]


@requires_falcon_data
def test_m1_direction_labels_never_cover_the_full_circle(audit: dict) -> None:
    coverage = audit["m1"]["direction_coverage"]
    assert coverage["covers_full_circle"] is False
    assert coverage["all_directions_within_one_half_plane"] is True
    assert coverage["max_angular_span_deg"] < 180.0
    for name, session in coverage["sessions"].items():
        # The design is usable; the limitation is arc coverage, not rank.
        assert session["support_design_rank"] == 3
        assert session["angular_span_deg"] < 180.0
        expected = 4 if name.startswith("held-out-calib/") else 8
        assert session["n_unique_directions"] == expected


@requires_falcon_data
def test_claim_revisions_keep_the_negative_and_withdraw_the_content_claim(audit: dict) -> None:
    revisions = audit["claim_revisions"]
    assert "Retained and strengthened" in revisions["m1_t4_minus_f0_no_net_gain"]
    assert "Withdrawn as evidence" in revisions["m1_t4_content_distinguishable_from_ts4"]
    assert "Permanently void" in revisions["m1_local_heldout_effect_sizes"]
    assert "M1 native MUA lacks functional calibration information" in audit["explicitly_not_claimed"]


@requires_falcon_data
def test_v2_withdraws_the_no_endpoint_and_evalai_only_overclaims(audit: dict) -> None:
    withdrawn = audit["withdrawn_claims"]
    assert "no_local_disjoint_endpoint" in withdrawn
    assert "evalai_only_route" in withdrawn
    assert audit["supersedes"]["sha256"] == (
        "8fc07e6fc1bb697aabae2487bc3c0d8d50f9ad9a4396ff1f61f7f2f6bc5b4b5a"
    )
    assert audit["constructible_local_endpoint"]["requires_datamodule_plumbing_change"] is True
    assert audit["constructible_local_endpoint"]["status"] == "available_not_evaluated"
    assert "only_remaining_m1_endpoint" not in audit
    for conclusion in audit["conclusions"]:
        assert "only route to a support/query-disjoint M1 result" not in conclusion


@requires_falcon_data
def test_frozen_input_hashes_are_bound(audit: dict) -> None:
    frozen = audit["frozen_inputs"]
    assert set(frozen) == {"internal_loso_aggregate_m1", "withdrawn_local_heldout_replay"}
    for record in frozen.values():
        assert len(record["sha256"]) == 64


@requires_falcon_data
def test_audit_refuses_to_overwrite_an_existing_directory(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "audit"
    module.run(out, include_datamodule_check=False)
    assert (out / "audit.json").is_file()
    assert (out / "audit.sha256").is_file()
    with pytest.raises(FileExistsError):
        module.run(out, include_datamodule_check=False)


@requires_falcon_data
def test_audit_refuses_to_write_when_a_frozen_input_drifts(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    path, _ = module.FROZEN_INPUTS["internal_loso_aggregate_m1"]
    monkeypatch.setitem(module.FROZEN_INPUTS, "internal_loso_aggregate_m1", (path, "0" * 64))
    with pytest.raises(ValueError, match="SHA-256 drift"):
        module.run(tmp_path / "drift", include_datamodule_check=False)


@requires_falcon_data
def test_written_audit_matches_its_recorded_digest(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "audit"
    written = module.run(out, include_datamodule_check=False)
    recorded = (out / "audit.sha256").read_text().split()[0]
    assert recorded == module.sha256(written)
    assert json.loads(written.read_text())["m1"]["local_heldout_calib_endpoint"][
        "eligible_session_count"
    ] == 0


@requires_falcon_data
def test_v2_committed_artifact_matches_recorded_sha256() -> None:
    module = load_module()
    assert V2_ARTIFACT.is_file()
    digest = module.sha256(V2_ARTIFACT)
    assert digest == V2_ARTIFACT_SHA256
    recorded = (V2_ARTIFACT.parent / "audit.sha256").read_text().split()[0]
    assert recorded == V2_ARTIFACT_SHA256
    payload = json.loads(V2_ARTIFACT.read_text())
    assert payload["schema_version"] == 2
