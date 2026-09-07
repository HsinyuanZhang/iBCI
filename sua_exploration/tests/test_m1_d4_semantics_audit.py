from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_m1_d4_semantics.py"
DATA = ROOT / "SPINT-main/data/000941"
ARTIFACT = ROOT / "sua_exploration/results/m1_d4_semantics_v2/audit.json"
requires_data = pytest.mark.skipif(not DATA.is_dir(), reason="local FALCON M1 data absent")


def module():
    spec = importlib.util.spec_from_file_location("m1_d4_semantics_audit", SCRIPT)
    assert spec and spec.loader
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def audit():
    return json.loads(ARTIFACT.read_text()) if ARTIFACT.exists() else module().build_audit()


@requires_data
def test_object_semantic_gate_is_data_derived_and_fails_at_heldout_m10(audit):
    gate = audit["semantic_gate"]
    assert gate["object_factor_identifiable_at_deployment_m10"] is False
    assert gate["decision"] == "rejected_as_object_descriptor"
    tables = audit["held_out_m10_exact_joint_tables"]
    assert len(tables) == 3
    for table in tables:
        assert len({row["tgt_obj"] for row in table["joint_table"]}) == 1


@requires_data
def test_all_requested_prefixes_and_input_hashes_are_recorded(audit):
    assert len(audit["input_files"]) == 7
    assert len(audit["prefix_coverage"]) == 7
    for entry in audit["prefix_coverage"]:
        for support in ("10", "40", "90", "200"):
            assert support in entry["prefixes"]
            assert entry["prefixes"][support]["status"] in {"available", "unavailable"}
            if entry["prefixes"][support]["status"] == "available":
                assert "design_diagnostics" in entry["prefixes"][support]


@requires_data
def test_model_audit_has_both_estimands_and_chronological_cv(audit):
    models = audit["trial_mean_model_audit"]
    for target in ("emg", "neural"):
        assert len(models[target]["sessions"]) == 4
        first = models[target]["sessions"][0]["models"]
        assert "cosine_direction" in first and "obj_id" in first
        assert "chronological_expanding_block_cv" in first["cosine_direction"]


@requires_data
def test_native_mua_rate_loader_matches_frozen_boolean_interval_reference():
    m = module()
    reference_script = ROOT / "sua_exploration/scripts/audit_m1_t4_mechanism.py"
    spec = importlib.util.spec_from_file_location("frozen_m1_rate_reference", reference_script)
    assert spec and spec.loader
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    path = m.files_for("held-in-calib")[0]
    actual = m.movement_window_neural_rates(path)
    expected = reference.m1_spike_time_firing_rates(path)
    assert actual.shape == expected.shape == (414, 64)
    assert actual.max() < 200.0
    assert (actual == expected).all()


@requires_data
def test_reference_neural_insample_values_reproduce_boolean_count_baseline(audit):
    session = next(row for row in audit["trial_mean_model_audit"]["neural"]["sessions"] if row["session"] == "20120924")
    assert session["models"]["cosine_direction"]["in_sample"]["r2"] == pytest.approx(0.140038, abs=1.0e-4)
    assert session["models"]["obj_id"]["in_sample"]["r2"] == pytest.approx(0.369019, abs=1.0e-4)


@requires_data
def test_d4_eligibility_is_predeclared_and_not_decoder_metric(audit):
    decision = audit["predeclared_decision"]
    assert decision["D4_categorical_profile_gpu_eligibility"] in {"eligible_for_minimal_gpu_pilot", "stop_cpu_gate_not_met", "indeterminate_insufficient_defined_m10_sessions"}
    assert decision["Gate_P"]["status"] in {"pass", "fail", "indeterminate"}
    assert "not decoder R2" in audit["deployment_support_neural_prediction"]["rate_estimand"]


@requires_data
def test_run_refuses_overwrite_and_digest_is_exact(tmp_path):
    m = module()
    out = tmp_path / "audit"
    written = m.run(out)
    assert (out / "audit.sha256").read_text().split()[0] == m.sha256(written)
    with pytest.raises(FileExistsError):
        m.run(out)
