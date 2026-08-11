"""Focused tests for the portable T4/Ridge supervision-density audit.

The fixture is tracked with the test.  In particular, these tests deliberately do
not read the ignored ``sua_exploration/results`` tree, which makes the audit
meaningful in a clean clone.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_t4_ridge_supervision_density.py"
PROTOCOL = ROOT / "sua_exploration/tests/fixtures/t4_ridge_supervision_density_protocol_v2.json"


def _module():
    spec = importlib.util.spec_from_file_location("t4_ridge_supervision_density", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_protocol(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_clean_clone_happy_path_uses_only_tracked_protocol(tmp_path: Path) -> None:
    """The default CLI succeeds in a minimal clone with no results tree at all."""
    clean_root = tmp_path / "clean_clone"
    clean_script = clean_root / "sua_exploration/scripts/audit_t4_ridge_supervision_density.py"
    clean_protocol = clean_root / "sua_exploration/tests/fixtures/t4_ridge_supervision_density_protocol_v2.json"
    clean_script.parent.mkdir(parents=True)
    clean_protocol.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, clean_script)
    shutil.copy2(PROTOCOL, clean_protocol)
    completed = subprocess.run(
        [sys.executable, str(clean_script), "--print"],
        cwd=clean_root,
        check=True,
        text=True,
        capture_output=True,
    )
    report = json.loads(completed.stdout)
    assert report["schema"] == "t4_ridge_supervision_density_v2"
    assert report["audit_mode"] == "cpu_only_portable_protocol_audit"
    assert report["protocol_binding"] == {
        "protocol_id": "t4_ridge_supervision_density_portable_v2_20260811",
        "source_snapshot_sha256": "760b542aae5d1f99fb3b8dda086927bc9afc73ea57527c71f43167071b9a285c",
    }
    assert not (clean_root / "sua_exploration/results").exists()


def test_source_snapshot_mutation_fails_closed(tmp_path: Path) -> None:
    """Changing recorded source facts without a versioned checksum is rejected."""
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["source_snapshot"]["external_subject_m"]["rows"][0]["ridge50_finite_velocity_2d_rows"] += 1
    mutated = tmp_path / "mutated_protocol.json"
    _write_protocol(mutated, protocol)
    with pytest.raises(ValueError, match="source_snapshot_sha256 mismatch"):
        _module().build_report(mutated)


def test_recorded_receipt_binding_mutation_fails_closed(tmp_path: Path) -> None:
    """Historical receipt binding values are part of the structured snapshot too."""
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["source_snapshot"]["native_m2_m24"]["recorded_receipt_bindings"]["native_ridge_aggregate"]["sha256"] = "0" * 64
    mutated = tmp_path / "mutated_binding_protocol.json"
    _write_protocol(mutated, protocol)
    with pytest.raises(ValueError, match="source_snapshot_sha256 mismatch"):
        _module().build_report(mutated)


def test_external_counts_are_view_deduplicated_and_exact() -> None:
    report = _module().build_report()["external_subject_m"]
    assert report["pooled"] == {
        "session_count": 15,
        "t4_direction_scalars": 750,
        "ridge_finite_velocity_2d_rows": 149725,
        "ridge_finite_velocity_scalar_coordinates": 299450,
        "ridge_2d_rows_per_t4_direction_scalar": 199.633333333333,
        "ridge_scalar_coordinates_per_t4_direction_scalar": 399.266666666667,
    }
    assert all(row["t4_direction_scalars"] == 50 for row in report["per_session"])
    assert report["per_session"][0]["session_id"] == "sub-M_ses-CO-20140307"
    assert report["per_session"][0]["ridge50_finite_velocity_2d_rows"] == 9744
    assert report["per_session"][-1]["session_id"] == "sub-M_ses-CO-20150626"
    assert report["per_session"][-1]["ridge50_finite_velocity_2d_rows"] == 7543


def test_native_m24_counts_and_claim_scope_are_preserved() -> None:
    report = _module().build_report()
    native = report["native_m2_m24"]
    assert native["pooled"] == {
        "session_count": 6,
        "t4_direction_scalars": 72,
        "ridge_finite_velocity_2d_rows": 9360,
        "ridge_finite_velocity_scalar_coordinates": 18720,
        "ridge_2d_rows_per_t4_direction_scalar": 130.0,
        "ridge_scalar_coordinates_per_t4_direction_scalar": 260.0,
    }
    assert "descriptive closure" in native["scope"]
    assert report["terminology"]["quantity"] == "algorithmic target-supervision consumption"
    excluded = report["terminology"]["not_claimed"]
    assert "human annotation cost or effort" in excluded
    assert "independent samples or effective sample size" in excluded
    assert "a causal explanation for accuracy differences" in excluded


def test_scientific_binding_is_structured_not_a_narrative_markdown_hash() -> None:
    report = _module().build_report()
    bindings = json.dumps(
        [
            report["external_subject_m"]["source_bindings"],
            report["native_m2_m24"]["source_bindings"],
        ],
        sort_keys=True,
    )
    assert ".md" not in bindings
    assert "native_t4_directional_trial_audit" not in bindings
