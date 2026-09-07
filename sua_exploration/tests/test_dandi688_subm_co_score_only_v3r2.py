"""Score-blind tests for the append-only external sub-M V3R2 package."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
WRITER = ROOT / "sua_exploration/scripts/write_dandi688_subm_co_score_only_prelaunch_v3r2.py"
CORE = ROOT / "sua_exploration/mc_maze/subm_co_score_only_v3r2.py"
RUNNER = ROOT / "sua_exploration/scripts/run_dandi688_subm_co_score_only_v3r2.py"
PREPARER = ROOT / "sua_exploration/scripts/prepare_dandi688_subm_co_score_only_authorization_v3r2.py"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_banner_only_transition_is_exact_and_body_recovers_historical_sha() -> None:
    writer = _module("subm_score_only_v3r2_writer_banner_test", WRITER)
    assert len(writer.SUPERSEDED_BANNER) == 276
    for relative, transition in writer.BANNER_ONLY_TRANSITIONS.items():
        evidence = writer._verify_banner_transition(ROOT, relative, transition)
        assert evidence["transition"] == "exact_276_byte_superseded_banner_prefix_only"
        assert evidence["historical_body_sha256"] == transition["historical_sha256"]
        assert evidence["current_full_file_sha256"] == transition["current_sha256"]
        assert evidence["scientific_or_adapter_body_changed"] is False


def test_static_authority_freezes_30_50_50_matrix_and_claim(tmp_path: Path) -> None:
    writer = _module("subm_score_only_v3r2_writer_static_test", WRITER)
    authority = writer.static_authority(ROOT)
    assert authority["deployment_budget"] == {
        "activity_identity_trials": 30,
        "t4_fit_pool_trials": 50,
        "query_start": "strictly_after_rewarded_trial_50",
        "chronological": True,
    }
    assert authority["score_only_v2"]["matrix"]["sealed_cell_count"] == 180
    assert authority["claim_boundary"]["frozen_arms_only"] == ["shared_t4", "shared_ts4"]
    assert authority["claim_boundary"]["absolute_t4_over_spint_claim"] == "UNSUPPORTED_NO_SHARED_B0_CONTROL"
    assert authority["claim_boundary"]["same_dandiset_cross_animal_only"] is True
    assert len(authority["v5r2_parity_execution"]["banner_only_transitions"]) == 2

    output = tmp_path / "v3r2-prelaunch"
    result = writer.write_prelaunch(output, ROOT)
    assert result["status"] == "STATIC_V3R2_READY_BUT_NOT_AUTHORIZED"
    stored = writer.load_stored_prelaunch(output, ROOT)
    assert stored["execution_policy"]["authorization_bindings"]["deployment_budget"] == authority["deployment_budget"]
    assert stored["execution_policy"]["authorization_bindings"]["claim_boundary"] == authority["claim_boundary"]
    assert all((path.stat().st_mode & 0o777) == 0o444 for path in output.iterdir())


def test_review_template_is_explicitly_non_authorizing(tmp_path: Path) -> None:
    writer = _module("subm_score_only_v3r2_writer_template_test", WRITER)
    preparer = _module("subm_score_only_v3r2_preparer_test", PREPARER)
    prelaunch = tmp_path / "prelaunch"
    writer.write_prelaunch(prelaunch, ROOT)
    template = preparer.review_template(prelaunch)
    assert template["status"] == "REVIEW_PENDING_NATIVE_M2_NOT_YET_AUTHORIZED"
    assert template["not_an_executable_authorization"] is True
    assert template["do_not_edit_status_to_authorize"] is True
    assert template["deployment_budget"]["activity_identity_trials"] == 30
    assert template["deployment_budget"]["t4_fit_pool_trials"] == 50
    assert template["matrix"]["cells"] == 180
    assert "authorization" not in template


def test_runner_dry_run_has_zero_data_score_or_nonce_actions(tmp_path: Path) -> None:
    writer = _module("subm_score_only_v3r2_writer_runner_test", WRITER)
    prelaunch = tmp_path / "prelaunch"
    writer.write_prelaunch(prelaunch, ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--mode",
            "dry-run",
            "--prelaunch-dir",
            str(prelaunch),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "V3R2_READY_BUT_NOT_AUTHORIZED_PENDING_NATIVE_M2"
    assert payload["authorization_created"] is False
    assert payload["nonce_claimed"] is False
    assert payload["external_subm_data_access_allowed"] is False
    assert payload["model_forward_calls"] == 0
    assert payload["r2_computations"] == 0


def test_score_mode_without_detached_capability_fails_before_output(tmp_path: Path) -> None:
    writer = _module("subm_score_only_v3r2_writer_missing_cap_test", WRITER)
    prelaunch = tmp_path / "prelaunch"
    writer.write_prelaunch(prelaunch, ROOT)
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--mode", "score", "--prelaunch-dir", str(prelaunch)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "detached --authorization/--signature" in completed.stderr
    assert not (tmp_path / "runs").exists()


def test_preauthorization_import_surface_has_no_numpy_torch_or_data_owners() -> None:
    tree = ast.parse(CORE.read_text(encoding="utf-8"), filename=str(CORE))
    top_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    forbidden = ("numpy", "torch", "pynwb", "multisession_datamodule", "unit_side_features")
    assert not [name for name in top_imports if any(fragment in name for fragment in forbidden)]


def test_preparer_does_not_accept_or_generate_a_private_key() -> None:
    source = PREPARER.read_text(encoding="utf-8")
    assert "private-key" not in source
    assert "Ed25519PrivateKey" not in source
    assert "private_bytes" not in source
    assert "detached_signature_created\": False" in source
