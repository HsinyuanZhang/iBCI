"""Focused no-data tests for the PMC-D matched-score V3 successor.

The tests exercise the V3 deployment seam only.  They do not open NWB,
target/formal data, reserve the canonical V3 root, mint a live capability, or
initialize CUDA.  The actual sealed-SWA regression is weights-only and CPU
strict-load, inherited from the reviewed V2 parser.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "tfpd_exploration"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
TEST_ROOT = PACKAGE_ROOT / "tests"
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p
from src.posterior_marginalized_cell_d_v2 import matched_score_physical as v2p
from src.posterior_marginalized_cell_d_v3 import matched_score as score
from src.posterior_marginalized_cell_d_v3 import matched_score_physical as physical
import test_posterior_marginalized_cell_d_matched_score as v1_fixture
import test_posterior_marginalized_cell_d_v2 as v2_fixture


ENVIRONMENT = dict(score.LAUNCH_ENVIRONMENT)


def _v3_identity_capability():
    base = v1_fixture._identity()
    provenance = v1_fixture._verified_provenance(base)
    graph = physical.validate_failed_v2_graph(ROOT)
    closure = score.implementation_closure(ROOT, failed_v2_graph=graph.payload())
    identity = physical.score_identity_from_provenance_v3(
        provenance=provenance, closure=closure, failed_v2_graph=graph,
    )
    capability = physical.issue_root_reviewed_execution_capability_v3(
        ROOT,
        root_review_capability=physical.issue_v3_root_review_capability(),
        device_profile=score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"],
        environ=ENVIRONMENT,
        provenance_loader=lambda _root: provenance,
        failed_graph_loader=lambda _root: graph,
        closure_loader=lambda _root, _graph: closure,
    )
    return identity, closure, capability, graph, provenance


def test_dry_cli_is_stdlib_only_and_rejects_public_execution() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score_v3.py"
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": "",
        "CUDA_VISIBLE_DEVICES": "",
    }
    rendered = subprocess.run(
        [sys.executable, "-S", str(script), "--dry-run"],
        env=environment, capture_output=True, text=True, check=True,
    )
    payload = json.loads(rendered.stdout)
    assert payload["status"].startswith("DRY_ONLY__")
    assert payload["score_root"] == score.SCORE_ROOT_RELATIVE
    assert payload["failed_v2_root"] == score.FAILED_V2_ROOT_RELATIVE
    rejected = subprocess.run(
        [sys.executable, "-S", str(script), "--execute"],
        env=environment, capture_output=True, text=True,
    )
    assert rejected.returncode != 0
    assert "root-reviewed" in rejected.stderr


def test_failed_v2_graph_is_exact_input_failure_with_no_forwards() -> None:
    graph = physical.validate_failed_v2_graph(ROOT)
    assert graph.root_relative == score.FAILED_V2_ROOT_RELATIVE
    assert graph.body_sha256 == physical.FAILED_V2_BODY_SHA256
    assert graph.sidecar_sha256 == physical.FAILED_V2_SIDECAR_SHA256
    assert graph.identity_sha256 == physical.FAILED_V2_IDENTITY_SHA256
    assert graph.failure_stage == "input"
    assert graph.error_sha256 == physical.FAILED_V2_ERROR_SHA256
    assert graph.no_input_authority is True
    assert graph.no_forwards is True
    assert set(graph.payload()) == {
        "schema", "root_relative", "body_sha256", "sidecar_sha256", "identity_sha256",
        "failure_stage", "error_sha256", "no_input_authority", "no_forwards",
    }


def test_actual_sealed_swa_payload_is_strict_cpu_weights_only_and_does_not_init_cuda() -> None:
    import torch

    was_initialized = torch.cuda.is_initialized()
    actual = physical.load_actual_sealed_swa_cpu(ROOT)
    assert actual.body_sha256 == v2p.ACTUAL_SEALED_SWA_SHA256
    assert actual.state_sha256 == v2p.ACTUAL_SEALED_DERIVED_STATE_SHA256
    assert actual.manifest["floating_tensor_count"] == 29
    assert actual.manifest["uninitialized_lazy_tensor_count"] == 2
    assert actual.manifest["buffer_tensor_count"] == 0
    assert actual.manifest["fp64_arithmetic"] is True
    assert actual.manifest["optimizer_state_included"] is False
    assert actual.lazy_keys == tuple(sorted(v2p.ACTUAL_SEALED_LAZY_KEYS))
    assert actual.floating_tensor_count == 29
    assert len(actual.state_keys) == 31
    assert actual.state_shapes
    assert torch.cuda.is_initialized() is was_initialized


def test_missing_or_wrong_environment_fails_before_v3_reservation(tmp_path: Path) -> None:
    graph = physical.validate_failed_v2_graph(ROOT)
    calls: list[tuple[Path, str]] = []

    def reserve(parent: Path, name: str, *, topology):
        calls.append((parent, name))
        path = parent / name
        path.mkdir(parents=True)
        return type("Reserved", (), {"directory": path, "topology": topology})()

    with pytest.raises(physical.V3PhysicalScoreError, match="launch environment"):
        physical.reserve_v3_root(
            tmp_path, failed_graph=graph, reserve_fn=reserve,
            environ={"SUBC_DATA_ROOT": score.SUBC_DATA_ROOT},
        )
    with pytest.raises(physical.V3PhysicalScoreError, match="launch environment"):
        physical.reserve_v3_root(
            tmp_path, failed_graph=graph, reserve_fn=reserve,
            environ={**ENVIRONMENT, "SUBM_DATA_ROOT": "/wrong"},
        )
    assert calls == []
    assert not (tmp_path / score.SCORE_ROOT_RELATIVE).exists()


def test_v3_issuer_binds_predecessor_closure_and_rejects_v2_capability() -> None:
    identity, closure, capability, graph, _provenance = _v3_identity_capability()
    assert identity.payload()["closure"] == closure.payload()
    assert capability.identity_sha256 == v1p._identity_digest(identity)
    assert capability.closure_sha256 == closure.payload()["closure_sha256"]
    assert capability.failed_v2_graph_sha256 == physical._digest(  # type: ignore[attr-defined]
        physical._json(graph.payload()),  # type: ignore[attr-defined]
    )
    capability.verify(identity)

    _v2_identity, _v2_closure, v2_capability, _v2_provenance = v2_fixture._v2_identity_capability()
    with pytest.raises(physical.V3PhysicalScoreError, match="V3 root-reviewed"):
        physical.execute_authorized_v3(
            ROOT, capability=v2_capability, environ=ENVIRONMENT,
        )


def test_v3_issuer_rejects_stale_or_forged_closure_leaf() -> None:
    _identity, closure, _capability, graph, provenance = _v3_identity_capability()
    hashes = dict(closure.sha256_by_path)
    drift_path = next(path for path in hashes if path != score.WORKORDER_RELATIVE)
    hashes[drift_path] = "0" * 64
    forged = score.V3ImplementationClosure(
        base=closure.base, sha256_by_path=hashes, failed_v2_graph=closure.failed_v2_graph,
    )
    with pytest.raises(physical.V3PhysicalScoreError, match="closure leaf drift"):
        physical.issue_root_reviewed_execution_capability_v3(
            ROOT,
            root_review_capability=physical.issue_v3_root_review_capability(),
            device_profile=score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"],
            environ=ENVIRONMENT,
            provenance_loader=lambda _root: provenance,
            failed_graph_loader=lambda _root: graph,
            closure_loader=lambda _root, _graph: forged,
        )


def test_fresh_v3_reservation_never_targets_v2_failed_root(tmp_path: Path) -> None:
    graph = physical.validate_failed_v2_graph(ROOT)
    result_parent = tmp_path / "tfpd_exploration" / "results"
    result_parent.mkdir(parents=True)
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def reserve(parent: Path, name: str, *, topology):
        calls.append((parent, name, topology))
        path = parent / name
        path.mkdir()
        return type("Reserved", (), {"directory": path, "topology": topology})()

    artifact = physical.reserve_v3_root(
        tmp_path, failed_graph=graph, reserve_fn=reserve, environ=ENVIRONMENT,
    )
    assert artifact.directory == tmp_path / score.SCORE_ROOT_RELATIVE
    assert artifact.directory != tmp_path / score.FAILED_V2_ROOT_RELATIVE
    assert not (tmp_path / score.FAILED_V2_ROOT_RELATIVE).exists()
    assert calls == [(result_parent, Path(score.SCORE_ROOT_RELATIVE).name, v2p.SCORE_TOPOLOGY)]


def test_synthetic_complete_inherited_lifecycle_binds_v3_identity_and_closure() -> None:
    identity, closure, capability, _graph, _provenance = _v3_identity_capability()
    events: list[str] = []
    runtime = v1_fixture._MockRuntime(identity, events)
    backend = v1_fixture._MockBackend(identity, runtime, events)
    artifact = v1_fixture._MemoryArtifact(events)
    assets = v1_fixture._physical_assets(identity)
    terminal = v1p.run_physical_score_lifecycle(
        artifact=artifact,
        identity=identity,
        capability=capability,
        backend=backend,
        assets=assets,
        preflight=v1p.build_preflight_payload(identity=identity, assets=assets),
        authorization=v1p.build_authorization_payload(identity=identity, capability=capability),
        final_reverify=lambda: closure.payload(),
    )
    expected_identity_sha = v1p._identity_digest(identity)
    for name in ("preflight.json", "authorization.json", "attempt.json"):
        assert json.loads(artifact.bodies[name])["identity_sha256"] == expected_identity_sha
    score_payload = json.loads(artifact.bodies["score.json"])
    terminal_payload = json.loads(artifact.bodies["terminal.json"])
    assert score_payload["identity"] == identity.payload()
    assert terminal_payload["identity"] == identity.payload()
    assert score_payload["identity"]["closure"] == closure.payload()
    assert terminal_payload["final_closure"] == closure.payload()
    assert terminal["identity"] == identity.payload()
    assert len(runtime.materialize_calls) == 21
    assert len(runtime.forward_calls) == 21 * len(score.BUDGETS) * len(score.SYSTEMS)
    assert "terminal.json" in artifact.bodies
    assert "failure.json" not in artifact.bodies
