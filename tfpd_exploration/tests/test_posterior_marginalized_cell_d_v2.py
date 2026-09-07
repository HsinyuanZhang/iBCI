"""Focused V2 successor tests.

The tests inspect metadata and the frozen sealed SWA bytes only.  They never
open NWB/target/formal data, initialize CUDA, launch a scorer, or mint a live
capability.  The reservation test uses a temporary result parent, never the
canonical V2 root.
"""
from __future__ import annotations

import hashlib
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

from src.posterior_marginalized_cell_d_v2 import matched_score as score
from src.posterior_marginalized_cell_d_v2 import matched_score_physical as physical
from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1_physical
import test_posterior_marginalized_cell_d_matched_score as v1_fixture


def _v2_identity_capability():
    base = v1_fixture._identity()
    provenance = v1_fixture._verified_provenance(base)
    closure = score.implementation_closure(ROOT)
    identity = physical.score_identity_from_provenance_v2(provenance=provenance, closure=closure)
    root_review = physical.issue_v2_root_review_capability()
    capability = physical.issue_root_reviewed_execution_capability_v2(
        ROOT,
        root_review_capability=root_review,
        device_profile=score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"],
        provenance_loader=lambda _root: provenance,
        closure_loader=lambda _root: closure,
    )
    return identity, closure, capability, provenance


def test_dry_cli_is_stdlib_only_and_rejects_public_execution() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score_v2.py"
    env = {**os.environ, "PYTHONPATH": "", "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""}
    rendered = subprocess.run(
        [sys.executable, "-S", str(script), "--dry-run"],
        env=env, capture_output=True, text=True, check=True,
    )
    payload = json.loads(rendered.stdout)
    assert payload["status"].startswith("DRY_ONLY__")
    assert payload["score_root"] == score.SCORE_ROOT_RELATIVE
    rejected = subprocess.run(
        [sys.executable, "-S", str(script), "--execute"],
        env=env, capture_output=True, text=True,
    )
    assert rejected.returncode != 0
    assert "root-reviewed" in rejected.stderr


def test_failed_v1_graph_is_exact_failed_prepare_and_has_no_input_or_terminal() -> None:
    graph = physical.validate_failed_v1_graph(ROOT)
    assert graph.root_relative == score.FAILED_V1_ROOT_RELATIVE
    assert graph.body_sha256 == physical.FAILED_V1_BODY_SHA256
    assert graph.sidecar_sha256 == physical.FAILED_V1_SIDECAR_SHA256
    assert graph.failure_stage == "prepare"
    assert graph.status == "SCORE_FAILED"
    assert graph.payload()["terminal_present"] is False
    assert graph.payload()["input_authority_present"] is False


def test_actual_sealed_swa_payload_strict_cpu_shape_and_no_cuda() -> None:
    import torch

    was_initialized = torch.cuda.is_initialized()
    actual = physical.load_actual_sealed_swa_cpu(ROOT)
    assert actual.body_sha256 == physical.ACTUAL_SEALED_SWA_SHA256
    assert actual.state_sha256 == physical.ACTUAL_SEALED_DERIVED_STATE_SHA256
    assert actual.manifest["floating_tensor_count"] == 29
    assert actual.manifest["uninitialized_lazy_tensor_count"] == 2
    assert actual.manifest["buffer_tensor_count"] == 0
    assert actual.manifest["fp64_arithmetic"] is True
    assert actual.manifest["optimizer_state_included"] is False
    assert actual.lazy_keys == tuple(sorted(physical.ACTUAL_SEALED_LAZY_KEYS))
    assert actual.floating_tensor_count == 29
    assert len(actual.state_keys) == 31
    assert actual.state_shapes
    assert torch.cuda.is_initialized() is was_initialized


def test_actual_payload_rejects_forged_top_level_schema() -> None:
    with pytest.raises(physical.V2PhysicalScoreError):
        physical._load_actual_payload(  # type: ignore[attr-defined]
            b"not-a-pytorch-payload", expected_manifest={}
        )


def test_reserve_v2_root_only_and_never_failed_v1(tmp_path: Path) -> None:
    graph = physical.validate_failed_v1_graph(ROOT)
    result_parent = tmp_path / "tfpd_exploration" / "results"
    result_parent.mkdir(parents=True)
    calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def reserve(parent: Path, name: str, *, topology: tuple[str, ...]):
        calls.append((parent, name, topology))
        path = parent / name
        path.mkdir()
        return type("Reserved", (), {"directory": path, "topology": topology})()

    artifact = physical.reserve_v2_root(tmp_path, failed_graph=graph, reserve_fn=reserve)
    assert artifact.directory == result_parent / "posterior_marginalized_cell_d_score_v2"
    assert artifact.directory.is_dir()
    assert not (tmp_path / score.FAILED_V1_ROOT_RELATIVE).exists()
    assert calls == [(result_parent, "posterior_marginalized_cell_d_score_v2", physical.SCORE_TOPOLOGY)]


def test_execute_rejects_non_root_capability_before_v2_reservation() -> None:
    with pytest.raises(physical.V2PhysicalScoreError, match="root-reviewed"):
        physical.execute_authorized_v2(ROOT, capability=object())  # type: ignore[arg-type]


def test_v1_capability_is_rejected_and_v2_issuer_binds_exact_v2_closure() -> None:
    identity, closure, capability, _provenance = _v2_identity_capability()
    assert identity.payload()["closure"] == closure.payload()
    assert capability.closure_sha256 == closure.payload()["closure_sha256"]
    assert capability.identity_sha256 == v1_physical._identity_digest(identity)
    capability.verify(identity)
    v1_capability = v1_physical.RootReviewedExecutionCapability(
        identity_sha256=v1_physical._identity_digest(identity),
        device_profile=score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"],
        provenance_sha256=v1_physical._provenance_digest(identity),
        closure_sha256=closure.payload()["closure_sha256"],
        seal=v1_physical._ROOT_EXECUTION_SEAL,
    )
    with pytest.raises(physical.V2PhysicalScoreError, match="V2 root-reviewed"):
        physical.execute_authorized_v2(ROOT, capability=v1_capability)  # type: ignore[arg-type]


def test_v2_issuer_rejects_closure_leaf_drift() -> None:
    _identity, closure, _capability, provenance = _v2_identity_capability()
    hashes = dict(closure.sha256_by_path)
    drift_path = next(path for path in hashes if path != score.WORKORDER_RELATIVE)
    hashes[drift_path] = "0" * 64
    forged = score.V2ImplementationClosure(base=closure.base, sha256_by_path=hashes)
    with pytest.raises(physical.V2PhysicalScoreError, match="closure leaf drift"):
        physical.issue_root_reviewed_execution_capability_v2(
            ROOT,
            root_review_capability=physical.issue_v2_root_review_capability(),
            device_profile=score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"],
            provenance_loader=lambda _root: provenance,
            closure_loader=lambda _root: forged,
        )


def test_synthetic_complete_lifecycle_receipts_bind_v2_identity_and_closure() -> None:
    identity, closure, capability, _provenance = _v2_identity_capability()
    events: list[str] = []
    runtime = v1_fixture._MockRuntime(identity, events)
    backend = v1_fixture._MockBackend(identity, runtime, events)
    artifact = v1_fixture._MemoryArtifact(events)
    assets = v1_fixture._physical_assets(identity)
    terminal = v1_physical.run_physical_score_lifecycle(
        artifact=artifact,
        identity=identity,
        capability=capability,
        backend=backend,
        assets=assets,
        preflight=v1_physical.build_preflight_payload(identity=identity, assets=assets),
        authorization=v1_physical.build_authorization_payload(identity=identity, capability=capability),
        final_reverify=lambda: closure.payload(),
    )
    expected_identity_sha = v1_physical._identity_digest(identity)
    for name in ("preflight.json", "authorization.json", "attempt.json"):
        assert json.loads(artifact.bodies[name])["identity_sha256"] == expected_identity_sha
    score_payload = json.loads(artifact.bodies["score.json"])
    terminal_payload = json.loads(artifact.bodies["terminal.json"])
    assert score_payload["identity"] == identity.payload()
    assert terminal_payload["identity"] == identity.payload()
    assert score_payload["identity"]["closure"] == closure.payload()
    assert terminal_payload["final_closure"] == closure.payload()
    assert terminal["identity"] == identity.payload()
