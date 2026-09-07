"""No-data/no-CUDA regression tests for the thin seed43 score V3 repair."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v3/score_v3.py"
CLI = ROOT / "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score_v3.py"
V2_TEST = ROOT / "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed43_score_v2.py"


def _load(path: Path, package_name: str) -> ModuleType:
    package = ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route = _load(MODULE, "_test_tfsr_seed43_phase_e_v3")
base = _load(V2_TEST, "_test_tfsr_seed43_phase_e_v2_helpers_for_v3")


def _sha(value: str | bytes) -> str:
    return route._sha(value.encode("utf-8") if isinstance(value, str) else value)


def _v3_identity(tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    """Build synthetic completed predecessor evidence without a target asset."""
    upstream, _expectation, _payloads = base._seed42_v2_graph(tmp_path)
    training = base._seed43_training()
    v2 = route._v2()
    v2_closure = v2.phase_e_seed43_v2_closure(ROOT)
    v2_authorization = v2.Seed43V2Authorization(
        preflight_sha256=_sha("synthetic-v2-preflight"),
        root_authorization_sha256=_sha("synthetic-v2-authorization"),
        preflight={"synthetic": True}, root_authorization={"synthetic": True}, closure=v2_closure,
    )
    v2_identity = v2.Seed43V2Identity(
        training=training, upstream=upstream, closure=v2_closure, authorization=v2_authorization.payload(),
    )
    predecessor = route.FailedV2Predecessor(
        upstream=upstream, training=training, authorization=v2_authorization, v2_identity=v2_identity,
        authority_root_identity=(11, 12), failed_score_root_identity=(13, 14),
    )
    closure = route.phase_e_seed43_v3_closure(ROOT)
    authorization = route.Seed43V3Authorization(
        preflight_sha256=_sha("synthetic-v3-preflight"), root_authorization_sha256=_sha("synthetic-v3-authorization"),
        preflight={"synthetic": True}, root_authorization={"synthetic": True}, closure=closure,
    )
    identity = route.Seed43V3Identity(predecessor=predecessor, closure=closure, authorization=authorization.payload())
    return identity, route._issue_execution_capability(authorization), upstream, training


def _physical_synthetic_inputs(physical: ModuleType, tmp_path: Path) -> tuple[tuple[Any, ...], tuple[Any, ...], Any]:
    """Translate synthetic receipt facts into the real physical score types.

    No NWB, model, CUDA, or parser is involved.  The result nevertheless
    makes V3 invoke the actual frozen input/mode validators from the module
    which owns the public ``ScoreSpec``.
    """
    static_within = base._within_bindings(tmp_path)
    static_external = base._external_bindings(tmp_path)
    within = tuple(
        physical.EvaluationAssetBinding(
            surface=item.surface, asset_id=item.asset_id, session=item.session, local_path=item.local_path,
            expected_bytes=item.expected_bytes, expected_sha256=item.expected_sha256, frozen_path=item.frozen_path,
        )
        for item in static_within
    )
    external = tuple(
        physical.ExternalAssetBinding(
            asset_id=item.asset_id, session=item.session, local_path=item.local_path,
            expected_bytes=item.expected_bytes, expected_sha256=item.expected_sha256, frozen_path=item.frozen_path,
        )
        for item in static_external
    )
    static_evidence = base._input_evidence(static_within, static_external)
    evidence = physical.InputAuthorityEvidence(
        records=tuple(physical.SessionInputAuthority(**item.payload()) for item in static_evidence.records),
        source_normalizer_body_sha256=static_evidence.source_normalizer_body_sha256,
        t4_normalizer_semantic_sha256=static_evidence.t4_normalizer_semantic_sha256,
        behavior_normalizer_semantic_sha256=static_evidence.behavior_normalizer_semantic_sha256,
        strict_manifest_sha256=static_evidence.strict_manifest_sha256,
        raw_to_normalized_exact=True, no_cache_readonly_adapter=True,
    )
    return within, external, evidence


def test_static_cli_is_no_torch_and_partial_flags_fail_before_authority() -> None:
    sentinel = ROOT / ".seed43-v3-torch-sentinel"
    sentinel.mkdir(exist_ok=True)
    try:
        (sentinel / "torch.py").write_text("raise RuntimeError('TORCH_IMPORTED')\n")
        env = {
            **os.environ, "PYTHONPATH": str(sentinel), "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        }
        dry = subprocess.run([sys.executable, str(CLI)], cwd=ROOT, env=env, text=True, capture_output=True)
        assert dry.returncode == 0, dry.stderr
        assert json.loads(dry.stdout)["phase"] == route.PHASE
        assert "TORCH_IMPORTED" not in dry.stdout + dry.stderr
        partial = subprocess.run([sys.executable, str(CLI), "--execute"], cwd=ROOT, env=env, text=True, capture_output=True)
        assert partial.returncode != 0
        assert "TORCH_IMPORTED" not in partial.stdout + partial.stderr
    finally:
        for path in sorted(sentinel.glob("*")):
            path.unlink()
        sentinel.rmdir()


def test_v3_binds_exact_backend_owned_score_spec_not_value_equivalent_static_spec() -> None:
    physical_module = route.physical_seed42_score_module()
    physical = route.expected_physical_score_spec()
    static = route._v2()._seed42_v2()._v1().PUBLIC_SPEC
    assert physical is physical_module.PUBLIC_SPEC
    assert physical.payload() == static.payload()
    assert type(physical) is not type(static)
    assert physical is not static
    assert physical != static


def test_held_pair_reader_rejects_topology_mode_and_body_tampering(tmp_path: Path) -> None:
    root = tmp_path / "immutable"
    root.mkdir()
    expectation = {"attempt.json": _sha("attempt"), "failure.json": _sha("failure")}
    for name, body in (("attempt.json", b"{}\n"), ("failure.json", b"{}\n")):
        # Use the expected literal digest only after regenerating it from the
        # body, so this exercises descriptor structure rather than a hash echo.
        digest = route._sha(body)
        (root / name).write_bytes(body)
        (root / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
        os.chmod(root / name, 0o444); os.chmod(root / f"{name}.sha256", 0o444)
        expectation[name] = digest
    loaded, identity = route._read_exact_pairs(root, expectation)
    assert set(loaded) == set(expectation) and identity == route._directory_identity(root)
    (root / "extra.json").write_text("{}\n")
    os.chmod(root / "extra.json", 0o444)
    with pytest.raises(route.FailClosedError, match="topology"):
        route._read_exact_pairs(root, expectation)
    (root / "extra.json").unlink()
    os.chmod(root / "failure.json", 0o644)
    with pytest.raises(route.FailClosedError, match="mode"):
        route._read_exact_pairs(root, expectation)


def test_real_v2_authority_and_failed_pair_metadata_match_frozen_literals() -> None:
    """Metadata-only check: no NWB, model state, CUDA, or authority write."""
    v2 = route._v2()
    authority, _ = route._read_exact_pairs(
        ROOT / v2.AUTHORITY_ROOT_RELATIVE,
        {
            "official_preflight.json": route.V2_AUTHORITY_PREFLIGHT_SHA256,
            "root_authorization.json": route.V2_AUTHORITY_ROOT_AUTHORIZATION_SHA256,
        },
    )
    failed, _ = route._read_exact_pairs(
        ROOT / v2.SCORE_ROOT_RELATIVE,
        {"attempt.json": route.V2_FAILED_ATTEMPT_SHA256, "failure.json": route.V2_FAILED_FAILURE_SHA256},
    )
    assert authority["root_authorization.json"]["status"] == "ROOT_AUTHORIZED"
    assert failed["attempt.json"]["schema"] == "tfsr_seed43_phase_e_v2_attempt_v1"
    route._validate_v2_failure_facts(failed["failure.json"])


def test_failure_predecessor_requires_exact_failure_stage_and_zero_forward_map() -> None:
    failure = {
        "schema": "tfsr_seed43_phase_e_v2_failure_v1",
        "stage": "resolve_identical_seed42_v2_inputs_after_attempt", "input_replay_sha256": None,
        "terminal_published": False, "backward_calls": 0, "optimizer_calls": 0, "cell_d_rerun": False,
        "resolved": {"source": False, "within": True, "external": True, "formal": False},
        "opened": {"source": False, "within": False, "external": False, "formal": False},
        "tfsr_forward_calls": {
            "within": {"aligned": 0, "zero": 0, "wrong_pair": 0},
            "external": {"aligned": 0, "zero": 0, "wrong_pair": 0},
        },
    }
    route._validate_v2_failure_facts(failure)
    forged = json.loads(json.dumps(failure)); forged["tfsr_forward_calls"]["within"]["aligned"] = 1
    with pytest.raises(route.FailClosedError, match="boundary"):
        route._validate_v2_failure_facts(forged)


def test_closure_is_explicit_and_v3_preflight_binds_exact_spec_and_predecessor(tmp_path: Path) -> None:
    identity, _capability, _upstream, _training = _v3_identity(tmp_path)
    closure = route.phase_e_seed43_v3_closure(ROOT)
    assert route.validate_phase_e_seed43_v3_closure(closure) == closure
    assert route.SOURCE_RELATIVE in closure["paths"]
    assert "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v2/score_v2.py" in closure["paths"]
    preflight = route.build_target_free_preflight(ROOT, predecessor=identity.predecessor, closure=closure)
    assert route.validate_target_free_preflight(preflight, predecessor=identity.predecessor, closure=closure) == preflight
    forged = json.loads(json.dumps(preflight)); forged["physical_score_spec_owner"] = "forged"
    with pytest.raises(route.FailClosedError, match="preflight"):
        route.validate_target_free_preflight(forged, predecessor=identity.predecessor, closure=closure)
    forged_closure = json.loads(json.dumps(closure)); forged_closure["sha256_by_path"].pop(route.CLI_RELATIVE)
    with pytest.raises(route.FailClosedError, match="topology"):
        route.validate_phase_e_seed43_v3_closure(forged_closure)


def test_data_roots_fail_before_predecessor_or_output_reservation() -> None:
    calls: list[str] = []
    with pytest.raises(route.FailClosedError, match="exact canonical"):
        route._prepare_execution(
            ROOT,
            environment={"SUBC_DATA_ROOT": "/wrong", "SUBM_DATA_ROOT": route.SUBM_DATA_ROOT},
            predecessor_loader=lambda _root: calls.append("predecessor") or None,
            authorization_loader=lambda _root, _predecessor: calls.append("authorization") or None,
        )
    assert calls == []


def test_injected_lifecycle_reaches_past_resolve_inputs_with_exact_supplied_spec(tmp_path: Path) -> None:
    """A complete CPU lifecycle demonstrates the repaired call is not dead code."""
    identity, capability, _upstream, _training = _v3_identity(tmp_path)
    physical = route.physical_seed42_score_module()
    artifact = physical.reserve_artifact_root(tmp_path, "seed43-v3-success", route.SCORE_TOPOLOGY)
    within, external, evidence = _physical_synthetic_inputs(physical, tmp_path)
    s43 = route._v2()._seed43_v1()

    class Backend:
        def __init__(self) -> None:
            self.received_spec: object | None = None
            self.calls: list[str] = []
            self.closed = False
            self.parity = s43.AcceleratedParityManifest()

        def resolve_inputs(self, *, spec: object, flags: Any, **_: object) -> Any:
            self.received_spec = spec; self.calls.append("resolve")
            flags.within_opened = True; flags.external_opened = True
            return evidence

        def score_tfsr(self, *, surface: str, mode: str, input_authority_sha256: str, flags: Any) -> Any:
            self.calls.append(f"tfsr:{surface}:{mode}")
            flags.record_forward("tfsr", surface, mode)
            self.parity.record(surface=surface, mode=mode, eager_bytes=b"exact-cpu", accelerated_bytes=b"exact-cpu", shape=(2, 50, 2))
            # Existing V2 fixture gives deterministic valid cell payloads;
            # reconstruct them through the actual physical module class.
            return physical.mode_evidence_from_payload(
                base._mode("tfsr", surface, mode, input_authority_sha256, offset=0.03 if mode == "aligned" else -0.02).payload()
            )

        def build_parity_manifest(self, *, flags: Any) -> Mapping[str, object]:
            return self.parity.payload(flags)

        def reverify_after_forwards(self, **_: object) -> None: self.calls.append("reverify")

        def resource_disclosure(self) -> Mapping[str, object]:
            return {
                "schema": "tfsr_seed43_phase_e_replication_resources_v1",
                "runtime": {**dict(physical.FROZEN_SCORE_DEVICE), "torchmetrics_version": "1.5.1"},
                "seed43_training_peak_memory_bytes": 17, "score_peak_memory_bytes": 1,
                "aligned_eager_latency_ms": 1.0, "tfsr_parameters": 10, "build": s43.BUILD_LABEL,
                "cell_d_replayed_upstream_only": True, "cell_d_forward_calls": 0,
                "forward_only": True, "no_grad": True,
            }

        def close(self) -> None: self.closed = True

    backend = Backend()
    terminal = route.run_seed43_v3_lifecycle(
        artifact=artifact, identity=identity, execution_capability=capability,
        within_roster=base.WITHIN, external_sessions=base.EXTERNAL,
        within_assets_factory=lambda: within, external_assets_factory=lambda: external,
        backend=backend, final_reverify=lambda: identity.closure, physical_score_module=physical,
    )
    assert backend.received_spec is physical.PUBLIC_SPEC
    assert backend.calls[0] == "resolve"
    assert len([item for item in backend.calls if item.startswith("tfsr:")]) == 6
    assert terminal["status"] == "REPLICATION_SCORE_COMPLETE"
    assert not artifact.has_name("failure.json")
    persisted = artifact.reload_json("score.json", route._sha(artifact.reload_pair("score.json")))
    route.validate_score_payload(persisted, identity=identity, input_sha=persisted["input_replay_sha256"])


def test_failure_after_attempt_publishes_failure_only(tmp_path: Path) -> None:
    identity, capability, _upstream, _training = _v3_identity(tmp_path)
    static_score = route.physical_seed42_score_module()
    artifact = static_score.reserve_artifact_root(tmp_path, "seed43-v3-failure", route.SCORE_TOPOLOGY)
    within = base._within_bindings(tmp_path); external = base._external_bindings(tmp_path)

    class FailingBackend:
        def resolve_inputs(self, **_: object) -> Any:
            raise RuntimeError("after-attempt synthetic failure")
        def score_tfsr(self, **_: object) -> Any: raise AssertionError("must not score")
        def build_parity_manifest(self, **_: object) -> Mapping[str, object]: raise AssertionError("must not build parity")
        def reverify_after_forwards(self, **_: object) -> None: raise AssertionError("must not reverify")
        def resource_disclosure(self) -> Mapping[str, object]: raise AssertionError("must not disclose")
        def close(self) -> None: pass

    with pytest.raises(RuntimeError, match="after-attempt"):
        route.run_seed43_v3_lifecycle(
            artifact=artifact, identity=identity, execution_capability=capability,
            within_roster=base.WITHIN, external_sessions=base.EXTERNAL,
            within_assets_factory=lambda: within, external_assets_factory=lambda: external,
            backend=FailingBackend(), final_reverify=lambda: identity.closure, physical_score_module=static_score,
        )
    assert artifact.has_name("attempt.json") and artifact.has_name("failure.json")
    assert not artifact.has_name("input_replay.json") and not artifact.has_name("score.json") and not artifact.has_name("terminal.json")
