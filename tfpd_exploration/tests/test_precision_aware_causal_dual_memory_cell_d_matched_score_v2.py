"""Synthetic/no-CUDA tests for the Precision-Aware CDM-D score V2 successor."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
import inspect
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration/src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.causal_dual_memory_cell_d_score_v1 import plan as baseplan  # noqa: E402
from src.causal_dual_memory_cell_d_score_v1 import score as sharedscore  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import plan as predecessor_plan  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import score as predecessor_score  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_score_v2 import physical, plan, score  # noqa: E402


def _sha(letter: str = "a") -> str:
    return letter * 64


def _witness() -> dict[str, object]:
    cells = {
        f"m{budget}:{surface}": _sha("b")
        for budget in (30, 10, 4) for surface in predecessor_plan.SURFACES
    }
    body = {
        "schema": "precision_aware_cdmd_v8_binding_witness_v1",
        "contract": predecessor_plan.V8_PREDECESSOR.payload(),
        "directory_identity": [101, 103],
        "v8_identity_sha256": _sha("c"),
        "v8_source_gate_binding": {"binding_sha256": _sha("d")},
        "input_records_sha256": _sha("e"),
        "sealed_cell_sha256s": cells,
    }
    return {**body, "binding_sha256": predecessor_plan.sha256_bytes(predecessor_plan.canonical_json_bytes(body))}


def _historical_v1_identity() -> predecessor_plan.ScoreIdentity:
    return predecessor_plan.ScoreIdentity(
        closure=predecessor_plan.implementation_closure(ROOT).payload(),
        v8_binding=_witness(),
    )


def _binding() -> score.V1FailedPredecessorBinding:
    return score.V1FailedPredecessorBinding(
        authority_directory_identity=(107, 109),
        score_directory_identity=(113, 127),
        historical_identity=_historical_v1_identity(),
    )


def _identity() -> plan.ScoreIdentity:
    historical = _historical_v1_identity()
    binding = _binding()
    return plan.ScoreIdentity(
        closure=historical.closure,
        v8_binding=historical.v8_binding,
        selected_device_profile=historical.selected_device_profile,
        v8_predecessor=historical.v8_predecessor,
        successor_closure=plan.implementation_closure(ROOT).payload(),
        v1_failed_predecessor_binding=binding.payload(),
    )


def _environment(identity: plan.ScoreIdentity) -> dict[str, str]:
    selected = identity.payload()["selected_device_profile"]
    assert isinstance(selected, Mapping)
    return {
        "CUDA_VISIBLE_DEVICES": str(selected["cuda_visible_devices"]),
        "CUDA_DEVICE_ORDER": str(selected["cuda_device_order"]),
        **plan.CANONICAL_SOURCE_ROOTS,
    }


def _write_pair(directory: Path, name: str, payload: object, *, mode: int = 0o444) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    body_path = directory / name
    sidecar_path = directory / f"{name}.sha256"
    body_path.write_bytes(body)
    sidecar_path.write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(body_path, mode)
    os.chmod(sidecar_path, mode)
    return digest


def _load_v1_synthetic_support() -> object:
    location = ROOT / "tfpd_exploration/tests/test_precision_aware_causal_dual_memory_cell_d_matched_score_v1.py"
    specification = importlib.util.spec_from_file_location("_precision_v1_synthetic_support_for_v2", location)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _semantic_v1_failure_graph(tmp_path: Path) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Actual-shaped in-memory V1 graph for semantic validator adversaries.

    The V2 production reader additionally requires the literal immutable body
    SHA values.  This helper intentionally tests semantics independently of
    those unforgeable live bytes, so it never accesses a result root.
    """
    support = _load_v1_synthetic_support()
    binding, _authority, fixed, _ = support._lifecycle_v8_binding()
    identity = predecessor_plan.ScoreIdentity(
        closure=predecessor_plan.implementation_closure(ROOT).payload(), v8_binding=binding.payload(),
    )
    preflight = predecessor_score.build_target_free_preflight(
        root=tmp_path, identity=identity, predecessor=binding, fixed_authority=fixed,
    )
    authorization = predecessor_score.build_root_authorization(
        official_preflight_sha256=plan.V1_AUTHORITY_SHAS["official_preflight.json"], preflight=preflight,
    )
    attempt = predecessor_score._attempt_payload(
        identity,
        plan.V1_AUTHORITY_SHAS["official_preflight.json"],
        plan.V1_AUTHORITY_SHAS["root_authorization.json"],
    )
    error = type("PhysicalCDMDScoreError", (RuntimeError,), {})("synthetic missing roots")
    failure = predecessor_score._failure_payload(
        identity, plan.V1_FAILED_SCORE_SHAS["attempt.json"], None, "materialize_inputs", error,
        {
            "within_assets_opened": False, "external_assets_opened": False, "checkpoint_opened": True,
            "cuda_initialized": True, "full_system_forward_count": 0, "group_forward_count": 0,
        },
    )
    failure["error_class"] = "PhysicalCDMDScoreError"
    failure["error_sha256"] = plan.V1_FAILURE_ERROR_SHA256
    return (
        {"official_preflight.json": preflight, "root_authorization.json": authorization},
        {"attempt.json": attempt, "failure.json": failure},
    )


def test_dry_cli_current_closure_and_identity_are_torch_free() -> None:
    assert "torch" not in sys.modules
    closure = plan.implementation_closure(ROOT).payload()
    assert len(closure["paths"]) == len(plan.IMPLEMENTATION_PATHS)
    assert plan.validate_implementation_closure(closure) == closure
    identity = _identity()
    payload = identity.payload()
    assert payload["schema"] == "precision_aware_cdmd_matched_score_identity_v2"
    assert payload["historical_v1_science_identity"]["schema"] == "precision_aware_cdmd_matched_score_identity_v1"
    assert payload["canonical_source_environment"] == plan.canonical_source_environment_payload()
    script = ROOT / "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_matched_score_v2.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=True,
        env={
            "PATH": os.environ["PATH"], "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": "",
        },
    )
    dry = json.loads(completed.stdout)
    assert dry["no_torch_import"] is True
    assert dry["canonical_source_environment"]["roots"] == plan.CANONICAL_SOURCE_ROOTS
    assert "torch" not in sys.modules


def test_v2_closure_binds_current_v1_runtime_and_new_v2_only_paths() -> None:
    closure = plan.implementation_closure(ROOT).payload()
    paths = {row["path"] for row in closure["paths"]}
    assert set(predecessor_plan.IMPLEMENTATION_PATHS).issubset(paths)
    assert "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v2/physical.py" in paths
    tampered = copy.deepcopy(closure)
    tampered["paths"][-1]["sha256"] = _sha("0")
    with pytest.raises(plan.PrecisionMatchedScoreV2PlanError, match="canonical|closure"):
        plan.validate_implementation_closure(tampered)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda values: values.pop("SUBC_DATA_ROOT"),
        lambda values: values.__setitem__("SUBC_DATA_ROOT", plan.CANONICAL_SOURCE_ROOTS["SUBM_DATA_ROOT"]),
        lambda values: values.__setitem__("SUBC_DATA_ROOT", "relative/sub-C"),
        lambda values: values.__setitem__("SUBM_DATA_ROOT", "/tmp/symlink-to-sub-M"),
        lambda values: values.__setitem__("CUDA_VISIBLE_DEVICES", "0"),
    ],
)
def test_canonical_environment_rejects_missing_swapped_relative_symlink_spelling_and_device_drift(mutate: object) -> None:
    identity = _identity()
    values = _environment(identity)
    assert score.validate_selected_launch_environment(identity, values)["canonical_source_environment"] == plan.canonical_source_environment_payload()
    assert callable(mutate)
    mutate(values)
    with pytest.raises(score.PrecisionMatchedScoreV2Error, match="environment|CVD|PCI"):
        score.validate_selected_launch_environment(identity, values)


def test_environment_gate_runs_before_any_authority_root_reservation_or_predecessor_read(tmp_path: Path) -> None:
    identity = _identity()
    bad = _environment(identity)
    bad.pop("SUBM_DATA_ROOT")
    with pytest.raises(score.PrecisionMatchedScoreV2Error, match="canonical SUBC/SUBM"):
        score.reserve_authority_artifact(tmp_path, object(), identity=identity, environ=bad)
    assert not (tmp_path / plan.AUTHORITY_ROOT_RELATIVE).exists()
    assert not (tmp_path / plan.SCORE_ROOT_RELATIVE).exists()


def test_mutating_authority_api_rejects_a_forged_valid_mapping_when_actual_process_env_drifts(tmp_path: Path) -> None:
    identity = _identity()
    claimed = _environment(identity)
    # The isolated test process deliberately has CUDA_VISIBLE_DEVICES=''; a
    # caller must not use this otherwise-valid mapping to reserve an output
    # for a different physical process environment.
    assert os.environ.get("CUDA_VISIBLE_DEVICES") != claimed["CUDA_VISIBLE_DEVICES"]
    with pytest.raises(score.PrecisionMatchedScoreV2Error, match="supplied/actual|CVD"):
        score.reserve_authority_artifact(tmp_path, object(), identity=identity, environ=claimed)
    assert not (tmp_path / plan.AUTHORITY_ROOT_RELATIVE).exists()


def test_v1_contract_freezes_authority_attempt_failure_topology_and_honest_prepare_boundary() -> None:
    contract = plan.V1_FAILED_PREDECESSOR.payload()
    assert contract["authority_exact_leaf_count"] == 4
    assert contract["failed_score_exact_leaf_count"] == 4
    assert contract["failure_stage"] == "materialize_inputs"
    assert contract["checkpoint_opened"] is True and contract["cuda_initialized"] is True
    assert contract["within_external_assets_opened"] is False
    assert contract["full_system_forward_count"] == contract["group_forward_count"] == 0
    altered = plan.V1FailedPredecessorContract(error_sha256=_sha("f"))
    with pytest.raises(plan.PrecisionMatchedScoreV2PlanError, match="literal"):
        altered.payload()


def test_descriptor_pair_reader_rejects_symlink_mode_and_topology_drift(tmp_path: Path) -> None:
    directory = tmp_path / "pairs"
    directory.mkdir()
    first = _write_pair(directory, "first.json", {"a": 1})
    second = _write_pair(directory, "second.json", {"b": 2})
    expected = {"first.json": first, "second.json": second}
    identity, bodies = score._read_exact_directory(directory, label="synthetic", expected_sha256s=expected)
    assert identity[0] > 0 and set(bodies) == set(expected)

    sidecar = directory / "first.json.sha256"
    os.chmod(sidecar, 0o644)
    with pytest.raises(score.PrecisionMatchedScoreV2Error, match="mode/link"):
        score._read_exact_directory(directory, label="synthetic", expected_sha256s=expected)
    os.chmod(sidecar, 0o444)
    sidecar.unlink()
    os.symlink(directory / "second.json.sha256", sidecar)
    with pytest.raises(score.PrecisionMatchedScoreV2Error, match="mode/link|body/sidecar"):
        score._read_exact_directory(directory, label="synthetic", expected_sha256s=expected)


def test_v1_failure_semantics_rejects_stage_and_zero_forward_boundary_drift(tmp_path: Path) -> None:
    authority, failed = _semantic_v1_failure_graph(tmp_path)
    identity = score._validate_v1_failed_semantics(authority, failed)
    assert isinstance(identity, predecessor_plan.ScoreIdentity)
    altered = copy.deepcopy(failed)
    altered["failure.json"]["stage"] = "prepare"
    with pytest.raises(score.PrecisionMatchedScoreV2Error, match="failure-stage|semantic"):
        score._validate_v1_failed_semantics(authority, altered)
    altered = copy.deepcopy(failed)
    altered["failure.json"]["full_system_forward_count"] = 1
    with pytest.raises(score.PrecisionMatchedScoreV2Error, match="failure-stage|semantic"):
        score._validate_v1_failed_semantics(authority, altered)


def test_v2_identity_rejects_forged_or_substituted_v1_binding() -> None:
    identity = _identity()
    payload = identity.payload()
    forged = copy.deepcopy(payload)
    forged["v1_failed_predecessor_binding"]["score_directory_identity"] = [1, 1]
    with pytest.raises(plan.PrecisionMatchedScoreV2PlanError, match="canonical"):
        plan.validate_v1_failed_predecessor_binding(forged["v1_failed_predecessor_binding"])


def test_physical_route_is_a_v1_delegating_environment_guard_not_a_parser_or_forward_copy() -> None:
    source = inspect.getsource(physical.PhysicalPrecisionMatchedScoreV2Backend)
    assert "predecessor_physical.build_reviewed_physical_backend" in inspect.getsource(physical.__dict__["PhysicalPrecisionMatchedScoreV2Backend"].__init__)
    assert "def revalidate" in source and "self._gate" in source
    assert "def _parse_session" not in source and "def _forward" not in source and "torch" not in source
    assert issubclass(physical.PhysicalPrecisionMatchedScoreV2Error, score.PrecisionMatchedScoreV2Error)


def test_execution_adapter_requires_actual_environment_agreement_before_backend_construction() -> None:
    identity = _identity()
    values = _environment(identity)
    values["SUBC_DATA_ROOT"] = "relative/sub-C"
    with pytest.raises(score.PrecisionMatchedScoreV2Error, match="canonical SUBC/SUBM"):
        physical._assert_actual_environment_matches(identity, values)


def _lifecycle_identity(binding: object) -> plan.ScoreIdentity:
    assert isinstance(binding, predecessor_score.V8PredecessorBinding)
    historical = predecessor_plan.ScoreIdentity(
        closure=predecessor_plan.implementation_closure(ROOT).payload(), v8_binding=binding.payload(),
    )
    failed = score.V1FailedPredecessorBinding(
        authority_directory_identity=(131, 137), score_directory_identity=(139, 149), historical_identity=historical,
    )
    return plan.ScoreIdentity(
        closure=historical.closure, v8_binding=historical.v8_binding,
        selected_device_profile=historical.selected_device_profile, v8_predecessor=historical.v8_predecessor,
        successor_closure=plan.implementation_closure(ROOT).payload(),
        v1_failed_predecessor_binding=failed.payload(),
    )


def _synthetic_v2_preflight(
    identity: plan.ScoreIdentity, binding: score.V1FailedPredecessorBinding, fixed: object,
) -> dict[str, object]:
    assert isinstance(fixed, sharedscore.FixedEvaluationAuthority)
    return {
        "schema": "precision_aware_cdmd_matched_score_target_free_preflight_v2",
        "status": "PREFLIGHT_ACCEPTED",
        "identity": identity.payload(),
        "source_gate": binding.payload(),
        "v8_predecessor": identity.payload()["v8_predecessor_binding"],
        "v1_failed_predecessor": binding.payload(),
        "evaluation_authority": fixed.payload(),
        "metric": dict(baseplan.METRIC_CONTRACT),
        "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": plan.SCORE_ROOT_RELATIVE,
        "canonical_launch_environment": {
            "selected_device_profile": identity.payload()["selected_device_profile"],
            "canonical_source_environment": plan.canonical_source_environment_payload(),
        },
        "target_free": True,
        "target_paths_resolved": False,
        "model_or_checkpoint_opened": False,
        "cuda_initialized": False,
        "boundaries": {
            **baseplan.EXECUTION_BOUNDARIES,
            "precision_v2_target_state": False,
        },
        "receipt_codec": "precision_v1_compatible_science_codec__v2_identity_and_environment_bound",
    }


class _Artifact:
    topology = score.SCORE_TOPOLOGY

    def __init__(self, events: list[str]) -> None:
        self.items: dict[str, bytes] = {}
        self.events = events

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        assert name not in self.items
        body = plan.canonical_json_bytes(payload)
        self.items[name] = body
        self.events.append(f"publish:{name}")
        return hashlib.sha256(body).hexdigest()

    def publish_group(self, bodies: Mapping[str, bytes], *, post_publish: Any = None) -> Mapping[str, str]:
        assert not (set(bodies) & set(self.items))
        digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        if post_publish is not None:
            post_publish(bodies, digests)
        self.items.update(bodies)
        self.events.append("publish:score_terminal")
        return digests

    def reload_json(self, name: str, expected_sha256: str | None = None) -> Mapping[str, object]:
        body = self.reload_pair(name, expected_sha256=expected_sha256)
        value = json.loads(body)
        assert isinstance(value, Mapping)
        return value

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        body = self.items[name]
        if expected_sha256 is not None:
            assert hashlib.sha256(body).hexdigest() == expected_sha256
        return body

    def has_name(self, name: str) -> bool:
        return name in self.items


def test_v2_shared_lifecycle_keeps_v1_science_codec_but_binds_v2_attempt_environment_and_predecessor(tmp_path: Path) -> None:
    """Exercise the actual shared lifecycle with no data, CUDA, or live root.

    The fake backend exposes only V1's already-reviewed precision cells.  It
    proves the successor does not copy the score loop and that V2-only attempt
    provenance survives a complete atomic terminal graph.
    """
    support = _load_v1_synthetic_support()
    v8_binding, authority, fixed, support_module = support._lifecycle_v8_binding()
    identity = _lifecycle_identity(v8_binding)
    historical = predecessor_plan.ScoreIdentity(
        closure=identity.closure, v8_binding=identity.v8_binding,
        selected_device_profile=identity.selected_device_profile,
    )
    failed = score.V1FailedPredecessorBinding((151, 157), (163, 167), historical)
    # Rebuild the outer identity around this exact hook result so the source
    # gate and durable identity use one typed V1-failure binding.
    identity = plan.ScoreIdentity(
        closure=historical.closure, v8_binding=historical.v8_binding,
        selected_device_profile=historical.selected_device_profile, v8_predecessor=historical.v8_predecessor,
        successor_closure=plan.implementation_closure(ROOT).payload(),
        v1_failed_predecessor_binding=failed.payload(),
    )
    preflight = _synthetic_v2_preflight(identity, failed, fixed)
    assert score.validate_target_free_preflight(preflight, identity) == preflight
    pre_sha = score._digest(preflight)
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = score._digest(authorization)
    capability = sharedscore.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=sharedscore._issue_root_publication_capability(),
    )
    events: list[str] = []

    class Backend:
        def preflight(self, *, root: Path, identity: plan.ScoreIdentity) -> Mapping[str, object]:
            events.append("preflight")
            return {"target_paths_resolved": False, "target_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

        def prepare(self, *, root: Path, identity: plan.ScoreIdentity) -> object:
            assert artifact.has_name("attempt.json")
            events.append("prepare")
            return self

        def materialize_inputs(self, runtime: object, *, identity: plan.ScoreIdentity,
                               evaluation_authority: sharedscore.FixedEvaluationAuthority) -> sharedscore.InputAuthority:
            assert runtime is self
            events.append("materialize")
            return authority

        def score_budget(self, runtime: object, *, budget: int, input_authority_sha256: str,
                         identity: plan.ScoreIdentity) -> Any:
            assert runtime is self
            events.append(f"budget_m{budget}")
            return support._precision_lifecycle_cells(
                authority=authority, binding=v8_binding, support=support_module,
                budget=budget, input_authority_sha256=input_authority_sha256,
            )

        def revalidate(self, runtime: object, *, root: Path, identity: plan.ScoreIdentity) -> None:
            assert runtime is self
            events.append("revalidate")

        def failure_progress(self, runtime: object | None) -> Mapping[str, object]:
            return {
                "within_assets_opened": False, "external_assets_opened": False,
                "checkpoint_opened": False, "cuda_initialized": False,
                "full_system_forward_count": 0, "group_forward_count": 0,
            }

        def close(self, runtime: object | None) -> None:
            events.append("close")

    artifact = _Artifact(events)
    hooks = replace(
        score.V2_LIFECYCLE_HOOKS,
        implementation_closure=lambda _root: identity.payload()["closure"],
        validate_source_gate=lambda _root: failed,
        validate_reserved_score_artifact=lambda _root, _artifact, _identity: None,
    )
    result = sharedscore.run_profiled_score_lifecycle(
        tmp_path, identity=identity, capability=capability, backend=Backend(), artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, hooks=hooks,
    )
    assert result["verdict"] == "SCREEN_COMPLETE_NO_FORMAL_VERDICT"
    assert events[:4] == ["preflight", "publish:attempt.json", "prepare", "materialize"]
    assert [event for event in events if event.startswith("budget_")] == ["budget_m10", "budget_m4"]
    attempt = json.loads(artifact.items["attempt.json"])
    assert attempt["schema"] == "precision_aware_cdmd_matched_score_attempt_v2"
    assert attempt["canonical_source_environment"] == plan.canonical_source_environment_payload()
    assert attempt["v1_failed_predecessor_binding"] == identity.payload()["v1_failed_predecessor_binding"]
    terminal = json.loads(artifact.items["terminal.json"])
    assert terminal["identity"]["schema"] == "precision_aware_cdmd_matched_score_identity_v2"
    assert set(artifact.items) == {"attempt.json", "input_authority.json", "score.json", "terminal.json"}
