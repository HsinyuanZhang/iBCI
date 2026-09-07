"""No-data/no-CUDA regressions for the V1-failure diagnostic V2 successor."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.causal_dual_memory_cell_d_score_v1 import score as shared_score
from src.precision_aware_causal_dual_memory_cell_d_speed_smoke_v1 import physical as v1_physical
from src.precision_aware_causal_dual_memory_cell_d_speed_smoke_v2 import physical, plan, score


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _write_pair(directory: Path, name: str, value: object) -> str:
    body = _json(value)
    digest = hashlib.sha256(body).hexdigest()
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _raw_comparison(*, max_abs: float = 1.5e-6, r2_delta: float = 1.5e-7, speedup: float = 0.8) -> dict[str, object]:
    """A raw physical comparison that may fail V1 but remains inspectable in V2."""
    baseline_r2 = 0.3249187469482422
    optimized_r2 = baseline_r2 - r2_delta
    baseline_wall = 4.0
    optimized_wall = baseline_wall / speedup
    return {
        "schema": "precision_aware_cdmd_speed_smoke_numerical_comparison_v1",
        "baseline_physical_eval_batch_size": 128,
        "optimized_physical_eval_batch_size": 1024,
        "baseline_prediction_sha256": "a" * 64,
        "optimized_prediction_sha256": "b" * 64,
        "max_abs_prediction_error": max_abs,
        "prediction_max_abs_tolerance": 1e-6,
        "baseline_governing_r2": baseline_r2,
        "optimized_governing_r2": optimized_r2,
        "r2_abs_error": abs(baseline_r2 - optimized_r2),
        "r2_absolute_tolerance": 1e-7,
        "baseline_transition_records_sha256": "c" * 64,
        "optimized_transition_records_sha256": "c" * 64,
        "transition_sequence_exact": True,
        "baseline_wall_seconds": baseline_wall,
        "optimized_wall_seconds": optimized_wall,
        "speedup_ratio": baseline_wall / optimized_wall,
        "baseline_full_system_forward_count": 1016,
        "baseline_group_forward_count": 3493,
        "optimized_full_system_forward_count": 9,
        "optimized_group_forward_count": 29,
        "smoke_min_speedup_ratio_exclusive": 1.0,
        "recommended_speedup_ratio": 1.5,
    }


@contextlib.contextmanager
def _synthetic_v1_failed_root() -> object:
    """Create a held-FD-compatible V1 attempt/input/failure topology in a temp root."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / plan.V1_SCORE_ROOT_RELATIVE
        directory.mkdir(parents=True)
        identity = {"schema": "synthetic_v1_speed_identity", "ordinal": 1}
        attempt = {"identity": identity}
        attempt_sha = _sha(attempt)
        input_authority = {"identity_sha256": _sha(identity)}
        input_sha = _sha(input_authority)
        failure = {
            "schema": "precision_aware_cdmd_speed_smoke_failure_v1",
            "status": "FAILED",
            "identity": identity,
            "attempt_sha256": attempt_sha,
            "input_authority_sha256": input_sha,
            "stage": "budget_m4",
            "error_class": "SpeedSmokeError",
            "error_sha256": plan.V1_FAILURE_ERROR_SHA256,
            "terminal_published": False,
            "checkpoint_opened": True,
            "cuda_initialized": True,
            "full_system_forward_count": 1016,
            "group_forward_count": 3493,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }
        body_shas = {
            "attempt.json": _write_pair(directory, "attempt.json", attempt),
            "input_authority.json": _write_pair(directory, "input_authority.json", input_authority),
            "failure.json": _write_pair(directory, "failure.json", failure),
        }
        contract = plan.V1FailureContract(body_sha256s=body_shas)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(plan, "V1_FAILURE_SHAS", body_shas))
            stack.enter_context(mock.patch.object(plan, "V1_FAILURE", contract))
            yield SimpleNamespace(root=root, directory=directory, body_shas=body_shas, identity=identity)


class _MemoryArtifact:
    """The shared lifecycle protocol, without a filesystem/root reservation."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.bodies: dict[str, bytes] = {}

    def publish_json(self, name: str, payload: dict[str, object]) -> str:
        self.events.append(f"publish:{name}")
        body = _json(payload)
        self.bodies[name] = body
        return hashlib.sha256(body).hexdigest()

    def publish_group(self, bodies: dict[str, bytes], *, post_publish: object | None = None) -> dict[str, str]:
        self.events.append("publish_group")
        self.bodies.update(bodies)
        digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        if callable(post_publish):
            post_publish(bodies, digests)
        return digests

    def reload_json(self, name: str, expected_sha256: str | None = None) -> dict[str, object]:
        body = self.reload_pair(name, expected_sha256)
        loaded = json.loads(body)
        assert isinstance(loaded, dict)
        return loaded

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        body = self.bodies[name]
        if expected_sha256 is not None and hashlib.sha256(body).hexdigest() != expected_sha256:
            raise AssertionError("synthetic artifact SHA drift")
        return body

    def has_name(self, name: str) -> bool:
        return name in self.bodies


class _FailAfterComparisonBackend:
    def __init__(self, events: list[str], comparison: dict[str, object]) -> None:
        self.events = events
        self.comparison = comparison

    def preflight(self, *, root: Path, identity: object) -> dict[str, object]:
        self.events.append("backend:preflight")
        return {
            "target_paths_resolved": False, "target_opened": False,
            "checkpoint_opened": False, "cuda_initialized": False,
        }

    def prepare(self, *, root: Path, identity: object) -> object:
        self.events.append("backend:prepare")
        return object()

    def materialize_inputs(self, runtime: object, *, identity: object, evaluation_authority: object) -> object:
        self.events.append("backend:materialize")
        return object()

    def score_budget(self, runtime: object, *, budget: int, input_authority_sha256: str, identity: object) -> tuple[object, ...]:
        self.events.append(f"backend:score:{budget}")
        raise RuntimeError("synthetic post-comparison policy failure")

    def revalidate(self, runtime: object, *, root: Path, identity: object) -> None:
        raise AssertionError("failure lifecycle must not final-revalidate")

    def failure_progress(self, runtime: object | None) -> dict[str, object]:
        return {
            "within_assets_opened": True, "external_assets_opened": True,
            "checkpoint_opened": True, "cuda_initialized": True,
            "full_system_forward_count": 1016, "group_forward_count": 3493,
            "diagnostic_numeric_comparison": score.diagnostic_numeric_comparison(self.comparison),
        }

    def close(self, runtime: object | None) -> None:
        self.events.append("backend:close")


class SpeedSmokeV2StaticTest(unittest.TestCase):
    def test_dry_cli_is_torch_free_and_exposes_relaxed_diagnostic_policy(self) -> None:
        self.assertEqual(plan.dry_plan()["optimized_o1_o2"]["v2_terminal_thresholds"], {
            "prediction_max_abs": 2e-6, "r2_abs": 2e-7, "speedup_is_descriptive": True,
        })
        root = Path(__file__).resolve().parents[2]
        script = root / "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_speed_smoke_v2.py"
        environment = dict(os.environ)
        environment.update({
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": f"{root / 'tfpd_exploration'}:{root / 'tfpd_exploration/src'}",
        })
        probe = (
            "import importlib.util,sys; p=sys.argv[1]; s=importlib.util.spec_from_file_location('v2_dry',p); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.main(['--dry-run']); "
            "raise SystemExit(0 if 'torch' not in sys.modules else 17)"
        )
        completed = subprocess.run([sys.executable, "-c", probe, str(script)], env=environment,
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_v1_failed_graph_is_held_exactly_and_topology_tamper_fails(self) -> None:
        with _synthetic_v1_failed_root() as fixture:
            binding = score.validate_v1_failed_speed_smoke(fixture.root)
            self.assertEqual(binding.historical_v1_identity, fixture.identity)
            self.assertEqual(score.validate_v1_failure_binding(binding.payload()), binding.payload())
            (fixture.directory / "injected.json").write_text("{}", encoding="utf-8")
            os.chmod(fixture.directory / "injected.json", 0o444)
            with self.assertRaisesRegex(score.SpeedSmokeV2Error, "descriptor read failed"):
                score.validate_v1_failed_speed_smoke(fixture.root)

    def test_v1_predicates_are_disclosed_v2_accepts_bounded_numeric_drift_and_speedup_is_nonblocking(self) -> None:
        comparison = score.diagnostic_numeric_comparison(_raw_comparison())
        self.assertEqual(comparison["v1_predicates"], {
            "max_abs_prediction_error_passed": False,
            "r2_absolute_error_passed": False,
            "speedup_strictly_gt_one_passed": False,
        })
        self.assertTrue(comparison["v2_predicates"]["accepted"])
        self.assertFalse(comparison["recommendation_reached"])
        rejected = score.diagnostic_numeric_comparison(_raw_comparison(max_abs=2.1e-6, r2_delta=0.0, speedup=2.0))
        self.assertFalse(rejected["v2_predicates"]["accepted"])
        forged = _raw_comparison()
        forged["r2_abs_error"] = 0.0
        with self.assertRaisesRegex(score.SpeedSmokeV2Error, "literal drift"):
            score.diagnostic_numeric_comparison(forged)

    def test_diagnostic_cell_keeps_comparison_for_terminal_and_rejects_only_v2_overage(self) -> None:
        """The new limits apply after durable V1 predicate disclosure, not before it."""
        transitions = [{"carrier_transition_committed": False}]
        raw = _raw_comparison()
        raw["baseline_transition_records_sha256"] = score._digest(transitions)
        raw["optimized_transition_records_sha256"] = score._digest(transitions)
        baseline = {
            "prediction_sha256": raw["baseline_prediction_sha256"],
            "governing_r2": raw["baseline_governing_r2"], "transition_records": transitions,
            "input_record_sha256": "e" * 64, "full_system_forward_count": 1016,
            "group_forward_count": 3493,
        }
        optimized = {
            "prediction_sha256": raw["optimized_prediction_sha256"],
            "governing_r2": raw["optimized_governing_r2"], "transition_records": transitions,
            "input_record_sha256": "e" * 64, "full_system_forward_count": 9,
            "group_forward_count": 29,
        }
        witness = SimpleNamespace(
            budget=4, surface="external", session="synthetic-session", canonical_row_sha256="f" * 64,
            input_record_sha256="e" * 64, model_state_sha256="d" * 64,
            sealed_model_load_proof_sha256="c" * 64, carrier_commits=0, transition_count=1,
        )
        evidence = {
            "stage0_o1_o2_engine_evidence": {
                "actual_model_forward_count": 9,
                "identity_encoder_forward_count": 1,
                "actual_model_forward_count_by_path": {"full": 9},
                "logical_chunk_count_by_path": {"full": 9},
            },
        }
        with (
            mock.patch.object(score, "_baseline_row", return_value=(baseline, transitions)),
            mock.patch.object(score, "_optimized_row", return_value=(optimized, transitions)),
            mock.patch.object(score, "_validate_speed_evidence", side_effect=lambda value: dict(value)),
            mock.patch.object(score, "_validate_resources", side_effect=lambda value, _identity: dict(value)),
        ):
            cell = score.build_diagnostic_cell(
                identity=object(), input_authority_sha256="b" * 64,
                baseline_session_row=baseline, optimized_session_row=optimized, raw_comparison=raw,
                speed_evidence=evidence, resources={"synthetic": True}, witness=witness,
            )
            self.assertTrue(cell["v2_numerical_equivalence_accepted"])
            self.assertFalse(cell["diagnostic_numeric_comparison"]["v1_predicates"]["speedup_strictly_gt_one_passed"])
            too_large = dict(raw)
            too_large["max_abs_prediction_error"] = 2.1e-6
            with self.assertRaisesRegex(score.SpeedSmokeV2Error, "limits exceeded"):
                score.build_diagnostic_cell(
                    identity=object(), input_authority_sha256="b" * 64,
                    baseline_session_row=baseline, optimized_session_row=optimized, raw_comparison=too_large,
                    speed_evidence=evidence, resources={"synthetic": True}, witness=witness,
                )

    def test_post_comparison_failure_receipt_persists_typed_live_numeric_evidence(self) -> None:
        identity = object.__new__(score.SpeedSmokeV2Identity)
        identity_payload = {
            "schema": "synthetic_v2_identity", "closure": {"closure_sha256": "a" * 64},
            "v1_failed_predecessor_binding_sha256": "b" * 64,
        }
        comparison = score.diagnostic_numeric_comparison(_raw_comparison())
        progress = {
            "within_assets_opened": True, "external_assets_opened": True,
            "checkpoint_opened": True, "cuda_initialized": True,
            "full_system_forward_count": 1016, "group_forward_count": 3493,
            "diagnostic_numeric_comparison": comparison,
        }
        with mock.patch.object(score.SpeedSmokeV2Identity, "payload", return_value=identity_payload):
            failure = score._failure_payload(
                identity, "c" * 64, "d" * 64, "budget_m4", RuntimeError("diagnostic failed"), progress,
            )
            self.assertEqual(failure["diagnostic_numeric_comparison"], comparison)
            self.assertEqual(
                score.validate_failure_payload(failure, identity=identity, attempt_sha256="c" * 64,
                                               input_authority_sha256="d" * 64),
                failure,
            )

    def test_physical_adapter_captures_comparison_before_v2_cell_policy_can_fail(self) -> None:
        comparison = _raw_comparison()

        class _Runtime:
            def _require_state(self) -> object:
                witness = plan.SMOKE_ROWS[0]
                return SimpleNamespace(sessions={(witness.surface, witness.session): object()})

            def score_eager_baseline_and_optimized(self, *, session: object, budget: int) -> tuple[object, object, object, object]:
                return ({}, {}, comparison, {})

        class _Delegate:
            @staticmethod
            def failure_progress(runtime: object | None) -> dict[str, object]:
                return {
                    "within_assets_opened": True, "external_assets_opened": True,
                    "checkpoint_opened": True, "cuda_initialized": True,
                    "full_system_forward_count": 1016, "group_forward_count": 3493,
                }

        runtime = _Runtime()
        backend = object.__new__(physical.PhysicalPrecisionSpeedSmokeV2Backend)
        backend._runtime = runtime
        backend._delegate = _Delegate()
        backend._identity = object()
        backend._root = Path("/synthetic-no-data")
        backend._last_diagnostic_comparison = None
        with (
            mock.patch.object(physical.PhysicalPrecisionSpeedSmokeV2Backend, "_gate", return_value=None),
            mock.patch.object(v1_physical, "_ComparativeSpeedPrecisionV2Runtime", _Runtime),
            mock.patch.object(score, "build_diagnostic_cell", side_effect=score.SpeedSmokeV2Error("V2 limits exceeded")),
        ):
            with self.assertRaisesRegex(physical.PhysicalSpeedSmokeV2Error, "comparison failed"):
                backend.score_budget(runtime, budget=4, input_authority_sha256="e" * 64, identity=backend._identity)
        progress = backend.failure_progress(runtime)
        self.assertEqual(progress["diagnostic_numeric_comparison"], score.diagnostic_numeric_comparison(comparison))

    def test_shared_lifecycle_publishes_attempt_before_prepare_and_failure_keeps_comparison(self) -> None:
        """Exercise the shared transaction engine with V2's actual failure codec."""
        identity = object.__new__(score.SpeedSmokeV2Identity)
        identity_payload = {
            "schema": "synthetic_v2_identity", "closure": {"closure_sha256": "a" * 64},
            "v1_failed_predecessor_binding_sha256": "b" * 64,
        }
        events: list[str] = []
        comparison = _raw_comparison()
        binding = SimpleNamespace(payload=lambda: {"binding": "held"})
        artifact = _MemoryArtifact(events)
        approved = SimpleNamespace(official_preflight_sha256="c" * 64, root_authorization_sha256="d" * 64)
        backend = _FailAfterComparisonBackend(events, comparison)
        hooks = replace(
            score.SPEED_SMOKE_V2_LIFECYCLE_HOOKS,
            require_capability=lambda capability, _identity: capability,
            validate_preflight=lambda value, _identity: value,
            validate_authorization=lambda value, _pre_sha, _preflight, _identity: value,
            implementation_closure=lambda _root: identity_payload["closure"],
            validate_source_gate=lambda _root: binding,
            validate_reserved_score_artifact=lambda _root, _artifact, _identity: None,
            make_attempt=lambda _identity, _pre_sha, _auth_sha: {"schema": "synthetic_attempt"},
            fixed_authority_from_preflight=lambda _preflight: object(),
            input_payload=lambda _authority, _identity: {"schema": "synthetic_input"},
            validate_input_payload=lambda value, _identity, _fixed: value,
        )
        with mock.patch.object(score.SpeedSmokeV2Identity, "payload", return_value=identity_payload):
            with self.assertRaisesRegex(RuntimeError, "post-comparison"):
                shared_score.run_profiled_score_lifecycle(
                    Path("/synthetic-no-data"), identity=identity, capability=approved, backend=backend, artifact=artifact,
                    official_preflight_sha256="c" * 64, root_authorization_sha256="d" * 64,
                    preflight={"source_gate": binding.payload()}, authorization={}, hooks=hooks,
                )
        self.assertLess(events.index("publish:attempt.json"), events.index("backend:prepare"))
        self.assertIn("failure.json", artifact.bodies)
        failure = json.loads(artifact.bodies["failure.json"])
        self.assertEqual(failure["stage"], "budget_m4")
        self.assertEqual(failure["diagnostic_numeric_comparison"], score.diagnostic_numeric_comparison(comparison))
        self.assertNotIn("terminal.json", artifact.bodies)

    def test_environment_gate_rejects_missing_swapped_relative_or_alias_roots_before_any_write(self) -> None:
        good = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **plan.CANONICAL_SOURCE_ROOTS}
        self.assertEqual(plan.validate_selected_launch_environment(good)["selected_device_profile"], plan.GPU0_PROFILE)
        for key, bad in (
            ("CUDA_VISIBLE_DEVICES", "1"), ("SUBC_DATA_ROOT", "relative/sub-C"),
            ("SUBM_DATA_ROOT", plan.CANONICAL_SOURCE_ROOTS["SUBC_DATA_ROOT"]),
            ("SUBC_DATA_ROOT", "/tmp/symlink-alias-sub-C"),
        ):
            candidate = dict(good)
            candidate[key] = bad
            with self.assertRaises(plan.SpeedSmokeV2PlanError):
                plan.validate_selected_launch_environment(candidate)

    def test_explicit_closure_is_workorder_bound_and_no_torch_cuda_is_initialized(self) -> None:
        root = Path(__file__).resolve().parents[2]
        closure = plan.implementation_closure(root).payload()
        self.assertEqual(
            next(item["sha256"] for item in closure["paths"] if item["path"] == plan.WORKORDER_RELATIVE),
            plan.WORKORDER_SHA256,
        )
        self.assertNotIn("*", "|".join(plan.IMPLEMENTATION_PATHS))
        if "torch" in sys.modules:
            self.assertFalse(sys.modules["torch"].cuda.is_initialized())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
