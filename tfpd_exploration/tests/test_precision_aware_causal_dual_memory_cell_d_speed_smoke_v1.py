"""No-data/no-CUDA adversaries for the bounded Precision-V2 speed smoke."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.precision_aware_causal_dual_memory_cell_d_score_v2 import physical as precision_v2_physical
from src.precision_aware_causal_dual_memory_cell_d_speed_smoke_v1 import plan, score


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


def _transition_rows(commits: int, total: int = 143) -> list[dict[str, object]]:
    return [{"carrier_transition_committed": index < commits} for index in range(total)]


@contextlib.contextmanager
def _synthetic_completed_v2() -> object:
    """Patch only test literals, then create an actual held-FD pair topology."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        authority_dir = root / plan.V2_AUTHORITY_ROOT_RELATIVE
        score_dir = root / plan.V2_SCORE_ROOT_RELATIVE
        authority_dir.mkdir(parents=True)
        score_dir.mkdir(parents=True)
        identity = {
            "schema": "historical_v2_identity_test",
            "closure": {"closure_sha256": "a" * 64},
            "selected_device_profile": dict(plan.HISTORICAL_V2_SELECTED_PROFILE),
            "historical_v1_science_identity": {"test": True},
        }
        records = [
            {"surface": "within", "session": "sub-C_ses-CO-20151109", "synthetic": "m10"},
            {"surface": "external", "session": "sub-M_ses-CO-20150615", "synthetic": "m4"},
        ]
        row10 = {
            "budget": 10, "session": records[0]["session"], "governing_r2": 0.625,
            "prediction_sha256": "1" * 64, "input_record_sha256": _sha(records[0]), "n_windows": 23,
            "model_state_before_sha256": "2" * 64, "model_state_after_sha256": "2" * 64,
            "sealed_model_load_proof_sha256": "3" * 64, "transition_records": _transition_rows(7),
            "full_system_forward_count": 286, "group_forward_count": 572,
        }
        row4 = {
            "budget": 4, "session": records[1]["session"], "governing_r2": 0.25,
            "prediction_sha256": "4" * 64, "input_record_sha256": _sha(records[1]), "n_windows": 34,
            "model_state_before_sha256": "2" * 64, "model_state_after_sha256": "2" * 64,
            "sealed_model_load_proof_sha256": "3" * 64, "transition_records": _transition_rows(5),
            "full_system_forward_count": 286, "group_forward_count": 572,
        }
        witnesses = (
            plan.SmokeRowWitness(4, "external", row4["session"], _sha(row4), row4["governing_r2"],
                                 row4["prediction_sha256"], row4["input_record_sha256"], row4["n_windows"], 5, 143,
                                 row4["model_state_before_sha256"], row4["sealed_model_load_proof_sha256"]),
        )
        input_body = {"identity_sha256": _sha(identity), "records": records}
        input_sha = _sha(input_body)
        preflight = {"identity": identity}
        pre_sha = _sha(preflight)
        authorization = {"official_preflight_sha256": pre_sha, "identity_sha256": _sha(identity)}
        auth_sha = _sha(authorization)
        attempt = {"identity": identity, "preflight_sha256": pre_sha, "authorization_sha256": auth_sha}
        attempt_sha = _sha(attempt)
        score_body = {
            "identity": identity, "input_authority_sha256": input_sha,
            "budget_summaries": {
                "10": {"budget": 10, "precision_cells": [{"surface": "within", "budget": 10, "sessions": [row10]}]},
                "4": {"budget": 4, "precision_cells": [{"surface": "external", "budget": 4, "sessions": [row4]}]},
            },
        }
        score_sha = _sha(score_body)
        terminal = {
            "identity": identity, "attempt_sha256": attempt_sha, "input_authority_sha256": input_sha,
            "score_sha256": score_sha, "launch_closure_sha256": "a" * 64,
            "final_closure_sha256": "a" * 64, "target_optimizer_backward_update": 0,
        }
        terminal_sha = _sha(terminal)
        authority_shas = {
            "official_preflight.json": _write_pair(authority_dir, "official_preflight.json", preflight),
            "root_authorization.json": _write_pair(authority_dir, "root_authorization.json", authorization),
        }
        result_shas = {
            "attempt.json": _write_pair(score_dir, "attempt.json", attempt),
            "input_authority.json": _write_pair(score_dir, "input_authority.json", input_body),
            "score.json": _write_pair(score_dir, "score.json", score_body),
            "terminal.json": _write_pair(score_dir, "terminal.json", terminal),
        }
        self_contract = plan.CompletedV2Contract(
            authority_body_sha256s=authority_shas, score_body_sha256s=result_shas,
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(plan, "V2_AUTHORITY_SHAS", authority_shas))
            stack.enter_context(mock.patch.object(plan, "V2_RESULT_SHAS", result_shas))
            stack.enter_context(mock.patch.object(plan, "V2_ATTEMPT_IDENTITY_SHA256", _sha(identity)))
            stack.enter_context(mock.patch.object(plan, "V2_FINAL_CLOSURE_SHA256", "a" * 64))
            stack.enter_context(mock.patch.object(plan, "SMOKE_ROWS", witnesses))
            stack.enter_context(mock.patch.object(plan, "COMPLETED_V2", self_contract))
            yield SimpleNamespace(
                root=root, identity=identity, witnesses=witnesses, records=records, rows=(row4,),
                authority_shas=authority_shas, result_shas=result_shas, score_dir=score_dir,
            )


def _speed_evidence() -> dict[str, object]:
    engine = {
        "schema": "causal_dual_memory_cell_d_speed_v1_identity_cache_v1",
        "logical_eval_batch_size": 128,
        "identity_cache_entry_count": 2,
        "identity_encoder_forward_count": 2,
        "actual_model_forward_count": 7,
        "actual_model_forward_count_by_path": {"full": 3, "held_group_0": 2, "held_group_1": 1, "held_group_2": 1},
        "logical_chunk_count_by_path": {"full": 2, "held_group_0": 1, "held_group_1": 1, "held_group_2": 1},
        "cache_key_semantics": "path_plus_contiguous_activity_prefix_normalized_t4_and_held_view_bytes",
        "repeat_audit": {
            "schema": "causal_dual_memory_cell_d_speed_v1_repeat_audit_v1",
            "canonical_full_first_chunk_per_exact_state": True,
            "held_group_0_fixed_mid_session_first_chunk": True,
            "repeat_audit_uses_no_target_values": True,
            "coordinates": [
                {"path": "full", "reasons": ["canonical_full_first_state"], "outputs_bitwise_equal": True},
                {"path": "held_group_0", "reasons": ["held_group_0_fixed_mid_session"], "outputs_bitwise_equal": True},
            ],
        },
        "numeric_batch_variants_deferred": [1024, 2048],
        "physical_eval_batch_size": 1024,
    }
    return {
        "schema": "precision_aware_cdmd_speed_smoke_o1_o2_evidence_v2",
        "stage0_o1_o2_engine_evidence": engine,
        "physical_eval_batch_size": 1024,
        "physical_eval_batch_candidates": [1024, 512, 128],
        "fallback_attempts": [{"physical_eval_batch_size": 1024, "outcome": "selected"}],
    }


def _comparison(baseline: dict[str, object], optimized: dict[str, object]) -> dict[str, object]:
    transitions = baseline["transition_records"]
    return {
        "schema": "precision_aware_cdmd_speed_smoke_numerical_comparison_v1",
        "baseline_physical_eval_batch_size": 128,
        "optimized_physical_eval_batch_size": 1024,
        "baseline_prediction_sha256": baseline["prediction_sha256"],
        "optimized_prediction_sha256": optimized["prediction_sha256"],
        "max_abs_prediction_error": 5e-7,
        "prediction_max_abs_tolerance": 1e-6,
        "baseline_governing_r2": baseline["governing_r2"],
        "optimized_governing_r2": optimized["governing_r2"],
        "r2_abs_error": abs(float(baseline["governing_r2"]) - float(optimized["governing_r2"])),
        "r2_absolute_tolerance": 1e-7,
        "baseline_transition_records_sha256": _sha(transitions),
        "optimized_transition_records_sha256": _sha(transitions),
        "transition_sequence_exact": True,
        "baseline_wall_seconds": 2.0,
        "optimized_wall_seconds": 1.0,
        "speedup_ratio": 2.0,
        "baseline_full_system_forward_count": baseline["full_system_forward_count"],
        "baseline_group_forward_count": baseline["group_forward_count"],
        "optimized_full_system_forward_count": optimized["full_system_forward_count"],
        "optimized_group_forward_count": optimized["group_forward_count"],
        "smoke_min_speedup_ratio_exclusive": 1.0,
        "recommended_speedup_ratio": 1.5,
    }


class _FakeIdentity:
    def payload(self) -> dict[str, object]:
        return {"selected_device_profile": dict(plan.GPU0_PROFILE)}


class _FakeRuntimeIdentity:
    def payload(self) -> dict[str, object]:
        return {"selected_device_profile": dict(plan.GPU0_PROFILE), "synthetic_runtime_identity": True}


class _MemoryArtifact:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.bodies: dict[str, bytes] = {}
        self.topology = ("attempt.json", "input_authority.json", "score.json", "terminal.json", "failure.json")

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
        return json.loads(body)

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        body = self.bodies[name]
        if expected_sha256 is not None and hashlib.sha256(body).hexdigest() != expected_sha256:
            raise AssertionError("synthetic artifact digest drift")
        return body

    def has_name(self, name: str) -> bool:
        return name in self.bodies


def _fixed_authority_and_records() -> tuple[object, list[dict[str, object]]]:
    """Build a typed 6/15 metadata-only authority with exact input-record rows."""
    from src.causal_dual_memory_cell_d_score_v1 import score as shared_score
    within_names = ["sub-C_ses-CO-20151109", *[f"sub-C_ses-CO-201511{day:02d}" for day in range(10, 15)]]
    external_names = ["sub-M_ses-CO-20150615", *[f"sub-M_ses-CO-201506{day:02d}" for day in range(16, 30)]]
    def asset(surface: str, session: str, index: int) -> object:
        return shared_score.EvaluationAsset(
            surface, session, f"asset-{surface}-{index}", f"{surface}/{session}_behavior+ecephys.nwb",
            100 + index, hashlib.sha256(f"{surface}:{session}".encode()).hexdigest(),
        )
    within = tuple(asset("within", session, index) for index, session in enumerate(within_names))
    external = tuple(asset("external", session, index) for index, session in enumerate(external_names))
    fixed = shared_score.FixedEvaluationAuthority(within, external, {"fixture": {}}, {})
    raw_axis = {
        "schema": "causal_dual_memory_cell_d_score_raw_t4_axis_v1", "budget": 30,
        "raw_t4_sha256": "a" * 64, "channel_order_sha256": "b" * 64, "valid_mask_sha256": "c" * 64,
        "source_unit_count": 4, "feature_group": "t4", "signal_view": "sua",
        "channel_ids_are_exact_int64_arange": True, "validity_rule": "raw_t4_modulation_m_gt_modulation_eps",
        "modulation_eps": 0.0, "raw_before_normalization": True,
    }
    records: list[dict[str, object]] = []
    for index, item in enumerate((*within, *external)):
        trials = tuple(f"{item.session}-trial-{trial:02d}" for trial in range(31))
        record = shared_score.InputRecord(
            surface=item.surface, session=item.session, asset_id=item.asset_id, frozen_path=item.frozen_path,
            asset_bytes=item.bytes, asset_sha256=item.sha256, chronological_trial_ids=trials,
            support_trial_ids_by_budget={"30": trials[:30], "10": trials[:10], "4": trials[:4]},
            query_trial_ids_by_budget={"30": trials[30:], "10": trials[30:], "4": trials[30:]},
            neural_sha256=hashlib.sha256(f"n{index}".encode()).hexdigest(),
            calibration_sha256=hashlib.sha256(f"c{index}".encode()).hexdigest(),
            target_last_bin_sha256_by_budget={key: hashlib.sha256(f"t{index}{key}".encode()).hexdigest() for key in ("30", "10", "4")},
            valid_last_bin_mask_sha256_by_budget={key: hashlib.sha256(f"m{index}{key}".encode()).hexdigest() for key in ("30", "10", "4")},
            valid_last_bin_count_by_budget={"30": 1, "10": 1, "4": 1},
            raw_m30_t4_axis_proof=raw_axis, theta_recovery_sha256=hashlib.sha256(f"theta{index}".encode()).hexdigest(),
        )
        records.append(record.payload())
    fixed.payload()
    return fixed, records


class _FakeInputAuthority:
    def __init__(self, records: list[dict[str, object]], fixed_sha: str) -> None:
        self.records = records
        self.fixed_sha = fixed_sha

    def payload(self, *, identity: object) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_score_input_authority_v1",
            "identity_sha256": identity.sha256 if hasattr(identity, "sha256") else "runtime",
            "fixed_evaluation_authority_sha256": self.fixed_sha,
            "records": self.records,
            "same_materialized_input_for_both_systems": True,
            "target_labels_metric_only": True,
            "cache_read_or_write": False,
        }


class _FakeBackend:
    def __init__(self, *, identity: object, authority: object, records: list[dict[str, object]], witness_rows: dict[int, tuple[object, dict[str, object]]], events: list[str]) -> None:
        self.identity, self.authority, self.records, self.witness_rows, self.events = identity, authority, records, witness_rows, events

    def preflight(self, *, root: Path, identity: object) -> dict[str, object]:
        self.events.append("backend:preflight")
        return {"target_paths_resolved": False, "target_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

    def prepare(self, *, root: Path, identity: object) -> object:
        self.events.append("backend:prepare")
        return object()

    def materialize_inputs(self, runtime: object, *, identity: object, evaluation_authority: object) -> object:
        self.events.append("backend:materialize")
        return _FakeInputAuthority(self.records, _sha(self.authority.payload()))

    def score_budget(self, runtime: object, *, budget: int, input_authority_sha256: str, identity: object) -> tuple[dict[str, object], ...]:
        self.events.append(f"backend:score:{budget}")
        witness, row = self.witness_rows[budget]
        resources = {
            "runtime_environment": {**plan.GPU0_PROFILE, "visible_devices": 1, "attested": True,
                                    "torch_cuda_matmul_allow_tf32": False, "torch_cudnn_allow_tf32": False},
            "current_cuda_allocated_bytes": 1, "current_cuda_reserved_bytes": 1,
            "peak_cuda_allocated_bytes": 2, "peak_cuda_reserved_bytes": 2, "rss_bytes": 1024,
            "wall_seconds": 1.0, "full_and_group_forward_chunks": 7, "completed_query_trials": 1,
            "windows_or_trials_per_s": 1.0,
        }
        return (score.build_speed_cell(
            identity=identity, input_authority_sha256=input_authority_sha256,
            baseline_session_row=row, optimized_session_row={
                **row, "prediction_sha256": "f" * 64,
                "governing_r2": float(row["governing_r2"]) + 1e-8,
            }, comparison=_comparison(row, {
                **row, "prediction_sha256": "f" * 64,
                "governing_r2": float(row["governing_r2"]) + 1e-8,
            }), speed_evidence=_speed_evidence(), resources=resources, witness=witness,
        ),)

    def revalidate(self, runtime: object, *, root: Path, identity: object) -> None:
        self.events.append("backend:revalidate")

    def failure_progress(self, runtime: object | None) -> dict[str, object]:
        return {"within_assets_opened": False, "external_assets_opened": False, "checkpoint_opened": False,
                "cuda_initialized": False, "full_system_forward_count": 0, "group_forward_count": 0}

    def close(self, runtime: object | None) -> None:
        self.events.append("backend:close")


class SpeedSmokeStaticTest(unittest.TestCase):
    def test_public_dry_plan_is_one_m4_external_cell_gpu0_only_and_never_imports_torch(self) -> None:
        self.assertEqual(plan.dry_plan()["current_gpu0_required"]["cuda_visible_devices"], "0")
        self.assertEqual(len(plan.dry_plan()["smoke_execution_order"]), 1)
        self.assertEqual(plan.dry_plan()["smoke_execution_order"][0]["budget"], 4)
        self.assertFalse(plan.dry_plan()["m30_rerun"])
        root = Path(__file__).resolve().parents[2]
        script = root / "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_speed_smoke_v1.py"
        environment = dict(os.environ)
        environment.update({
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": f"{root / 'tfpd_exploration'}:{root / 'tfpd_exploration/src'}",
        })
        probe = (
            "import importlib.util,sys; p=sys.argv[1]; s=importlib.util.spec_from_file_location('dry_probe', p); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.main(['--dry-run']); "
            "raise SystemExit(0 if 'torch' not in sys.modules else 17)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe, str(script)], env=environment, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launch_environment_rejects_gpu1_missing_relative_and_swapped_roots_before_write(self) -> None:
        base = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **plan.CANONICAL_SOURCE_ROOTS}
        self.assertEqual(plan.validate_selected_launch_environment(base)["selected_device_profile"], plan.GPU0_PROFILE)
        for key, bad in (("CUDA_VISIBLE_DEVICES", "1"), ("SUBC_DATA_ROOT", "relative/sub-C"), ("SUBM_DATA_ROOT", "/tmp/sub-M")):
            mutated = dict(base)
            mutated[key] = bad
            with self.assertRaises(plan.SpeedSmokePlanError):
                plan.validate_selected_launch_environment(mutated)

    def test_mutating_gate_rejects_claimed_gpu0_when_actual_process_is_not_gpu0(self) -> None:
        claimed = {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **plan.CANONICAL_SOURCE_ROOTS}
        actual = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **plan.CANONICAL_SOURCE_ROOTS}
        with mock.patch.dict(os.environ, actual, clear=False):
            with self.assertRaisesRegex(score.SpeedSmokeError, "supplied/actual"):
                score._validate_mutating_launch_environment(claimed)

    def test_fresh_root_gate_rejects_collision_before_attempt_or_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan.assert_fresh_prospective_root(root, plan.SCORE_ROOT_RELATIVE)
            collision = root / plan.SCORE_ROOT_RELATIVE
            collision.mkdir(parents=True)
            with self.assertRaisesRegex(plan.SpeedSmokePlanError, "already exists"):
                plan.assert_fresh_prospective_root(root, plan.SCORE_ROOT_RELATIVE)

    def test_descriptor_reader_reconstructs_exact_v2_rows_and_binding(self) -> None:
        with _synthetic_completed_v2() as fixture:
            observed = score.validate_completed_precision_v2(fixture.root)
            payload = observed.payload()
            self.assertEqual(payload["historical_selected_rows"], list(fixture.rows))
            self.assertEqual(payload["selected_row_witnesses"], [row.payload() for row in fixture.witnesses])
            self.assertEqual(score.validate_completed_v2_binding(payload), payload)

    def test_descriptor_reader_rejects_extra_leaf_and_row_rehash_drift(self) -> None:
        with _synthetic_completed_v2() as fixture:
            (fixture.score_dir / "extra.json").write_text("{}", encoding="utf-8")
            os.chmod(fixture.score_dir / "extra.json", 0o444)
            with self.assertRaisesRegex(score.SpeedSmokeError, "topology"):
                score.validate_completed_precision_v2(fixture.root)
        with _synthetic_completed_v2() as fixture:
            mutated = dict(fixture.rows[0])
            mutated["governing_r2"] = 0.0
            fake = {"budget_summaries": {"4": {"budget": 4, "precision_cells": [
                {"surface": "external", "budget": 4, "sessions": [mutated]},
            ]}}}
            with self.assertRaisesRegex(score.SpeedSmokeError, "canonical digest"):
                score._row_from_score(fake, fixture.witnesses[0])

    def test_speed_evidence_requires_o1_o2_repeat_coordinates_and_b1024_fallback_contract(self) -> None:
        evidence = _speed_evidence()
        self.assertEqual(score._validate_speed_evidence(evidence), evidence)
        for mutate in (
            lambda item: item["stage0_o1_o2_engine_evidence"].__setitem__("logical_eval_batch_size", 2048),
            lambda item: item["stage0_o1_o2_engine_evidence"]["repeat_audit"].__setitem__("coordinates", []),
            lambda item: item.__setitem__("physical_eval_batch_size", 257),
        ):
            candidate = json.loads(json.dumps(evidence))
            mutate(candidate)
            with self.assertRaises(score.SpeedSmokeError):
                score._validate_speed_evidence(candidate)

    def test_cell_requires_exact_eager_anchor_and_optimized_numerical_parity(self) -> None:
        with _synthetic_completed_v2() as fixture:
            witness, row = fixture.witnesses[0], fixture.rows[0]
            optimized = {**row, "prediction_sha256": "f" * 64, "governing_r2": row["governing_r2"] + 1e-8}
            resources = {
                "runtime_environment": {**plan.GPU0_PROFILE, "visible_devices": 1, "attested": True,
                                        "torch_cuda_matmul_allow_tf32": False, "torch_cudnn_allow_tf32": False},
                "current_cuda_allocated_bytes": 1, "current_cuda_reserved_bytes": 1,
                "peak_cuda_allocated_bytes": 2, "peak_cuda_reserved_bytes": 2, "rss_bytes": 1024,
                "wall_seconds": 1.0, "full_and_group_forward_chunks": 7, "completed_query_trials": 1,
                "windows_or_trials_per_s": 1.0,
            }
            cell = score.build_speed_cell(
                identity=_FakeIdentity(), input_authority_sha256="f" * 64,
                baseline_session_row=row, optimized_session_row=optimized,
                comparison=_comparison(row, optimized), speed_evidence=_speed_evidence(),
                resources=resources, witness=witness,
            )
            self.assertTrue(cell["baseline_exact_prediction_sha256_parity"])
            self.assertTrue(cell["optimized_prediction_numerically_equivalent"])
            bad = dict(optimized)
            bad["governing_r2"] = row["governing_r2"] + 2e-7
            with self.assertRaisesRegex(score.SpeedSmokeError, "R2"):
                score.build_speed_cell(
                    identity=_FakeIdentity(), input_authority_sha256="f" * 64,
                    baseline_session_row=row, optimized_session_row=bad,
                    comparison=_comparison(row, bad), speed_evidence=_speed_evidence(),
                    resources=resources, witness=witness,
                )

    def test_speed_physical_module_keeps_v2_environment_wrapper_and_stage0_runtime_factory(self) -> None:
        import inspect
        from src.precision_aware_causal_dual_memory_cell_d_speed_smoke_v1 import physical
        self.assertTrue(issubclass(physical._V2EnvironmentSpeedDelegate, precision_v2_physical.PhysicalPrecisionMatchedScoreV2Backend))
        self.assertIn("_ComparativeSpeedPrecisionV2Runtime", inspect.getsource(physical._V2EnvironmentSpeedDelegate.__init__))
        self.assertTrue(issubclass(
            physical._ComparativeSpeedPrecisionV2Runtime,
            physical.speed_stage0_physical.SpeedPrecisionV2ReviewedCDMScoreRuntime,
        ))
        if "torch" in sys.modules:
            self.assertFalse(sys.modules["torch"].cuda.is_initialized())

    def test_closure_is_explicit_no_glob_and_workorder_bound(self) -> None:
        root = Path(__file__).resolve().parents[2]
        closure = plan.implementation_closure(root).payload()
        self.assertEqual(closure["paths"][0]["path"], plan.IMPLEMENTATION_PATHS[0])
        self.assertEqual(
            next(row["sha256"] for row in closure["paths"] if row["path"] == plan.WORKORDER_RELATIVE),
            plan.WORKORDER_SHA256,
        )
        self.assertNotIn("*", "|".join(plan.IMPLEMENTATION_PATHS))

    def test_shared_lifecycle_publishes_attempt_before_prepare_and_terminal_after_one_cell_comparison(self) -> None:
        """Exercise the successor hooks, not a copied lifecycle loop."""
        from src.causal_dual_memory_cell_d_score_v1 import score as shared_score
        fixed, records = _fixed_authority_and_records()
        selected = {
            ("external", "sub-M_ses-CO-20150615"): _sha(next(item for item in records if item["surface"] == "external")),
        }
        rows: dict[int, dict[str, object]] = {}
        witnesses: list[plan.SmokeRowWitness] = []
        for budget, surface, session, r2, commits in (
            (4, "external", "sub-M_ses-CO-20150615", 0.25, 2),
        ):
            row = {
                "budget": budget, "session": session, "governing_r2": r2,
                "prediction_sha256": hashlib.sha256(f"pred-{budget}".encode()).hexdigest(),
                "input_record_sha256": selected[(surface, session)], "n_windows": 10 + budget,
                "model_state_before_sha256": "2" * 64, "model_state_after_sha256": "2" * 64,
                "sealed_model_load_proof_sha256": "3" * 64, "transition_records": _transition_rows(commits),
                "full_system_forward_count": 286, "group_forward_count": 572,
            }
            rows[budget] = row
            witnesses.append(plan.SmokeRowWitness(
                budget, surface, session, _sha(row), r2, row["prediction_sha256"], row["input_record_sha256"],
                row["n_windows"], commits, 143, "2" * 64, "3" * 64,
            ))
        binding = {
            "binding_sha256": "b" * 64,
            "historical_v2_input_authority": {"records": records},
            "historical_selected_rows": [rows[4]],
        }
        closure = {"schema": "synthetic_speed_closure", "paths": [], "closure_sha256": "c" * 64}
        runtime = _FakeRuntimeIdentity()
        events: list[str] = []

        class _Binding:
            def payload(self) -> dict[str, object]:
                return dict(binding)

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(plan, "SMOKE_ROWS", tuple(witnesses)))
            stack.enter_context(mock.patch.object(plan, "validate_implementation_closure", side_effect=lambda value: dict(value)))
            stack.enter_context(mock.patch.object(score, "validate_completed_v2_binding", side_effect=lambda value: dict(value)))
            stack.enter_context(mock.patch.object(score, "_runtime_v1_identity_from_completed", return_value=runtime))
            stack.enter_context(mock.patch.object(score, "validate_completed_precision_v2", side_effect=lambda _root: _Binding()))
            identity = score.SpeedSmokeIdentity(closure=closure, completed_v2_binding=binding, runtime_v1_identity=runtime.payload())
            preflight = {
                "schema": "precision_aware_cdmd_speed_smoke_target_free_preflight_v1", "status": "PREFLIGHT_ACCEPTED",
                "identity": identity.payload(), "source_gate": binding, "completed_precision_v2": binding,
                "evaluation_authority": fixed.payload(),
                "metric": dict(score.base_plan_metric()), "authority_root_relative": plan.AUTHORITY_ROOT_RELATIVE,
                "score_root_relative": plan.SCORE_ROOT_RELATIVE,
                "canonical_launch_environment": plan.validate_selected_launch_environment(
                    {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", **plan.CANONICAL_SOURCE_ROOTS}
                ),
                "target_free": True, "target_paths_resolved": False, "model_or_checkpoint_opened": False,
                "cuda_initialized": False,
                "boundaries": {"target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
                               "normalizer_refit": False, "m30_rerun": False, "sealed_comparator_rerun": False},
                "receipt_codec": "speed_smoke_v1__baseline_exact_optimized_numerical_equivalence",
            }
            authorization = score.build_root_authorization(
                official_preflight_sha256=_sha(preflight), preflight=preflight,
            )
            root_capability = shared_score._issue_root_publication_capability()
            capability = shared_score.issue_execution_capability(
                durable_preflight_sha256=_sha(preflight), durable_authorization_sha256=_sha(authorization),
                identity=identity, root_capability=root_capability,
            )
            artifact = _MemoryArtifact(events)
            backend = _FakeBackend(
                identity=identity, authority=fixed, records=records,
                witness_rows={4: (witnesses[0], rows[4])}, events=events,
            )
            hooks = replace(
                score.SPEED_SMOKE_LIFECYCLE_HOOKS,
                implementation_closure=lambda _root: closure,
                validate_source_gate=lambda _root: _Binding(),
                validate_reserved_score_artifact=lambda _root, _artifact, _identity: None,
            )
            result = shared_score.run_profiled_score_lifecycle(
                Path("/synthetic-no-data"), identity=identity, capability=capability, backend=backend, artifact=artifact,
                official_preflight_sha256=_sha(preflight), root_authorization_sha256=_sha(authorization),
                preflight=preflight, authorization=authorization, hooks=hooks,
            )
        self.assertEqual(result["verdict"], "SPEED_SMOKE_NUMERICAL_EQUIVALENCE_COMPLETE_NON_GOVERNING")
        self.assertLess(events.index("backend:preflight"), events.index("publish:attempt.json"))
        self.assertLess(events.index("publish:attempt.json"), events.index("backend:prepare"))
        self.assertIn("backend:score:4", events)
        self.assertIn("score.json", artifact.bodies)
        self.assertIn("terminal.json", artifact.bodies)
        self.assertNotIn("failure.json", artifact.bodies)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
