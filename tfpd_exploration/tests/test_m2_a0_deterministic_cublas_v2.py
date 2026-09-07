"""Focused no-data/no-CUDA tests for M2 deterministic-cuBLAS A0 V2."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tfpd_exploration.src.m2_a0_deterministic_cublas_v2 import binding, execution, lifecycle, plan
from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import physical as v1_physical
from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import plan as v1_plan
from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import receipts


def _environment() -> dict[str, str]:
    return dict(plan.DETERMINISTIC_ENVIRONMENT)


def _v1_failure_graph(repo: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    root = repo / plan.V1_EXTERNAL_ROOT_RELATIVE; root.mkdir(parents=True)
    attempt = receipts.publish_pair(root, "attempt.json", {
        "schema": "m2_a0_attempt_v1", "source_closure": {"closure_sha256": "f" * 64}})
    launch = receipts.publish_pair(root, "launch.json", {
        "schema": "m2_a0_launch_v1", "attempt_sha256": attempt.sha256})
    failure = receipts.publish_pair(root, "failure.json", {
        "schema": "m2_a0_failure_v1", "status": "FAIL_CLOSED", "stage": "replay",
        "attempt_sha256": attempt.sha256,
        "error": "RuntimeError: deterministic CUDA needs CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "progress": {"parameter_updates": 0, "target_gradients": 0}})
    monkeypatch.setattr(plan, "V1_EXTERNAL_ATTEMPT_SHA256", attempt.sha256)
    monkeypatch.setattr(plan, "V1_EXTERNAL_LAUNCH_SHA256", launch.sha256)
    monkeypatch.setattr(plan, "V1_EXTERNAL_FAILURE_SHA256", failure.sha256)
    monkeypatch.setattr(plan, "V1_HISTORICAL_CLOSURE_SHA256", "f" * 64)
    return {"attempt_sha256": attempt.sha256, "launch_sha256": launch.sha256,
            "failure_sha256": failure.sha256}


def test_deterministic_environment_rejects_before_live_root_reservation(tmp_path: Path) -> None:
    env = _environment(); del env["CUBLAS_WORKSPACE_CONFIG"]
    root = tmp_path / "external"; root.parent.mkdir(exist_ok=True)
    with pytest.raises(binding.BindingError, match="deterministic cuBLAS"):
        binding.issue_live_capability(repo_root=tmp_path / "missing-repo", root=root,
                                      surface="external_post30_local", roster=("s",), environ=env)
    assert not root.exists()
    env = _environment(); env["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    with pytest.raises(binding.BindingError, match="deterministic cuBLAS"):
        binding.validate_deterministic_environment(env)


def test_v1_external_failure_descriptor_codec_accepts_exact_graph_and_rejects_mode_sidecar_or_semantic_drift(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"; expected = _v1_failure_graph(repo, monkeypatch)
    witness = binding.validate_v1_external_failure(repo)
    assert {key: witness[key] for key in expected} == expected
    root = repo / plan.V1_EXTERNAL_ROOT_RELATIVE
    (root / "failure.json.sha256").chmod(0o644)
    with pytest.raises(receipts.ReceiptError, match="0444"):
        binding.validate_v1_external_failure(repo)


def test_v2_roots_are_disjoint_and_science_profile_is_exact_v1_parity() -> None:
    roots = tuple(plan.V2_ROOTS.values())
    assert len(set(roots)) == 3 and all("_v2/" in item for item in roots)
    profile = binding.v1_science_profile()
    binding.assert_science_profile_parity(profile)
    with pytest.raises(binding.BindingError, match="science profile"):
        binding.assert_science_profile_parity(replace(profile, decoder_window_bins=100))


def test_v2_synthetic_lifecycle_is_current_closure_bound_and_environment_rechecked(tmp_path: Path) -> None:
    closure = binding.source_closure(_ROOT)
    assert closure["closure_sha256"] != plan.V1_HISTORICAL_CLOSURE_SHA256
    env = _environment()
    v1 = {"root_relative": plan.V1_EXTERNAL_ROOT_RELATIVE, "attempt_sha256": "a" * 64,
          "launch_sha256": "b" * 64, "failure_sha256": "c" * 64,
          "historical_closure_sha256": "d" * 64}
    cpre = {"terminal_sha256": "e" * 64, "metadata_inventory_sha256": "f" * 64,
            "historical_cpre_closure_sha256": "1" * 64}
    cap = binding._mint_test_capability({
        "root_relative": plan.V2_ROOTS["external_post30_local"], "closure_sha256": closure["closure_sha256"],
        "deterministic_environment": env, "v1_external_failure_predecessor": v1,
        "historical_cpre_v2_witness": cpre, "science_profile": binding.v1_science_profile().__dict__})
    parent = tmp_path / "roots"; parent.mkdir(); root = parent / "external"
    result = lifecycle.execute_synthetic(root=root, repo_root=_ROOT, capability=cap,
                                         surface="external_post30_local", v1_failure=v1, cpre_v2=cpre,
                                         environ=env)
    assert set(result) == {"attempt", "launch", "terminal"}
    terminal = receipts.descriptor_read(root, "terminal.json").payload
    assert terminal["source_closure"] == closure
    assert terminal["deterministic_environment"] == env
    bad_cap = binding._mint_test_capability({
        "root_relative": plan.V2_ROOTS["within_post30"], "closure_sha256": closure["closure_sha256"],
        "deterministic_environment": env, "v1_external_failure_predecessor": v1,
        "historical_cpre_v2_witness": cpre})
    bad_env = {**env, "CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    with pytest.raises(binding.BindingError, match="deterministic"):
        binding.consume(bad_cap, expected_root=plan.V2_ROOTS["within_post30"], closure=closure,
                        environ=bad_env)


def test_v2_synthetic_failure_revalidates_env_and_preserves_attempt_launch_prefix(tmp_path: Path) -> None:
    closure, env = binding.source_closure(_ROOT), _environment()
    v1 = {"root_relative": plan.V1_EXTERNAL_ROOT_RELATIVE, "attempt_sha256": "a" * 64,
          "launch_sha256": "b" * 64, "failure_sha256": "c" * 64,
          "historical_closure_sha256": "d" * 64}
    cpre = {"terminal_sha256": "e" * 64, "metadata_inventory_sha256": "f" * 64,
            "historical_cpre_closure_sha256": "1" * 64}
    cap = binding._mint_test_capability({"root_relative": plan.V2_ROOTS["external_post30_local"],
                                         "closure_sha256": closure["closure_sha256"],
                                         "deterministic_environment": env,
                                         "v1_external_failure_predecessor": v1,
                                         "historical_cpre_v2_witness": cpre})
    root = tmp_path / "external"
    with pytest.raises(lifecycle.LifecycleError, match="lifecycle"):
        lifecycle.execute_synthetic(root=root, repo_root=_ROOT, capability=cap,
                                    surface="external_post30_local", v1_failure=v1, cpre_v2=cpre,
                                    environ=env, fail_after_launch=True)
    verified = receipts.verify_terminal_xor(root, success_bodies=("attempt.json", "launch.json", "terminal.json"))
    assert set(verified) == {"attempt.json", "launch.json", "failure.json"}
    assert verified["failure.json"].payload["progress"]["deterministic_environment"] == env


def test_v2_public_cli_is_inert() -> None:
    import subprocess
    script = _ROOT / "tfpd_exploration/scripts/run_m2_a0_deterministic_cublas_v2.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--dry-run"], check=True,
                            capture_output=True, text=True)
    assert "inert" in result.stdout
    blocked = subprocess.run([sys.executable, "-S", str(script), "--execute"], capture_output=True, text=True)
    assert blocked.returncode == 2 and "capability" in blocked.stderr


def _held_profile_binding(closure: dict[str, object], env: dict[str, str], *, surface: str = "external_post30_local") -> dict[str, object]:
    roster = ("s",)
    segments = [{"first_ordinal": 0, "last_ordinal": 4, "scored_prediction_count": 5,
                 "pre_update_identity_sha256": "a" * 64, "governed_starts_sha256": "b" * 64}]
    arm = {"frozen_batch_size": 32, "first_prediction_parity_singleton_batch": True,
           "actual_scored_prediction_count": 5, "identity_segments": segments,
           "identity_segments_sha256": v1_physical._canonical_json_sha({"identity_segments": segments})}
    query = {"s": {"governed_post30_cpre_int64_sha256": "4" * 64,
                    "governed_post30_anchor_core_sha256": "5" * 64,
                    "governed_post30_window_count": 5,
                    "cpre_anchor_reference_int64_sha256": "6" * 64,
                    "anchor_score_ordered_window_starts_sha256": "7" * 64,
                    "anchor_reference_window_count": 9,
                    "governed_post30_subset_of_anchor_full_query": True,
                    "within_post30_exact_anchor_parity": None,
                    "batch_decode_evidence": {"frozen_batch_size": 32,
                                                "arms": {key: dict(arm) for key in (
                                                    "static", "true_trial", "chunk_phase0", "chunk_phase50")}}}}
    anchor = {"schema": "m2_t4_activity_budget_screen_v1", "status": "TERMINAL",
              "checkpoint_sha256": v1_plan.ANCHOR_CHECKPOINT_SHA256,
              "normalization_sha256": v1_plan.ANCHOR_NORMALIZER_SHA256,
              "parameter_updates": 0, "target_gradients": 0,
              "rows": [{"cell": v1_plan.ANCHOR_CELL}]}
    return {"mode": "test", "root_relative": plan.V2_ROOTS[surface],
            "closure_sha256": closure["closure_sha256"], "deterministic_environment": env,
            "surface": surface, "roster": list(roster), "cpre_terminal_sha256": "c" * 64,
            "cpre_metadata_inventory_sha256": "d" * 64, "static_anchor_payload": anchor,
            "static_anchor_descriptor_sha256": "e" * 64,
            "static_anchor_payload_sha256": binding._canonical_sha(anchor),
            "query_window_authority": query, "query_window_authority_sha256": binding._canonical_sha(query),
            "device_profile": v1_physical.static_device_profile(),
            "scheduler_profile": v1_physical.scheduler_profile(surface),
            "scheduler_attestation": {"synthetic": True}, "frozen_decode_batch_size": 32,
            "resolved_window_size": 50, "science_profile": binding.v1_science_profile().__dict__,
            "v1_external_failure_predecessor": {"root_relative": plan.V1_EXTERNAL_ROOT_RELATIVE,
                                                  "attempt_sha256": "1" * 64, "launch_sha256": "2" * 64,
                                                  "failure_sha256": "3" * 64,
                                                  "historical_closure_sha256": "4" * 64},
            "historical_cpre_v2_witness": {"terminal_sha256": "c" * 64,
                                            "metadata_inventory_sha256": "d" * 64,
                                            "historical_cpre_closure_sha256": "5" * 64}}


class _FakeCuda:
    @staticmethod
    def is_initialized() -> bool:
        return False


class _FakeTorch:
    cuda = _FakeCuda()


def _runtime_factory(surface: str) -> dict[str, object]:
    return {"torch": _FakeTorch(), "scheduler_attestation": {"synthetic": True},
            "gpu_uuid_attestation": {"raw_uuid": plan.GPU0_UUID, "canonical_uuid": plan.GPU0_UUID}}


def _session_factory(*, context: object, surface: str, session: str, batch_size: int,
                     anchor_window_evidence: object, fail: bool = False):
    if fail:
        raise RuntimeError("typed synthetic replay failure")
    digest = "d" * 64
    rows = [v1_physical.A0Row(surface=surface, session_id=session, arm=arm, phase=phase, r2=r2,
                              first_prediction_sha256=digest, prediction_sha256=digest,
                              initial_b3s_identity_sha256=digest, final_b3s_identity_sha256=digest,
                              initial_activity_stack_sha256=digest, final_activity_stack_sha256=digest,
                              query_starts_sha256=digest, target_window_sha256=digest, window_count=5,
                              update_count=0, eviction_count=0, partial_final_bins_discarded=0,
                              parameter_updates=0, target_gradients=0, wall_seconds=.01, peak_memory_bytes=0)
            for arm, phase, r2 in (("static", None, .2), ("true_trial", None, .3),
                                   ("chunk", 0, .299), ("chunk", 50, .29))]
    carrier = {session: {"budget": 10, "selection": "chronological_first_m", "selected_indices": list(range(10)),
                         "selected_indices_sha256": v1_plan.M10_CHRONOLOGICAL_SELECTED_INDICES_SHA256,
                         "usable_directional_trials": 3, "raw_t4_sha256": "e" * 64,
                         "normalized_t4_sha256": "f" * 64}}
    seed = {session: {"seed_source": "dataset.calib_trialized_neural_features[:30]", "activity_rows": 30,
                      "reconstruction_law": "frozen_cubic_interpolation",
                      "sealed_first30_activity_stack_sha256": "1" * 64,
                      "used_seed_activity_stack_sha256": "1" * 64,
                      "reconstructed_first30_activity_stack_sha256": "2" * 64,
                      "reconstructed_vs_sealed_max_abs": 0.0}}
    return rows, carrier, seed, {session: anchor_window_evidence}


def test_v2_adapter_runs_the_real_shared_production_loop_with_typed_runtime_success_and_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _environment()
    for name, value in env.items(): monkeypatch.setenv(name, value)
    closure = binding.source_closure(_ROOT); held = _held_profile_binding(closure, env)
    root = tmp_path / "external"
    cap = binding._mint_test_capability(held)
    result = execution.execute_production(root=root, repo_root=_ROOT, capability=cap,
                                          surface="external_post30_local", roster=("s",),
                                          runtime_factory=_runtime_factory, session_replay_factory=_session_factory)
    terminal = receipts.descriptor_read(root, "terminal.json").payload
    assert result["terminal"] == receipts.descriptor_read(root, "terminal.json").sha256
    assert terminal["schema"] == "m2_a0_terminal_v2"
    assert terminal["execution_profile_bindings"]["deterministic_environment"] == env
    with pytest.raises(binding.BindingError, match="one-shot"):
        execution.execute_production(root=tmp_path / "again", repo_root=_ROOT, capability=cap,
                                     surface="external_post30_local", roster=("s",), runtime_factory=_runtime_factory,
                                     session_replay_factory=_session_factory)
    v1_cap = v1_physical._mint_test_capability({"root_relative": plan.V2_ROOTS["external_post30_local"],
                                                 "closure_sha256": closure["closure_sha256"]})
    with pytest.raises(binding.BindingError, match="opaque"):
        execution.execute_production(root=tmp_path / "wrong-token", repo_root=_ROOT, capability=v1_cap,
                                     surface="external_post30_local", roster=("s",), runtime_factory=_runtime_factory,
                                     session_replay_factory=_session_factory)
    failure_root = tmp_path / "failure" / "external"; failure_root.parent.mkdir()
    failed_cap = binding._mint_test_capability(_held_profile_binding(closure, env))
    with pytest.raises(v1_physical.PhysicalContractError, match="failed"):
        execution.execute_production(root=failure_root, repo_root=_ROOT, capability=failed_cap,
                                     surface="external_post30_local", roster=("s",), runtime_factory=_runtime_factory,
                                     session_replay_factory=lambda **kwargs: _session_factory(**kwargs, fail=True))
    failure = receipts.descriptor_read(failure_root, "failure.json").payload
    assert failure["schema"] == "m2_a0_failure_v2"
    assert failure["progress"]["execution_profile_bindings"]["deterministic_environment"] == env
