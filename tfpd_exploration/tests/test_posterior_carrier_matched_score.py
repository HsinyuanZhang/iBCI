"""No-data/no-CUDA tests for the Posterior Carrier matched-score scaffold."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tfpd_exploration" / "src" / "posterior_carrier_v1" / "matched_score.py"
CLI = ROOT / "tfpd_exploration" / "scripts" / "run_posterior_carrier_matched_score.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("posterior_carrier_matched_score_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


score = _load()


def _physical() -> object:
    return score._physical_module()


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _immutable_pair(directory: Path, name: str, body: bytes) -> str:
    path = directory / name
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    sidecar = directory / f"{name}.sha256"
    sidecar.write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(path, 0o444)
    os.chmod(sidecar, 0o444)
    return digest


def _synthetic_completed_full_mirror(tmp_path: Path) -> tuple[object, Path, object, dict[str, str], dict[str, object]]:
    """Create only synthetic immutable receipt bytes; never a real result."""
    physical = _physical()
    mirror = tmp_path / "import_mirror"
    mirror.mkdir()

    def closure(label: str) -> dict[str, object]:
        path = f"synthetic/{label}.py"
        hashes = {path: _hash(f"closure:{label}")}
        body = {"paths": [path], "sha256_by_path": hashes}
        return {**body, "closure_sha256": score._digest(score._json(body))}

    source_roster = [f"synthetic-source-{index:02d}" for index in range(27)]
    remote = dict(physical._REMOTE_TORCH_AUTHORITY)
    v1_boundaries = {
        "source_only": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "scientific_score": False,
    }
    v2_boundaries = {**v1_boundaries, "v1_result_mutated": False}
    v3_boundaries = {**v2_boundaries, "v2_result_mutated": False}
    v1_closure, v2_closure, v3_closure, full_closure = (
        closure("phase-b-v1"), closure("phase-b-v2"), closure("phase-b-v3"), closure("full"),
    )
    tf32_enforcement = {
        "schema": "posterior_carrier_tf32_enforcement_v3",
        "observed_pre_state": {
            "cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True, "amp_enabled": False,
        },
        "enforced_post_state": {
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "amp_enabled": False,
        },
        "enforced_before_model_construction": True,
        "enforced_before_optimizer_construction": True,
        "route_local_restore_on_close": True,
        "acceptance_uses_enforced_post_state": True,
    }

    def v3_identity(*, stage_metadata_label: str) -> dict[str, object]:
        source_identity = {
            "schema": "posterior_carrier_target_free_source_identity_v1",
            "roster": list(source_roster),
            "roster_sha256": score._digest(score._json(source_roster)),
            "strict_source_metadata_sha256": _hash(stage_metadata_label),
            "manifest_sha256": _hash("strict-manifest"),
            "ordinary_raw_t4_semantic_sha256": _hash("ordinary-t4"),
            "behavior_normalizer_semantic_sha256": _hash("behavior"),
            "source_lineage_sha256": _hash("source-lineage"),
            "source_data_root": {"synthetic": "external-source-root"},
            "source_only": True,
            "target_opened": False,
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
            "h1_opened": False,
        }
        v1 = {
            "cell": physical._FULL_CELL,
            "phase": "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V1",
            "handoff": {"path": "synthetic/handoff.md", "sha256": _hash("phase-b-v1-handoff")},
            "phase_b_normalizer_amendment": "synthetic posterior-specific source-only normalizer",
            "source_authority": source_identity,
            "closure": v1_closure,
            "remote_device": remote,
            "boundaries": v1_boundaries,
        }
        theta_topology = {"synthetic": "same-prefix-only"}
        v2 = {
            "cell": physical._FULL_CELL,
            "phase": "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V2",
            "handoff": {"path": "synthetic/handoff.md", "sha256": _hash("phase-b-v2-handoff")},
            "v1_base_source_identity": v1,
            "v1_failed_predecessor": {"synthetic": "v1-failure"},
            "theta_recovery": {
                "schema": "posterior_carrier_same_prefix_theta_recovery_v2",
                "fallback_topology": theta_topology,
                "fallback_topology_sha256": score._digest(score._json(theta_topology)),
                "snap_tolerance_rad": 0.1,
                "same_prefix_only": True,
                "no_later_row_substitution": True,
            },
            "closure": v2_closure,
            "remote_device": remote,
            "boundaries": v2_boundaries,
        }
        return {
            "cell": physical._FULL_CELL,
            "phase": "POSTERIOR_CARRIER_DISTRIBUTIONAL_IDENTITY_PHASE_B_SOURCE_SMOKE_V3",
            "handoff": {"path": "synthetic/handoff.md", "sha256": _hash("phase-b-v3-handoff")},
            "v2_source_identity": v2,
            "v1_failed_predecessor": {"synthetic": "v1-failure"},
            "v2_failed_predecessor": {"synthetic": "v2-failure"},
            "closure": v3_closure,
            "remote_device": remote,
            "tf32_contract": {
                "amp": False,
                "cuda_matmul_allow_tf32_enforced": False,
                "cudnn_allow_tf32_enforced": False,
                "enforcement_before_model_and_optimizer": True,
                "pre_state_disclosed_not_gating": True,
                "route_local_restore": True,
            },
            "boundaries": v3_boundaries,
        }

    completed_v3 = v3_identity(stage_metadata_label="completed-stage-metadata")
    fresh_v3 = v3_identity(stage_metadata_label="fresh-stage-metadata")
    binding = {
        "schema": "posterior_carrier_completed_smoke_to_full_source_identity_binding_v1",
        "stage_bound_field": "v2_source_identity.v1_base_source_identity.source_authority.strict_source_metadata_sha256",
        "stage_bound_difference_reason": "synthetic fresh stage descriptor identity differs",
        "stage_bound_metadata_must_differ": True,
        "completed_smoke_strict_source_metadata_sha256": _hash("completed-stage-metadata"),
        "fresh_full_stage_strict_source_metadata_sha256": _hash("fresh-stage-metadata"),
        "completed_smoke_v3_identity_sha256": score._digest(score._json(completed_v3)),
        "fresh_full_stage_v3_identity_sha256": score._digest(score._json(fresh_v3)),
        "stage_agnostic_v3_identity_sha256": _hash("stage-agnostic-v3"),
        "accepted_phase_b_v3_closure_sha256": v3_closure["closure_sha256"],
        "source_authority_asset_sha256s": {"synthetic/source-authority.json": _hash("source-asset")},
        "strict_roster_sha256": score._digest(score._json(source_roster)),
        "source_data_root": {"synthetic": "external-source-root"},
    }
    identity = {
        "cell": physical._FULL_CELL,
        "phase": physical._FULL_PHASE,
        "handoff": {"path": "synthetic/handoff.md", "sha256": _hash("full-handoff")},
        "completed_smoke_v3_identity": completed_v3,
        "fresh_full_stage_v3_identity": fresh_v3,
        "smoke_to_full_source_identity_binding": binding,
        "phase_b_v3_closure": v3_closure,
        "full_closure": full_closure,
        "source_smoke_lineage": {
            "root_relative": "synthetic/smoke-v3",
            "attempt_sha256": _hash("smoke-attempt"),
            "launch_sha256": _hash("smoke-launch"),
            "source_authority_sha256": _hash("smoke-source"),
            "step100_sha256": _hash("smoke-step100"),
            "terminal_sha256": _hash("smoke-terminal"),
            "step100_status": "SOURCE_SMOKE_V3_100_STEPS_COMPLETE",
            "terminal_status": "SOURCE_SMOKE_V3_COMPLETE__NON_AUTHORITATIVE",
            "phase_b_v3_closure_sha256": v3_closure["closure_sha256"],
            "completed_smoke_identity_sha256": score._digest(score._json(completed_v3)),
        },
        "remote_torch_authority": remote,
        "boundaries": physical._source_only_boundaries(),
    }
    bodies: dict[str, bytes] = {}
    spec = physical._full_training_spec()
    bodies["attempt.json"] = score._json({
        "schema": "posterior_carrier_full_train_attempt_v1",
        "cell": physical._FULL_CELL,
        "phase": physical._FULL_PHASE,
        "spec": spec,
        "identity": identity,
        "topology": physical._full_receipt_topology(),
        "source_opened": False,
        "remote_initialized": False,
        "optimizer_steps_completed": 0,
        "boundaries": physical._source_only_boundaries(),
        "status": "ATTEMPT_STARTED_SOURCE_ONLY",
    })
    attempt_sha = score._digest(bodies["attempt.json"])
    bodies["launch.json"] = score._json({
        "schema": "posterior_carrier_full_train_launch_v1",
        "cell": physical._FULL_CELL,
        "phase": physical._FULL_PHASE,
        "spec": spec,
        "identity": identity,
        "attempt_sha256": attempt_sha,
        "full_launch_closure": full_closure,
        "phase_b_v3_launch_closure": v3_closure,
        "remote_torch_authority": remote,
        "optimizer": physical._FULL_OPTIMIZER_LITERAL,
        "execution_policy": physical._FULL_EXECUTION_POLICY_LITERAL,
        "dropout_contract": physical._FULL_DROPOUT_CONTRACT,
        "boundaries": physical._source_only_boundaries(),
        "status": "FULL_TRAINING_LAUNCHED",
    })
    launch_sha = score._digest(bodies["launch.json"])
    normalizer_body = {
        "schema": "posterior_carrier_source_normalizer_v1",
        "source_roster": source_roster,
        "source_roster_sha256": score._digest(score._json(source_roster)),
        "row_order": "budget_major_M4_M10_M30_then_strict_roster_then_unit_index",
        "row_count": 81,
        "per_budget_row_counts": {"4": 27, "10": 27, "30": 27},
        "per_budget_raw_rows_sha256": {"4": _hash("m4"), "10": _hash("m10"), "30": _hash("m30")},
        "raw_rows_sha256": _hash("all-raw"),
        "mean_float64": [0.0, 0.0, 0.0, 0.0],
        "std_float64": [1.0, 1.0, 1.0, 1.0],
        "ddof": 0, "source_only": True, "contains_only_deterministic_posterior_means": True,
        "all_zero_raw_rows_retained": True,
    }
    normalizer_payload = {
        **normalizer_body,
        "body_sha256": score._digest(score._json(normalizer_body)),
    }
    nested_v1_authority = {
        "schema": "posterior_carrier_source_authority_v1",
        "posterior_prior": {"schema": "posterior_carrier_source_prior_v1"},
        "posterior_normalizer": normalizer_payload,
    }
    nested_v2_authority = {"schema": "posterior_carrier_source_authority_v2", "v1_compatible_authority": nested_v1_authority}
    nested_v3_authority = {
        "schema": "posterior_carrier_source_authority_v3",
        "cell": physical._FULL_CELL,
        "v2_compatible_authority": nested_v2_authority,
        "tf32_enforcement": tf32_enforcement,
        "closure": v3_closure,
        "launch_sha256": launch_sha,
        "v2_failed_predecessor": fresh_v3["v2_failed_predecessor"],
    }
    bodies["source_authority.json"] = score._json({
        "schema": "posterior_carrier_full_source_authority_v1",
        "cell": physical._FULL_CELL,
        "phase": physical._FULL_PHASE,
        "spec": spec,
        "identity": identity,
        "full_launch_sha256": launch_sha,
        "phase_b_v3_source_authority": nested_v3_authority,
        "phase_b_v3_source_authority_sha256": score._digest(score._json(nested_v3_authority)),
        "schedule": physical._expected_source_schedule(source_roster),
        "remote_torch_authority": remote,
        "boundaries": physical._source_only_boundaries(),
        "status": "STRICT27_POSTERIOR_SOURCE_AUTHORITY_READY",
    })
    source_sha = score._digest(bodies["source_authority.json"])
    resources = {
        "rss_bytes": 1,
        "current_allocated_bytes": 2,
        "current_reserved_bytes": 3,
        "peak_allocated_bytes": 4,
        "peak_reserved_bytes": 5,
    }
    bodies["throughput100.json"] = score._json({
        "schema": "posterior_carrier_full_throughput_v1",
        "cell": physical._FULL_CELL,
        "phase": physical._FULL_PHASE,
        "spec": spec,
        "identity": identity,
        "steps": 100,
        "epoch": 0,
        "elapsed_seconds": 1.0,
        "steps_per_second": 100.0,
        "resources": resources,
        "boundaries": physical._source_only_boundaries(),
        "status": "ENGINEERING_THROUGHPUT_ONLY",
    })
    for epoch in range(48):
        global_step = (epoch + 1) * physical._SOURCE_STEPS_PER_EPOCH
        cache = {
            "posterior_fit_calls": 81,
            "posterior_inverse_calls": 81,
            "deterministic_mean_view_builds": 81,
            "epoch_sampled_view_builds": 27 * (epoch + 1),
            "device_epoch_view_builds": 27 * (epoch + 1),
            "normalized_view_builds": 81 + 27 * (epoch + 1),
            "batch_loop_requests": global_step,
            "batch_loop_inverse_calls": 0,
            "source_sessions": 27,
            "scheduled_session_epochs": 27 * (epoch + 1),
        }
        bodies[f"epoch-{epoch:02d}.json"] = score._json({
            "schema": "posterior_carrier_full_epoch_v1",
            "cell": physical._FULL_CELL,
            "phase": physical._FULL_PHASE,
            "spec": spec,
            "identity": identity,
            "epoch": epoch,
            "cumulative_optimizer_steps": global_step,
            "schedule": physical._expected_epoch_schedule(source_roster, epoch),
            "loss": {"mean": 1.0, "min": 0.5, "max": 1.5},
            "lr": {
                "first": physical._full_lr_at_step(epoch * physical._SOURCE_STEPS_PER_EPOCH),
                "last": physical._full_lr_at_step(global_step - 1),
                "expected_first": physical._full_lr_at_step(epoch * physical._SOURCE_STEPS_PER_EPOCH),
                "expected_last": physical._full_lr_at_step(global_step - 1),
            },
            "batch_count_by_budget": {"4": physical._SOURCE_STEPS_PER_EPOCH, "10": 0, "30": 0},
            "epoch_boundary_proof": {
                "critical_gradients": {key: True for key in physical._FULL_CRITICAL_GRADIENT_KEYS},
                "finite_model": True,
                "finite_adam": True,
                "model_state_sha256": _hash(f"epoch-{epoch}-model"),
                "optimizer_state_sha256": _hash(f"epoch-{epoch}-adam"),
            },
            "posterior_cache": cache,
            "dropout_contract": physical._FULL_DROPOUT_CONTRACT,
            "resources": resources,
            "progress": {
                "epoch": epoch,
                "completed_epochs": epoch + 1,
                "epochs": 48,
                "optimizer_steps_completed": global_step,
                "total_optimizer_steps": physical._SOURCE_TOTAL_STEPS,
                "source_opened": True,
                "remote_initialized": True,
            },
            "boundaries": physical._source_only_boundaries(),
            "elapsed_seconds": 1.0,
            "throughput_steps_per_second": float(physical._SOURCE_STEPS_PER_EPOCH),
        })
    checkpoint_state = {str(epoch): _hash(f"state-{epoch}") for epoch in (44, 45, 46, 47)}
    for epoch in (44, 45, 46, 47):
        bodies[f"checkpoint-{epoch:02d}.pt"] = f"synthetic-checkpoint-{epoch}".encode("ascii")
    swa_body = b"synthetic-swa-body"
    bodies["swa_final4.pt"] = swa_body
    artifact_sha = {name: score._digest(body) for name, body in bodies.items()}
    swa_state = _hash("synthetic-swa-state")
    terminal = {
        "schema": "posterior_carrier_full_train_terminal_v1",
        "cell": physical._FULL_CELL,
        "phase": physical._FULL_PHASE,
        "spec": spec,
        "identity": identity,
        "final_identity": identity,
        "status": "FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER",
        "attempt_sha256": attempt_sha,
        "launch_sha256": launch_sha,
        "source_authority_sha256": source_sha,
        "artifact_sha256s": dict(artifact_sha),
        "checkpoint_state_sha256s": checkpoint_state,
        "swa_state_sha256": swa_state,
        "swa_proof": {
            "checkpoint_epochs": [44, 45, 46, 47],
            "checkpoint_state_sha256s": checkpoint_state,
            "fresh_strict_load": True,
            "eval_mode": True,
            "repeat_bitwise_equal": True,
            "state_unchanged": True,
            "eval_no_sampling": True,
            "prediction_shape": [32, 50, 2],
            "prediction_sha256": _hash("prediction"),
            "state_digest_before_eval": _hash("before"),
            "state_digest_after_eval": _hash("before"),
            "boundaries": physical._source_only_boundaries(),
        },
        "full_launch_closure": full_closure,
        "full_final_closure": full_closure,
        "phase_b_v3_launch_closure": v3_closure,
        "phase_b_v3_final_closure": v3_closure,
        "boundaries": physical._source_only_boundaries(),
    }
    bodies["terminal.json"] = score._json(terminal)
    digests = {name: score._digest(body) for name, body in bodies.items()}
    assert set(digests) == set(physical.full_training_artifact_names())
    for name in physical.full_training_artifact_names():
        _immutable_pair(mirror, name, bodies[name])
    provenance = physical.ImportedFullMirrorProvenance(
        local_mirror_relative="import_mirror",
        remote_result_root="remote:/posterior/full_train",
        remote_root_descriptor_identity=(9001, 9002),
        remote_terminal_sha256=digests["terminal.json"],
        remote_artifact_sha256s=digests,
        imported_artifact_sha256s=digests,
        remote_full_closure_sha256=full_closure["closure_sha256"],
    )
    return physical, mirror, provenance, digests, {
        "swa_state": swa_state, "closure": full_closure, "normalizer": normalizer_payload,
    }


def _rewrite_immutable_json_pair(mirror: Path, name: str, payload: object) -> str:
    """Rewrite a synthetic temp receipt and re-freeze both owned leaves."""
    body = score._json(payload)
    for path in (mirror / name, mirror / f"{name}.sha256"):
        os.chmod(path, 0o644)
    return _immutable_pair(mirror, name, body)


def _reprovenance_from_synthetic_mirror(physical: object, mirror: Path, old: object) -> object:
    """Simulate a malicious but internally rehashed remote/import map."""
    digests = {
        name: score._digest((mirror / name).read_bytes())
        for name in physical.full_training_artifact_names()
    }
    return physical.ImportedFullMirrorProvenance(
        local_mirror_relative=old.local_mirror_relative,
        remote_result_root=old.remote_result_root,
        remote_root_descriptor_identity=old.remote_root_descriptor_identity,
        remote_terminal_sha256=digests["terminal.json"],
        remote_artifact_sha256s=digests,
        imported_artifact_sha256s=digests,
        remote_full_closure_sha256=old.remote_full_closure_sha256,
    )


def _rewire_synthetic_full_json(
    physical: object,
    mirror: Path,
    provenance: object,
    *,
    name: str,
    mutate: object,
) -> object:
    """Recompute every outer SHA that a forged semantic receipt can control."""
    payload = json.loads((mirror / name).read_text())
    mutate(payload)
    artifact_sha = _rewrite_immutable_json_pair(mirror, name, payload)
    if name != "terminal.json":
        terminal = json.loads((mirror / "terminal.json").read_text())
        terminal["artifact_sha256s"][name] = artifact_sha
        _rewrite_immutable_json_pair(mirror, "terminal.json", terminal)
    return _reprovenance_from_synthetic_mirror(physical, mirror, provenance)


def _load_synthetic_full(physical: object, mirror: Path, provenance: object, fields: dict[str, object]) -> object:
    digests = provenance.imported_artifact_sha256s
    return physical.load_completed_full_mirror(
        mirror,
        provenance=provenance,
        expected_terminal_sha256=digests["terminal.json"],
        expected_swa_sha256=digests["swa_final4.pt"],
        expected_swa_state_sha256=fields["swa_state"],
        expected_source_authority_sha256=digests["source_authority.json"],
        expected_checkpoint_sha256s={str(epoch): digests[f"checkpoint-{epoch:02d}.pt"] for epoch in (44, 45, 46, 47)},
        expected_full_closure_sha256=fields["closure"]["closure_sha256"],
    )


def _identity() -> object:
    closure = score.implementation_closure(ROOT)
    return score.ScoreIdentity(
        full_training=score.FullTrainingEvidence(
            terminal_sha256=_hash("full-terminal"),
            swa_sha256=_hash("full-swa"),
            swa_state_sha256=_hash("full-state"),
            source_authority_sha256=_hash("full-source"),
            checkpoint_sha256s={str(epoch): _hash(f"checkpoint-{epoch}") for epoch in (44, 45, 46, 47)},
            terminal_status="FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER",
            full_closure_sha256=_hash("full-closure"),
        ),
        sealed_cell_d=score.SealedCellDEvidence(
            terminal_sha256=_hash("d-terminal"),
            swa_sha256=_hash("d-swa"),
            m30_ols_point_last_bin_table_sha256=_hash("d-last-bin"),
            sealed_point_replay_authority_sha256=_hash("d-score-authority"),
        ),
        closure=closure,
        within_roster=tuple(f"within-{index:02d}" for index in range(6)),
        external_roster=tuple(f"external-{index:02d}" for index in range(15)),
    )


def _record(surface: str, session: str, index: int) -> object:
    hashes = {str(budget): _hash(f"{surface}:{session}:m{budget}") for budget in score.BUDGETS}
    prefix_rows = [f"{session}:trial:{1000 + position}" for position in range(30)]
    theta_evidence = {
        "schema": "posterior_carrier_same_prefix_theta_recovery_v2",
        "session_id": session,
        "prefix_row_ids": prefix_rows,
        "prefix_rows_sha256": score._digest(score._json(prefix_rows)),
        "theta_m30_sha256": _hash(f"{surface}:{session}:theta"),
        "theta_sources_by_prefix_position": ["native_target_dir"] * 30,
        "fallback_rows": [],
        "fallback_count": 0,
        "no_later_row_substitution": True,
        "design_matrix_rank": 3,
        "design_matrix_condition": 1.0,
    }
    theta_evidence["body_sha256"] = score._digest(score._json(theta_evidence))
    return score.SessionInput(
        surface=surface,
        session=session,
        n_windows=9 + index,
        neural_sha256=_hash(f"{surface}:{session}:neural"),
        calibration_m30_sha256=_hash(f"{surface}:{session}:calibration"),
        last_bin_target_sha256=_hash(f"{surface}:{session}:target"),
        last_bin_valid_mask_sha256=_hash(f"{surface}:{session}:mask"),
        last_bin_valid_count=9 + index,
        posterior_prefix_input_sha256s=hashes,
        ols_point_prefix_input_sha256s={key: _hash(f"point:{key}:{session}") for key in hashes},
        matched_prefix_row_ids_sha256s={
            str(budget): score._digest(score._json(prefix_rows[:budget])) for budget in score.BUDGETS
        },
        normalized_posterior_carrier_sha256s={key: _hash(f"carrier:{key}:{session}") for key in hashes},
        normalized_ols_point_carrier_sha256s={key: _hash(f"ols-normalized:{key}:{session}") for key in hashes},
        target_theta_recovery_evidence=theta_evidence,
    )


def _authority(identity: object) -> object:
    records = tuple(
        _record(surface, session, index)
        for surface, roster in ((score.WITHIN, identity.within_roster), (score.EXTERNAL, identity.external_roster))
        for index, session in enumerate(roster)
    )
    return score.InputAuthority(
        records=records,
        source_posterior_normalizer_sha256=_hash("posterior-normalizer"),
        behavior_normalizer_sha256=_hash("behavior-normalizer"),
        shared_input_pass=True,
        cache_read_or_write=False,
    )


class _Backend:
    def __init__(self, identity: object, *, fail_at: int | None = None) -> None:
        self.identity = identity
        self.authority = _authority(identity)
        self.fail_at = fail_at
        self.calls: list[object] = []
        self.closed = False

    def prepare(self, *, identity: object, flags: object) -> None:
        assert identity == self.identity

    def resolve_inputs(self, *, identity: object, flags: object) -> object:
        assert identity == self.identity
        flags.within_opened = True
        flags.external_opened = True
        return self.authority

    def score_cell(self, *, cell: object, input_payload: object, flags: object) -> object:
        self.calls.append(cell)
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("synthetic forward failure")
        roster = self.identity.within_roster if cell.surface == score.WITHIN else self.identity.external_roster
        records = {
            row["session"]: row for row in input_payload["records"] if row["surface"] == cell.surface
        }
        base = {
            # The true comparator is the sealed Cell-D checkpoint with OLS
            # point carriers, not this posterior-trained model fed a point
            # input.  Its intentionally steeper M30-to-M4 curve makes the
            # synthetic PASS gate fully exercised.
            score.SEALED_POINT_MODE: {30: 0.10, 10: 0.08, 4: 0.05}[cell.budget],
            score.POSTERIOR_MODE: {30: 0.20, 10: 0.19, 4: 0.16}[cell.budget],
            score.ZERO_MODE: -0.05,
            score.WRONG_PAIR_MODE: -0.10,
        }[cell.mode]
        sessions = tuple(
            score.SessionScore(
                session=session,
                n_windows=records[session]["n_windows"],
                r2=base + index * 0.001,
                prediction_sha256=_hash(f"prediction:{cell.surface}:{cell.mode}:{cell.budget}:{session}"),
                input_record_sha256=score._digest(score._json(records[session])),
            )
            for index, session in enumerate(roster)
        )
        credibility = {
            score.SEALED_POINT_MODE: "sealed_cell_d_ols_point_no_posterior_bias",
            score.POSTERIOR_MODE: "posterior_precision_logit_bias",
            score.ZERO_MODE: "posterior_zero_carrier_m30_diagnostic",
            score.WRONG_PAIR_MODE: "posterior_cyclic_wrong_pair_m30_diagnostic",
        }[cell.mode]
        return score.ModeEvidence(
            cell=cell,
            model_system=(
                "sealed_cell_d_checkpoint" if cell.mode == score.SEALED_POINT_MODE else "posterior_carrier_full_swa"
            ),
            model_swa_sha256=(
                self.identity.sealed_cell_d.swa_sha256
                if cell.mode == score.SEALED_POINT_MODE else self.identity.full_training.swa_sha256
            ),
            sessions=sessions,
            input_authority_sha256=score._digest(score._json(input_payload)),
            model_state_before_sha256=_hash("same-model-state"),
            model_state_after_sha256=_hash("same-model-state"),
            eval_mode=True,
            dropout_disabled=True,
            gradients_none=True,
            finite_outputs=True,
            repeated_fixed_batch_bitwise_equal=True,
            b3s_m30_recomputed=True,
            posterior_mean_eval=cell.mode != score.SEALED_POINT_MODE,
            credibility_mode=credibility,
        )

    def reverify_after_forwards(self, *, identity: object, flags: object) -> object:
        assert identity == self.identity
        return self.identity.closure

    def close(self) -> None:
        self.closed = True


def _run(tmp_path: Path, *, backend: _Backend | None = None) -> tuple[object, object, object]:
    identity = _identity()
    artifact = score.reserve_artifact_root(tmp_path, "synthetic-score")
    backend = backend or _Backend(identity)
    terminal = score.run_score_lifecycle(
        artifact=artifact,
        identity=identity,
        execution_capability=score._issue_execution_capability(identity),
        backend=backend,
    )
    return artifact, backend, terminal


def test_static_plan_is_matrix_complete_and_has_no_h1_or_execution() -> None:
    plan = score.dry_plan()
    assert plan["status"].startswith("DRY_ONLY")
    assert len(plan["matrix"]) == 16
    assert plan["h1"] == "not_included_without_separate_authority"
    assert plan["boundaries"]["target_optimizer_steps"] == 0
    assert [item["mode"] for item in plan["matrix"][:8]] == [
        score.SEALED_POINT_MODE, score.SEALED_POINT_MODE, score.SEALED_POINT_MODE,
        score.POSTERIOR_MODE, score.POSTERIOR_MODE, score.POSTERIOR_MODE,
        score.ZERO_MODE, score.WRONG_PAIR_MODE,
    ]
    for surface in (score.WITHIN, score.EXTERNAL):
        rows = [item for item in plan["matrix"] if item["surface"] == surface]
        assert [(item["mode"], item["budget"]) for item in rows] == [
            (score.SEALED_POINT_MODE, 30), (score.SEALED_POINT_MODE, 10), (score.SEALED_POINT_MODE, 4),
            (score.POSTERIOR_MODE, 30), (score.POSTERIOR_MODE, 10), (score.POSTERIOR_MODE, 4),
            (score.ZERO_MODE, 30), (score.WRONG_PAIR_MODE, 30),
        ]


def test_closure_binds_exact_workorder_and_all_additive_files() -> None:
    closure = score.implementation_closure(ROOT).payload()
    assert closure["sha256_by_path"][score.WORKORDER_RELATIVE] == score.WORKORDER_SHA256
    assert set(closure["sha256_by_path"]) == set(score.IMPLEMENTATION_CLOSURE)
    assert closure["closure_sha256"] == score._digest(score._json({
        "paths": list(score.IMPLEMENTATION_CLOSURE), "sha256_by_path": closure["sha256_by_path"],
    }))


def test_success_lifecycle_is_complete_transaction_and_primary_comparisons_are_fixed(tmp_path: Path) -> None:
    artifact, backend, terminal = _run(tmp_path)
    assert backend.closed is True
    assert len(backend.calls) == 16
    assert terminal["score_terminal_transactional_group"] is True
    assert artifact.has_name("score.json") is True
    assert artifact.has_name("terminal.json") is True
    assert artifact.has_name("failure.json") is False
    score_payload = artifact.reload_json("score.json")
    external = score_payload["paired_posterior_minus_sealed_point"][score.EXTERNAL]
    assert external["m30"]["mean"] == pytest.approx(0.1)
    assert external["m4"]["mean"] == pytest.approx(0.11)
    assert external["m4"]["paired_bootstrap_ci"]["draws"] == 10_000
    assert score_payload["external_verdict"]["status"] == "PASS"
    assert set(score_payload["system_budget_r2"][score.WITHIN]) == {
        "sealed_cell_d_ols_point", "posterior_carrier",
    }


def test_forward_failure_after_durable_input_publishes_honest_failure_only(tmp_path: Path) -> None:
    identity = _identity()
    artifact = score.reserve_artifact_root(tmp_path, "synthetic-failure")
    backend = _Backend(identity, fail_at=3)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        score.run_score_lifecycle(
            artifact=artifact,
            identity=identity,
            execution_capability=score._issue_execution_capability(identity),
            backend=backend,
        )
    assert backend.closed is True
    assert artifact.has_name("attempt.json") is True
    assert artifact.has_name("input_authority.json") is True
    assert artifact.has_name("score.json") is False
    assert artifact.has_name("terminal.json") is False
    failure = artifact.reload_json("failure.json")
    assert failure["flags"]["forward_cells"] == [list(item) for item in [
        (score.WITHIN, score.SEALED_POINT_MODE, 30),
        (score.WITHIN, score.SEALED_POINT_MODE, 10),
    ]]
    score.validate_failure_payload(failure, identity=identity)


def test_group_rollback_leaves_no_partial_scientific_score(tmp_path: Path) -> None:
    artifact = score.reserve_artifact_root(tmp_path, "group-rollback")
    with pytest.raises(RuntimeError, match="post-publish"):
        artifact.publish_group(
            {"score.json": b"{}", "terminal.json": b"{}"},
            post_publish=lambda _bodies, _digests: (_ for _ in ()).throw(RuntimeError("post-publish")),
        )
    assert artifact.has_name("score.json") is False
    assert artifact.has_name("terminal.json") is False


def test_input_authority_rejects_lexical_or_roster_order_substitution() -> None:
    identity = _identity()
    authority = _authority(identity)
    payload = authority.payload(identity=identity)
    mutated = json.loads(json.dumps(payload))
    mutated["records"][0], mutated["records"][1] = mutated["records"][1], mutated["records"][0]
    with pytest.raises(score.ScoreError, match="roster-defined order"):
        score.validate_input_authority_payload(mutated, identity=identity)


def test_target_theta_fallback_is_target_local_and_binds_the_selected_prefix_row() -> None:
    """A target fallback is allowed only on its already selected M30 row."""
    session = "external-00"
    prefix_rows = [f"{session}:trial:{2000 + index}" for index in range(30)]
    evidence = {
        "schema": "posterior_carrier_same_prefix_theta_recovery_v2",
        "session_id": session,
        "prefix_row_ids": prefix_rows,
        "prefix_rows_sha256": score._digest(score._json(prefix_rows)),
        "theta_m30_sha256": _hash("target-theta"),
        "theta_sources_by_prefix_position": [
            "same_prefix_target_corners_canonical_snap" if index == 7 else "native_target_dir"
            for index in range(30)
        ],
        "fallback_rows": [{
            "prefix_position": 7,
            "trial_index": 2007,
            "source": "same_prefix_trial_target_corners_xyxy_center",
            "geometry": {"synthetic": "same selected row only"},
        }],
        "fallback_count": 1,
        "no_later_row_substitution": True,
        "design_matrix_rank": 3,
        "design_matrix_condition": 1.0,
    }
    evidence["body_sha256"] = score._digest(score._json(evidence))
    checked = score._validate_target_theta_recovery_evidence(evidence, session=session)
    assert checked["fallback_rows"][0]["trial_index"] == 2007

    # Recomputing an otherwise syntactically valid evidence digest cannot move
    # the geometry-derived label to another selected or later row.
    forged = json.loads(json.dumps(evidence))
    forged["fallback_rows"][0]["trial_index"] = 2008
    forged.pop("body_sha256")
    forged["body_sha256"] = score._digest(score._json(forged))
    with pytest.raises(score.ScoreError, match="selected prefix row"):
        score._validate_target_theta_recovery_evidence(forged, session=session)


def test_input_authority_binds_shared_point_and_posterior_prefix_rows() -> None:
    identity = _identity()
    payload = _authority(identity).payload(identity=identity)
    first = payload["records"][0]
    assert first["point_and_posterior_prefix_rows_identical"] is True
    assert first["target_theta_fallback_count"] == 0
    assert first["matched_prefix_row_ids_sha256s"]["30"] == first[
        "target_theta_recovery_evidence"
    ]["prefix_rows_sha256"]

    forged = json.loads(json.dumps(payload))
    forged["records"][0]["matched_prefix_row_ids_sha256s"]["4"] = _hash("different-short-prefix")
    with pytest.raises(score.ScoreError, match="shared prefix-row"):
        score.validate_input_authority_payload(forged, identity=identity)


def test_score_aggregate_rejects_forged_diagnostic_or_budget_selection(tmp_path: Path) -> None:
    artifact, _backend, _terminal = _run(tmp_path)
    identity = _identity()
    input_payload = artifact.reload_json("input_authority.json")
    value = artifact.reload_json("score.json")
    forged = json.loads(json.dumps(value))
    forged["paired_posterior_minus_sealed_point"][score.EXTERNAL]["m4"] = (
        forged["paired_posterior_minus_sealed_point"][score.EXTERNAL]["m30"]
    )
    with pytest.raises(score.ScoreError, match="aggregate/selection"):
        score.validate_score_payload(forged, identity=identity, input_payload=input_payload)


def test_external_verdict_thresholds_are_exact_and_diagnostics_cannot_rescue() -> None:
    def paired(mean: float, positives: int) -> dict[str, object]:
        return {"mean": mean, "n_positive": positives, "n_total": 15}

    degradation = {
        "posterior_m30_to_m4": {"equal_session_mean": 0.010},
        "sealed_point_m30_to_m4": {"equal_session_mean": 0.020},
    }
    diagnostics = {
        "posterior_aligned_m30": {"equal_session_mean": 0.20},
        "posterior_zero_m30": {"equal_session_mean": -0.01},
        "posterior_cyclic_wrong_pair_m30": {"equal_session_mean": -0.02},
    }
    exact_pass = score._verdict(
        paired={"m30": paired(-0.02, 8), "m4": paired(0.03, 9)},
        degradation=degradation, diagnostics=diagnostics,
    )
    assert exact_pass["status"] == "PASS"
    stop = score._verdict(
        paired={"m30": paired(-0.0200001, 8), "m4": paired(0.5, 15)},
        degradation=degradation, diagnostics=diagnostics,
    )
    assert stop["status"] == "STOP"
    # Strong M30 diagnostics cannot repair a just-missed M4 headline.
    hold = score._verdict(
        paired={"m30": paired(0.1, 15), "m4": paired(0.0299999, 15)},
        degradation=degradation, diagnostics=diagnostics,
    )
    assert hold["status"] == "HOLD"
    assert hold["gates"]["m30_anti_triviality_posterior_above_zero_and_wrong_pair"] is True


def test_paired_bootstrap_is_replayable_and_does_not_use_global_rng() -> None:
    left = {"a": 0.3, "b": 0.2, "c": 0.1}
    right = {"a": 0.1, "b": 0.1, "c": 0.0}
    first = score._paired_summary(left, right, label="synthetic")
    second = score._paired_summary(left, right, label="synthetic")
    assert first["paired_bootstrap_ci"] == second["paired_bootstrap_ci"]
    assert first["paired_bootstrap_ci"]["method"].startswith("sha256_domain")


def test_sealed_point_mode_cannot_be_forged_as_the_posterior_swa() -> None:
    identity = _identity()
    backend = _Backend(identity)
    flags = score.ScoreFlags(within_opened=True, external_opened=True)
    input_payload = backend.authority.payload(identity=identity)
    cell = score.ScoreCell(score.WITHIN, score.SEALED_POINT_MODE, 30, "sealed_point_system", False)
    value = backend.score_cell(cell=cell, input_payload=input_payload, flags=flags).payload(
        input_payload=input_payload, identity=identity,
    )
    value["model_swa_sha256"] = identity.full_training.swa_sha256
    with pytest.raises(score.ScoreError, match="forward-only/model-state/carrier semantics"):
        score.validate_mode_payload(value, input_payload=input_payload, identity=identity)


def test_capability_is_identity_bound_and_public_execution_fails_before_roots() -> None:
    identity = _identity()
    other = _identity()
    capability = score._issue_execution_capability(identity)
    # A separate but byte-equivalent identity is permitted; mutate a binding
    # to show the opaque capability is not merely a flag-shaped token.
    changed = score.ScoreIdentity(
        full_training=score.FullTrainingEvidence(
            terminal_sha256=_hash("other-terminal"), swa_sha256=identity.full_training.swa_sha256,
            swa_state_sha256=identity.full_training.swa_state_sha256,
            source_authority_sha256=identity.full_training.source_authority_sha256,
            checkpoint_sha256s=identity.full_training.checkpoint_sha256s,
            terminal_status=identity.full_training.terminal_status,
            full_closure_sha256=identity.full_training.full_closure_sha256,
        ),
        sealed_cell_d=identity.sealed_cell_d, closure=identity.closure,
        within_roster=other.within_roster, external_roster=other.external_roster,
    )
    with pytest.raises(score.ScoreError, match="identity drift"):
        score._require_execution_capability(capability, identity=changed)
    with pytest.raises(score.ScoreError, match="dry"):
        score.execute_authorized()


def test_cli_dry_is_no_torch_and_two_flags_fail_before_physical_work() -> None:
    probe = (
        "import importlib.util, sys; p=sys.argv[1]; s=importlib.util.spec_from_file_location('p',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.main([]); print('TORCH='+str('torch' in sys.modules))"
    )
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(CLI)], cwd=ROOT, env=env,
        check=True, capture_output=True, text=True,
    )
    assert '"status": "DRY_ONLY__NO_FULL_TERMINAL_OR_SWA_READ__NO_TORCH_NO_DATA_NO_CUDA_NO_WRITE"' in completed.stdout
    assert "TORCH=False" in completed.stdout
    failure = subprocess.run(
        [sys.executable, str(CLI), "--execute", "--i-have-root-reviewed-posterior-score-authorization"],
        cwd=ROOT, env=env, check=False, capture_output=True, text=True,
    )
    assert failure.returncode != 0
    assert "dry until a separate physical-route review" in failure.stderr


def test_imported_full_mirror_requires_exact_immutable_topology_graph_and_copy_provenance(tmp_path: Path) -> None:
    physical, mirror, provenance, digests, fields = _synthetic_completed_full_mirror(tmp_path)
    loaded = physical.load_completed_full_mirror(
        mirror,
        provenance=provenance,
        expected_terminal_sha256=digests["terminal.json"],
        expected_swa_sha256=digests["swa_final4.pt"],
        expected_swa_state_sha256=fields["swa_state"],
        expected_source_authority_sha256=digests["source_authority.json"],
        expected_checkpoint_sha256s={str(epoch): digests[f"checkpoint-{epoch:02d}.pt"] for epoch in (44, 45, 46, 47)},
        expected_full_closure_sha256=fields["closure"]["closure_sha256"],
    )
    assert loaded.artifact_sha256s == digests
    assert loaded.provenance.payload()["source_stage_identity_rebuilt_locally"] is False
    assert loaded.terminal["artifact_sha256s"]["swa_final4.pt"] == digests["swa_final4.pt"]

    # A copied local result may be valid only because every byte is committed
    # by the remote receipt map; changing a checkpoint while leaving the map
    # and its sidecar provenance behind is rejected before tensor loading.
    changed = mirror / "checkpoint-44.pt"
    os.chmod(changed, 0o644)
    changed.write_bytes(b"synthetic-checkpoint-44-tampered")
    os.chmod(changed, 0o444)
    with pytest.raises(physical.PhysicalScoreError, match="SHA drift"):
        physical.load_completed_full_mirror(
            mirror,
            provenance=provenance,
            expected_terminal_sha256=digests["terminal.json"],
            expected_swa_sha256=digests["swa_final4.pt"],
            expected_swa_state_sha256=fields["swa_state"],
            expected_source_authority_sha256=digests["source_authority.json"],
            expected_checkpoint_sha256s={str(epoch): digests[f"checkpoint-{epoch:02d}.pt"] for epoch in (44, 45, 46, 47)},
            expected_full_closure_sha256=fields["closure"]["closure_sha256"],
        )


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "epoch-07.json",
            lambda payload: payload.__setitem__(
                "cumulative_optimizer_steps", payload["cumulative_optimizer_steps"] + 1,
            ),
            "epoch 7 frozen binding",
        ),
        (
            "epoch-13.json",
            lambda payload: payload["epoch_boundary_proof"]["critical_gradients"].__setitem__("ffn", False),
            "epoch 13 critical-gradient",
        ),
        (
            "epoch-23.json",
            lambda payload: payload["posterior_cache"].__setitem__("batch_loop_inverse_calls", 1),
            "epoch 23 posterior-cache/inverse",
        ),
        (
            "epoch-31.json",
            lambda payload: payload["progress"].__setitem__("total_optimizer_steps", 1_628_399),
            "epoch 31 progress/boundary",
        ),
    ],
)
def test_imported_full_graph_rejects_semantic_epoch_forgery_after_all_outer_hashes_are_rewired(
    tmp_path: Path,
    name: str,
    mutate: object,
    message: str,
) -> None:
    """A rehashed provenance map cannot turn a bad 48-epoch proof into evidence."""
    physical, mirror, provenance, _digests, fields = _synthetic_completed_full_mirror(tmp_path)
    forged = _rewire_synthetic_full_json(
        physical, mirror, provenance, name=name, mutate=mutate,
    )
    with pytest.raises(physical.PhysicalScoreError, match=message):
        _load_synthetic_full(physical, mirror, forged, fields)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.__setitem__("final_identity", {"forged": "identity"}),
            "terminal graph/closure/spec",
        ),
        (
            lambda payload: payload["spec"].__setitem__("total_steps", 1_628_399),
            "terminal graph/closure/spec",
        ),
    ],
)
def test_imported_full_graph_rejects_rehashed_terminal_final_identity_or_spec_drift(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    physical, mirror, provenance, _digests, fields = _synthetic_completed_full_mirror(tmp_path)
    forged = _rewire_synthetic_full_json(
        physical, mirror, provenance, name="terminal.json", mutate=mutate,
    )
    with pytest.raises(physical.PhysicalScoreError, match=message):
        _load_synthetic_full(physical, mirror, forged, fields)


def test_imported_full_graph_rejects_rehashed_throughput100_boundary_drift(tmp_path: Path) -> None:
    physical, mirror, provenance, _digests, fields = _synthetic_completed_full_mirror(tmp_path)
    forged = _rewire_synthetic_full_json(
        physical,
        mirror,
        provenance,
        name="throughput100.json",
        mutate=lambda payload: payload.__setitem__("steps", 99),
    )
    with pytest.raises(physical.PhysicalScoreError, match="throughput100 frozen semantics"):
        _load_synthetic_full(physical, mirror, forged, fields)


def test_posterior_normalizer_is_derived_from_completed_full_source_authority(tmp_path: Path) -> None:
    physical, _mirror, provenance, digests, fields = _synthetic_completed_full_mirror(tmp_path)
    derived = physical.derive_posterior_normalizer_from_full_mirror(
        tmp_path, provenance=provenance,
        terminal_sha256=digests["terminal.json"], swa_sha256=digests["swa_final4.pt"],
        swa_state_sha256=fields["swa_state"], source_authority_sha256=digests["source_authority.json"],
        checkpoint_sha256s={str(epoch): digests[f"checkpoint-{epoch:02d}.pt"] for epoch in (44, 45, 46, 47)},
        full_closure_sha256=fields["closure"]["closure_sha256"],
    )
    assert derived == fields["normalizer"]
    assert derived["body_sha256"] == score._digest(score._json({
        key: value for key, value in derived.items() if key != "body_sha256"
    }))


def test_imported_mirror_provenance_rejects_remote_local_map_substitution() -> None:
    physical = _physical()
    names = physical.full_training_artifact_names()
    map_one = {name: _hash("remote:" + name) for name in names}
    map_two = dict(map_one)
    map_two["swa_final4.pt"] = _hash("substituted")
    provenance = physical.ImportedFullMirrorProvenance(
        local_mirror_relative="mirror",
        remote_result_root="remote:/result",
        remote_root_descriptor_identity=(1, 2),
        remote_terminal_sha256=map_one["terminal.json"],
        remote_artifact_sha256s=map_one,
        imported_artifact_sha256s=map_two,
        remote_full_closure_sha256=_hash("closure"),
    )
    with pytest.raises(physical.PhysicalScoreError, match="differ"):
        provenance.payload()


def test_held_input_descriptor_snapshot_rejects_path_or_body_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    physical = _physical()
    root = tmp_path / "source"
    nested = root / "sessions"
    nested.mkdir(parents=True)
    body = b"synthetic-nwb-bytes"
    asset = nested / "within-00_behavior+ecephys.nwb"
    asset.write_bytes(body)
    monkeypatch.setenv("SUBC_DATA_ROOT", str(root))
    held_root = physical.HeldDataRoot.from_environment("SUBC_DATA_ROOT")
    held = held_root.open_asset(
        relative="sessions/within-00_behavior+ecephys.nwb", expected_bytes=len(body),
        expected_sha256=hashlib.sha256(body).hexdigest(), surface="within", session="within-00",
    )
    snapshot = held.private_snapshot()
    try:
        assert snapshot.path.read_bytes() == body
        os.chmod(asset, 0o644)
        asset.write_bytes(b"replaced-body")
        os.chmod(asset, 0o444)
        with pytest.raises(physical.PhysicalScoreError, match="descriptor drift|named inode drift"):
            held.reverify()
    finally:
        snapshot.close()
        held.close()
        held_root.close()


def _physical_identity() -> object:
    physical = _physical()
    ordinary = _identity()
    return score.ScoreIdentity(
        full_training=ordinary.full_training,
        sealed_cell_d=score.SealedCellDEvidence(
            terminal_sha256=physical.SEALED_CELL_D_TERMINAL_SHA256,
            swa_sha256=physical.SEALED_CELL_D_SWA_SHA256,
            m30_ols_point_last_bin_table_sha256=_hash("sealed-governing-table"),
            sealed_point_replay_authority_sha256=_hash("sealed-point-authority"),
        ),
        closure=ordinary.closure,
        within_roster=ordinary.within_roster,
        external_roster=ordinary.external_roster,
    )


def _physical_preflight_args(identity: object) -> dict[str, object]:
    physical = _physical()
    names = physical.full_training_artifact_names()
    remote = {name: _hash("remote-preflight:" + name) for name in names}
    remote["terminal.json"] = identity.full_training.terminal_sha256
    provenance = physical.ImportedFullMirrorProvenance(
        local_mirror_relative=score.FULL_IMPORT_MIRROR_ROOT_RELATIVE,
        remote_result_root="remote:/posterior/full",
        remote_root_descriptor_identity=(10, 11),
        remote_terminal_sha256=identity.full_training.terminal_sha256,
        remote_artifact_sha256s=remote,
        imported_artifact_sha256s=remote,
        remote_full_closure_sha256=identity.full_training.full_closure_sha256,
    ).payload()
    within_authority = {
        "kind": "c1_paired_view_manifest_val6",
        "manifest_relative": "sua_exploration/results/t4_paired_view_c1_fresh_prelaunch_v1/c1_train_val_33_manifest.json",
        "manifest_sha256": "bb3440b688b6d16dabbf91db3ce43e91711f1e9e389de4b241e80a827fbcab7d",
        "selected_split": "val",
    }
    external_authority = {
        "kind": "dandi688_fixed_v2_uuid_ledger_join",
        "ledger_relative": "sua_exploration/results/dandi_000688_subm_co_schema_preflight_v2/receipt.json",
        "ledger_sha256": "1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283",
        "scope_relative": "sua_exploration/manifests/dandi_000688_v0.250122.1735_subm_co_scope_freeze_v2.json",
        "scope_sha256": "68503c7b2985182f821a0c896be68bd4a2f957304f2487fa3a9947b740689c55",
        "eligible_session_ids_semantics": "asset_uuid_then_exact_disposition_verified_download_scope_join",
    }
    def rows(surface: str, roster: tuple[str, ...], authority: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "schema": score.PHYSICAL_INPUT_ASSET_SCHEMA,
                "surface": surface,
                "asset_id": f"{surface}:{session}",
                "session": session,
                "frozen_path": f"sessions/{session}_behavior+ecephys.nwb",
                "bytes": 17 + index,
                "sha256": _hash(f"{surface}:{session}:asset"),
                "authority": dict(authority),
            }
            for index, session in enumerate(roster)
        ]
    posterior_payload = {"body_sha256": _hash("posterior-normalizer-body")}
    normalizers = {
        "ordinary_point": {
            "semantic_sha256": score.SEALED_CELL_D_OLS_T4_NORMALIZER_SEMANTIC_SHA256,
            "sealed_cell_d_point_replay_authority_sha256": (
                identity.sealed_cell_d.sealed_point_replay_authority_sha256
            ),
            "mean": list(score.SEALED_CELL_D_OLS_T4_MEAN_FLOAT32),
            "std": list(score.SEALED_CELL_D_OLS_T4_STD_FLOAT32),
            "raw_before_normalization": True,
            "system_comparison_role": (
                "sealed_cell_d_training_time_ols_normalizer_reused_for_m30_m10_m4_replay"
            ),
            "not_point_budget_mix_trained_control": True,
        },
        "posterior_distribution": {
            "body_sha256": posterior_payload["body_sha256"],
            "source_authority_sha256": identity.full_training.source_authority_sha256,
            "payload": posterior_payload,
            "raw_before_normalization": True,
        },
        "behavior": {
            "semantic_sha256": score.SEALED_BEHAVIOR_NORMALIZER_SEMANTIC_SHA256,
            "source_authority_sha256": identity.full_training.source_authority_sha256,
            "mean": list(score.SEALED_BEHAVIOR_MEAN_FLOAT32),
            "std": list(score.SEALED_BEHAVIOR_STD_FLOAT32),
        },
        "no_target_refit": True,
    }
    return {
        "full_import_provenance": provenance,
        "sealed_cell_d_artifacts": {
            "terminal_relative": physical.SEALED_CELL_D_TERMINAL_RELATIVE,
            "terminal_sha256": identity.sealed_cell_d.terminal_sha256,
            "swa_relative": physical.SEALED_CELL_D_SWA_RELATIVE,
            "swa_sha256": identity.sealed_cell_d.swa_sha256,
            "baseline_relative": physical.SEALED_CELL_D_BASELINE_RELATIVE,
            "baseline_receipt_sha256": physical.SEALED_CELL_D_BASELINE_SHA256,
            "m30_ols_point_last_bin_table_sha256": identity.sealed_cell_d.m30_ols_point_last_bin_table_sha256,
            "sealed_point_replay_authority_sha256": identity.sealed_cell_d.sealed_point_replay_authority_sha256,
        },
        "input_assets": {
            score.WITHIN: rows(score.WITHIN, identity.within_roster, within_authority),
            score.EXTERNAL: rows(score.EXTERNAL, identity.external_roster, external_authority),
        },
        "normalizers": normalizers,
        "device_contract": {
            "cuda_visible_devices": "0", "cuda_device_order": "PCI_BUS_ID", "logical_device": "cuda:0",
            "uuid": "GPU-synthetic", "bdf": "00000000:01:00.0", "name": "Synthetic GPU",
            "nvidia_smi_memory_total_mib": 1, "torch_total_memory_bytes": 1024,
            "torch_version": "synthetic", "torch_cuda_version": "synthetic", "cudnn_version": 1,
        },
    }


def _synthetic_fixed_input_authorities(tmp_path: Path) -> tuple[object, object, Path, dict[str, object]]:
    """Make descriptor-readable metadata only; never make an NWB file."""
    physical = _physical()
    root = tmp_path / "fixed-authorities"
    root.mkdir()
    within_rows = (
        ("within-a", "sub-C/within-a_behavior+ecephys.nwb", 101, _hash("within-a")),
        ("within-b", "sub-C/within-b_behavior+ecephys.nwb", 102, _hash("within-b")),
    )
    formal = ("formal-a",)
    manifest_relative = "authorities/paired.json"
    ledger_relative = "authorities/ledger.json"
    scope_relative = "authorities/scope.json"
    manifest = {
        "schema_version": 1, "task": "CO", "file_count": 33,
        "split_counts": [27, 2, 1], "max_units_exclusive": 100,
        "source_manifest_sha256": _hash("strict-manifest"),
        "formal_test_file_hashes": [], "formal_test_file_paths": [], "formal_test_paths_resolved": False,
        "session_splits": {"train": [f"train-{i:02d}" for i in range(27)],
                           "val": [row[0] for row in within_rows], "test": list(formal)},
        "file_inventory": {
            "train": [{"session": f"train-{i:02d}"} for i in range(27)],
            "val": [
                {"session": session, "filename": Path(path).name, "bytes": count, "sha256": digest}
                for session, path, count, digest in within_rows
            ],
        },
    }
    external = (("uuid-a", "external-a"), ("uuid-b", "external-b"))
    ledger = {
        "eligible_session_ids": [item[0] for item in external], "eligible_session_count": 2,
        "asset_disposition_ledger": [
            {"asset_id": asset_id, "eligible": True, "disposition": "ELIGIBLE", "session_id": session,
             "frozen_path": f"sub-M/{session}_behavior+ecephys.nwb"}
            for asset_id, session in external
        ],
        "verified_downloads": [
            {"asset_id": asset_id, "bytes": 200 + index, "sha256": _hash(f"{asset_id}:body"),
             "size_and_sha256_verified_before_nwb_open": True}
            for index, (asset_id, _session) in enumerate(external)
        ],
    }
    scope = {
        "selected_assets": [
            {"asset_id": asset_id, "session_id": session, "path": f"sub-M/{session}_behavior+ecephys.nwb",
             "size": 200 + index, "sha256": _hash(f"{asset_id}:body")}
            for index, (asset_id, session) in enumerate(external)
        ],
    }
    def write(relative: str, value: dict[str, object], mode: int) -> str:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        body = score._json(value)
        path.write_bytes(body)
        os.chmod(path, mode)
        return score._digest(body)
    manifest_sha = write(manifest_relative, manifest, 0o600)
    ledger_sha = write(ledger_relative, ledger, 0o444)
    scope_sha = write(scope_relative, scope, 0o444)
    spec = physical.FixedInputAuthoritySpec(
        within_manifest_relative=manifest_relative, within_manifest_sha256=manifest_sha,
        within_manifest_mode=0o600, strict_manifest_sha256=_hash("strict-manifest"), within_rows=within_rows,
        external_ledger_relative=ledger_relative, external_ledger_sha256=ledger_sha,
        external_scope_relative=scope_relative, external_scope_sha256=scope_sha,
        external_authority_mode=0o444, formal_test_sessions=formal, external_count=2,
    )
    rows = physical.derive_fixed_input_assets(
        root, within_roster=("within-a", "within-b"), external_roster=("external-a", "external-b"), spec=spec,
    )
    return physical, spec, root, rows


def test_fixed_asset_authorities_are_descriptor_derived_not_caller_rows(tmp_path: Path) -> None:
    physical, spec, root, rows = _synthetic_fixed_input_authorities(tmp_path)
    checked = physical.validate_fixed_input_assets(
        rows, within_roster=("within-a", "within-b"), external_roster=("external-a", "external-b"), spec=spec,
    )
    assert checked == rows
    assert rows["within"][0]["authority"]["manifest_read_once"] is True
    assert rows["external"][0]["authority"]["eligible_session_ids_semantics"].startswith("asset_uuid")

    # Changing the ledger's semantic path while retaining a plausible basename
    # cannot be made authoritative: the exact descriptor body SHA is checked
    # before any durable preflight can be formed.
    ledger_path = root / spec.external_ledger_relative
    os.chmod(ledger_path, 0o644)
    altered = json.loads(ledger_path.read_bytes())
    altered["asset_disposition_ledger"][0]["frozen_path"] = "sub-M/rebound_behavior+ecephys.nwb"
    ledger_path.write_bytes(score._json(altered))
    os.chmod(ledger_path, 0o444)
    with pytest.raises(physical.PhysicalScoreError, match="SHA drift"):
        physical.derive_fixed_input_assets(
            root, within_roster=("within-a", "within-b"), external_roster=("external-a", "external-b"), spec=spec,
        )


def test_physical_preflight_binds_two_source_only_normalizer_families_and_fixed_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _physical_identity()
    kwargs = _physical_preflight_args(identity)
    physical = _physical()
    # Build itself accepts no ``input_assets`` parameter.  A proxy here only
    # keeps this contract-level test no-data; the preceding test exercises the
    # real descriptor semantic derivation against a temporary authority root.
    rows = json.loads(json.dumps(kwargs.pop("input_assets")))
    kwargs.pop("normalizers")
    posterior_payload = {"schema": "posterior_carrier_source_normalizer_v1"}
    posterior_payload["body_sha256"] = score._digest(score._json(posterior_payload))
    monkeypatch.setattr(physical, "derive_fixed_input_assets", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(physical, "validate_fixed_input_assets", lambda value, **_kwargs: value)
    monkeypatch.setattr(
        physical, "derive_posterior_normalizer_from_full_mirror",
        lambda *_args, **_kwargs: dict(posterior_payload),
    )
    monkeypatch.setattr(score, "_physical_module", lambda: physical)
    preflight = score.build_target_free_preflight(root=tmp_path, identity=identity, **kwargs)
    assert preflight["normalizers"]["ordinary_point"]["mean"] != preflight["normalizers"]["posterior_distribution"]["payload"]
    assert preflight["input_assets"][score.WITHIN][0]["authority"]["selected_split"] == "val"
    assert preflight["input_assets"][score.EXTERNAL][0]["authority"]["eligible_session_ids_semantics"].startswith("asset_uuid")
    with pytest.raises(TypeError):
        score.build_target_free_preflight(root=tmp_path, identity=identity, input_assets=rows, **kwargs)
    # The exact literal sealed ordinary moments cannot be replaced merely by
    # keeping the correct semantic/source digest strings.
    normalizers = json.loads(json.dumps(preflight["normalizers"]))
    normalizers["ordinary_point"]["mean"][0] += 0.01
    with pytest.raises(score.ScoreError, match="cross-binding"):
        score._validate_normalizer_bindings(
            normalizers,
            full_source_authority_sha256=identity.full_training.source_authority_sha256,
            sealed_point_replay_authority_sha256=identity.sealed_cell_d.sealed_point_replay_authority_sha256,
            expected_posterior_payload=posterior_payload,
        )
    different_posterior = {"schema": "posterior_carrier_source_normalizer_v1", "different": True}
    different_posterior["body_sha256"] = score._digest(score._json(different_posterior))
    normalizers = json.loads(json.dumps(preflight["normalizers"]))
    normalizers["posterior_distribution"]["payload"] = different_posterior
    normalizers["posterior_distribution"]["body_sha256"] = different_posterior["body_sha256"]
    with pytest.raises(score.ScoreError, match="unique completed-full"):
        score._validate_normalizer_bindings(
            normalizers,
            full_source_authority_sha256=identity.full_training.source_authority_sha256,
            sealed_point_replay_authority_sha256=identity.sealed_cell_d.sealed_point_replay_authority_sha256,
            expected_posterior_payload=posterior_payload,
        )

    # The import is not just a self-consistent remote/local SHA map.  It has
    # one reviewed local mirror namespace, so a future score stage cannot
    # silently point the loader at a copied or historical result root.
    bad_mirror = json.loads(json.dumps(kwargs))
    bad_mirror["full_import_provenance"]["local_mirror_relative"] = "some_other_import"
    rebuilt = dict(bad_mirror["full_import_provenance"])
    rebuilt.pop("provenance_sha256")
    bad_mirror["full_import_provenance"]["provenance_sha256"] = score._digest(score._json(rebuilt))
    with pytest.raises(score.ScoreError, match="local mirror root drift"):
        score.build_target_free_preflight(root=tmp_path, identity=identity, **bad_mirror)


def test_physical_backend_constructor_is_inert_and_does_not_open_or_import_runtime() -> None:
    physical = _physical()
    backend = physical.PhysicalPosteriorMatchedBackend(root=ROOT, preflight={"synthetic": True}, contract=score)
    assert backend._runtime is None
    assert backend._full is None
    assert backend._held_assets == []


def test_real_cpu_cell_d_and_posterior_wrapper_share_the_frozen_b3s_interface() -> None:
    """CPU-only interface proof; no checkpoint, data path, or score surface."""
    script = r'''
import hashlib
import sys
import torch
repo = sys.argv[1]
sys.path.insert(0, repo + "/tfpd_exploration")
from src.posterior_carrier_v1 import core
from src.tfpd_lane.pop_robust import build_population_robustness_model
sha = lambda text: hashlib.sha256(text.encode("ascii")).hexdigest()
torch.manual_seed(9)
base = build_population_robustness_model(seed=42, cell="D").eval()
wrapper = core.CellDPosteriorWrapper(base).eval()
units = 4
prior = core.SourcePrior(0.0, 1.0, 1.0, ("synthetic-source",), sha("synthetic-raw-t4"))
carrier = core.fit_conjugate_posterior(
    counts=torch.tensor([[1,2,1,3],[2,1,3,1],[1,3,2,1],[3,1,1,2]], dtype=torch.int64),
    exposure=torch.ones(4, dtype=torch.float32),
    theta=torch.tensor([0.0,1.57079632679,3.14159265359,4.71238898038], dtype=torch.float32),
    prior=prior,
)
normalizer = core.FrozenSourceT4Normalizer(
    torch.zeros(4, dtype=torch.float32), torch.ones(4, dtype=torch.float32), sha("synthetic-frozen-normalizer")
)
view = core.posterior_mean_view(carrier, normalizer)
neural = torch.randn(1, 50, units)
calibration = torch.randn(1, 30, 100, units)
with torch.no_grad():
    first, first_identity = wrapper(neural, calib_trials_m30=calibration, carrier=view)
    second, second_identity = wrapper(neural, calib_trials_m30=calibration, carrier=view)
assert tuple(first.shape) == (1, 50, 2)
assert tuple(first_identity.shape) == (1, units, 50)
assert torch.isfinite(first).all() and torch.isfinite(first_identity).all()
assert torch.equal(first, second) and torch.equal(first_identity, second_identity)
audit = wrapper.preservation_audit()
assert audit.wrapper_new_parameter_count == 0 and audit.dynamic_dropout is True
print("CPU_FORWARD_OK")
'''
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(ROOT)], cwd=ROOT, env=environment,
        check=True, capture_output=True, text=True,
    )
    assert completed.stdout.strip() == "CPU_FORWARD_OK"


def test_posterior_authority_cpu_and_runtime_view_share_fit_to_view_device_dtype_contract() -> None:
    """The physical path keeps authority CPU-only but consumes a local view.

    This deliberately exercises the same sequence as physical prepare/parse:
    reconstruct the immutable authority payload, create a device-local frozen
    view, fit a posterior, then call ``posterior_mean_view``.  A schema-only
    test would miss the exact device/dtype guard that motivated the split.
    """
    script = r'''
import importlib.util
import pathlib
import sys
import torch

repo = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(repo / "tfpd_exploration"))
physical_path = repo / "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py"
spec = importlib.util.spec_from_file_location("synthetic_physical", physical_path)
physical = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = physical
spec.loader.exec_module(physical)
from src.posterior_carrier_v1 import core

roster = ("source-a",)
raw_by_budget = {
    budget: {"source-a": torch.tensor([[1.0, 2.0, 2.2360679775, 3.0], [3.0, 4.0, 5.0, 6.0]], dtype=torch.float64)}
    for budget in (4, 10, 30)
}
fitted = core.PosteriorSourceT4Normalizer.fit(source_roster=roster, raw_mean_t4_by_budget=raw_by_budget)
authority = physical._posterior_normalizer_from_payload(core, torch, fitted.payload())
runtime_view = physical._posterior_runtime_view_from_authority(
    core, torch, authority, device=torch.device("cpu"),
)
assert authority.mean.device.type == "cpu" and authority.mean.dtype == torch.float64
assert runtime_view.mean.device.type == "cpu" and runtime_view.mean.dtype == torch.float64
assert runtime_view.authority_sha256 == authority.body_sha256
prior = core.SourcePrior(0.0, 1.0, 1.0, roster, "0" * 64)
carrier = core.fit_conjugate_posterior(
    counts=torch.tensor([[1, 2, 1, 3], [2, 1, 3, 1]], dtype=torch.int64),
    exposure=torch.ones(4, dtype=torch.float64),
    theta=torch.tensor([0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469], dtype=torch.float64),
    prior=prior,
)
view = core.posterior_mean_view(carrier, runtime_view)
assert carrier.mean.device == runtime_view.mean.device
assert carrier.mean.dtype == runtime_view.mean.dtype
assert view.normalized_t4.device == carrier.mean.device
assert view.normalized_t4.dtype == carrier.mean.dtype
assert view.normalizer_authority_sha256 == authority.body_sha256
assert torch.isfinite(view.normalized_t4).all()
print("POSTERIOR_AUTHORITY_RUNTIME_VIEW_OK")
'''
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(ROOT)], cwd=ROOT, env=environment,
        check=True, capture_output=True, text=True,
    )
    assert completed.stdout.strip() == "POSTERIOR_AUTHORITY_RUNTIME_VIEW_OK"
