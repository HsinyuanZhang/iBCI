"""No-data tests for the additive PMC-D matched-score contract.

These tests use only synthetic JSON-shaped provenance and score rows.  They do
not open a result, NWB, checkpoint, target/formal asset, or CUDA device.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "tfpd_exploration"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.posterior_marginalized_cell_d_v1 import matched_score as score
from src.posterior_marginalized_cell_d_v1 import matched_score_physical as physical


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _closure() -> score.ImplementationClosure:
    return score.ImplementationClosure({path: _sha(f"closure/{path}") for path in score.scorer_closure_paths()}
                                       | {score.WORKORDER_RELATIVE: score.WORKORDER_SHA256})


def _training() -> score.FinalFourSWAProvenance:
    return score.FinalFourSWAProvenance(
        terminal_sha256=_sha("pmc-terminal"),
        swa_sha256=_sha("pmc-swa"),
        swa_manifest_sha256=_sha("pmc-manifest"),
        source_authority_sha256=_sha("pmc-source-authority"),
        checkpoint_sha256s={str(epoch): _sha(f"pmc-checkpoint-{epoch}") for epoch in score.plan.CHECKPOINT_EPOCHS},
        training_closure_sha256=_sha("pmc-training-closure"),
    )


def _sealed() -> score.SealedCellDEvidence:
    # Center the synthetic rows exactly on the frozen sealed Cell-D baseline
    # means; the small symmetric offsets preserve a non-degenerate per-session
    # roster while keeping the aggregate contract exact.
    within_mean = score.SEALED_CELL_D_WITHIN_MEAN_R2
    external_mean = score.SEALED_CELL_D_EXTERNAL_MEAN_R2
    return score.SealedCellDEvidence(
        baseline_receipt_sha256=score.SEALED_CELL_D_BASELINE_SHA256,
        within=tuple(
            score.BaselineSessionScore(
                f"within-{index:02d}", 2 + (index % 2), within_mean + (index - 2.5) / 1000.0
            )
            for index in range(6)
        ),
        external=tuple(
            score.BaselineSessionScore(
                f"external-{index:02d}", 2 + (index % 2), external_mean + (index - 7) / 1000.0
            )
            for index in range(15)
        ),
    )


def _identity() -> score.ScoreIdentity:
    return score.ScoreIdentity(
        pmc_training=_training(),
        sealed_cell_d=_sealed(),
        closure=_closure(),
        within_roster=tuple(f"within-{index:02d}" for index in range(6)),
        external_roster=tuple(f"external-{index:02d}" for index in range(15)),
    )


def _physical_closure(identity: score.ScoreIdentity, *, salt: str = "physical") -> physical.PhysicalImplementationClosure:
    dependencies = {
        path: _sha(f"{salt}/{path}") for path in physical.PHYSICAL_RUNTIME_DEPENDENCIES
    }
    return physical.PhysicalImplementationClosure(
        base=identity.closure,
        backend_sha256=dependencies[physical.PHYSICAL_BACKEND_RELATIVE],
        runtime_dependencies=dependencies,
    )


def _physical_identity() -> score.ScoreIdentity:
    base = _identity()
    closure = _physical_closure(base)
    return score.ScoreIdentity(
        pmc_training=base.pmc_training,
        sealed_cell_d=base.sealed_cell_d,
        closure=closure,
        within_roster=base.within_roster,
        external_roster=base.external_roster,
    )


def _verified_provenance(identity: score.ScoreIdentity) -> physical.VerifiedProvenance:
    import types

    return physical.VerifiedProvenance(
        pmc=identity.pmc_training,
        terminal={"status": "TERMINAL"},
        swa_manifest={"schema": "synthetic"},
        source_authority={"source_only": True},
        pmc_swa_body=b"synthetic-pmc-swa",
        checkpoint_bodies={str(epoch): f"checkpoint-{epoch}".encode("ascii")
                           for epoch in score.plan.CHECKPOINT_EPOCHS},
        sealed_material=types.SimpleNamespace(swa_body=b"synthetic-sealed-swa"),
        sealed=identity.sealed_cell_d,
        artifact_sha256s={},
    )


def _input_authority(identity: score.ScoreIdentity) -> score.InputAuthority:
    records = []
    for surface, roster in ((score.WITHIN, identity.within_roster), (score.EXTERNAL, identity.external_roster)):
        for index, session in enumerate(roster):
            records.append(score.InputRecord(
                surface=surface,
                session=session,
                n_windows=2 + (index % 2),
                neural_sha256=_sha(f"neural/{surface}/{session}"),
                calibration_m30_sha256=_sha(f"calibration/{surface}/{session}"),
                target_sha256=_sha(f"target/{surface}/{session}"),
                valid_mask_sha256=_sha(f"valid/{surface}/{session}"),
                ordinary_ols_point_carrier_sha256s={str(budget): _sha(f"carrier/{surface}/{session}/M{budget}")
                                                     for budget in score.BUDGETS},
            ))
    return score.InputAuthority(tuple(records))


def _evidence(identity: score.ScoreIdentity, authority: score.InputAuthority) -> list[score.ModeEvidence]:
    input_payload = authority.payload(identity=identity)
    input_sha = hashlib.sha256(json.dumps(input_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record_rows = {(row["surface"], row["session"]): row for row in input_payload["records"]}
    evidence: list[score.ModeEvidence] = []
    for surface, roster in ((score.WITHIN, identity.within_roster), (score.EXTERNAL, identity.external_roster)):
        baseline = {row.session: row.r2 for row in identity.sealed_cell_d.rows(surface)}
        for budget in score.BUDGETS:
            for system in score.SYSTEMS:
                rows = []
                for index, session in enumerate(roster):
                    base = baseline[session]
                    # External M4 is intentionally positive and broad in this
                    # synthetic contract fixture; all rows remain metadata only.
                    offset = 0.04 if system == score.SYSTEM_PMC and budget == 4 else 0.0
                    rows.append(score.SessionScore(
                        session=session,
                        n_windows=record_rows[(surface, session)]["n_windows"],
                        r2=base + offset + index / 10000.0 if system == score.SYSTEM_PMC else base,
                        prediction_sha256=_sha(f"prediction/{surface}/M{budget}/{system}/{session}"),
                        input_record_sha256=hashlib.sha256(
                            json.dumps(record_rows[(surface, session)], sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                    ))
                evidence.append(score.ModeEvidence(
                    cell=score.ScoreCell(surface, budget, system),
                    sessions=tuple(rows),
                    input_authority_sha256=input_sha,
                    model_swa_sha256=(identity.pmc_training.swa_sha256 if system == score.SYSTEM_PMC
                                      else identity.sealed_cell_d.swa_sha256),
                    model_state_before_sha256=_sha(f"state/{surface}/M{budget}/{system}"),
                    model_state_after_sha256=_sha(f"state/{surface}/M{budget}/{system}"),
                ))
    return evidence


def test_dry_cli_is_stdlib_only_and_rejects_execution() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score.py"
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
    rendered = subprocess.run(
        [sys.executable, "-S", str(script), "--dry-run"],
        env=environment, capture_output=True, text=True, check=True,
    )
    payload = json.loads(rendered.stdout)
    assert payload["status"].startswith("DRY_ONLY__")
    assert payload["inference"]["carrier"] == score.INFERENCE_SEMANTICS
    assert payload["surfaces"] == {"within": 6, "external": 15}
    rejected = subprocess.run(
        [sys.executable, "-S", str(script), "--execute"],
        env=environment, capture_output=True, text=True,
    )
    assert rejected.returncode != 0
    assert "scaffold-only" in rejected.stderr


def test_terminal_and_manifest_require_exact_full_final_four_and_point_inference() -> None:
    checkpoints = {str(epoch): _sha(f"checkpoint-{epoch}") for epoch in score.plan.CHECKPOINT_EPOCHS}
    state_components = {str(epoch): _sha(f"state-component-{epoch}") for epoch in score.plan.CHECKPOINT_EPOCHS}
    plan_closure = score.plan.implementation_closure(ROOT)
    identity = {
        "schema": "posterior_marginalized_cell_d_identity_v1",
        "cell": score.CELL,
        "phase": "SOURCE_TRAINING_POSTERIOR_MARGINALIZATION",
        "closure": plan_closure,
        "predecessor_phase1_closure_sha256": score.plan.PHASE1_ACCEPTED_CLOSURE_SHA256,
        "approved_phase_b_v2_authority_sha256": _sha("phase-b-v2-authority"),
        "approved_phase_b_v2_closure_sha256": _sha("phase-b-v2-closure"),
        "pmc_source_binding_sha256": _sha("pmc-source-binding"),
        "inference": score.INFERENCE_SEMANTICS,
        "sealed_cell_d": {
            "canonical_initial_state_artifact_sha256": score.plan.CANONICAL_INITIAL_STATE_SHA256,
            "canonical_initial_state_state_sha256": score.plan.CANONICAL_INITIAL_STATE_STATE_SHA256,
            "ordinary_ols_t4_normalizer_sha256": score.plan.SEALED_OLS_T4_NORMALIZER_SHA256,
            "ordinary_ols_t4_mean_float32": list(score.plan.SEALED_OLS_T4_MEAN_FLOAT32),
            "ordinary_ols_t4_std_float32": list(score.plan.SEALED_OLS_T4_STD_FLOAT32),
            "initialized_trainable_parameters": score.plan.SEALED_CELL_D_INITIALIZED_PARAMETERS,
            "uninitialized_lazy_keys": list(score.plan.SEALED_CELL_D_LAZY_KEYS),
        },
        "sole_intervention": "source_training_side_cached_posterior_sample_only",
        "gpu": dict(score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"]),
        "boundaries": {
            "source_only": True, "target_opened": False, "within_opened": False,
            "external_opened": False, "formal_opened": False,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "posterior_normalizer_used": False, "posterior_credibility_or_attention_bias_used": False,
        },
    }
    swa_proof = {
        "window_epochs": list(score.plan.CHECKPOINT_EPOCHS),
        # Artifact-body SHAs and the SWA proof's state-dict SHAs are distinct
        # authorities; the scorer must bind both topologies without conflating
        # them.
        "component_state_sha256": state_components,
        "fp64_arithmetic": True, "fresh_strict_load": True, "eval_mode": True,
        "eval_no_mask": True, "repeat_bitwise_equal": True, "state_unchanged": True,
        "prediction_shape": [4, 50, 2], "prediction_sha256": _sha("prediction"),
        "state_sha256": _sha("state-before"),
        "state_sha256_before_eval": _sha("state-before"),
        "state_sha256_after_eval": _sha("state-before"),
        "uninitialized_lazy_keys": list(score.plan.SEALED_CELL_D_LAZY_KEYS),
    }
    artifacts = {
        "epoch_sha256": {str(epoch): _sha(f"epoch-{epoch}") for epoch in range(score.plan.EPOCHS)},
        "smoke_sha256": None,
        "checkpoint_sha256": checkpoints,
        "swa_sha256": _sha("swa"),
        "swa_manifest_sha256": _sha("manifest"),
        "swa_proof": swa_proof,
    }
    terminal = {
        "schema": "posterior_marginalized_cell_d_terminal_v1",
        "cell": score.CELL,
        "status": "TERMINAL",
        "spec": score._expected_full_spec(),
        "identity": identity,
        "source_authority_sha256": _sha("source-authority"),
        "optimizer_steps_completed": score.plan.TOTAL_STEPS,
        "epoch_count": score.plan.EPOCHS,
        "launch_final_closure_equal": True,
        "source_only": True,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "artifacts": artifacts,
    }
    manifest = {
        "schema": "posterior_marginalized_cell_d_swa_manifest_v1",
        "cell": score.CELL,
        "swa_sha256": _sha("swa"),
        "proof": swa_proof,
        "binding": {"run_spec": score._expected_full_spec(), "identity": identity},
        "checkpoint_sha256": checkpoints,
    }
    provenance = score.validate_pmc_terminal_and_swa(
        terminal, manifest,
        terminal_sha256=_sha("terminal"), swa_sha256=_sha("swa"), swa_manifest_sha256=_sha("manifest"),
    )
    assert provenance.payload()["checkpoint_epochs"] == [44, 45, 46, 47]
    forged = dict(identity, inference="posterior_mean")
    terminal["identity"] = forged
    with pytest.raises(score.ScoreError, match="inference identity"):
        score.validate_pmc_terminal_and_swa(
            terminal, manifest,
            terminal_sha256=_sha("terminal"), swa_sha256=_sha("swa"), swa_manifest_sha256=_sha("manifest"),
        )


def test_physical_swa_loader_accepts_actual_pmc_and_sealed_producer_shapes_and_binds_state_sha() -> None:
    state_sha = _sha("actual-swa-state")
    pmc_proof = {
        "window_epochs": [44, 45, 46, 47],
        "component_state_sha256": {str(epoch): _sha(f"component-{epoch}") for epoch in (44, 45, 46, 47)},
        "fp64_arithmetic": True, "fresh_strict_load": True, "eval_mode": True,
        "eval_no_mask": True, "repeat_bitwise_equal": True, "state_unchanged": True,
        "prediction_shape": [4, 50, 2], "prediction_sha256": _sha("pmc-prediction"),
        "state_sha256_before_eval": state_sha, "state_sha256_after_eval": state_sha,
        "uninitialized_lazy_keys": list(score.plan.SEALED_CELL_D_LAZY_KEYS),
    }
    pmc_payload = {
        "schema": "cell_d_equal_session_swa_v1", "cell": score.CELL,
        "state_dict": {"weight": object()}, "state_dict_sha256": state_sha,
        "binding": {}, "proof": pmc_proof,
    }
    assert physical.DefaultPhysicalRuntime._extract_state(pmc_payload, label="synthetic PMC") == pmc_payload["state_dict"]

    boundaries = {
        "source_only": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "scientific_score": False, "cache_read_or_write": False,
    }
    sealed_proof = {
        "checkpoint_epochs": [44, 45, 46, 47],
        "checkpoint_state_sha256s": {str(epoch): _sha(f"sealed-component-{epoch}") for epoch in (44, 45, 46, 47)},
        "fresh_strict_load": True, "eval_mode": True, "repeat_bitwise_equal": True,
        "state_unchanged": True, "eval_no_sampling": True,
        "prediction_shape": [32, 50, 2], "prediction_sha256": _sha("sealed-prediction"),
        "state_digest_before_eval": state_sha, "state_digest_after_eval": state_sha,
        "boundaries": boundaries,
    }
    sealed_payload = {
        "schema": "posterior_carrier_full_swa_v1", "state": {"weight": object()},
        "state_sha256": state_sha, "proof": sealed_proof, "binding": {},
    }
    assert physical.DefaultPhysicalRuntime._extract_state(sealed_payload, label="synthetic sealed") == sealed_payload["state"]
    bad = {**sealed_payload, "proof": {**sealed_proof, "prediction_shape": [4, 50, 2]}}
    with pytest.raises(physical.PhysicalScoreError, match="proof"):
        physical.DefaultPhysicalRuntime._extract_state(bad, label="bad sealed shape")


def test_gpu_attestation_selects_exact_uuid_for_both_reviewed_profiles() -> None:
    rows = [
        ", ".join(str(profile[key]) for key in ("uuid", "bdf", "name", "nvidia_smi_memory_total_mib"))
        for profile in score.plan.COMPATIBLE_DEVICE_PROFILES.values()
    ]
    for profile in score.plan.COMPATIBLE_DEVICE_PROFILES.values():
        selected = physical._select_attested_gpu_row(rows, profile=profile)
        assert selected["uuid"] == profile["uuid"]
        assert selected["bdf"] == profile["bdf"]
    wrong = dict(score.plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
    wrong["uuid"] = score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"]["uuid"]
    with pytest.raises(physical.PhysicalScoreError, match="attestation"):
        physical._select_attested_gpu_row(rows, profile=wrong)


def test_bound_module_loader_executes_held_bytes_after_path_substitution(tmp_path, monkeypatch) -> None:
    module_path = tmp_path / "adversarial_module.py"
    replacement_path = tmp_path / "replacement.py"
    module_path.write_text("VALUE = 'held-old'\n", encoding="utf-8")
    replacement_path.write_text("VALUE = 'path-reopened-new'\n", encoding="utf-8")
    original = physical._read_bound_module_source

    def substitute_after_read(path):
        body, digest, identity = original(path)
        os.replace(replacement_path, path)
        return body, digest, identity

    monkeypatch.setattr(physical, "_read_bound_module_source", substitute_after_read)
    loaded = physical._load_exact_module("_pmc_adversarial_substitution", module_path)
    assert loaded.VALUE == "held-old"
    assert loaded.__bound_source_sha256__ == _sha("VALUE = 'held-old'\n")


def test_fixed_bin_score_binds_detached_cpu_targets_and_mask_to_prepared_authority() -> None:
    torch = pytest.importorskip("torch")
    import numpy as np
    import types

    runtime = object.__new__(physical.DefaultPhysicalRuntime)
    runtime._closed = False
    runtime._runtime = {
        "torch": torch,
        "np": np,
        "metric": types.SimpleNamespace(session_r2=lambda predictions, targets: float(predictions[:, 0].mean())),
    }
    private = physical._TorchPreparedSession(
        asset=None, held_root=None, held_asset=None, neural=None, behavior=None,
        calibration=None, starts=None, side_by_budget={},
        last_targets=np.ascontiguousarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        last_valid_mask=np.ascontiguousarray([True, True], dtype=np.bool_),
    )
    target_sha = hashlib.sha256(private.last_targets.tobytes()).hexdigest()
    mask_sha = hashlib.sha256(private.last_valid_mask.astype(np.uint8).tobytes()).hexdigest()
    session = physical.PreparedSession(
        surface=score.WITHIN, session="within-00", n_windows=2,
        neural_sha256=_sha("n"), calibration_m30_sha256=_sha("c"),
        target_sha256=target_sha, valid_mask_sha256=mask_sha,
        ordinary_ols_point_carrier_sha256s={str(budget): _sha(f"carrier-{budget}") for budget in score.BUDGETS},
        input_token_sha256=_sha("token"), opaque=private,
    )
    result = physical.ForwardResult(
        input_token_sha256=session.input_token_sha256, prediction_sha256=_sha("prediction"),
        output_shape=(2, 50, 2), last_bin_predictions=torch.tensor([[5.0, 0.0], [7.0, 0.0]]),
        last_bin_targets=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        last_bin_valid_mask=torch.tensor([True, True]),
    )
    result.validate(session=session)
    assert runtime.score_result(result, session=session) == pytest.approx(6.0)
    tampered = physical.ForwardResult(
        **{**result.__dict__, "last_bin_targets": torch.tensor([[99.0, 2.0], [3.0, 4.0]])},
    )
    with pytest.raises(physical.PhysicalScoreError, match="target/mask"):
        runtime.score_result(tampered, session=session)
    tampered_mask = physical.ForwardResult(
        **{**result.__dict__, "last_bin_valid_mask": torch.tensor([True, False])},
    )
    with pytest.raises(physical.PhysicalScoreError, match="mask"):
        runtime.score_result(tampered_mask, session=session)


def test_build_score_pairs_exact_sessions_and_recomputes_verdict_and_deltas() -> None:
    identity = _identity()
    authority = _input_authority(identity)
    payload = score.build_score_payload(identity=identity, input_authority=authority, evidence=_evidence(identity, authority))
    checked = score.validate_score_payload(payload, identity=identity)
    assert checked["boundaries"]["target_optimizer_steps"] == 0
    assert checked["boundaries"]["posterior_sample_at_inference"] is False
    assert checked["paired_pmc_minus_sealed"]["external"]["4"]["paired_pmc_minus_sealed"]["n_total"] == 15
    assert checked["paired_pmc_minus_sealed"]["external"]["4"]["paired_pmc_minus_sealed"]["n_positive"] == 15
    assert checked["verdict"]["external_governing"]["breadth_passed"] is True


def test_inference_flags_and_paired_delta_are_fail_closed() -> None:
    identity = _identity()
    authority = _input_authority(identity)
    evidence = _evidence(identity, authority)
    evidence[0] = score.ModeEvidence(
        **{**evidence[0].__dict__, "posterior_normalizer_used": True},
    )
    with pytest.raises(score.ScoreError, match="posterior"):
        score.build_score_payload(identity=identity, input_authority=authority, evidence=evidence)

    delta = score.PairedSessionDelta(
        surface=score.WITHIN, budget=30, session="within-00", n_windows=2,
        pmc_r2=0.3, sealed_r2=0.2, input_record_sha256=_sha("input"),
    )
    assert delta.payload()["delta_r2"] == pytest.approx(0.1)
    assert delta.payload()["sign"] == "+"


class _MemoryArtifact:
    """No-files transactional sink used only by the physical seam tests."""

    topology = physical.SCORE_TOPOLOGY

    def __init__(self, events: list[str] | None = None) -> None:
        self.bodies: dict[str, bytes] = {}
        self.events: list[str] = events if events is not None else []

    def publish_group(self, bodies, *, post_publish=None):
        names = tuple(sorted(bodies))
        if any(name in self.bodies for name in names):
            raise physical.PhysicalScoreError("memory artifact collision")
        digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        self.bodies.update(bodies)
        self.events.append("publish:" + ",".join(names))
        if post_publish is not None:
            post_publish(dict(bodies), dict(digests))
        return digests

    def publish_json(self, name, payload):
        return self.publish_group({name: json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()})[name]

    def reload_pair(self, name, expected_sha256=None):
        if name not in self.bodies:
            raise physical.PhysicalScoreError("memory artifact missing")
        body = self.bodies[name]
        actual = hashlib.sha256(body).hexdigest()
        if expected_sha256 is not None and expected_sha256 != actual:
            raise physical.PhysicalScoreError("memory artifact SHA drift")
        return body

    def reload_json(self, name, expected_sha256=None):
        return json.loads(self.reload_pair(name, expected_sha256))

    def has_name(self, name):
        return name in self.bodies


def _physical_assets(identity: score.ScoreIdentity) -> dict[str, tuple[physical.PhysicalAsset, ...]]:
    result: dict[str, tuple[physical.PhysicalAsset, ...]] = {}
    for surface, roster in ((score.WITHIN, identity.within_roster), (score.EXTERNAL, identity.external_roster)):
        result[surface] = tuple(
            physical.PhysicalAsset(
                surface=surface,
                session=session,
                frozen_path=f"{session}_behavior+ecephys.nwb",
                bytes=100 + index,
                sha256=_sha(f"asset/{surface}/{session}"),
                asset_id=f"asset-{surface}-{index:02d}",
                authority={"kind": "synthetic-no-data-test"},
            )
            for index, session in enumerate(roster)
        )
    return result


def _prepared(asset: physical.PhysicalAsset, index: int) -> physical.PreparedSession:
    carriers = {str(budget): _sha(f"carrier/{asset.surface}/{asset.session}/M{budget}") for budget in score.BUDGETS}
    return physical.PreparedSession(
        surface=asset.surface,
        session=asset.session,
        n_windows=2 + (index - 1) % 2,
        neural_sha256=_sha(f"neural/{asset.surface}/{asset.session}"),
        calibration_m30_sha256=_sha(f"calibration/{asset.surface}/{asset.session}"),
        target_sha256=_sha(f"target/{asset.surface}/{asset.session}"),
        valid_mask_sha256=_sha(f"mask/{asset.surface}/{asset.session}"),
        ordinary_ols_point_carrier_sha256s=carriers,
        input_token_sha256=_sha(f"input-token/{asset.surface}/{asset.session}"),
    )


class _MockRuntime:
    def __init__(self, identity: score.ScoreIdentity, events: list[str], *, bad_token: bool = False,
                 posterior_flag: bool = False, mutating_state: bool = False) -> None:
        self.identity = identity
        self.events = events
        self.bad_token = bad_token
        self.posterior_flag = posterior_flag
        self.mutating_state = mutating_state
        self.materialize_calls: list[str] = []
        self.prepared_sessions: list[physical.PreparedSession] = []
        self.forward_calls: list[tuple[str, str, int, int]] = []
        self._state_counter = 0

    def load_models(self, *, profile, pmc_swa_body, sealed_swa_body, provenance):
        self.events.append("runtime.load_models")
        return {score.SYSTEM_PMC: object(), score.SYSTEM_SEALED: object()}

    def materialize_session(self, *, asset, normalizer):
        assert normalizer["semantic_sha256"] == score.plan.SEALED_OLS_T4_NORMALIZER_SHA256
        self.materialize_calls.append(asset.session)
        self.events.append(f"materialize:{asset.surface}:{asset.session}")
        local_index = sum(1 for prior in self.prepared_sessions if prior.surface == asset.surface) + 1
        prepared = _prepared(asset, local_index)
        self.prepared_sessions.append(prepared)
        return prepared

    def forward(self, *, system, budget, model, session):
        self.forward_calls.append((system, session.surface, budget, id(session)))
        self.events.append(f"forward:{system}:{session.surface}:M{budget}")
        baseline = {
            row.session: row.r2
            for row in self.identity.sealed_cell_d.rows(session.surface)
        }[session.session]
        offset = 0.04 if system == score.SYSTEM_PMC and session.surface == score.EXTERNAL and budget == 4 else 0.0
        token = _sha(f"wrong/{session.session}") if self.bad_token and system == score.SYSTEM_SEALED else session.input_token_sha256
        if self.mutating_state:
            self._state_counter += 1
        return physical.ForwardResult(
            input_token_sha256=token,
            prediction_sha256=_sha(f"prediction/{system}/{session.surface}/M{budget}/{session.session}"),
            r2=baseline + offset,
            output_shape=(session.n_windows, 50, 2),
            posterior_normalizer_used=self.posterior_flag,
        )

    def score_result(self, result, *, session=None):
        assert result.r2 is not None
        return result.r2

    def state_digest(self, model):
        return _sha(f"state/{id(model)}/{self._state_counter if self.mutating_state else 0}")

    def repeat_probe(self, *, system, budget, model, session):
        return True

    def close(self):
        self.events.append("runtime.close")


class _MockBackend(physical.PhysicalMatchedScoreBackend):
    def __init__(self, identity, runtime, events):
        import types

        provenance = types.SimpleNamespace(
            pmc=identity.pmc_training,
            sealed=identity.sealed_cell_d,
            pmc_swa_body=b"pmc",
            sealed_material=types.SimpleNamespace(swa_body=b"sealed"),
        )
        super().__init__(
            root=ROOT,
            runtime=runtime,
            provenance_loader=lambda _root: provenance,
        )
        self._mock_provenance = provenance
        self.events = events

    def prepare(self, *, identity):
        self.events.append("backend.prepare")
        self._provenance = self._mock_provenance
        self._identity = identity
        self._models = self.runtime.load_models(
            profile=self.device_profile,
            pmc_swa_body=b"pmc",
            sealed_swa_body=b"sealed",
            provenance=self._mock_provenance,
        )
        return self._mock_provenance


def _run_mock_physical(identity, *, runtime=None, backend=None, artifact=None):
    events: list[str] = []
    if runtime is None:
        runtime = _MockRuntime(identity, events)
    if backend is None:
        backend = _MockBackend(identity, runtime, events)
    if artifact is None:
        artifact = _MemoryArtifact(events)
    assets = _physical_assets(identity)
    capability = physical._issue_test_capability(identity)
    terminal = physical.run_physical_score_lifecycle(
        artifact=artifact,
        identity=identity,
        capability=capability,
        backend=backend,
        assets=assets,
        preflight=physical.build_preflight_payload(identity=identity, assets=assets),
        authorization=physical.build_authorization_payload(identity=identity, capability=capability),
        final_reverify=lambda: identity.closure.payload(),
    )
    return terminal, artifact, runtime, events


def test_physical_lifecycle_attempt_precedes_strict_load_and_materializes_once_with_shared_tokens() -> None:
    identity = _identity()
    terminal, artifact, runtime, events = _run_mock_physical(identity)
    assert terminal["status"] == "SCORE_COMPLETE"
    first_group = "publish:attempt.json,authorization.json,preflight.json"
    assert first_group in artifact.events
    assert artifact.events.index(first_group) < events.index("backend.prepare")
    assert len(runtime.materialize_calls) == 21
    assert len(set(runtime.materialize_calls)) == 21
    assert len(runtime.forward_calls) == 21 * len(score.BUDGETS) * len(score.SYSTEMS)
    prepared_ids = {id(session) for session in runtime.prepared_sessions}
    forward_ids = {item[3] for item in runtime.forward_calls}
    assert forward_ids == prepared_ids
    assert all(sum(item[3] == id(session) for item in runtime.forward_calls) == 6
               for session in runtime.prepared_sessions)
    assert "terminal.json" in artifact.bodies and "failure.json" not in artifact.bodies
    assert events[-1] == "runtime.close"


def test_sealed_m30_live_parity_is_persisted_and_reload_validated() -> None:
    identity = _identity()
    terminal, artifact, _runtime, _events = _run_mock_physical(identity)
    input_sha = terminal["input_authority_sha256"]
    parity = artifact.reload_json("sealed_m30_parity.json")
    checked = physical._validate_sealed_m30_parity(
        parity, identity=identity, input_authority_sha256=input_sha,
    )
    assert checked["all_exact"] is True
    assert [row["session"] for row in checked["rows"][score.WITHIN]] == list(identity.within_roster)
    assert [row["session"] for row in checked["rows"][score.EXTERNAL]] == list(identity.external_roster)
    forged = json.loads(json.dumps(parity))
    forged["rows"][score.EXTERNAL][0]["live_r2"] += 1e-6
    with pytest.raises(physical.PhysicalScoreError, match="live row"):
        physical._validate_sealed_m30_parity(
            forged, identity=identity, input_authority_sha256=input_sha,
        )


def test_deferred_input_derivation_runs_after_attempt_and_failure_is_durable() -> None:
    identity = _identity()
    events: list[str] = []
    runtime = _MockRuntime(identity, events)
    backend = _MockBackend(identity, runtime, events)
    artifact = _MemoryArtifact(events)
    capability = physical._issue_test_capability(identity)

    def derive_and_fail():
        events.append("derive_fixed_input_assets")
        raise RuntimeError("synthetic fixed-input derivation failure")

    with pytest.raises(physical.PhysicalScoreError, match="derivation failed"):
        physical.run_physical_score_lifecycle(
            artifact=artifact,
            identity=identity,
            capability=capability,
            backend=backend,
            assets=None,
            assets_factory=derive_and_fail,
            preflight=physical.build_preflight_payload(identity=identity, assets=None),
            authorization=physical.build_authorization_payload(identity=identity, capability=capability),
        )
    assert artifact.events.index("publish:attempt.json,authorization.json,preflight.json") \
        < events.index("derive_fixed_input_assets")
    failure = artifact.reload_json("failure.json")
    assert failure["stage"] == "input_authority"
    assert failure["terminal_published"] is False
    assert "terminal.json" not in artifact.bodies


def test_physical_lifecycle_rejects_posterior_flags_and_publishes_failure() -> None:
    identity = _identity()
    events: list[str] = []
    runtime = _MockRuntime(identity, events, posterior_flag=True)
    artifact = _MemoryArtifact()
    backend = _MockBackend(identity, runtime, events)
    with pytest.raises(physical.PhysicalScoreError, match="posterior"):
        _run_mock_physical(identity, runtime=runtime, backend=backend, artifact=artifact)
    assert "failure.json" in artifact.bodies
    assert "terminal.json" not in artifact.bodies
    assert artifact.reload_json("failure.json")["terminal_published"] is False


def test_physical_lifecycle_rejects_model_input_token_mismatch_and_never_mints_terminal() -> None:
    identity = _identity()
    events: list[str] = []
    runtime = _MockRuntime(identity, events, bad_token=True)
    artifact = _MemoryArtifact()
    backend = _MockBackend(identity, runtime, events)
    with pytest.raises(physical.PhysicalScoreError, match="input token"):
        _run_mock_physical(identity, runtime=runtime, backend=backend, artifact=artifact)
    assert "failure.json" in artifact.bodies
    assert "terminal.json" not in artifact.bodies


def test_physical_lifecycle_rejects_model_state_mutation_during_forward() -> None:
    identity = _identity()
    events: list[str] = []
    runtime = _MockRuntime(identity, events, mutating_state=True)
    artifact = _MemoryArtifact()
    backend = _MockBackend(identity, runtime, events)
    with pytest.raises(physical.PhysicalScoreError, match="state changed"):
        _run_mock_physical(identity, runtime=runtime, backend=backend, artifact=artifact)
    assert "failure.json" in artifact.bodies
    assert "terminal.json" not in artifact.bodies


def test_physical_backend_rejects_second_materialization_and_missing_terminal_before_runtime() -> None:
    identity = _identity()
    events: list[str] = []
    runtime = _MockRuntime(identity, events)
    backend = _MockBackend(identity, runtime, events)
    assets = _physical_assets(identity)
    capability = physical._issue_test_capability(identity)
    backend.prepare(identity=identity)
    backend.materialize_inputs(identity=identity, assets=assets)
    with pytest.raises(physical.PhysicalScoreError, match="requested twice"):
        backend.materialize_inputs(identity=identity, assets=assets)

    class _MissingBackend(_MockBackend):
        def prepare(self, *, identity):
            self.events.append("backend.prepare.missing")
            raise physical.PhysicalScoreError("PMC terminal/SWA is missing")

    missing_runtime = _MockRuntime(identity, events)
    missing = _MissingBackend(identity, missing_runtime, events)
    artifact = _MemoryArtifact()
    with pytest.raises(physical.PhysicalScoreError, match="terminal/SWA"):
        _run_mock_physical(identity, runtime=missing_runtime, backend=missing, artifact=artifact)
    assert missing_runtime.materialize_calls == []
    assert "failure.json" in artifact.bodies
    assert "terminal.json" not in artifact.bodies


def test_root_reviewed_issuer_reconstructs_identity_and_rejects_stale_authorities() -> None:
    identity = _physical_identity()
    provenance = _verified_provenance(identity)
    events: list[str] = []
    root_review = physical.issue_root_review_capability()
    capability = physical.issue_root_reviewed_execution_capability(
        ROOT / "synthetic-no-data-root",
        root_review_capability=root_review,
        device_profile=score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"],
        provenance_loader=lambda _root: events.append("provenance") or provenance,
        closure_loader=lambda _root: events.append("closure") or identity.closure,
    )
    assert events == ["provenance", "closure"]
    assert isinstance(capability, physical.RootReviewedExecutionCapability)
    assert not isinstance(capability, physical.ExecutionCapability)
    capability.verify(identity)

    stale_terminal = replace(identity, pmc_training=replace(
        identity.pmc_training, terminal_sha256=_sha("stale-terminal"),
    ))
    with pytest.raises(physical.PhysicalScoreError, match="identity binding"):
        capability.verify(stale_terminal)

    original_closure_sha = capability.closure_sha256
    capability.closure_sha256 = _sha("stale-closure")
    with pytest.raises(physical.PhysicalScoreError, match="closure binding"):
        capability.verify(identity)
    capability.closure_sha256 = original_closure_sha
    capability.verify(identity)

    original_profile = dict(capability.device_profile)
    capability.device_profile["cuda_visible_devices"] = "1"
    with pytest.raises(physical.PhysicalScoreError, match="device profile drift"):
        capability.verify(identity)
    capability.device_profile.clear()
    capability.device_profile.update(original_profile)
    capability.verify(identity)

    invalid_profile = dict(score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"])
    invalid_profile["uuid"] = score.plan.COMPATIBLE_DEVICE_PROFILES["gpu1"]["uuid"]
    with pytest.raises(physical.PhysicalScoreError, match="compatible profile"):
        physical.issue_root_reviewed_execution_capability(
            ROOT / "synthetic-no-data-root",
            root_review_capability=root_review,
            device_profile=invalid_profile,
            provenance_loader=lambda _root: provenance,
            closure_loader=lambda _root: identity.closure,
        )
    with pytest.raises(physical.PhysicalScoreError, match="root-review capability"):
        physical.issue_root_reviewed_execution_capability(
            ROOT / "synthetic-no-data-root",
            root_review_capability=object(),
            device_profile=score.plan.COMPATIBLE_DEVICE_PROFILES["gpu0"],
            provenance_loader=lambda _root: provenance,
            closure_loader=lambda _root: identity.closure,
        )


def test_physical_execute_is_closed_before_capability_or_result_root_access() -> None:
    with pytest.raises(physical.PhysicalScoreError, match="capability"):
        physical.execute_authorized(ROOT / "path-that-is-not-read", capability=None)
    with pytest.raises(physical.PhysicalScoreError, match="production execution capability"):
        physical.execute_authorized(
            ROOT / "path-that-is-not-read",
            capability=physical._issue_test_capability(_physical_identity()),
        )
