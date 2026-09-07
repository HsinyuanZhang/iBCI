"""Focused CPU/synthetic gates for the additive PIRG candidate.

These tests never resolve a source/evaluation file, load a sealed checkpoint,
initialise CUDA, or create a canonical result root.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.posterior_identity_residual_gate_v1 import plan
from src.posterior_identity_residual_gate_v1.core import (
    CredibilityGateCache,
    PIRGError,
    PosteriorIdentityResidualGate,
    build_gate_schedule,
    centered_log_credibility,
    residual_gate,
)
from src.posterior_identity_residual_gate_v1 import score
from src.posterior_identity_residual_gate_v1 import train
from src.posterior_identity_residual_gate_v1 import physical
from src.posterior_identity_residual_gate_v1.remote_stage import (
    PIRGStagePlanError,
    build_remote_stage_plan,
    score_overlay_paths,
    validate_remote_stage_plan,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity() -> train.PIRGIdentity:
    return train.PIRGIdentity(train.implementation_closure(ROOT))


def _source_authority(identity: train.PIRGIdentity) -> dict[str, object]:
    roster = [f"source-{index:02d}" for index in range(27)]
    controls = []
    for epoch, row in enumerate(train.build_schedule_for_count(27)):
        for index, budget in enumerate(row):
            controls.append({
                "session": roster[index], "session_index": index, "epoch": epoch,
                "budget": budget, "posterior_sha256": _sha(f"posterior-{epoch}-{index}"),
            })
    return {
        "schema": "posterior_identity_residual_gate_source_authority_v1",
        "identity": identity.payload(), "source_only": True, "roster": roster,
        "budget_schedule": [list(row) for row in train.build_schedule_for_count(27)],
        "posterior_credibility_only": True, "ordinary_ols_t4_held": True,
        "posterior_mean_used": False, "posterior_normalizer_used": False,
        "posterior_sampling_used": False, "posterior_attention_bias_used": False,
        "batch_loop_inverse_calls": 0,
        "gate_cache": {
            "source_sessions": 27, "logical_epochs": 3, "controls_built": 81,
            "optimizer_batch_requests": 0, "optimizer_batch_inverse_calls": 0,
            "posterior_mean_view_builds": 0, "posterior_sampling_view_builds": 0,
            "posterior_normalizer_view_builds": 0,
        },
        "control_digest_rows": controls,
        "base_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
        "model_dropout_rng_seed": plan.SEED,
        "source_adapter_metadata_sha256": _sha("source-adapter-metadata"),
        "remote_torch_authority": {"synthetic": "no-cuda"},
        "tf32_enforcement": {"synthetic": "not-executed"},
        "source_opened": True, "target_opened": False, "within_opened": False,
        "external_opened": False, "formal_opened": False,
    }


def _epoch(epoch: int) -> train.EpochSummary:
    stats = {str(budget): {"min": 0.9, "mean": 1.0, "max": 1.1} for budget in plan.BUDGETS}
    return train.EpochSummary(
        epoch=epoch, budget_schedule=train.build_schedule_for_count(27)[epoch],
        alpha_before=0.0, alpha_after=0.01 * (epoch + 1), loss_first=1.0,
        loss_last=0.9, loss_min=0.8, loss_max=1.1,
        optimizer_steps=train.SOURCE_STEPS_PER_EPOCH, only_alpha_gradient=True,
        alpha_gradient_nonzero=True, finite_model=True, finite_optimizer=True,
        gate_stats_by_budget=stats, throughput_steps_per_second=1.0,
        cache={"synthetic": 1},
    )


class _MockTrainBackend:
    def __init__(self, identity: train.PIRGIdentity, *, fail_stage: str | None = None) -> None:
        self.identity = identity
        self.fail_stage = fail_stage
        self.closed = False

    def prepare(self, *, identity: train.PIRGIdentity) -> None:
        assert identity == self.identity
        if self.fail_stage == "prepare":
            raise RuntimeError("synthetic prepare failure")

    def source_authority(self, *, identity: train.PIRGIdentity):
        if self.fail_stage == "authority":
            raise RuntimeError("synthetic authority failure")
        return _source_authority(identity)

    def run_epoch(self, *, epoch: int, identity: train.PIRGIdentity) -> train.EpochSummary:
        if self.fail_stage == f"epoch-{epoch}":
            raise RuntimeError("synthetic epoch failure")
        return _epoch(epoch)

    def final_artifact(self, *, identity: train.PIRGIdentity):
        body = b"synthetic-final-alpha"
        return body, {"artifact_sha256": hashlib.sha256(body).hexdigest(), "cpu_reload_equal": True}

    def final_reverify(self, *, identity: train.PIRGIdentity) -> train.ImplementationClosure:
        return identity.closure

    def progress(self):
        return {"source_opened": self.fail_stage != "prepare", "cuda_initialized": False}

    def close(self) -> None:
        self.closed = True


def test_rotation_exposes_each_source_session_to_each_budget_once() -> None:
    roster = [f"s{index}" for index in range(27)]
    schedule = build_gate_schedule(roster=roster)
    assert len(schedule) == 3
    assert all(len(row) == 27 for row in schedule)
    for index in range(27):
        assert set(row[index] for row in schedule) == set(plan.BUDGETS)


def test_centered_log_credibility_uniform_cancels_exactly_and_gate_is_bounded() -> None:
    alpha = torch.tensor(1.2, requires_grad=True)
    uniform = torch.full((7,), 0.4)
    gate, z = residual_gate(alpha=alpha, credibility=uniform)
    assert torch.equal(z, torch.zeros_like(z))
    assert torch.equal(gate, torch.ones_like(gate))
    nonuniform = torch.tensor([0.1, 0.2, 0.3, 0.7, 0.9])
    gate, z = residual_gate(alpha=alpha, credibility=nonuniform)
    assert torch.all(gate >= 0.5) and torch.all(gate <= 1.5)
    assert not torch.equal(z, torch.zeros_like(z))


def test_alpha_zero_is_exactly_neutral_but_has_nonzero_gradient_on_nonuniform_fixture() -> None:
    alpha = torch.zeros((), requires_grad=True)
    credibility = torch.tensor([0.1, 0.2, 0.3, 0.7, 0.9])
    gate, _z = residual_gate(alpha=alpha, credibility=credibility)
    assert torch.equal(gate, torch.ones_like(gate))
    (gate * torch.arange(1.0, 6.0)).sum().backward()
    assert alpha.grad is not None and alpha.grad.item() != 0.0


def test_credibility_range_and_shape_fail_closed() -> None:
    with pytest.raises(PIRGError):
        centered_log_credibility(torch.tensor([0.0, 0.2]))
    with pytest.raises(PIRGError):
        centered_log_credibility(torch.ones(1, 2, 3))


class _Carrier:
    def __init__(self, credibility: torch.Tensor, label: str) -> None:
        self.credibility = credibility
        self.label = label

    def digest(self) -> str:
        return _sha(self.label)


class _Bank:
    def __init__(self, roster: list[str]) -> None:
        self.calls: list[tuple[str, int]] = []
        self.roster = roster

    def posterior_for(self, *, session: str, epoch: int) -> _Carrier:
        self.calls.append((session, epoch))
        return _Carrier(torch.linspace(0.1, 0.9, 5), f"{session}/{epoch}")

    def training_view(self, **_kwargs):  # pragma: no cover - must never be used
        raise AssertionError("PIRG must not request posterior training view")


def test_gate_cache_prewarm_is_session_epoch_static_and_does_not_consume_global_torch_rng() -> None:
    roster = [f"s{index}" for index in range(27)]
    bank = _Bank(roster)
    cache = CredibilityGateCache(roster=roster, posterior_bank=bank)
    before = torch.get_rng_state().clone()
    for epoch in range(3):
        cache.prewarm_epoch(epoch=epoch, device="cpu")
    assert torch.equal(before, torch.get_rng_state())
    assert len(bank.calls) == 81
    assert cache.observer().controls_built == 81
    control = cache.control_for_optimizer_batch(session=roster[0], epoch=0, device="cpu")
    assert control.budget == 4
    assert len(bank.calls) == 81
    assert cache.observer().optimizer_batch_inverse_calls == 0
    assert cache.observer().posterior_mean_view_builds == 0
    assert cache.observer().posterior_sampling_view_builds == 0
    assert cache.observer().posterior_normalizer_view_builds == 0


def test_gate_cache_refuses_an_unprepared_optimizer_batch() -> None:
    cache = CredibilityGateCache(roster=["s"], posterior_bank=_Bank(["s"]))
    with pytest.raises(PIRGError):
        cache.control_for_optimizer_batch(session="s", epoch=0, device="cpu")


def test_real_cell_d_wrapper_preserves_graph_shapes_dropout_and_bitwise_zero_gate_forward() -> None:
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    torch.manual_seed(7)
    base = build_population_robustness_model(seed=42, cell="D")
    wrapper = PosteriorIdentityResidualGate(base)
    preservation = wrapper.preservation()
    assert preservation.base_live_parameters == 3_510_842
    assert preservation.wrapper_live_parameters == 3_510_843
    assert preservation.new_trainable_parameter_names == ("alpha",)
    assert preservation.base_lazy_parameter_names == (
        "decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight",
    )
    assert preservation.dynamic_dropout and preservation.dropout_low == 0.0 and preservation.dropout_high == 1.0
    neural = torch.randn(1, 50, 5)
    calibration = torch.randn(1, 30, 100, 5)
    ordinary_side = torch.randn(1, 5, 4)
    credibility = torch.linspace(0.1, 0.9, 5)
    received: dict[str, torch.Tensor] = {}
    original = base.decode_with_identity

    def observed_decode(activity, identity, *args, **kwargs):
        received["activity"] = activity
        return original(activity, identity, *args, **kwargs)

    base.decode_with_identity = observed_decode  # type: ignore[method-assign]
    base.eval()
    wrapper.eval()
    with torch.no_grad():
        expected = original(neural, base.compute_identity(calibration, side_features=ordinary_side))
        actual, _identity, gate, _z = wrapper(
            neural, calib_trials_m30=calibration, ordinary_ols_side_features=ordinary_side,
            directional_credibility=credibility,
        )
    assert received["activity"] is neural
    assert torch.equal(gate, torch.ones_like(gate))
    assert torch.equal(expected, actual)
    assert tuple(name for name, item in wrapper.named_parameters() if item.requires_grad) == ("alpha",)
    # This is an actual Cell-D decode graph (not merely the algebraic gate).
    # Eval mode removes the inherited dynamic dropout for the fixture, while
    # the signed deterministic objective makes an accidental zero derivative
    # detectable without changing the frozen decoder or OLS carrier.
    train_actual, _identity, _gate, _z = wrapper(
        neural, calib_trials_m30=calibration, ordinary_ols_side_features=ordinary_side,
        directional_credibility=credibility,
    )
    objective = (train_actual * (torch.arange(train_actual.numel(), dtype=train_actual.dtype).reshape_as(train_actual) / 100.0)).sum()
    objective.backward()
    assert wrapper.alpha.grad is not None and wrapper.alpha.grad.item() != 0.0
    assert all(parameter.grad is None for name, parameter in base.named_parameters() if name != "alpha")


def test_real_cell_d_pirg_gate_preserves_unit_permutation_equivariance() -> None:
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    torch.manual_seed(19)
    wrapper = PosteriorIdentityResidualGate(build_population_robustness_model(seed=42, cell="D"))
    with torch.no_grad():
        wrapper.alpha.fill_(0.73)
    wrapper.eval()
    neural = torch.randn(1, 50, 6)
    calibration = torch.randn(1, 30, 100, 6)
    point_side = torch.randn(1, 6, 4)
    credibility = torch.tensor([0.08, 0.13, 0.21, 0.34, 0.55, 0.89])
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    with torch.no_grad():
        original, identity, gate, _z = wrapper(
            neural, calib_trials_m30=calibration, ordinary_ols_side_features=point_side,
            directional_credibility=credibility,
        )
        permuted, permuted_identity, permuted_gate, _permuted_z = wrapper(
            neural[:, :, permutation], calib_trials_m30=calibration[:, :, :, permutation],
            ordinary_ols_side_features=point_side[:, permutation, :], directional_credibility=credibility[permutation],
        )
    torch.testing.assert_close(permuted_identity, identity[:, permutation, :], rtol=0.0, atol=2e-6)
    torch.testing.assert_close(permuted_gate, gate[:, permutation], rtol=0.0, atol=0.0)
    torch.testing.assert_close(permuted, original, rtol=0.0, atol=2e-6)


def test_training_source_authority_rejects_forged_control_row() -> None:
    identity = _identity()
    authority = _source_authority(identity)
    train._validate_source_authority(authority, identity=identity)
    authority["control_digest_rows"][0]["budget"] = 30
    with pytest.raises(train.PIRGTrainError):
        train._validate_source_authority(authority, identity=identity)


def test_training_source_authority_binds_the_cell_d_model_dropout_seed() -> None:
    identity = _identity()
    authority = _source_authority(identity)
    authority["model_dropout_rng_seed"] = 43
    with pytest.raises(train.PIRGTrainError):
        train._validate_source_authority(authority, identity=identity)


def test_mock_training_lifecycle_publishes_all_three_epoch_receipts_and_final_pair(tmp_path: Path) -> None:
    identity = _identity()
    backend = _MockTrainBackend(identity)
    result = train.run_training_lifecycle(
        root=tmp_path, identity=identity, backend=backend,
        execution_capability=train._issue_root_review_capability(identity),
    )
    root = tmp_path / train.RESULT_ROOT_RELATIVE
    assert set(result) == {"attempt_sha256", "source_authority_sha256", "final_alpha_sha256", "terminal_sha256"}
    for name in ("attempt.json", "source_authority.json", "epoch-00.json", "epoch-01.json", "epoch-02.json", "final_alpha.pt", "terminal.json"):
        assert (root / name).exists() and (root / f"{name}.sha256").exists()
        assert stat_mode(root / name) == 0o444
    assert backend.closed


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_mock_training_failure_is_honest_and_has_no_terminal(tmp_path: Path) -> None:
    identity = _identity()
    backend = _MockTrainBackend(identity, fail_stage="epoch-1")
    with pytest.raises(RuntimeError):
        train.run_training_lifecycle(
            root=tmp_path, identity=identity, backend=backend,
            execution_capability=train._issue_root_review_capability(identity),
        )
    root = tmp_path / train.RESULT_ROOT_RELATIVE
    assert (root / "failure.json").exists()
    assert not (root / "terminal.json").exists()
    assert backend.closed


def test_train_pair_writer_rolls_back_a_body_if_the_sidecar_cannot_be_created(tmp_path: Path, monkeypatch) -> None:
    artifact = train._ArtifactRoot(tmp_path / "train-pair")
    artifact.reserve()
    real_open = train.os.open

    def failing_open(path, *args, **kwargs):
        if path == "broken.json.sha256":
            raise OSError("synthetic sidecar failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(train.os, "open", failing_open)
    try:
        with pytest.raises(OSError, match="synthetic sidecar failure"):
            artifact.publish_json("broken.json", {"synthetic": True})
    finally:
        artifact.close()
    assert not (tmp_path / "train-pair" / "broken.json").exists()
    assert not (tmp_path / "train-pair" / "broken.json.sha256").exists()


def _score_identity() -> score.PIRGScoreIdentity:
    # Keep this focused suite no-data/no-result-root: production computes the
    # descriptor-safe score closure only after a reviewed capability, while
    # this synthetic fixture exercises its fixed topology and digest rules.
    hashes = {path: _sha(f"score-closure/{path}") for path in train.SCORE_IMPLEMENTATION_CLOSURE}
    body = {"paths": list(train.SCORE_IMPLEMENTATION_CLOSURE), "sha256_by_path": hashes}
    closure = score.ScoreClosure({**body, "closure_sha256": hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()})
    return score.PIRGScoreIdentity(
        closure=closure, source_terminal_sha256=_sha("source-terminal"),
        final_alpha_sha256=_sha("final-alpha"), source_authority_sha256=_sha("source-authority"),
    )


def test_score_closure_is_fixed_and_separate_from_source_training_closure() -> None:
    score_identity = _score_identity()
    assert score_identity.closure.payload()["paths"] == list(train.SCORE_IMPLEMENTATION_CLOSURE)
    assert "tfpd_exploration/src/posterior_identity_residual_gate_v1/score_physical.py" in train.SCORE_IMPLEMENTATION_CLOSURE
    # A self-consistent source-only closure cannot license an evaluator which
    # composes the V3 physical parser/device seam.
    with pytest.raises(score.PIRGScoreError):
        score.ScoreClosure(_identity().closure.payload()).payload()


def _input_authority() -> score.InputAuthority:
    return score.InputAuthority({
        surface: [
            {
                "session": session, "n_windows": index + 10,
                "input_sha256": _sha(f"{surface}/{session}/input"),
                "last_bin_target_sha256": _sha(f"{surface}/{session}/target"),
                "last_bin_mask_sha256": _sha(f"{surface}/{session}/mask"),
                "point_side_sha256s": {
                    str(budget): _sha(f"{surface}/{session}/point/m{budget}")
                    for budget in score.BUDGETS
                },
                "directional_credibility_sha256s": {
                    str(budget): _sha(f"{surface}/{session}/credibility/m{budget}")
                    for budget in score.BUDGETS
                },
                "prefix_row_ids_sha256s": {
                    str(budget): _sha(f"{surface}/{session}/prefix/m{budget}")
                    for budget in score.BUDGETS
                },
            }
            for index, session in enumerate(score.FIXED_SESSIONS[surface])
        ]
        for surface in score.SURFACES
    })


def _cell_evidence(identity: score.PIRGScoreIdentity, cell: score.ScoreCell, *, delta: float) -> score.CellEvidence:
    baseline = 0.2
    value = baseline + (delta if cell.mode == score.PIRG_MODE else 0.0)
    input_rows = {
        row["session"]: row
        for row in _input_authority().payload(identity=identity)["surfaces"][cell.surface]
    }
    rows = tuple(score.SessionScore(
        session=session, n_windows=input_rows[session]["n_windows"], r2=value + index * 0.001,
        prediction_sha256=_sha(f"prediction/{cell.surface}/{cell.mode}/{cell.budget}/{session}"),
        input_sha256=score._digest(score._json(input_rows[session])),
    ) for index, session in enumerate(score.FIXED_SESSIONS[cell.surface]))
    state = _sha(f"state/{cell.mode}")
    return score.CellEvidence(
        cell=cell, model_state_before_sha256=state, model_state_after_sha256=state,
        model_artifact_sha256=(plan.SEALED_CELL_D_SWA_SHA256 if cell.mode == score.BASELINE_MODE else identity.final_alpha_sha256),
        base_cell_d_swa_sha256=plan.SEALED_CELL_D_SWA_SHA256,
        rows=rows,
    )


def test_score_payload_has_deterministic_safety_and_short_prefix_screen() -> None:
    identity = _score_identity()
    inputs = _input_authority().payload(identity=identity)
    evidence = [
        _cell_evidence(identity, cell, delta=(0.04 if cell.budget == 4 else 0.0)).payload(
            identity=identity, input_authority_sha256=score._digest(score._json(inputs)),
        )
        for cell in score.score_matrix()
    ]
    payload = score.build_score_payload(identity=identity, input_payload=inputs, evidence=evidence)
    assert payload["screen"]["verdict"] == "PROMISING__M4_SCREEN"
    score.validate_score_payload(payload, identity=identity, input_payload=inputs)
    payload["screen"]["verdict"] = "HOLD__DESCRIPTIVE_SCREEN"
    with pytest.raises(score.PIRGScoreError):
        score.validate_score_payload(payload, identity=identity, input_payload=inputs)


def test_score_input_authority_binds_all_three_point_credibility_and_prefix_controls() -> None:
    identity = _score_identity()
    authority = _input_authority()
    payload = authority.payload(identity=identity)
    row = payload["surfaces"][score.WITHIN][0]
    assert payload["within_opened"] is True and payload["external_opened"] is True
    assert set(row["point_side_sha256s"]) == {"30", "10", "4"}
    assert set(row["directional_credibility_sha256s"]) == {"30", "10", "4"}
    assert set(row["prefix_row_ids_sha256s"]) == {"30", "10", "4"}
    row["point_side_sha256s"].pop("10")
    with pytest.raises(score.PIRGScoreError):
        score.validate_input_authority(payload, identity=identity)


def test_score_cell_receipt_binds_pirg_alpha_artifact_and_shared_sealed_base() -> None:
    identity = _score_identity()
    inputs = _input_authority().payload(identity=identity)
    input_sha = score._digest(score._json(inputs))
    cell = score.ScoreCell(score.EXTERNAL, score.PIRG_MODE, 10)
    payload = _cell_evidence(identity, cell, delta=0.01).payload(
        identity=identity, input_authority_sha256=input_sha,
    )
    score.validate_cell_evidence_payload(payload, identity=identity, input_authority_sha256=input_sha)
    payload["base_cell_d_swa_sha256"] = _sha("wrong-base")
    with pytest.raises(score.PIRGScoreError):
        score.validate_cell_evidence_payload(payload, identity=identity, input_authority_sha256=input_sha)


def test_score_payload_rejects_a_cell_row_bound_to_the_wrong_same_input_record() -> None:
    identity = _score_identity()
    inputs = _input_authority().payload(identity=identity)
    input_sha = score._digest(score._json(inputs))
    evidence = [
        _cell_evidence(identity, cell, delta=0.0).payload(identity=identity, input_authority_sha256=input_sha)
        for cell in score.score_matrix()
    ]
    evidence[0]["session_scores"][0]["input_sha256"] = _sha("wrong-per-session-input")
    with pytest.raises(score.PIRGScoreError):
        score.build_score_payload(identity=identity, input_payload=inputs, evidence=evidence)


class _MockScoreBackend:
    def __init__(self, identity: score.PIRGScoreIdentity, *, fail: bool = False) -> None:
        self.identity = identity
        self.fail = fail
        self.closed = False

    def prepare(self, *, identity: score.PIRGScoreIdentity) -> None:
        assert identity == self.identity

    def resolve_inputs(self, *, identity: score.PIRGScoreIdentity) -> score.InputAuthority:
        return _input_authority()

    def score_cell(self, *, cell: score.ScoreCell, input_payload):
        if self.fail and cell.budget == 10:
            raise RuntimeError("synthetic score failure")
        return _cell_evidence(self.identity, cell, delta=0.01)

    def final_reverify(self, *, identity: score.PIRGScoreIdentity) -> score.ScoreClosure:
        return identity.closure

    def progress(self):
        return {"within_opened": True, "external_opened": True, "cuda_initialized": False,
                "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
                "formal_opened": False, "h1_opened": False}

    def close(self) -> None:
        self.closed = True


def test_mock_score_lifecycle_publishes_atomic_score_and_terminal(tmp_path: Path) -> None:
    identity = _score_identity()
    backend = _MockScoreBackend(identity)
    result = score.run_score_lifecycle(
        root=tmp_path, identity=identity, backend=backend,
        execution_capability=score._issue_root_review_capability(identity),
    )
    root = tmp_path / score.RESULT_ROOT_RELATIVE
    assert {"attempt_sha256", "input_authority_sha256", "score.json", "terminal.json"} == set(result)
    assert (root / "score.json").exists() and (root / "terminal.json").exists()
    assert not (root / "failure.json").exists()
    assert backend.closed


def test_score_terminal_rejects_any_target_update_or_missing_evaluation_open_fact() -> None:
    identity = _score_identity()
    progress = {
        "within_opened": True, "external_opened": True, "cuda_initialized": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "formal_opened": False, "h1_opened": False,
    }
    terminal = score._terminal_payload(
        identity=identity, attempt_sha256=_sha("attempt"), input_authority_sha256=_sha("input"),
        score_sha256=_sha("score"), final_closure=identity.closure.payload(), progress=progress,
    )
    score.validate_score_terminal_payload(terminal, identity=identity, score_sha256=_sha("score"))
    terminal["execution_progress"]["target_update_calls"] = 1
    with pytest.raises(score.PIRGScoreError):
        score.validate_score_terminal_payload(terminal, identity=identity, score_sha256=_sha("score"))


def test_mock_score_failure_never_publishes_a_partial_scientific_score(tmp_path: Path) -> None:
    identity = _score_identity()
    backend = _MockScoreBackend(identity, fail=True)
    with pytest.raises(RuntimeError):
        score.run_score_lifecycle(
            root=tmp_path, identity=identity, backend=backend,
            execution_capability=score._issue_root_review_capability(identity),
        )
    root = tmp_path / score.RESULT_ROOT_RELATIVE
    assert (root / "failure.json").exists()
    assert not (root / "score.json").exists()
    assert not (root / "terminal.json").exists()


def test_score_pair_writer_rolls_back_a_body_if_the_sidecar_cannot_be_created(tmp_path: Path, monkeypatch) -> None:
    artifact = score._ArtifactRoot(tmp_path / "score-pair")
    artifact.reserve()
    real_open = score.os.open

    def failing_open(path, *args, **kwargs):
        if path == "broken.json.sha256":
            raise OSError("synthetic sidecar failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(score.os, "open", failing_open)
    try:
        with pytest.raises(OSError, match="synthetic sidecar failure"):
            artifact.publish_json("broken.json", {"synthetic": True})
    finally:
        artifact.close()
    assert not (tmp_path / "score-pair" / "broken.json").exists()
    assert not (tmp_path / "score-pair" / "broken.json.sha256").exists()


def test_physical_backend_construction_is_inert_and_needs_no_source_or_cuda() -> None:
    from src.posterior_identity_residual_gate_v1.physical import PhysicalPIRGBackend

    backend = PhysicalPIRGBackend(root=ROOT, source_data=object())
    assert backend.progress() == {"source_opened": False, "cuda_initialized": False}
    backend.close()


def test_physical_score_backend_construction_is_inert_and_needs_no_evaluation_or_cuda() -> None:
    from src.posterior_identity_residual_gate_v1.score_physical import PhysicalPIRGQuickScoreBackend

    backend = PhysicalPIRGQuickScoreBackend(root=ROOT)
    assert backend.progress() == {
        "within_opened": False, "external_opened": False, "cuda_initialized": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }
    backend.close()


def test_route_local_physical_score_seam_forwards_real_cell_d_at_m4_m10_m30_without_posterior_mean() -> None:
    """Exercise the actual private-score seam, not a mock-only scorer.

    This is CPU-only and intentionally has no V3/data/checkpoint action.  A
    synthetic V3-shaped session exposes exactly the fields used by the seam:
    ordinary point-side carriers and directional credibility.  The posterior
    view object's mean/normalizer fields are absent, so an accidental consumer
    of posterior content fails rather than being silently tolerated.
    """
    from src.posterior_identity_residual_gate_v1.score_physical import PIRGForwardSeam
    from src.tfpd_lane import pop_robust
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    torch.manual_seed(123)
    sealed = build_population_robustness_model(seed=42, cell="D")
    gated_base = build_population_robustness_model(seed=42, cell="D")
    gated_base.load_state_dict(sealed.state_dict(), strict=True)
    for parameter in sealed.parameters():
        if not isinstance(parameter, torch.nn.parameter.UninitializedParameter):
            parameter.requires_grad_(False)
    wrapper = PosteriorIdentityResidualGate(gated_base)
    sealed.eval()
    wrapper.eval()

    class _PosteriorView:
        def __init__(self, credibility: torch.Tensor) -> None:
            self.credibility = credibility

    class _Session:
        def __init__(self) -> None:
            self.point_side = {budget: torch.randn(5, 4) for budget in plan.BUDGETS}
            self.posterior_view = {
                budget: _PosteriorView(torch.tensor([0.11, 0.19, 0.38, 0.64, 0.91]))
                for budget in plan.BUDGETS
            }

    seam = PIRGForwardSeam(sealed_cell_d=sealed, pirg=wrapper, torch=torch, pop_robust=pop_robust)
    session = _Session()
    neural = torch.randn(2, 50, 5)
    calibration = torch.randn(2, 30, 100, 5)
    for budget in plan.BUDGETS:
        baseline, gated, gate = seam.forward_pair(
            session=session, budget=budget, neural=neural, calibration=calibration,
        )
        assert tuple(baseline.shape) == tuple(gated.shape) == (2, 50, 2)
        assert torch.equal(baseline, gated)  # alpha is exactly neutral at construction.
        assert torch.equal(gate, torch.ones_like(gate))


def test_physical_score_binds_the_exact_m4_m10_m30_point_credibility_and_prefix_controls() -> None:
    from src.posterior_identity_residual_gate_v1.score_physical import (
        PhysicalPIRGQuickScoreBackend,
        PhysicalPIRGScoreError,
    )

    class _Core:
        @staticmethod
        def tensor_digest(value: torch.Tensor) -> str:
            return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    class _View:
        def __init__(self, credibility: torch.Tensor) -> None:
            self.credibility = credibility

    class _Record:
        def __init__(self, prefixes) -> None:
            self._prefixes = prefixes

        def payload(self):
            return {"matched_prefix_row_ids_sha256s": self._prefixes}

    point_side = {budget: torch.full((4, 4), float(budget)) for budget in plan.BUDGETS}
    posterior_view = {budget: _View(torch.full((4,), budget / 30.0)) for budget in plan.BUDGETS}
    prefixes = {str(budget): _sha(f"prefix/m{budget}") for budget in plan.BUDGETS}
    session = type("SyntheticSession", (), {
        "point_side": point_side, "posterior_view": posterior_view, "input_record": _Record(prefixes),
    })()
    record = {
        "point_side_sha256s": {str(budget): _Core.tensor_digest(point_side[budget]) for budget in plan.BUDGETS},
        "directional_credibility_sha256s": {
            str(budget): _Core.tensor_digest(posterior_view[budget].credibility) for budget in plan.BUDGETS
        },
        "prefix_row_ids_sha256s": prefixes,
    }
    for budget in plan.BUDGETS:
        PhysicalPIRGQuickScoreBackend._validate_budget_control_binding(
            session=session, record=record, budget=budget, core=_Core,
        )
    record["point_side_sha256s"]["10"] = _sha("m10-control-swap")
    with pytest.raises(PhysicalPIRGScoreError, match="control binding drift"):
        PhysicalPIRGQuickScoreBackend._validate_budget_control_binding(
            session=session, record=record, budget=10, core=_Core,
        )


def test_physical_source_authority_uses_only_prewarmed_credibility_controls() -> None:
    from src.posterior_identity_residual_gate_v1.physical import PhysicalPIRGBackend, _Runtime

    identity = _identity()
    roster = tuple(f"source-{index:02d}" for index in range(27))
    cache = CredibilityGateCache(roster=roster, posterior_bank=_Bank(list(roster)))
    for epoch in range(3):
        cache.prewarm_epoch(epoch=epoch, device="cpu")

    adapter = type("SyntheticAdapter", (), {"roster": roster})()

    class _TF32:
        def payload(self):
            return {"synthetic": "no-cuda"}

        def restore(self, _torch):
            return None

    backend = PhysicalPIRGBackend(root=ROOT, source_data=object())
    backend._runtime = _Runtime(  # type: ignore[attr-defined]
        torch=torch, adapter=adapter, wrapper=None, optimizer=None, loader=None, device=torch.device("cpu"),
        arm_common=None, base_state_sha256=_sha("base"), cache=cache,
        source_authority_metadata_sha256=_sha("source-metadata"), remote_device={"synthetic": "remote"},
        tf32_enforcement=_TF32(), source_opened=True, cuda_initialized=False,
    )
    authority = backend.source_authority(identity=identity)
    train._validate_source_authority(authority, identity=identity)
    assert cache.observer().optimizer_batch_requests == 0
    assert len(authority["control_digest_rows"]) == 81
    backend.close()


def _copy_source_stage_closure(destination_root: Path) -> None:
    for relative in train.IMPLEMENTATION_CLOSURE:
        source = ROOT / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def test_remote_stage_plan_is_code_only_and_cannot_stage_source_nwbs(tmp_path: Path) -> None:
    """A fresh temp stage declares every Cell-D init pair before source/CUDA."""
    _copy_source_stage_closure(tmp_path)
    payload = build_remote_stage_plan(tmp_path)
    assert payload["nwb_assets_in_stage"] is False
    assert payload["source_data_policy"]["copy_source_nwb"] is False
    assert payload["source_data_policy"]["symlink_or_bind_source_nwb"] is False
    assert payload["closure"]["paths"] == list(train.IMPLEMENTATION_CLOSURE)
    authority_assets = payload["source_adapter_authority_assets"]
    assert len(authority_assets) == 5
    assert all(not item["relative_path"].endswith(".nwb") for item in authority_assets)
    staged_authorities = [item for item in payload["stage_files"] if item["role"] == "immutable_source_adapter_authority"]
    assert len(staged_authorities) == 5
    expected_assets = plan.sealed_cell_d_init_assets_payload()
    assert payload["sealed_cell_d_initialization_assets"] == expected_assets
    init_bodies = [item for item in payload["stage_files"] if item["role"] == "immutable_sealed_cell_d_initialization_body"]
    init_sidecars = [item for item in payload["stage_files"] if item["role"] == "immutable_sealed_cell_d_initialization_sidecar"]
    assert len(init_bodies) == len(init_sidecars) == 3
    for asset, body, sidecar in zip(expected_assets, init_bodies, init_sidecars, strict=True):
        assert body == {
            "relative_path": asset["relative_path"], "sha256": asset["sha256"],
            "source_mode": "0444", "destination_mode": "0444",
            "role": "immutable_sealed_cell_d_initialization_body",
        }
        assert sidecar == {
            "relative_path": asset["sidecar_relative_path"], "sha256": asset["sidecar_sha256"],
            "contents": asset["sidecar_contents"], "source_mode": "0444",
            "destination_mode": "0444", "role": "immutable_sealed_cell_d_initialization_sidecar",
        }
    missing = deepcopy(payload)
    missing["stage_files"] = [
        item for item in missing["stage_files"]
        if item.get("relative_path") != expected_assets[0]["sidecar_relative_path"]
    ]
    with pytest.raises(PIRGStagePlanError, match="initialization body/sidecar transport drift"):
        validate_remote_stage_plan(missing)
    substituted = deepcopy(payload)
    for item in substituted["stage_files"]:
        if item.get("relative_path") == expected_assets[1]["relative_path"]:
            item["sha256"] = _sha("substituted-sealed-cell-d-swa")
            break
    with pytest.raises(PIRGStagePlanError, match="initialization body/sidecar transport drift"):
        validate_remote_stage_plan(substituted)
    assert payload["quick_score_substrate"]["v3_stage_root"] == score.V3_STAGE_ROOT
    assert payload["quick_score_substrate"]["copy_or_restage_evaluation_nwb"] is False
    assert payload["quick_score_substrate"]["new_pirg_score_cells_reuse_v3_rows"] is False


def test_score_overlay_is_additive_to_the_source_stage_and_never_carries_an_nwb() -> None:
    overlay = score_overlay_paths()
    assert overlay
    assert set(overlay).isdisjoint(train.IMPLEMENTATION_CLOSURE)
    assert all(not path.endswith(".nwb") for path in overlay)
    assert "tfpd_exploration/src/posterior_identity_residual_gate_v1/score.py" in overlay
    assert "tfpd_exploration/src/posterior_identity_residual_gate_v1/score_physical.py" in overlay
    assert "tfpd_exploration/scripts/run_posterior_identity_residual_gate_score.py" in overlay


def test_source_stage_plan_module_imports_without_the_score_overlay_present(tmp_path: Path) -> None:
    """A source-only stage must not accidentally import score-only code."""
    _copy_source_stage_closure(tmp_path)
    stage_pythonpath = str(tmp_path / "tfpd_exploration")
    program = (
        "from src.posterior_identity_residual_gate_v1.remote_stage import score_overlay_paths; "
        "assert score_overlay_paths(); print('source-stage-plan-import-pass')"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=tmp_path,
        env={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": stage_pythonpath},
        text=True, capture_output=True, check=True,
    )
    assert result.stdout.strip() == "source-stage-plan-import-pass"


def test_physical_prepare_revalidates_sealed_cell_d_pairs_before_source_or_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing/substituted staged pair aborts before source/CUDA side effects."""
    events: list[str] = []

    def reject_sealed_material(_root: Path) -> object:
        events.append("sealed-cell-d-material")
        raise physical.PIRGPhysicalError("synthetic sealed initialization asset omission")

    def unexpected_source_adapter(*_args: object, **_kwargs: object) -> object:
        events.append("source-adapter")
        raise AssertionError("source adapter must not run after sealed init failure")

    class _UnexpectedCuda:
        def is_available(self) -> bool:
            events.append("cuda-query")
            raise AssertionError("CUDA must not be queried after sealed init failure")

    backend = physical.PhysicalPIRGBackend(root=ROOT, source_data=object())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(backend, "_load_modules", lambda: {
        "torch": SimpleNamespace(cuda=_UnexpectedCuda()),
        "phase_b": SimpleNamespace(), "phase_b_v3": SimpleNamespace(),
        "matched": SimpleNamespace(load_sealed_cell_d_material=reject_sealed_material),
        "source_adapter_v2": SimpleNamespace(build_physical_source_adapter_v2=unexpected_source_adapter),
        "pop_robust": SimpleNamespace(), "arm_common": SimpleNamespace(),
    })
    with pytest.raises(physical.PIRGPhysicalError, match="sealed initialization asset omission"):
        backend.prepare(identity=_identity())
    assert events == ["sealed-cell-d-material"]
    assert backend.progress() == {"source_opened": False, "cuda_initialized": False}
    backend.close()


def test_public_cli_is_static_dry_and_flags_fail_before_any_lifecycle_import() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_identity_residual_gate.py"
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    dry = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["status"].startswith("DRY_NO_TORCH")
    failed = subprocess.run([sys.executable, str(script), "--execute", "--root-reviewed"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "in-process root-reviewed capability" in failed.stderr


def test_public_score_cli_is_static_dry_and_flags_fail_before_score_import() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_identity_residual_gate_score.py"
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    dry = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["status"].startswith("DRY_NO_TORCH")
    assert payload["route"] == "PIRG_QUICK_SCORE_3WITHIN_3EXTERNAL_M30_M10_M4"
    failed = subprocess.run([sys.executable, str(script), "--execute", "--root-reviewed"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "in-process root-reviewed capability" in failed.stderr
