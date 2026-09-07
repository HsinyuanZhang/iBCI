"""Focused CPU/no-data gates for the additive PIRG score-v2 repair."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.posterior_identity_residual_gate_score_v2 import score_v2 as v2
from src.posterior_identity_residual_gate_score_v2 import score_physical_v2 as physical_v2
from src.posterior_identity_residual_gate_v1 import score as v1
from src.posterior_identity_residual_gate_v1 import score_physical as physical_v1
from src.posterior_carrier_v1 import core as posterior_core


ROOT = Path(__file__).resolve().parents[2]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_pair(directory: Path, name: str, payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    body_path, sidecar_path = directory / name, directory / f"{name}.sha256"
    body_path.write_bytes(body)
    sidecar_path.write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(body_path, 0o444)
    os.chmod(sidecar_path, 0o444)
    return digest


def _copy_v2_closure(stage_root: Path) -> None:
    for relative in v2.V2_CLOSURE_PATHS:
        source, destination = ROOT / relative, stage_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _base_identity(root: Path) -> v1.PIRGScoreIdentity:
    return v1.PIRGScoreIdentity(
        closure=v1.score_implementation_closure(root),
        source_terminal_sha256=_sha("source-terminal"),
        final_alpha_sha256=_sha("final-alpha"),
        source_authority_sha256=_sha("source-authority"),
    )


def _v1_failure_pair(root: Path, *, identity: v1.PIRGScoreIdentity) -> v2.V1FailedPredecessor:
    attempt = {
        "schema": "posterior_identity_residual_gate_score_attempt_v1",
        "identity": identity.payload(),
        "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_INPUTS",
        "within_opened": False, "external_opened": False, "formal_opened": False, "h1_opened": False,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
    }
    failure = {
        "schema": "posterior_identity_residual_gate_score_failure_v1",
        "identity": identity.payload(),
        "attempt_sha256": _sha("placeholder"), "input_authority_sha256": None,
        "stage": "prepare",
        "progress": {
            "within_opened": False, "external_opened": False, "cuda_initialized": True,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "formal_opened": False, "h1_opened": False,
        },
        "terminal_published": False,
        "error_class": "RuntimeError",
        "error_sha256": _sha("scalar-view-failure"),
        "traceback_sha256": _sha("traceback"),
    }
    directory = root / v2.V1_RESULT_ROOT_RELATIVE
    attempt_sha = _write_pair(directory, "attempt.json", attempt)
    failure["attempt_sha256"] = attempt_sha
    failure_sha = _write_pair(directory, "failure.json", failure)
    return v2.V1FailedPredecessor(
        attempt_sha256=attempt_sha, failure_sha256=failure_sha, error_sha256=failure["error_sha256"],
    )


def _reference_scalar_digest(value: torch.Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    if detached.is_floating_point():
        detached = detached + 0
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(str(tuple(detached.shape)).encode("utf-8"))
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_scalar_digest_preserves_original_rank_zero_shape_and_changes_on_mutation() -> None:
    alpha = torch.tensor(-0.0, dtype=torch.float32)
    got = physical_v2.scalar_safe_tensor_digest(core=posterior_core, tensor=alpha)
    assert got == _reference_scalar_digest(alpha)
    assert got != physical_v2.scalar_safe_tensor_digest(core=posterior_core, tensor=alpha.reshape(1))
    alpha.add_(0.25)
    assert physical_v2.scalar_safe_tensor_digest(core=posterior_core, tensor=alpha) != got


def test_physical_v2_subclass_changes_only_wrapper_scalar_digest() -> None:
    from src.posterior_identity_residual_gate_v1.core import PosteriorIdentityResidualGate
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    assert issubclass(physical_v2.PhysicalPIRGQuickScoreBackendV2, physical_v1.PhysicalPIRGQuickScoreBackend)
    assert physical_v2.PhysicalPIRGQuickScoreBackendV2.prepare is physical_v1.PhysicalPIRGQuickScoreBackend.prepare
    assert physical_v2.PhysicalPIRGQuickScoreBackendV2.resolve_inputs is physical_v1.PhysicalPIRGQuickScoreBackend.resolve_inputs
    assert physical_v2.PhysicalPIRGQuickScoreBackendV2.score_cell is physical_v1.PhysicalPIRGQuickScoreBackend.score_cell
    assert physical_v2.PhysicalPIRGQuickScoreBackendV2.final_reverify is physical_v1.PhysicalPIRGQuickScoreBackend.final_reverify

    class _ArmCommon:
        @staticmethod
        def state_sha256(_model: object) -> str:
            return _sha("frozen-cell-d")

    wrapper = PosteriorIdentityResidualGate(build_population_robustness_model(seed=42, cell="D"))
    with pytest.raises(RuntimeError, match="cannot be 0"):
        physical_v1.PhysicalPIRGQuickScoreBackend._wrapper_state_digest(
            arm_common=_ArmCommon, core=posterior_core, wrapper=wrapper,
        )
    first = physical_v2.PhysicalPIRGQuickScoreBackendV2._wrapper_state_digest(
        arm_common=_ArmCommon, core=posterior_core, wrapper=wrapper,
    )
    with torch.no_grad():
        wrapper.alpha.add_(0.01)
    second = physical_v2.PhysicalPIRGQuickScoreBackendV2._wrapper_state_digest(
        arm_common=_ArmCommon, core=posterior_core, wrapper=wrapper,
    )
    assert first != second


def test_held_fd_v1_predecessor_validator_rejects_semantic_tamper(tmp_path: Path) -> None:
    _copy_v2_closure(tmp_path)
    identity = _base_identity(tmp_path)
    predecessor = _v1_failure_pair(tmp_path, identity=identity)
    assert v2.validate_v1_failed_predecessor(
        tmp_path, expected_identity=identity, predecessor=predecessor,
    ).payload() == identity.payload()

    directory = tmp_path / v2.V1_RESULT_ROOT_RELATIVE
    failure_path = directory / "failure.json"
    failure = json.loads(failure_path.read_bytes())
    failure["stage"] = "forwards"
    os.chmod(failure_path, 0o644)
    os.chmod(directory / "failure.json.sha256", 0o644)
    forged_failure_sha = _write_pair(directory, "failure.json", failure)
    forged = v2.V1FailedPredecessor(
        attempt_sha256=predecessor.attempt_sha256,
        failure_sha256=forged_failure_sha,
        error_sha256=predecessor.error_sha256,
    )
    with pytest.raises(v2.PIRGScoreV2Error, match="failure semantics drift"):
        v2.validate_v1_failed_predecessor(tmp_path, expected_identity=identity, predecessor=forged)


def _input_authority() -> v1.InputAuthority:
    return v1.InputAuthority({
        surface: [
            {
                "session": session, "n_windows": index + 5,
                "input_sha256": _sha(f"{surface}/{session}/input"),
                "last_bin_target_sha256": _sha(f"{surface}/{session}/target"),
                "last_bin_mask_sha256": _sha(f"{surface}/{session}/mask"),
                "point_side_sha256s": {str(budget): _sha(f"{surface}/{session}/point/{budget}") for budget in v1.BUDGETS},
                "directional_credibility_sha256s": {str(budget): _sha(f"{surface}/{session}/cred/{budget}") for budget in v1.BUDGETS},
                "prefix_row_ids_sha256s": {str(budget): _sha(f"{surface}/{session}/prefix/{budget}") for budget in v1.BUDGETS},
            }
            for index, session in enumerate(v1.FIXED_SESSIONS[surface])
        ]
        for surface in v1.SURFACES
    })


def _cell(identity: v1.PIRGScoreIdentity, cell: v1.ScoreCell, input_payload: Mapping[str, object]) -> v1.CellEvidence:
    rows_by_session = {row["session"]: row for row in input_payload["surfaces"][cell.surface]}
    rows = tuple(v1.SessionScore(
        session=session,
        n_windows=rows_by_session[session]["n_windows"],
        r2=0.2 + 0.01 * index + (0.002 if cell.mode == v1.PIRG_MODE else 0.0),
        prediction_sha256=_sha(f"prediction/{cell.surface}/{cell.mode}/{cell.budget}/{session}"),
        input_sha256=v1._digest(v1._json(rows_by_session[session])),
    ) for index, session in enumerate(v1.FIXED_SESSIONS[cell.surface]))
    state = _sha(f"state/{cell.mode}")
    return v1.CellEvidence(
        cell=cell, model_state_before_sha256=state, model_state_after_sha256=state,
        model_artifact_sha256=(v1.plan.SEALED_CELL_D_SWA_SHA256 if cell.mode == v1.BASELINE_MODE else identity.final_alpha_sha256),
        base_cell_d_swa_sha256=v1.plan.SEALED_CELL_D_SWA_SHA256, rows=rows,
    )


class _MockV2Backend:
    def __init__(self, identity: v1.PIRGScoreIdentity, *, fail: bool = False) -> None:
        self.identity, self.fail, self.closed = identity, fail, False

    def prepare(self, *, identity: v1.PIRGScoreIdentity) -> None:
        assert identity == self.identity

    def resolve_inputs(self, *, identity: v1.PIRGScoreIdentity) -> v1.InputAuthority:
        assert identity == self.identity
        return _input_authority()

    def score_cell(self, *, cell: v1.ScoreCell, input_payload: Mapping[str, object]) -> v1.CellEvidence:
        if self.fail and cell.budget == 10:
            raise RuntimeError("synthetic V2 forward failure")
        return _cell(self.identity, cell, input_payload)

    def final_reverify(self, *, identity: v1.PIRGScoreIdentity) -> v1.ScoreClosure:
        assert identity == self.identity
        return identity.closure

    def progress(self) -> Mapping[str, object]:
        return {
            "within_opened": True, "external_opened": True, "cuda_initialized": False,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "formal_opened": False, "h1_opened": False,
        }

    def close(self) -> None:
        self.closed = True


def test_v2_lifecycle_binds_predecessor_and_publishes_only_v2_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_v2_closure(tmp_path)
    base_identity = _base_identity(tmp_path)
    predecessor = _v1_failure_pair(tmp_path, identity=base_identity)
    monkeypatch.setattr(v2, "V1_FAILURE_PREDECESSOR", predecessor)
    identity = v2.PIRGScoreV2Identity(base_identity, v2.score_v2_implementation_closure(tmp_path))
    backend = _MockV2Backend(base_identity)
    before = (tmp_path / v2.V1_RESULT_ROOT_RELATIVE / "failure.json").read_bytes()
    result = v2.run_v2_score_lifecycle(
        root=tmp_path, identity=identity, backend=backend,
        execution_capability=v2._issue_root_review_capability(identity),
    )
    output = tmp_path / v2.RESULT_ROOT_RELATIVE
    assert set(result) == {"attempt_sha256", "input_authority_sha256", "score.json", "terminal.json"}
    assert (output / "score.json").exists() and (output / "terminal.json").exists()
    assert not (output / "failure.json").exists()
    assert (tmp_path / v2.V1_RESULT_ROOT_RELATIVE / "failure.json").read_bytes() == before
    assert backend.closed


def test_v2_lifecycle_failure_never_publishes_partial_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _copy_v2_closure(tmp_path)
    base_identity = _base_identity(tmp_path)
    predecessor = _v1_failure_pair(tmp_path, identity=base_identity)
    monkeypatch.setattr(v2, "V1_FAILURE_PREDECESSOR", predecessor)
    identity = v2.PIRGScoreV2Identity(base_identity, v2.score_v2_implementation_closure(tmp_path))
    backend = _MockV2Backend(base_identity, fail=True)
    with pytest.raises(RuntimeError, match="synthetic V2 forward failure"):
        v2.run_v2_score_lifecycle(
            root=tmp_path, identity=identity, backend=backend,
            execution_capability=v2._issue_root_review_capability(identity),
        )
    output = tmp_path / v2.RESULT_ROOT_RELATIVE
    assert (output / "failure.json").exists()
    assert not (output / "score.json").exists() and not (output / "terminal.json").exists()
    assert backend.closed


def test_v2_closure_and_cli_are_dry_and_fail_closed() -> None:
    closure = v2.score_v2_implementation_closure(ROOT).payload()
    assert tuple(closure["paths"]) == v2.V2_CLOSURE_PATHS
    assert "tfpd_exploration/src/posterior_identity_residual_gate_score_v2/score_physical_v2.py" in closure["paths"]
    plan = v2.dry_plan()
    assert plan["status"].startswith("DRY_NO_TORCH")
    assert plan["result_root_relative"] == v2.RESULT_ROOT_RELATIVE
    script = ROOT / "tfpd_exploration/scripts/run_posterior_identity_residual_gate_score_v2.py"
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    dry = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert json.loads(dry.stdout)["status"].startswith("DRY_NO_TORCH")
    blocked = subprocess.run(
        [sys.executable, str(script), "--execute", "--root-reviewed"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert blocked.returncode != 0
    assert "in-process root-reviewed capability" in blocked.stderr
