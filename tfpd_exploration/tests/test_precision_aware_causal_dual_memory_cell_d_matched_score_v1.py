"""Focused synthetic/no-CUDA contract tests for the Precision-V2 score route."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import inspect
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration/src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan  # noqa: E402
from src.causal_dual_memory_cell_d_score_v1 import score as v1score  # noqa: E402
from src.causal_dual_memory_cell_d_score_v5 import plan as v5plan  # noqa: E402
from src.causal_dual_memory_cell_d_score_v8 import physical as v8physical  # noqa: E402
from src.causal_dual_memory_cell_d_score_v8 import plan as v8plan  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_v2 import transition as precision_transition  # noqa: E402
from src.precision_aware_causal_dual_memory_cell_d_score_v1 import physical, plan, score  # noqa: E402


def _sha(letter: str = "a") -> str:
    return letter * 64


def _witness(*, cell_shas: dict[str, str] | None = None) -> dict[str, object]:
    cells = {
        f"m{budget}:{surface}": _sha("b")
        for budget in (30, 10, 4) for surface in plan.SURFACES
    }
    if cell_shas is not None:
        cells.update(cell_shas)
    body = {
        "schema": "precision_aware_cdmd_v8_binding_witness_v1",
        "contract": plan.V8_PREDECESSOR.payload(),
        "directory_identity": [11, 13],
        "v8_identity_sha256": _sha("c"),
        "v8_source_gate_binding": {"binding_sha256": _sha("d")},
        "input_records_sha256": _sha("e"),
        "sealed_cell_sha256s": cells,
    }
    return {**body, "binding_sha256": plan.sha256_bytes(plan.canonical_json_bytes(body))}


def _identity() -> plan.ScoreIdentity:
    return plan.ScoreIdentity(
        closure=plan.implementation_closure(ROOT).payload(),
        v8_binding=_witness(),
    )


def _v5_synthetic_support() -> object:
    """Load synthetic V5 fixtures only; never a live authority or asset."""
    location = ROOT / "tfpd_exploration/tests/test_causal_dual_memory_cell_d_matched_score_v5.py"
    specification = importlib.util.spec_from_file_location("_precision_v5_synthetic_support", location)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _lifecycle_v8_binding() -> tuple[score.V8PredecessorBinding, v1score.InputAuthority, v1score.FixedEvaluationAuthority, object]:
    """Build a complete synthetic held-V8 witness without reading live roots.

    The raw V8 cells are built through V5's existing synthetic input codec so
    the successor's full lifecycle has to reconstruct both the row body SHA
    and exact input record digest—not merely accept a SHA-shaped placeholder.
    """
    support = _v5_synthetic_support()
    authority, fixed = support._input_authority()
    v5_identity = support._identity()
    v5_input = authority.payload(identity=v5_identity)
    sealed: dict[str, Mapping[str, object]] = {}
    for budget in (30, 10, 4):
        for candidate in support._cells(
            authority, budget=budget, delta=0.03,
            input_authority_sha256=plan.V8_EXPECTED_SHAS["input_authority.json"],
        ):
            payload = candidate.payload(input_payload=v5_input)
            if payload["system"] == v1plan.SYSTEM_SEALED:
                sealed[f"m{budget}:{payload['surface']}"] = payload
    binding = score.V8PredecessorBinding(
        directory_identity=(41, 43),
        v8_identity={"schema": "synthetic_v8_identity", "historical_closure": plan.V8_HISTORICAL_CLOSURE_SHA256},
        input_payload={"records": v5_input["records"]},
        score_payload={"schema": "synthetic_v8_score"}, terminal_payload={"status": "TERMINAL"},
        source_gate=support._source_gate_binding(), sealed_cells=sealed,
    )
    # This proves the typed source-gate binding itself—not a loose mapping—is
    # part of the V8 witness before the lifecycle sees it.
    assert binding.source_gate.contract is v5plan.SOURCE_GATE_CONTRACT
    return binding, authority, fixed, support


def _lifecycle_identity(binding: score.V8PredecessorBinding) -> plan.ScoreIdentity:
    return plan.ScoreIdentity(
        closure=plan.implementation_closure(ROOT).payload(), v8_binding=binding.payload(),
    )


def _base(*, session: str, budget: int, input_record_sha256: str = _sha("2")) -> v1score.SessionScore:
    return v1score.SessionScore(
        session=session, n_windows=3, r2=0.25, prediction_sha256=_sha("1"), input_record_sha256=input_record_sha256,
        model_state_before_sha256=_sha("3"), model_state_after_sha256=_sha("3"),
        initial_carrier_sha256=_sha("4"), group_assignment_sha256=_sha("5"),
        group_valid_mask_sha256=_sha("6"), initial_activity_sha256=_sha("7"),
        support_trial_ids_sha256=_sha("8"), raw_m30_t4_axis_proof_sha256=_sha("9"),
        sealed_normalizer_sha256=v1plan.SEALED_OLS_NORMALIZER_SHA256,
        sealed_model_load_proof_sha256=_sha("a"), target_last_bin_sha256=_sha("b"),
        valid_mask_sha256=_sha("c"), valid_last_bin_count=3, activity_fifo_capacity=v1plan.FIFO_CAPACITY[budget],
        accepted_updates=0, rejected_updates={}, group_forward_count=8, full_system_forward_count=2,
        dropout_calls=0, target_label_state_uses=0,
    )


def _posterior(*, budget: int) -> dict[str, object]:
    array = {"dtype": "float64", "shape": [budget, 4], "sha256": _sha("d")}
    return {
        "schema": "precision_aware_cdmd_v2_support_conditional_posterior_v1", "budget": budget,
        "fit_mode": "fixed_ridge_by_trial", "normalized_lambda": 0.1,
        "covariance_estimand": "conditional_gaussian_ridge_posterior_sigma2_A_inverse",
        "sampling_sandwich_covariance_used": False, "support_rates": dict(array),
        "support_direction_indices": {"dtype": "int64", "shape": [budget], "sha256": _sha("e")},
        "valid_mask": {"dtype": "bool", "shape": [4], "sha256": _sha("f")},
        "fixed_ridge_t4": {"dtype": "float32", "shape": [4, 4], "sha256": _sha("0")},
        "posterior_covariance_ac": {"dtype": "float64", "shape": [4, 2, 2], "sha256": _sha("1")},
        "residual_variance": {"dtype": "float64", "shape": [4], "sha256": _sha("2")},
        "design": {"dtype": "float64", "shape": [budget, 3], "sha256": _sha("3")},
        "normal_matrix": {"dtype": "float64", "shape": [3, 3], "sha256": _sha("4")},
        "normal_inverse": {"dtype": "float64", "shape": [3, 3], "sha256": _sha("5")},
        "residual_degrees_of_freedom": 1.0, "posterior_variance_floor": 1.0e-12,
        "groups_sha256_or_null": _sha("6"), "pseudo_labels_used": False, "decoder_token_used": False,
    }


def _transition(*, trial_id: str) -> dict[str, object]:
    return {
        "trial_id": trial_id, "activity_transition_committed": True, "activity_fifo_changed": True,
        "carrier_transition_committed": False, "activity_rejection_reason_or_null": None,
        "carrier_rejection_reason_or_null": "departure_freeze", "state_before_sha256": _sha("7"),
        "state_after_sha256": _sha("8"), "activity_before_sha256": _sha("9"),
        "activity_after_sha256": _sha("a"), "carrier_before_sha256": _sha("b"), "carrier_after_sha256": _sha("b"),
    }


def _record(*, surface: str, session: str, budget: int) -> dict[str, object]:
    return {
        "surface": surface, "session": session,
        "query_trial_ids_by_budget": {"4": ["q0"], "10": ["q0"], "30": ["q0"]},
        "support_trial_ids_sha256_by_budget": {"4": _sha("8"), "10": _sha("8"), "30": _sha("8")},
        "target_last_bin_sha256_by_budget": {"4": _sha("b"), "10": _sha("b"), "30": _sha("b")},
        "valid_last_bin_mask_sha256_by_budget": {"4": _sha("c"), "10": _sha("c"), "30": _sha("c")},
        "valid_last_bin_count_by_budget": {"4": 3, "10": 3, "30": 3},
    }


def _v8_session(*, session: str, budget: int, input_record_sha256: str = _sha("2")) -> dict[str, object]:
    base = _base(session=session, budget=budget, input_record_sha256=input_record_sha256)
    if budget == 30:
        base = _base(session=session, budget=30, input_record_sha256=input_record_sha256)
    payload = base.payload(budget=budget, system=v1plan.SYSTEM_CDMD)
    payload["system"] = v8plan.SYSTEM_SEALED
    payload["accepted_updates"] = 0
    payload["group_forward_count"] = 0
    return payload


def _v8_cell(*, surface: str, budget: int, sessions: list[str]) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_score_cell_evidence_v5", "surface": surface, "budget": budget,
        "system": v8plan.SYSTEM_SEALED, "input_authority_sha256": plan.V8_EXPECTED_SHAS["input_authority.json"],
        "model_swa_sha256": v1plan.SEALED_CELL_D_SWA_SHA256,
        "sessions": [
            _v8_session(
                session=item, budget=budget,
                input_record_sha256=score._digest(_record(surface=surface, session=item, budget=budget)),
            )
            for item in sessions
        ],
        "resources": {}, "eval_mode": True, "no_grad": True, "dropout_disabled": True,
        "same_sealed_model_state_for_both_systems": True,
    }


def _precision_session(
    *, session: str, budget: int, input_record_sha256: str = _sha("2"),
) -> score.PrecisionSessionEvidence:
    posterior = _posterior(budget=budget)
    return score.PrecisionSessionEvidence(
        base=_base(session=session, budget=budget, input_record_sha256=input_record_sha256),
        transition_records=(_transition(trial_id="q0"),),
        conditional_posterior=posterior, conditional_posterior_sha256=score._digest(posterior), decisions=(None,),
        frozen_support_reference_float64_sha256=_sha("7"),
    )


def _cell(*, surface: str, budget: int, sessions: list[str], binding_sha: str) -> score.PrecisionCellEvidence:
    reused = _v8_cell(surface=surface, budget=budget, sessions=sessions)
    m30_reference = _v8_cell(surface=surface, budget=30, sessions=sessions)
    return score.PrecisionCellEvidence(
        surface=surface, budget=budget, input_authority_sha256=_sha("f"), model_swa_sha256=v1plan.SEALED_CELL_D_SWA_SHA256,
        sessions=tuple(
            _precision_session(
                session=item, budget=budget,
                input_record_sha256=score._digest(_record(surface=surface, session=item, budget=budget)),
            )
            for item in sessions
        ), resources={},
        v8_reused_sealed_cell=reused,
        v8_reused_sealed_cell_sha256=score._digest(reused),
        v8_m30_reference_cell=m30_reference,
        v8_m30_reference_cell_sha256=score._digest(m30_reference), v8_predecessor_binding_sha256=binding_sha,
    )


def test_static_plan_dry_cli_and_current_closure_are_torch_free() -> None:
    assert "torch" not in sys.modules
    closure = plan.implementation_closure(ROOT).payload()
    assert len(closure["paths"]) == len(plan.IMPLEMENTATION_PATHS)
    assert plan.validate_implementation_closure(closure) == closure
    tampered = copy.deepcopy(closure)
    tampered["paths"][-1]["sha256"] = _sha("0")
    with pytest.raises(plan.PrecisionMatchedScorePlanError, match="canonical|closure"):
        plan.validate_implementation_closure(tampered)
    script = ROOT / "tfpd_exploration/scripts/run_precision_aware_causal_dual_memory_cell_d_matched_score_v1.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=True,
        env={"PATH": os.environ["PATH"], "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    dry = json.loads(completed.stdout)
    assert dry["no_torch_import"] is True and dry["spec"]["new_matrix_cell_count"] == 4
    assert "torch" not in sys.modules


def test_v8_witness_is_dynamic_held_evidence_not_a_caller_sha_shape() -> None:
    witness = _witness()
    assert plan.validate_v8_binding_witness(witness) == witness
    swapped = copy.deepcopy(witness)
    swapped["sealed_cell_sha256s"]["m10:within"] = _sha("f")
    with pytest.raises(plan.PrecisionMatchedScorePlanError, match="canonical"):
        plan.validate_v8_binding_witness(swapped)
    identity = plan.ScoreIdentity(closure=plan.implementation_closure(ROOT).payload(), v8_binding=witness)
    assert identity.payload()["v8_predecessor_binding"] == witness
    assert identity.payload()["closure"]["closure_sha256"] != plan.V8_HISTORICAL_CLOSURE_SHA256


def test_historical_v8_loader_never_rebuilds_a_current_v8_or_v5_closure() -> None:
    loader = inspect.getsource(score._read_v8_held_graph)
    assert "_v8_identity_from_attempt" in loader
    assert "implementation_closure" not in loader
    identity_codec = inspect.getsource(score._v8_identity_from_attempt)
    assert "v8plan.ScoreIdentity" in identity_codec
    runtime = inspect.getsource(physical.PrecisionV2ReviewedCDMScoreRuntime)
    # Runtime authentication consumes only the successor closure passed in
    # the durable identity; it must not substitute a current V2/V3/V4/V8
    # closure while trying to consume an immutable historical predecessor.
    assert "v8plan.implementation_closure" not in runtime


def test_precision_session_rejects_missing_decision_for_precision_reason() -> None:
    evidence = _precision_session(session="within-0", budget=10)
    payload = evidence.payload(budget=10, query_trial_ids=("q0",))
    assert payload["precision_decision_count"] == 0
    forged = copy.deepcopy(payload)
    forged["transition_records"][0]["carrier_rejection_reason_or_null"] = "precision_credible_region"
    with pytest.raises(score.PrecisionMatchedScoreError, match="omitted decision"):
        score._precision_session_from_payload(forged, budget=10, query_trial_ids=("q0",))

    forged_commit = copy.deepcopy(payload)
    forged_commit["transition_records"][0].update({
        "carrier_transition_committed": True,
        "carrier_rejection_reason_or_null": None,
        "carrier_after_sha256": _sha("f"),
    })
    with pytest.raises(score.PrecisionMatchedScoreError, match="carrier commit omitted decision"):
        score._precision_session_from_payload(forged_commit, budget=10, query_trial_ids=("q0",))


def test_precision_session_binds_frozen_support_and_rejects_current_carrier_substitution() -> None:
    evidence = _precision_session(session="within-0", budget=4)
    payload = evidence.payload(budget=4, query_trial_ids=("q0",))
    assert payload["precision_reference"] == "frozen_support_only_initial_fixed_ridge_not_current_active"
    assert payload["conditional_posterior"]["covariance_estimand"] == "conditional_gaussian_ridge_posterior_sigma2_A_inverse"
    forged = copy.deepcopy(payload)
    forged["conditional_posterior"]["sampling_sandwich_covariance_used"] = True
    with pytest.raises(score.PrecisionMatchedScoreError, match="posterior"):
        score._precision_session_from_payload(forged, budget=4, query_trial_ids=("q0",))


def test_precision_decision_receipt_cross_binds_the_exact_float64_frozen_support_reference() -> None:
    rates = np.asarray(
        [[4.0, 5.0, 6.0, 7.0], [5.0, 6.5, 7.0, 8.5], [6.0, 5.5, 8.0, 7.5], [4.5, 7.0, 6.5, 9.0]],
        dtype=np.float64,
    )
    posterior = precision_transition.SupportConditionalPosterior.from_support_only_fixed_ridge(
        support_rates=rates, support_direction_indices=np.asarray([0, 2, 4, 6], dtype=np.int64),
        valid_mask=np.ones((4,), dtype=np.bool_), groups_sha256=_sha("d"),
    )
    decision = precision_transition.decide_against_frozen_support(
        posterior, frozen_initial_active_t4=posterior.frozen_initial_active_t4,
        proposed_active_t4=posterior.frozen_initial_active_t4,
    )
    transition = _transition(trial_id="q0")
    transition.update({
        "carrier_transition_committed": True,
        "carrier_rejection_reason_or_null": None,
        "carrier_after_sha256": _sha("f"),
    })
    evidence = score.PrecisionSessionEvidence(
        base=_base(session="within-0", budget=4), transition_records=(transition,),
        conditional_posterior=posterior.payload(), conditional_posterior_sha256=posterior.digest,
        frozen_support_reference_float64_sha256=decision.reference_initial_sha256,
        decisions=(decision,),
    )
    payload = evidence.payload(budget=4, query_trial_ids=("q0",))
    assert payload["precision_decision_count"] == 1 and payload["precision_accept_count"] == 1
    forged = copy.deepcopy(payload)
    forged["frozen_support_reference_float64_sha256"] = _sha("e")
    with pytest.raises(score.PrecisionMatchedScoreError, match="frozen-support reference"):
        score._precision_session_from_payload(forged, budget=4, query_trial_ids=("q0",))


def test_physical_runtime_changes_only_transition_and_cell_hooks() -> None:
    runtime = physical.PrecisionV2ReviewedCDMScoreRuntime
    assert runtime._parse_session is v8physical.V8ReviewedCDMScoreRuntime._parse_session
    assert runtime._forward_full is v8physical.V8ReviewedCDMScoreRuntime._forward_full
    assert runtime._group_predictions is v8physical.V8ReviewedCDMScoreRuntime._group_predictions
    source = inspect.getsource(runtime._commit_completed_transition)
    assert "decide_against_frozen_support" in source and "commit_independent" in source
    # The explanatory docstring may name the inherited operation.  The route
    # itself must not issue a second observation after V1 has made the sole
    # proposal/forward boundary.
    assert ".observe_completed_trial(" not in source
    lifecycle = inspect.getsource(score.run_authorized_score_lifecycle)
    assert "v1score.run_profiled_score_lifecycle" in lifecycle
    assert "for budget" not in lifecycle
    inherited_parser = inspect.getsource(v8physical.V8ReviewedCDMScoreRuntime._parse_session)
    assert "tuple(ordered[30:])" in inherited_parser
    assert "torch" not in sys.modules


def test_held_v8_reader_rejects_symlink_leaf_before_any_semantic_acceptance(tmp_path: Path) -> None:
    directory = tmp_path / plan.V8_ROOT_RELATIVE
    directory.mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text("{}", encoding="utf-8")
    for name in plan.V8_EXPECTED_TOPOLOGY:
        if name == "attempt.json":
            (directory / name).symlink_to(target)
        else:
            (directory / name).write_text("{}", encoding="utf-8")
        if name != "attempt.json":
            (directory / f"{name}.sha256").write_text("0" * 64 + f"  {name}\n", encoding="ascii")
    (directory / "attempt.json.sha256").write_text("0" * 64 + "  attempt.json\n", encoding="ascii")
    for path in directory.iterdir():
        if not path.is_symlink():
            path.chmod(0o444)
    with pytest.raises(score.PrecisionMatchedScoreError, match="descriptor read|authority"):
        score.validate_completed_v8_predecessor(tmp_path)


def test_four_cell_summary_reuses_v8_sealed_rows_and_m30_without_rerun() -> None:
    within = [f"w{index}" for index in range(6)]
    external = [f"e{index}" for index in range(15)]
    binding = _witness()
    records = [
        *[_record(surface=plan.WITHIN, session=item, budget=10) for item in within],
        *[_record(surface=plan.EXTERNAL, session=item, budget=10) for item in external],
    ]
    cells = (
        _cell(surface=plan.WITHIN, budget=10, sessions=within, binding_sha=binding["binding_sha256"]),
        _cell(surface=plan.EXTERNAL, budget=10, sessions=external, binding_sha=binding["binding_sha256"]),
    )
    summary = score.summarize_budget(cells, 10, {"records": records})
    assert summary["m30_rerun"] is False
    assert set(summary["surfaces"]) == set(plan.SURFACES)
    assert summary["surfaces"][plan.EXTERNAL]["paired_precision_v2_minus_v8_sealed"]["n_sessions"] == 15
    assert summary["surfaces"][plan.WITHIN]["precision_transition_acceptance"]["conditional_posterior_decision_count"] == 0
    assert score.budget_gate(summary, 10)["formal_gate"] is False


def test_reused_v8_sealed_cell_body_and_m30_reference_sha_are_reconstructed_exactly() -> None:
    sessions = [f"w{index}" for index in range(6)]
    binding = _witness()
    cell = _cell(surface=plan.WITHIN, budget=4, sessions=sessions, binding_sha=binding["binding_sha256"])
    records = [_record(surface=plan.WITHIN, session=item, budget=4) for item in sessions]
    assert cell.payload(input_payload={"records": records})["v8_reused_sealed_cell_sha256"] == cell.v8_reused_sealed_cell_sha256
    forged = copy.deepcopy(cell.v8_reused_sealed_cell)
    forged["sessions"][0]["governing_r2"] = 0.26
    tampered = replace(cell, v8_reused_sealed_cell=forged)
    with pytest.raises(score.PrecisionMatchedScoreError, match="V8 reused sealed/M30 cell identity"):
        tampered.payload(input_payload={"records": records})
    forged_m30 = copy.deepcopy(cell.v8_m30_reference_cell)
    forged_m30["sessions"][0]["model_state_after_sha256"] = _sha("f")
    tampered_m30 = replace(cell, v8_m30_reference_cell=forged_m30)
    with pytest.raises(score.PrecisionMatchedScoreError, match="V8 reused sealed/M30 cell identity"):
        tampered_m30.payload(input_payload={"records": records})


def test_new_matrix_is_exact_m10_m4_and_never_m30_execution() -> None:
    spec = plan.PUBLIC_SPEC.payload()
    assert spec["budgets_executed"] == [10, 4]
    assert spec["m30_reference_only"] is True
    assert spec["new_matrix_cell_count"] == 4
    assert all(row["budget"] != 30 for row in spec["new_cell_order"])
    assert score.terminal_verdict({10: {"screen_complete": True}, 4: {"screen_complete": True}}) == "SCREEN_COMPLETE_NO_FORMAL_VERDICT"


def test_input_equivalence_fails_closed_on_single_v8_record_digest_drift() -> None:
    identity = _identity()
    payload = {"records": []}
    with pytest.raises(score.PrecisionMatchedScoreError, match="differ"):
        score.validate_v8_input_equivalence(payload, identity)


def _precision_lifecycle_cells(
    *, authority: v1score.InputAuthority, binding: score.V8PredecessorBinding,
    support: object, budget: int, input_authority_sha256: str,
) -> tuple[score.PrecisionCellEvidence, ...]:
    """Turn exact V5 synthetic dynamic rows into V2-only successor cells.

    Every synthetic inherited carrier commit is converted to a pre-existing
    ``departure_freeze`` rejection.  That leaves the test focused on the
    shared score lifecycle / V8 bridge while the separate V2 suite covers the
    accepted and precision-rejected mathematical decision branches.
    """
    inherited = support._cells(
        authority, budget=budget, delta=0.03, input_authority_sha256=input_authority_sha256,
    )
    output: list[score.PrecisionCellEvidence] = []
    for surface in plan.SURFACES:
        dynamic = next(
            item for item in inherited
            if item.surface == surface and item.system == v1plan.SYSTEM_CDMD
        )
        sessions: list[score.PrecisionSessionEvidence] = []
        for inherited_session in dynamic.sessions:
            transitions: list[dict[str, object]] = []
            for raw in inherited_session.transition_records:
                row = dict(raw)
                if row["carrier_transition_committed"]:
                    row["carrier_transition_committed"] = False
                    row["carrier_rejection_reason_or_null"] = "departure_freeze"
                    row["carrier_after_sha256"] = row["carrier_before_sha256"]
                transitions.append(row)
            sessions.append(score.PrecisionSessionEvidence(
                base=inherited_session.base, transition_records=tuple(transitions),
                conditional_posterior=_posterior(budget=budget),
                conditional_posterior_sha256=score._digest(_posterior(budget=budget)),
                frozen_support_reference_float64_sha256=_sha("7"),
                decisions=tuple(None for _ in transitions),
            ))
        output.append(score.PrecisionCellEvidence(
            surface=surface, budget=budget, input_authority_sha256=input_authority_sha256,
            model_swa_sha256=v1plan.SEALED_CELL_D_SWA_SHA256, sessions=tuple(sessions),
            resources=support._resources(),
            v8_reused_sealed_cell=binding.sealed_cell(budget=budget, surface=surface),
            v8_reused_sealed_cell_sha256=binding.payload()["sealed_cell_sha256s"][f"m{budget}:{surface}"],
            v8_m30_reference_cell=binding.sealed_cell(budget=30, surface=surface),
            v8_m30_reference_cell_sha256=binding.payload()["sealed_cell_sha256s"][f"m30:{surface}"],
            v8_predecessor_binding_sha256=binding.payload()["binding_sha256"],
        ))
    return tuple(output)


class _LifecycleArtifact:
    """Small in-memory immutable artifact protocol with one shared event log."""

    topology = score.SCORE_TOPOLOGY

    def __init__(self, events: list[str]) -> None:
        self.items: dict[str, bytes] = {}
        self.events = events

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        assert name not in self.items
        body = score._json(payload)
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


class _LifecycleRuntime:
    """No-data backend exercising the actual successor hooks and order."""

    def __init__(
        self, *, authority: v1score.InputAuthority, binding: score.V8PredecessorBinding,
        support: object, events: list[str],
    ) -> None:
        self.authority, self.binding, self.support, self.events = authority, binding, support, events

    def preflight(self, *, root: Path, identity: plan.ScoreIdentity) -> Mapping[str, object]:
        self.events.append("preflight")
        return {
            "target_paths_resolved": False, "target_opened": False,
            "checkpoint_opened": False, "cuda_initialized": False,
        }

    def prepare(self, *, root: Path, identity: plan.ScoreIdentity) -> object:
        self.events.append("prepare")
        return self

    def materialize_inputs(
        self, runtime: object, *, identity: plan.ScoreIdentity,
        evaluation_authority: v1score.FixedEvaluationAuthority,
    ) -> v1score.InputAuthority:
        assert runtime is self
        self.events.append("materialize")
        return self.authority

    def score_budget(
        self, runtime: object, *, budget: int, input_authority_sha256: str,
        identity: plan.ScoreIdentity,
    ) -> tuple[score.PrecisionCellEvidence, ...]:
        assert runtime is self and budget in plan.BUDGETS
        self.events.append(f"budget_m{budget}")
        return _precision_lifecycle_cells(
            authority=self.authority, binding=self.binding, support=self.support,
            budget=budget, input_authority_sha256=input_authority_sha256,
        )

    def revalidate(self, runtime: object, *, root: Path, identity: plan.ScoreIdentity) -> None:
        assert runtime is self
        self.events.append("revalidate")

    def failure_progress(self, runtime: object | None) -> Mapping[str, object]:
        return {
            "within_assets_opened": False, "external_assets_opened": False,
            "checkpoint_opened": False, "cuda_initialized": False,
            "full_system_forward_count": 0, "group_forward_count": 0,
        }

    def close(self, runtime: object | None) -> None:
        self.events.append("close")


def test_shared_lifecycle_publishes_attempt_before_backend_and_runs_exact_four_new_cells(tmp_path: Path) -> None:
    binding, authority, fixed, support = _lifecycle_v8_binding()
    identity = _lifecycle_identity(binding)
    preflight = score.build_target_free_preflight(
        root=tmp_path, identity=identity, predecessor=binding, fixed_authority=fixed,
    )
    # The route-local V8 label and the shared lifecycle source-gate hook are
    # deliberately exact aliases of the same descriptor-derived witness.
    assert preflight["source_gate"] == preflight["v8_predecessor"] == binding.payload()
    pre_sha = score._digest(preflight)
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = score._digest(authorization)
    capability = v1score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=v1score._issue_root_publication_capability(),
    )
    hooks = replace(
        score.PRECISION_LIFECYCLE_HOOKS,
        implementation_closure=lambda _root: identity.payload()["closure"],
        validate_source_gate=lambda _root: binding,
        validate_reserved_score_artifact=lambda _root, _artifact, _identity: None,
    )
    events: list[str] = []
    artifact = _LifecycleArtifact(events)
    result = v1score.run_profiled_score_lifecycle(
        tmp_path, identity=identity, capability=capability,
        backend=_LifecycleRuntime(authority=authority, binding=binding, support=support, events=events),
        artifact=artifact, official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, hooks=hooks,
    )
    assert events[:4] == ["preflight", "publish:attempt.json", "prepare", "materialize"]
    assert events.index("publish:attempt.json") < events.index("prepare") < events.index("materialize")
    assert [item for item in events if item.startswith("budget_")] == ["budget_m10", "budget_m4"]
    assert result["verdict"] == "SCREEN_COMPLETE_NO_FORMAL_VERDICT"
    assert set(artifact.items) == {"attempt.json", "input_authority.json", "score.json", "terminal.json"}
    score_body = json.loads(artifact.items["score.json"])
    assert score_body["cell_execution_order"] == [
        {"budget": budget, "surface": surface, "system": plan.SYSTEM_PRECISION_V2}
        for budget in (10, 4) for surface in plan.SURFACES
    ]
    assert score_body["m30_execution"] == "v8_immutable_sealed_deployment_reference_only__not_rerun"
    assert score_body["target_optimizer_backward_update"] == 0
    assert "torch" not in sys.modules
