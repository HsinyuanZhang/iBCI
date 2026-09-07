"""Focused CPU/no-data tests for the V5 independent-activity matched scorer.

The fixtures below are deliberately synthetic receipt/model-state values.  They
never resolve an NWB path, inspect a checkpoint tensor, initialize CUDA, or
touch a canonical authority/result root.  A temporary descriptor graph is
used only to exercise the shared held-FD pair reader with 88 synthetic bodies.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
from typing import Mapping

import pytest

from src.causal_dual_memory_cell_d_score_v1 import physical as v1physical
from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_score_v1 import score as v1score
from src.causal_dual_memory_cell_d_score_v5 import physical, plan, score


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity() -> plan.ScoreIdentity:
    rows = {path: _sha(path) for path in plan.IMPLEMENTATION_PATHS}
    rows[plan.WORKORDER_RELATIVE] = plan.WORKORDER_SHA256
    return plan.ScoreIdentity(plan.ImplementationClosure(rows).payload())


def _source_gate_binding(*, contract: v1plan.CompletedSourceGateContract = plan.SOURCE_GATE_CONTRACT) -> v1score.SourceGateBinding:
    roster = tuple(f"sub-C_ses-CO-v5-synthetic-{index:02d}" for index in range(27))
    rows = dict(contract.fixed_body_sha256s)
    rows.update({
        f"budget_m{budget}__{session}.json": _sha(f"v5:{budget}:{session}")
        for budget in plan.BUDGETS for session in roster
    })
    return v1score.SourceGateBinding(
        directory_device=1, directory_inode=2, body_sha256s=rows,
        terminal_status=contract.terminal_status,
        source_gate_closure_sha256=contract.closure_sha256,
        strict_source_roster=roster, contract=contract,
    )


def _fixed_authority() -> v1score.FixedEvaluationAuthority:
    within = tuple(
        v1score.EvaluationAsset(
            v1plan.WITHIN, f"sub-C_ses-CO-within-{index:02d}", f"within-{index}",
            f"sub-C/sub-C_ses-CO-within-{index:02d}_behavior+ecephys.nwb", 1000 + index, _sha(f"within:{index}"),
        )
        for index in range(6)
    )
    external = tuple(
        v1score.EvaluationAsset(
            v1plan.EXTERNAL, f"sub-M_ses-CO-external-{index:02d}", f"external-{index}",
            f"sub-M/sub-M_ses-CO-external-{index:02d}_behavior+ecephys.nwb", 2000 + index, _sha(f"external:{index}"),
        )
        for index in range(15)
    )
    return v1score.FixedEvaluationAuthority(
        within=within,
        external=external,
        fixed_authority_bindings={"strict_manifest": {"sha256": _sha("within-manifest")}},
        within_manifest_binding={"manifest_sha256": _sha("external-ledger")},
    )


def _raw_axis(session: str) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_score_raw_t4_axis_v1",
        "budget": 30,
        "raw_t4_sha256": _sha(f"{session}:raw-m30"),
        "channel_order_sha256": _sha(f"{session}:channel-order"),
        "valid_mask_sha256": _sha(f"{session}:valid-mask"),
        "source_unit_count": 8,
        "feature_group": "t4",
        "signal_view": "sua",
        "channel_ids_are_exact_int64_arange": True,
        "validity_rule": "raw_t4_modulation_m_gt_modulation_eps",
        "modulation_eps": 1.0e-6,
        "raw_before_normalization": True,
    }


def _record(asset: v1score.EvaluationAsset) -> v1score.InputRecord:
    chronology = tuple(f"{asset.session}:trial:{index:02d}" for index in range(32))
    support = {"4": chronology[:4], "10": chronology[:10], "30": chronology[:30]}
    query = {key: chronology[30:] for key in support}
    return v1score.InputRecord(
        surface=asset.surface, session=asset.session, asset_id=asset.asset_id,
        frozen_path=asset.frozen_path, asset_bytes=asset.bytes, asset_sha256=asset.sha256,
        chronological_trial_ids=chronology,
        support_trial_ids_by_budget=support,
        query_trial_ids_by_budget=query,
        neural_sha256=_sha(f"{asset.session}:neural"),
        calibration_sha256=_sha(f"{asset.session}:calibration"),
        target_last_bin_sha256_by_budget={key: _sha(f"{asset.session}:target:{key}") for key in support},
        valid_last_bin_mask_sha256_by_budget={key: _sha(f"{asset.session}:mask:{key}") for key in support},
        valid_last_bin_count_by_budget={key: 3 for key in support},
        raw_m30_t4_axis_proof=_raw_axis(asset.session), theta_recovery_sha256=_sha(f"{asset.session}:theta"),
    )


def _input_authority() -> tuple[v1score.InputAuthority, v1score.FixedEvaluationAuthority]:
    fixed = _fixed_authority()
    records = tuple(_record(asset) for asset in (*fixed.within, *fixed.external))
    return v1score.InputAuthority(records, score._digest(fixed.payload())), fixed


def _resources() -> dict[str, object]:
    return {
        "runtime_environment": {
            **v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
            "visible_devices": 1,
            "attested": True,
            "torch_cuda_matmul_allow_tf32": False,
            "torch_cudnn_allow_tf32": False,
        },
        "current_cuda_allocated_bytes": 4,
        "current_cuda_reserved_bytes": 8,
        "peak_cuda_allocated_bytes": 12,
        "peak_cuda_reserved_bytes": 16,
        "rss_bytes": 4096,
        "wall_seconds": 2.0,
        "full_and_group_forward_chunks": 8,
        "completed_query_trials": 2,
        "windows_or_trials_per_s": 1.0,
    }


def _transition_rows(*, budget: int, query_ids: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    state_before = _sha(f"M{budget}:state:0")
    activity_before = _sha(f"M{budget}:activity:0")
    carrier_before = _sha(f"M{budget}:carrier:0")
    for index, trial_id in enumerate(query_ids):
        accepted_carrier = budget != 30 and index == len(query_ids) - 1
        state_after = _sha(f"M{budget}:state:{index + 1}")
        activity_after = (
            activity_before if budget == 30 else _sha(f"M{budget}:activity:{index + 1}")
        )
        carrier_after = _sha(f"M{budget}:carrier:{index + 1}") if accepted_carrier else carrier_before
        rows.append({
            "trial_id": trial_id,
            "activity_transition_committed": True,
            "activity_fifo_changed": budget != 30,
            "carrier_transition_committed": accepted_carrier,
            "activity_rejection_reason_or_null": None,
            "carrier_rejection_reason_or_null": None if accepted_carrier else "movement_too_short",
            "state_before_sha256": state_before,
            "state_after_sha256": state_after,
            "activity_before_sha256": activity_before,
            "activity_after_sha256": activity_after,
            "carrier_before_sha256": carrier_before,
            "carrier_after_sha256": carrier_after,
        })
        state_before, activity_before, carrier_before = state_after, activity_after, carrier_after
    return tuple(rows)


def _independent_session(
    record: v1score.InputRecord, *, budget: int, system: str, r2: float,
) -> score.IndependentSessionScore:
    record_payload = record.payload()
    key = str(budget)
    query_ids = tuple(record_payload["query_trial_ids_by_budget"][key])
    transitions = _transition_rows(budget=budget, query_ids=query_ids) if system == plan.SYSTEM_CDMD else ()
    carrier_committed = sum(bool(row["carrier_transition_committed"]) for row in transitions)
    rejected = sum(not bool(row["carrier_transition_committed"]) for row in transitions)
    base = v1score.SessionScore(
        session=record.session, n_windows=3, r2=r2,
        prediction_sha256=_sha(f"{system}:{record.session}:M{budget}:pred"),
        input_record_sha256=score._digest(record_payload),
        model_state_before_sha256=_sha("sealed-state"), model_state_after_sha256=_sha("sealed-state"),
        initial_carrier_sha256=_sha(f"{record.session}:M{budget}:carrier"),
        group_assignment_sha256=_sha(f"{record.session}:M{budget}:groups"),
        group_valid_mask_sha256=str(record_payload["raw_m30_t4_axis_proof"]["valid_mask_sha256"]),
        initial_activity_sha256=_sha(f"{record.session}:M{budget}:activity"),
        support_trial_ids_sha256=str(record_payload["support_trial_ids_sha256_by_budget"][key]),
        raw_m30_t4_axis_proof_sha256=score._digest(record_payload["raw_m30_t4_axis_proof"]),
        sealed_normalizer_sha256=v1plan.SEALED_OLS_NORMALIZER_SHA256,
        sealed_model_load_proof_sha256=_sha("sealed-strict-load"),
        target_last_bin_sha256=str(record_payload["target_last_bin_sha256_by_budget"][key]),
        valid_mask_sha256=str(record_payload["valid_last_bin_mask_sha256_by_budget"][key]),
        valid_last_bin_count=3, activity_fifo_capacity=plan.FIFO_CAPACITY[budget],
        accepted_updates=carrier_committed * plan.GROUP_COUNT if system == plan.SYSTEM_CDMD else 0,
        rejected_updates=(
            {"movement_too_short": rejected * plan.GROUP_COUNT}
            if system == plan.SYSTEM_CDMD and rejected else {}
        ),
        group_forward_count=8 if system == plan.SYSTEM_CDMD else 0,
        full_system_forward_count=4, dropout_calls=0, target_label_state_uses=0,
    )
    return score.IndependentSessionScore(
        base=base, transition_records=transitions,
        activity_transition_committed_count=len(transitions),
        activity_fifo_changed_count=sum(bool(row["activity_fifo_changed"]) for row in transitions),
        carrier_transition_committed_count=carrier_committed,
        activity_rejection_counts={},
        carrier_rejection_counts={"movement_too_short": rejected} if rejected else {},
    )


def _cells(
    authority: v1score.InputAuthority, *, budget: int, delta: float,
    input_authority_sha256: str,
) -> tuple[score.CellEvidence, ...]:
    payload = authority.payload(identity=_identity())
    rows = tuple(authority.records)
    result: list[score.CellEvidence] = []
    for surface in plan.SURFACES:
        surface_rows = tuple(row for row in rows if row.surface == surface)
        sealed = tuple(_independent_session(row, budget=budget, system=plan.SYSTEM_SEALED, r2=0.20) for row in surface_rows)
        cdmd = tuple(_independent_session(row, budget=budget, system=plan.SYSTEM_CDMD, r2=0.20 + delta) for row in surface_rows)
        result.extend((
            score.CellEvidence(surface, budget, plan.SYSTEM_SEALED, input_authority_sha256,
                               v1plan.SEALED_CELL_D_SWA_SHA256, sealed, _resources()),
            score.CellEvidence(surface, budget, plan.SYSTEM_CDMD, input_authority_sha256,
                               v1plan.SEALED_CELL_D_SWA_SHA256, cdmd, _resources()),
        ))
    # Force codec construction here, so an accidental first-30 query or
    # collapsed transition cannot hide until the lifecycle assertion below.
    assert all(cell.payload(input_payload=payload)["budget"] == budget for cell in result)
    return tuple(result)


def _synthetic_v5_semantic_graph() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    roster = [f"synthetic-{index:02d}" for index in range(27)]
    independent = dict(score._V3_INDEPENDENT_ACTIVITY_CONTRACT)
    binding = {
        "schema": "causal_dual_memory_cell_d_source_execution_v5_binding",
        "synthetic": True,
        "v3_independent_activity_contract": dict(independent),
    }
    identity = {"closure": {"closure_sha256": plan.SOURCE_GATE_CLOSURE_SHA256}}
    attempt = {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v5", "status": "ATTEMPT_RESERVED",
        "source_only": True, "within_external_formal_target_forbidden": True,
        "target_optimizer_backward_update": 0, "source_resolved_or_opened": False,
        "checkpoint_opened": False, "cuda_initialized": False, "identity": identity,
        "v5_binding": binding,
    }
    launch = {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v5", "status": "LAUNCHED",
        "attempt_sha256": plan.SOURCE_GATE_EXPECTED_SHAS["attempt.json"],
        "launch_closure_sha256": plan.SOURCE_GATE_CLOSURE_SHA256, "source_only": True,
        "target_optimizer_backward_update": 0, "identity": identity, "v5_binding": binding,
    }
    source = {
        "strict_train_roster": roster, "source_only": True, "v5_binding": binding,
        "independent_activity_contract": independent,
        "access": {"within_opened": False, "external_opened": False, "formal_opened": False,
                   "target_opened": False, "optimizer_steps": 0, "backward_calls": 0, "parameter_updates": 0},
    }
    evidence: dict[str, dict[str, object]] = {}
    for budget in plan.BUDGETS:
        for index, session in enumerate(roster):
            passed = not (budget == 4 and index == 26)
            evidence[f"budget_m{budget}__{session}.json"] = {
                "v5_binding": binding, "independent_activity_contract": independent,
                # Real source-gate rows retain V1's fixed-pool status and a
                # separate exact ``pass`` boolean; they are not synthetic
                # PASS/NONPASS string rows.
                "status": "COMPLETE_FIXED_POOL", "pass": passed,
            }
        row_hashes = [score._digest(evidence[f"budget_m{budget}__{session}.json"]) for session in roster]
        passed = plan.SOURCE_GATE_EXPECTED_PASSING_SESSION_COUNTS[budget]
        threshold = plan.SOURCE_GATE_BREADTH_MIN_PASSING_SESSIONS
        evidence[f"budget_m{budget}_aggregate.json"] = {
            "schema": "causal_dual_memory_cell_d_source_execution_budget_aggregate_v1", "budget": budget,
            "strict_source_session_count": plan.SOURCE_GATE_STRICT_SOURCE_SESSION_TOTAL,
            "session_body_sha256s": row_hashes,
            "passing_session_count": passed, "breadth_min_passing_sessions": threshold,
            "breadth_pass": passed >= threshold, "source_only": True,
            "v5_binding": binding, "independent_activity_contract": independent,
        }
    terminal = {
        "schema": "causal_dual_memory_cell_d_source_execution_terminal_v5", "status": plan.SOURCE_GATE_STATUS,
        "attempt_sha256": plan.SOURCE_GATE_EXPECTED_SHAS["attempt.json"],
        "launch_sha256": plan.SOURCE_GATE_EXPECTED_SHAS["launch.json"],
        "source_authority_sha256": plan.SOURCE_GATE_EXPECTED_SHAS["source_authority.json"],
        "launch_closure_sha256": plan.SOURCE_GATE_CLOSURE_SHA256,
        "final_closure_sha256": plan.SOURCE_GATE_CLOSURE_SHA256, "source_only": True,
        "target_optimizer_backward_update": 0, "v5_binding": binding, "identity": identity,
        "evidence_sha256s": {name: score._digest(row) for name, row in evidence.items()},
    }
    return attempt, launch, source, terminal, evidence


def test_v5_static_profile_matrix_and_public_cli_are_inert(tmp_path: Path) -> None:
    dry = plan.dry_plan()
    assert dry["workorder_sha256"] == plan.WORKORDER_SHA256
    assert plan.score_matrix() == tuple(
        (budget, surface, system)
        for budget in (30, 10, 4) for surface in plan.SURFACES for system in plan.SYSTEMS
    )
    assert dry["score_spec"]["matrix_cell_count"] == 12
    assert dry["source_gate_predecessor"]["exact_leaves"] == 176
    assert dry["source_gate_predecessor"]["required_pass_total"] == {
        "30": {"passing_session_count": 27, "strict_source_session_count": 27},
        "10": {"passing_session_count": 27, "strict_source_session_count": 27},
        "4": {"passing_session_count": 26, "strict_source_session_count": 27},
    }
    assert dry["source_gate_predecessor"]["breadth_min_passing_sessions"] == 14
    assert plan.ROUTE_PROFILE is not v1plan.V1_ROUTE_PROFILE
    assert plan.ROUTE_PROFILE.source_gate is plan.SOURCE_GATE_CONTRACT
    closure = plan.implementation_closure(Path.cwd()).payload()
    assert len(closure["paths"]) == len(plan.IMPLEMENTATION_PATHS)
    assert plan.validate_implementation_closure(closure) == closure
    drifted_rows = {row["path"]: row["sha256"] for row in closure["paths"]}
    drifted_rows.pop("tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v5.py")
    with pytest.raises(plan.V5PlanError, match="topology"):
        plan.ImplementationClosure(drifted_rows).payload()
    with pytest.raises(plan.V5PlanError, match="terminal identity"):
        plan.ScoreIdentity(closure, source_gate_terminal_sha256=_sha("forged-source-terminal")).payload()
    script = Path("tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v5.py").absolute()
    code = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('candidate',{str(script)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main([]);"
        "print('TORCH='+str('torch' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=Path.cwd(), text=True, capture_output=True, check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPATH": str(Path.cwd() / "tfpd_exploration")},
    )
    assert '"no_torch_import":true' in completed.stdout
    assert "TORCH=False" in completed.stdout
    denied = subprocess.run([sys.executable, str(script), "--execute"], cwd=Path.cwd(), text=True,
                            capture_output=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    assert denied.returncode != 0 and "both --execute" in denied.stderr


def test_v1_default_profile_and_coupled_runtime_are_not_repointed() -> None:
    assert v1plan.V1_ROUTE_PROFILE.payload() == {
        "route": "causal_dual_memory_cell_d_score_v1",
        "authority_root_relative": v1plan.AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": v1plan.SCORE_ROOT_RELATIVE,
        "source_gate": v1plan.V1_SOURCE_GATE_CONTRACT.payload(),
    }
    assert v1plan.V1_SOURCE_GATE_CONTRACT.root_relative.endswith("source_gate_v2")
    historical = inspect.getsource(v1physical.ReviewedCDMScoreRuntime._make_dual_memory)
    successor = inspect.getsource(physical.V5ReviewedCDMScoreRuntime._make_dual_memory)
    assert "CausalDualMemory" in historical and "IndependentActivity" not in historical
    assert "IndependentActivityCausalDualMemory" in successor
    assert "commit_independent" in inspect.getsource(physical.V5ReviewedCDMScoreRuntime._commit_completed_transition)


def test_v5_semantic_gate_validator_requires_the_v5_contract_breadth_and_retained_m4_failure() -> None:
    attempt, launch, source, terminal, evidence = _synthetic_v5_semantic_graph()
    # The immutable V5 producer places this canonical contract only under the
    # v5 binding in attempt/launch/terminal.  Source/evidence own their
    # top-level copies, so no scorer may invent a top-level attempt field.
    assert "independent_activity_contract" not in attempt
    assert "independent_activity_contract" not in launch
    assert "independent_activity_contract" not in terminal
    assert score._validate_v5_source_gate_semantics(
        attempt, launch, source, terminal, evidence, plan.SOURCE_GATE_CONTRACT,
    ) == tuple(source["strict_train_roster"])
    repaired = {name: dict(row) for name, row in evidence.items()}
    repaired["budget_m4__synthetic-26.json"]["pass"] = True
    repaired["budget_m4_aggregate.json"]["session_body_sha256s"] = [
        score._digest(repaired[f"budget_m4__synthetic-{index:02d}.json"])
        for index in range(27)
    ]
    terminal_repaired = dict(terminal)
    terminal_repaired["evidence_sha256s"] = {name: score._digest(row) for name, row in repaired.items()}
    with pytest.raises(score.V5ScoreError, match="sole M4"):
        score._validate_v5_source_gate_semantics(
            attempt, launch, source, terminal_repaired, repaired, plan.SOURCE_GATE_CONTRACT,
        )

    # The accepted source producer uses a fixed threshold 14, independently
    # of the literal retained pass/total counts.  All four fields must remain
    # exact even if a forger recomputes every affected body and terminal map.
    assert [
        (
            evidence[f"budget_m{budget}_aggregate.json"]["passing_session_count"],
            evidence[f"budget_m{budget}_aggregate.json"]["breadth_min_passing_sessions"],
            evidence[f"budget_m{budget}_aggregate.json"]["breadth_pass"],
        )
        for budget in plan.BUDGETS
    ] == [(27, 14, True), (27, 14, True), (26, 14, True)]

    def rehashed_terminal(rows: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
        replacement = dict(terminal)
        replacement["evidence_sha256s"] = {name: score._digest(row) for name, row in rows.items()}
        return replacement

    for field, value in (
        ("passing_session_count", 26),
        ("strict_source_session_count", 26),
        ("breadth_min_passing_sessions", 27),
        ("breadth_pass", False),
    ):
        forged_breadth = {name: dict(row) for name, row in evidence.items()}
        forged_breadth["budget_m30_aggregate.json"][field] = value
        with pytest.raises(score.V5ScoreError, match="M30 aggregate/breadth"):
            score._validate_v5_source_gate_semantics(
                attempt, launch, source, rehashed_terminal(forged_breadth), forged_breadth,
                plan.SOURCE_GATE_CONTRACT,
            )
    forged_nested_attempt = dict(attempt)
    forged_binding = dict(binding := attempt["v5_binding"])
    forged_contract = dict(forged_binding["v3_independent_activity_contract"])
    forged_contract["m10_activity_fifo_capacity"] = 19
    forged_binding["v3_independent_activity_contract"] = forged_contract
    forged_nested_attempt["v5_binding"] = forged_binding
    with pytest.raises(score.V5ScoreError, match="nested independent"):
        score._validate_v5_source_gate_semantics(
            forged_nested_attempt, launch, source, terminal, evidence, plan.SOURCE_GATE_CONTRACT,
        )
    forged_source = dict(source)
    forged_source_contract = dict(independent := source["independent_activity_contract"])
    forged_source_contract["m4_activity_fifo_capacity"] = 25
    forged_source["independent_activity_contract"] = forged_source_contract
    with pytest.raises(score.V5ScoreError, match="source-authority roster/contract"):
        score._validate_v5_source_gate_semantics(
            attempt, launch, forged_source, terminal, evidence, plan.SOURCE_GATE_CONTRACT,
        )
    forged_evidence = {name: dict(row) for name, row in evidence.items()}
    aggregate_name = "budget_m10_aggregate.json"
    forged_evidence_contract = dict(forged_evidence[aggregate_name]["independent_activity_contract"])
    forged_evidence_contract["m30_activity_fifo_capacity"] = 1
    forged_evidence[aggregate_name]["independent_activity_contract"] = forged_evidence_contract
    forged_terminal = dict(terminal)
    forged_terminal["evidence_sha256s"] = {name: score._digest(row) for name, row in forged_evidence.items()}
    with pytest.raises(score.V5ScoreError, match="independent activity evidence"):
        score._validate_v5_source_gate_semantics(
            attempt, launch, source, forged_terminal, forged_evidence, plan.SOURCE_GATE_CONTRACT,
        )
    with pytest.raises(score.V5ScoreError, match="contract identity"):
        score._validate_v5_source_gate_semantics(
            attempt, launch, source, terminal, evidence,
            v1plan.CompletedSourceGateContract(
                route="forged", root_relative="synthetic", closure_sha256=_sha("closure"),
                fixed_body_sha256s={name: _sha(name) for name in plan.SOURCE_GATE_EXPECTED_SHAS},
                expected_json_bodies=88, expected_leaves=176, terminal_status=plan.SOURCE_GATE_STATUS,
                binding_schema="forged",
            ),
        )


def test_v5_preflight_accepts_only_the_literal_v5_source_profile() -> None:
    identity = _identity()
    fixed = _fixed_authority()
    with pytest.raises(score.V5ScoreError, match="exact typed V5"):
        score.build_target_free_preflight(
            root=Path("/synthetic"), identity=identity,
            source_gate=_source_gate_binding(contract=v1plan.V1_SOURCE_GATE_CONTRACT), fixed_authority=fixed,
        )
    accepted = score.build_target_free_preflight(
        root=Path("/synthetic"), identity=identity, source_gate=_source_gate_binding(), fixed_authority=fixed,
    )
    assert accepted["source_gate"]["root_relative"] == plan.SOURCE_GATE_ROOT_RELATIVE
    # Authority-root reservation itself is an output boundary: it cannot be
    # reached with a caller's unvalidated source-gate summary and must receive
    # the exact typed identity used for its held V5 reload.
    assert "identity" in inspect.signature(score.reserve_authority_artifact).parameters
    reservation_source = inspect.getsource(score.reserve_authority_artifact)
    assert "validate_completed_source_gate" in reservation_source
    assert "assert_fresh_prospective_root" in reservation_source


def _write_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {Path(name).name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def test_shared_held_fd_loader_rejects_synthetic_88_pair_mode_or_topology_drift(tmp_path: Path) -> None:
    """Exercise the route-independent exact 88-pair loader without real receipts."""
    root_relative = "synthetic_completed_gate"
    directory = tmp_path / root_relative
    directory.mkdir()
    roster = tuple(f"synthetic-{index:02d}" for index in range(27))
    evidence: dict[str, dict[str, object]] = {
        f"budget_m{budget}__{session}.json": {"session": session, "budget": budget}
        for budget in plan.BUDGETS for session in roster
    }
    for budget in plan.BUDGETS:
        evidence[f"budget_m{budget}_aggregate.json"] = {"budget": budget, "aggregate": True}
    evidence_shas = {name: _write_pair(directory, name, row) for name, row in evidence.items()}
    attempt = {"attempt": True}
    launch = {"launch": True}
    source = {"source": True}
    terminal = {"status": "PASS_SYNTHETIC", "evidence_sha256s": evidence_shas}
    primary = {
        "attempt.json": _write_pair(directory, "attempt.json", attempt),
        "launch.json": _write_pair(directory, "launch.json", launch),
        "source_authority.json": _write_pair(directory, "source_authority.json", source),
        "budget_m30_aggregate.json": evidence_shas["budget_m30_aggregate.json"],
        "budget_m10_aggregate.json": evidence_shas["budget_m10_aggregate.json"],
        "budget_m4_aggregate.json": evidence_shas["budget_m4_aggregate.json"],
    }
    primary["terminal.json"] = _write_pair(directory, "terminal.json", terminal)
    contract = v1plan.CompletedSourceGateContract(
        route="synthetic", root_relative=root_relative, closure_sha256=_sha("closure"),
        fixed_body_sha256s=primary, expected_json_bodies=88, expected_leaves=176,
        terminal_status="PASS_SYNTHETIC", binding_schema="synthetic_binding",
    )

    def semantic(
        _attempt: Mapping[str, object], _launch: Mapping[str, object], _source: Mapping[str, object],
        _terminal: Mapping[str, object], loaded: Mapping[str, Mapping[str, object]],
        received: v1plan.CompletedSourceGateContract,
    ) -> tuple[str, ...]:
        assert received is contract and len(loaded) == 84
        return roster

    bound = v1score.load_completed_source_gate_contract(tmp_path, contract=contract, semantic_validator=semantic)
    assert bound.payload()["exact_leaf_count"] == 176
    os.chmod(directory / "budget_m10__synthetic-00.json", 0o644)
    with pytest.raises(v1score.ScoreError, match="mode"):
        v1score.load_completed_source_gate_contract(tmp_path, contract=contract, semantic_validator=semantic)
    os.chmod(directory / "budget_m10__synthetic-00.json", 0o444)
    (directory / "extra.json").write_text("{}")
    with pytest.raises(v1score.ScoreError, match="topology"):
        v1score.load_completed_source_gate_contract(tmp_path, contract=contract, semantic_validator=semantic)


def test_v5_independent_transition_codec_preserves_separate_activity_and_carrier_facts() -> None:
    authority, _fixed = _input_authority()
    record = authority.records[0]
    cdmd = _independent_session(record, budget=10, system=plan.SYSTEM_CDMD, r2=0.31)
    payload = cdmd.payload(budget=10, system=plan.SYSTEM_CDMD, query_trial_ids=record.query_trial_ids_by_budget["10"])
    assert "accepted_updates" not in payload and "rejected_updates" not in payload
    assert payload["activity_transition_committed_count"] == 2
    assert payload["carrier_transition_committed_count"] == 1
    assert score._session_from_payload(payload, query_trial_ids=record.query_trial_ids_by_budget["10"]).payload(
        budget=10, system=plan.SYSTEM_CDMD, query_trial_ids=record.query_trial_ids_by_budget["10"],
    ) == payload
    collapsed = dict(payload)
    rows = [dict(row) for row in payload["transition_records"]]
    rows[0]["carrier_transition_committed"] = True
    collapsed["transition_records"] = rows
    with pytest.raises(score.V5ScoreError, match="carrier"):
        score._session_from_payload(collapsed, query_trial_ids=record.query_trial_ids_by_budget["10"])
    forged_reason = dict(payload)
    forged_reason_rows = [dict(row) for row in payload["transition_records"]]
    forged_reason_rows[0]["carrier_rejection_reason_or_null"] = "invented_reason"
    forged_reason["transition_records"] = forged_reason_rows
    forged_reason["carrier_rejection_counts"] = {"invented_reason": 1}
    with pytest.raises(score.V5ScoreError, match="rejection reason"):
        score._session_from_payload(forged_reason, query_trial_ids=record.query_trial_ids_by_budget["10"])
    reordered = dict(payload)
    reordered_rows = [dict(row) for row in payload["transition_records"]]
    reordered_rows.reverse()
    reordered["transition_records"] = reordered_rows
    with pytest.raises(score.V5ScoreError, match="trial-ID/order|chain"):
        score._session_from_payload(reordered, query_trial_ids=record.query_trial_ids_by_budget["10"])
    m30 = _independent_session(record, budget=30, system=plan.SYSTEM_CDMD, r2=0.31)
    m30_payload = m30.payload(budget=30, system=plan.SYSTEM_CDMD, query_trial_ids=record.query_trial_ids_by_budget["30"])
    assert all(row["activity_before_sha256"] == row["activity_after_sha256"] for row in m30_payload["transition_records"])
    forged_m30 = dict(m30_payload)
    forged_rows = [dict(row) for row in m30_payload["transition_records"]]
    forged_rows[0]["activity_after_sha256"] = _sha("m30-mutated-activity")
    forged_m30["transition_records"] = forged_rows
    with pytest.raises(score.V5ScoreError, match="FIFO|M30"):
        score._session_from_payload(forged_m30, query_trial_ids=record.query_trial_ids_by_budget["30"])


def test_v5_cpu_independent_memory_semantics_cover_rejection_acceptance_invalid_and_m30() -> None:
    """Use the accepted core directly; the V5 physical hook calls this exact API."""
    import math
    import numpy as np
    from src.causal_dual_memory_cell_d_v1 import core

    session = "v5-core-session"
    channels = np.arange(8, dtype=np.int64)
    channel_sha = core.channel_order_digest(channels)
    directions = np.asarray((0, 2, 4, 3, 3, 3, 5, 7, 1, 6, 3), dtype=np.int64)
    theta = np.asarray([core.CANONICAL_DIRECTIONS_RAD[index] for index in directions], dtype=np.float64)
    a = np.linspace(0.7, 1.4, 8, dtype=np.float64)
    c = np.linspace(-0.8, 0.5, 8, dtype=np.float64)
    b = np.linspace(4.0, 5.0, 8, dtype=np.float64)
    rates = b[None, :] + np.cos(theta)[:, None] * a[None, :] + np.sin(theta)[:, None] * c[None, :]
    raw_t4 = np.column_stack((a, c, np.hypot(a, c), b)).astype(np.float64)
    query_rate = rates[3]

    def memory(budget: int) -> core.IndependentActivityCausalDualMemory:
        support_indices = np.resize(np.arange(rates.shape[0], dtype=np.int64), budget)
        config = core.CDMDConfig(
            support_budget_m=budget, activity_fifo_capacity=plan.FIFO_CAPACITY[budget],
            dt=1.0, minimum_movement_bins=2, minimum_displacement=0.01,
            minimum_mean_speed=0.01, max_canonical_distance_rad=math.pi / 8.0,
            max_group_direction_disagreement_rad=math.pi / 8.0,
            minimum_accepted_evidence=3, active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        carrier = core.CarrierMemory.from_support_trials(
            initial_raw_t4=raw_t4,
            channel_ids=channels, support_trial_rates=rates[support_indices],
            support_direction_indices=directions[support_indices], config=config,
        )
        support = tuple(
            core.B3SInterpolatedSpikeCountTrial(
                activity=np.full((100, 8), float(index), dtype=np.float32), session_id=session,
                trial_id=f"support-{index}", channel_order_sha256=channel_sha,
            ) for index in range(budget)
        )
        return core.IndependentActivityCausalDualMemory(
            activity=core.ActivityMemory.initialize(support, channel_ids=channels, fifo_capacity=plan.FIFO_CAPACITY[budget]),
            carrier=carrier,
        )

    def b3s(trial_id: str) -> core.B3SInterpolatedSpikeCountTrial:
        return core.B3SInterpolatedSpikeCountTrial(
            activity=np.repeat(query_rate[None, :], 100, axis=0), session_id=session, trial_id=trial_id,
            channel_order_sha256=channel_sha,
        )

    def native(trial_id: str) -> core.NativeRewardedTrialSpikeCounts:
        return core.NativeRewardedTrialSpikeCounts(
            counts=np.repeat((query_rate * 0.020)[None, :], 7, axis=0), session_id=session, trial_id=trial_id,
            channel_order_sha256=channel_sha, rewarded_interval_start_bin=10, rewarded_interval_stop_bin=17,
        )

    def predictions(trial_id: str, *, valid: bool) -> tuple[core.CompletedVelocityPrediction, ...]:
        validity = core.VelocityValidityEvidence(
            valid_mask=np.ones(6, dtype=np.bool_) if valid else np.zeros(6, dtype=np.bool_),
            session_id=session, trial_id=trial_id, prediction_interval_start_bin=20, prediction_interval_stop_bin=26,
        )
        theta = core.CANONICAL_DIRECTIONS_RAD[3]
        velocity = np.repeat([[math.cos(theta), math.sin(theta)]], 6, axis=0)
        return tuple(core.CompletedVelocityPrediction(velocity=velocity, validity=validity) for _ in range(4))

    rejected_memory = memory(10)
    before_carrier = rejected_memory.state.carrier.digest
    rejected = rejected_memory.commit_independent(rejected_memory.observe_completed_trial(
        b3s_trial_activity=b3s("rejected"), carrier_trial_counts=native("rejected"),
        complementary_predictions=predictions("rejected", valid=False),
    ))
    assert rejected.activity_transition_committed is True and rejected.activity_fifo_changed is True
    assert rejected.carrier_transition_committed is False
    assert rejected.carrier_before_sha256 == rejected.carrier_after_sha256 == before_carrier
    accepted_memory = memory(4)
    accepted = accepted_memory.commit_independent(accepted_memory.observe_completed_trial(
        b3s_trial_activity=b3s("accepted"), carrier_trial_counts=native("accepted"),
        complementary_predictions=predictions("accepted", valid=True),
    ))
    assert accepted.activity_transition_committed is True and accepted.carrier_transition_committed is True
    assert accepted.carrier_before_sha256 != accepted.carrier_after_sha256
    invalid_memory = memory(4)
    invalid = invalid_memory.commit_independent(invalid_memory.observe_completed_trial(
        b3s_trial_activity=object(), carrier_trial_counts=native("invalid"),
        complementary_predictions=predictions("invalid", valid=True),
    ))
    assert invalid.activity_transition_committed is False and invalid.carrier_transition_committed is False
    m30_memory = memory(30)
    before_activity = m30_memory.state.activity.digest
    m30 = m30_memory.commit_independent(m30_memory.observe_completed_trial(
        b3s_trial_activity=b3s("m30"), carrier_trial_counts=native("m30"),
        complementary_predictions=predictions("m30", valid=False),
    ))
    assert m30.activity_transition_committed is True and m30.activity_fifo_changed is False
    assert m30.activity_before_sha256 == m30.activity_after_sha256 == before_activity
    assert m30_memory.state.committed_query_trials == 1 and m30_memory.state.activity.query_count == 0


class _Artifact:
    topology = score.SCORE_TOPOLOGY

    def __init__(self, *, fail_group: bool = False, event_log: list[str] | None = None) -> None:
        self.items: dict[str, bytes] = {}
        self.fail_group = fail_group
        self.event_log = [] if event_log is None else event_log

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        if name in self.items:
            raise RuntimeError("synthetic artifact collision")
        body = score._json(payload)
        self.items[name] = body
        self.event_log.append(f"publish:{name}")
        return hashlib.sha256(body).hexdigest()

    def publish_group(self, bodies: Mapping[str, bytes], *, post_publish=None):
        if self.fail_group:
            raise RuntimeError("synthetic atomic publication failure")
        if any(name in self.items for name in bodies):
            raise RuntimeError("synthetic group collision")
        digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        if post_publish is not None:
            post_publish(bodies, digests)
        self.items.update(bodies)
        return digests

    def reload_json(self, name: str, expected_sha256: str | None = None):
        body = self.items[name]
        if expected_sha256 is not None and hashlib.sha256(body).hexdigest() != expected_sha256:
            raise RuntimeError("synthetic reload digest drift")
        return json.loads(body)

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        body = self.items[name]
        if expected_sha256 is not None and hashlib.sha256(body).hexdigest() != expected_sha256:
            raise RuntimeError("synthetic reload digest drift")
        return body

    def has_name(self, name: str) -> bool:
        return name in self.items


class _MockRuntime:
    def __init__(self, authority: v1score.InputAuthority, *, event_log: list[str] | None = None) -> None:
        self.authority = authority
        self.calls: list[str] = []
        self.event_log = [] if event_log is None else event_log

    def _record(self, event: str) -> None:
        self.calls.append(event)
        self.event_log.append(event)

    def preflight(self, *, root: Path, identity: plan.ScoreIdentity):
        self._record("preflight")
        return {"target_paths_resolved": False, "target_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

    def prepare(self, *, root: Path, identity: plan.ScoreIdentity):
        self._record("prepare")
        return self

    def materialize_inputs(self, runtime, *, identity: plan.ScoreIdentity, evaluation_authority: v1score.FixedEvaluationAuthority):
        self._record("materialize")
        return self.authority

    def score_budget(self, runtime, *, budget: int, input_authority_sha256: str, identity: plan.ScoreIdentity):
        self._record(f"budget{budget}")
        # Deliberately negative M30: the V5 matrix must still run M10 and M4.
        return _cells(self.authority, budget=budget, delta=(-0.02 if budget == 30 else 0.06),
                      input_authority_sha256=input_authority_sha256)

    def revalidate(self, runtime, *, root: Path, identity: plan.ScoreIdentity) -> None:
        self._record("revalidate")

    def failure_progress(self, runtime):
        return {
            "within_assets_opened": False, "external_assets_opened": False, "checkpoint_opened": False,
            "cuda_initialized": False, "full_system_forward_count": 0, "group_forward_count": 0,
        }

    def close(self, runtime) -> None:
        self._record("close")


def test_v5_shared_lifecycle_runs_all_budgets_after_negative_m30_and_is_atomic(tmp_path: Path) -> None:
    identity = _identity()
    binding = _source_gate_binding()
    authority, fixed = _input_authority()
    preflight = score.build_target_free_preflight(root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed)
    pre_sha = score._digest(preflight)
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = score._digest(authorization)
    root_capability = v1score._issue_root_publication_capability()
    capability = v1score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_capability,
    )
    hooks = replace(
        score.V5_LIFECYCLE_HOOKS,
        implementation_closure=lambda _root: identity.payload()["closure"],
        validate_source_gate=lambda _root: binding,
        assert_fresh_score_root=lambda _root: None,
    )
    event_log: list[str] = []
    runtime = _MockRuntime(authority, event_log=event_log)
    artifact = _Artifact(event_log=event_log)
    result = v1score.run_profiled_score_lifecycle(
        tmp_path, identity=identity, capability=capability, backend=runtime, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization, hooks=hooks,
    )
    assert result["verdict"] == "ADVANCE_SHORT_BUDGET"
    # The one shared event log proves the durable attempt pair exists before
    # any target/model/CUDA-reachable prepare or input-materialization work.
    # ``runtime.calls`` alone cannot establish that publication boundary.
    assert event_log[:4] == ["preflight", "publish:attempt.json", "prepare", "materialize"]
    assert event_log.index("publish:attempt.json") < event_log.index("prepare") < event_log.index("materialize")
    assert runtime.calls[:5] == ["preflight", "prepare", "materialize", "budget30", "budget10"]
    assert "budget4" in runtime.calls and "score.json" in artifact.items and "terminal.json" in artifact.items
    assert "failure.json" not in artifact.items
    result_score = json.loads(artifact.items["score.json"])
    assert result_score["budget_execution_order"] == [30, 10, 4]
    assert result_score["cell_execution_order"] == [
        {"budget": budget, "surface": surface, "system": system}
        for budget in plan.BUDGETS for surface in plan.SURFACES for system in plan.SYSTEMS
    ]
    assert result_score["budget_gates"]["30"]["external_gate_pass"] is False
    failing_runtime = _MockRuntime(authority)
    failing_artifact = _Artifact(fail_group=True)
    with pytest.raises(RuntimeError, match="atomic publication"):
        v1score.run_profiled_score_lifecycle(
            tmp_path, identity=identity, capability=capability, backend=failing_runtime, artifact=failing_artifact,
            official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
            preflight=preflight, authorization=authorization, hooks=hooks,
        )
    assert "attempt.json" in failing_artifact.items and "failure.json" in failing_artifact.items
    assert "score.json" not in failing_artifact.items and "terminal.json" not in failing_artifact.items


def test_v5_physical_seam_is_inherited_and_only_changes_loader_memory_commit_and_codec() -> None:
    assert issubclass(physical.V5ReviewedCDMScoreRuntime, v1physical.ReviewedCDMScoreRuntime)
    backend = physical.build_reviewed_physical_backend(
        root=Path.cwd(), selected_device_profile=v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
    )
    assert isinstance(backend, v1physical.PhysicalCDMDMatchedScoreBackend)
    implementation = inspect.getsource(physical)
    assert "V5OneShotFinalizedRowExecutor" in implementation
    assert "commit_independent" in implementation
    assert "type(state.executor) is not executor_type" in implementation
    assert "_closure_bound_runtime_module" in implementation
    assert "V5 runtime module bytes differ from durable closure" in implementation
    assert "sys.modules" not in implementation and "monkeypatch" not in implementation
    assert "tuple(ordered[30:])" in inspect.getsource(v1physical.ReviewedCDMScoreRuntime._parse_session)
    runtime = physical.V5ReviewedCDMScoreRuntime(
        root=Path.cwd(), selected_device_profile=v1plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
    )
    relative = physical.V5ReviewedCDMScoreRuntime._V5_RUNTIME_MODULES[
        "src.causal_dual_memory_cell_d_v1.source_execute_v5"
    ]
    runtime._v5_runtime_sha256_by_path = {relative: plan._read_regular_no_follow(Path.cwd() / relative)}
    exact = SimpleNamespace(
        __name__="src.causal_dual_memory_cell_d_v1.source_execute_v5",
        __file__=str(Path.cwd() / relative),
    )
    assert runtime._closure_bound_runtime_module(
        exact, module_name=exact.__name__, relative=relative,
    ) is exact
    wrong_path = SimpleNamespace(__name__=exact.__name__, __file__=str(Path.cwd() / "wrong.py"))
    with pytest.raises(physical.PhysicalV5ScoreError, match="identity/path"):
        runtime._closure_bound_runtime_module(wrong_path, module_name=exact.__name__, relative=relative)
    runtime._v5_runtime_sha256_by_path = {relative: _sha("forged-runtime-byte-binding")}
    with pytest.raises(physical.PhysicalV5ScoreError, match="bytes differ"):
        runtime._closure_bound_runtime_module(exact, module_name=exact.__name__, relative=relative)
    # The V1 regression module may already have imported Torch in this shared
    # pytest process.  Constructing the deferred physical seam itself must
    # still leave CUDA untouched.
    if "torch" in sys.modules:
        assert sys.modules["torch"].cuda.is_initialized() is False


def test_torch_cpu_fact_does_not_initialize_cuda() -> None:
    import torch

    assert torch.cuda.is_initialized() is False
