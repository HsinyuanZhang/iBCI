"""Focused CPU/no-data gates for CDM-D matched-score V1.

These tests intentionally use only typed synthetic receipts and injected
runtime objects.  They never inspect source/target data, checkpoint tensors,
CUDA state, or a canonical result root.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Mapping

import pytest

from src.causal_dual_memory_cell_d_score_v1 import physical, plan, score


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity() -> plan.ScoreIdentity:
    rows = {path: _sha(path) for path in plan.IMPLEMENTATION_PATHS}
    rows[plan.WORKORDER_RELATIVE] = plan.WORKORDER_SHA256
    return plan.ScoreIdentity(plan.ImplementationClosure(rows).payload())


def _gate_binding() -> score.SourceGateBinding:
    roster = tuple(f"sub-C_ses-CO-synthetic-{index:02d}" for index in range(27))
    rows = dict(plan.SOURCE_GATE_EXPECTED_SHAS)
    rows.update({
        f"budget_m{budget}__{session}.json": _sha(f"{budget}:{session}")
        for budget in plan.BUDGETS for session in roster
    })
    return score.SourceGateBinding(
        directory_device=1, directory_inode=2, body_sha256s=rows,
        terminal_status=plan.SOURCE_GATE_STATUS,
        source_gate_closure_sha256=plan.SOURCE_GATE_CLOSURE_SHA256,
        strict_source_roster=roster,
    )


def _authority() -> score.FixedEvaluationAuthority:
    within = tuple(
        score.EvaluationAsset(plan.WITHIN, f"sub-C_ses-CO-w{index:02d}", f"within-{index}",
                              f"sub-C/sub-C_ses-CO-w{index:02d}_behavior+ecephys.nwb", 100 + index, _sha(f"w{index}"))
        for index in range(6)
    )
    external = tuple(
        score.EvaluationAsset(plan.EXTERNAL, f"sub-M_ses-CO-e{index:02d}", f"external-{index}",
                              f"sub-M/sub-M_ses-CO-e{index:02d}_behavior+ecephys.nwb", 200 + index, _sha(f"e{index}"))
        for index in range(15)
    )
    return score.FixedEvaluationAuthority(
        within, external, {"strict_manifest": {"sha256": _sha("manifest")}},
        {"manifest_sha256": _sha("paired-view")},
    )


def _record(
    surface: str, session: str, *, asset: score.EvaluationAsset | None = None,
) -> score.InputRecord:
    chronology = tuple(f"{session}:trial:{index}" for index in range(8))
    support = {
        "4": chronology[:4], "10": chronology[:1] * 0 + chronology[:4] + chronology[4:8][:0],
        "30": chronology[:4],
    }
    # Synthetic chronology has only eight rows; a literal M10/M30 support
    # cardinality is still required by the contract, so use long unique IDs
    # while retaining the same authoritatively ordered chronology below.
    chronology = tuple(f"{session}:trial:{index}" for index in range(32))
    support = {"4": chronology[:4], "10": chronology[:10], "30": chronology[:30]}
    query = {key: chronology[30:] for key in support}
    # There is exactly one raw M30 unit/channel/validity authority.  M4/M10
    # support carriers are selected ridge fits, not chronological raw pools.
    raw_axis = {
        "schema": "causal_dual_memory_cell_d_score_raw_t4_axis_v1",
        "budget": 30,
        "raw_t4_sha256": _sha(f"{session}:raw:m30"),
        "channel_order_sha256": _sha(f"{session}:channels:m30"),
        "valid_mask_sha256": _sha(f"{session}:valid:m30"),
        "source_unit_count": 8,
        "feature_group": "t4",
        "signal_view": "sua",
        "channel_ids_are_exact_int64_arange": True,
        "validity_rule": "raw_t4_modulation_m_gt_modulation_eps",
        "modulation_eps": 1e-6,
        "raw_before_normalization": True,
    }
    return score.InputRecord(
        surface=surface, session=session,
        asset_id=(f"asset:{session}" if asset is None else asset.asset_id),
        frozen_path=(f"asset/{session}_behavior+ecephys.nwb" if asset is None else asset.frozen_path),
        asset_bytes=(123 if asset is None else asset.bytes),
        asset_sha256=(_sha(session + "asset") if asset is None else asset.sha256),
        chronological_trial_ids=chronology,
        support_trial_ids_by_budget=support, query_trial_ids_by_budget=query,
        neural_sha256=_sha(session + "neural"), calibration_sha256=_sha(session + "calibration"),
        target_last_bin_sha256_by_budget={key: _sha(session + "target" + key) for key in support},
        valid_last_bin_mask_sha256_by_budget={key: _sha(session + "mask" + key) for key in support},
        valid_last_bin_count_by_budget={key: 3 for key in support},
        raw_m30_t4_axis_proof=raw_axis, theta_recovery_sha256=_sha(session + "theta"),
    )


def _session_score(record: score.InputRecord, *, budget: int, system: str, r2: float) -> score.SessionScore:
    payload = record.payload()
    key = str(budget)
    state = _sha("model-state")
    return score.SessionScore(
        session=record.session, n_windows=3, r2=r2, prediction_sha256=_sha(f"{system}:{record.session}:{budget}"),
        input_record_sha256=score._digest(payload), model_state_before_sha256=state, model_state_after_sha256=state,
        initial_carrier_sha256=_sha(f"carrier:{record.session}:{budget}"),
        group_assignment_sha256=_sha(f"groups:{record.session}:{budget}"),
        group_valid_mask_sha256=payload["raw_m30_t4_axis_proof"]["valid_mask_sha256"],
        initial_activity_sha256=_sha(f"activity:{record.session}:{budget}"),
        support_trial_ids_sha256=payload["support_trial_ids_sha256_by_budget"][key],
        raw_m30_t4_axis_proof_sha256=score._digest(payload["raw_m30_t4_axis_proof"]),
        sealed_normalizer_sha256=plan.SEALED_OLS_NORMALIZER_SHA256,
        sealed_model_load_proof_sha256=_sha("strict-sealed-load-proof"),
        target_last_bin_sha256=payload["target_last_bin_sha256_by_budget"][key],
        valid_mask_sha256=payload["valid_last_bin_mask_sha256_by_budget"][key], valid_last_bin_count=3,
        activity_fifo_capacity=plan.FIFO_CAPACITY[budget], accepted_updates=0 if system == plan.SYSTEM_SEALED else 4,
        rejected_updates={} if system == plan.SYSTEM_SEALED else {"insufficient_design_rank": 1},
        group_forward_count=0 if system == plan.SYSTEM_SEALED else 8,
        full_system_forward_count=2, dropout_calls=0, target_label_state_uses=0,
    )


def _resources() -> dict[str, object]:
    return {
        "runtime_environment": {
            **plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
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
        "full_and_group_forward_chunks": 4,
        "completed_query_trials": 2,
        "windows_or_trials_per_s": 1.0,
    }


def _cells(
    authority: score.InputAuthority, *, budget: int, delta: float = 0.04,
    input_authority_sha256: str | None = None,
) -> tuple[score.CellEvidence, ...]:
    input_payload = authority.payload(identity=_identity())
    rows_by_surface: dict[str, list[score.InputRecord]] = {surface: [] for surface in plan.SURFACES}
    for item in authority.records:
        rows_by_surface[item.surface].append(item)
    result: list[score.CellEvidence] = []
    input_sha = _sha("input-authority") if input_authority_sha256 is None else input_authority_sha256
    for surface in plan.SURFACES:
        sealed = tuple(_session_score(row, budget=budget, system=plan.SYSTEM_SEALED, r2=0.2) for row in rows_by_surface[surface])
        cdmd = tuple(_session_score(row, budget=budget, system=plan.SYSTEM_CDMD, r2=0.2 + delta) for row in rows_by_surface[surface])
        result.extend((
            score.CellEvidence(surface, budget, plan.SYSTEM_SEALED, input_sha, plan.SEALED_CELL_D_SWA_SHA256, sealed,
                               _resources()),
            score.CellEvidence(surface, budget, plan.SYSTEM_CDMD, input_sha, plan.SEALED_CELL_D_SWA_SHA256, cdmd,
                               _resources()),
        ))
    # The cell implementation cross-checks the actual argument SHA only; the
    # synthetic caller uses a deterministic SHA-shaped durable placeholder.
    return tuple(result)


def _input_authority() -> score.InputAuthority:
    authority = _authority()
    records = tuple(
        _record(asset.surface, asset.session, asset=asset)
        for asset in (*authority.within, *authority.external)
    )
    return score.InputAuthority(records, score._digest(authority.payload()))


def test_plan_is_static_and_exact_contracts_are_literal() -> None:
    dry = plan.dry_plan()
    assert dry["workorder_sha256"] == plan.WORKORDER_SHA256
    assert dry["score_spec"]["budgets"] == [30, 10, 4]
    assert dry["score_spec"]["systems"] == [plan.SYSTEM_SEALED, plan.SYSTEM_CDMD]
    assert dry["no_torch_import"] is True
    assert plan.validate_compatible_device_profile(plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])["logical_device"] == "cuda:0"
    with pytest.raises(plan.PlanError, match="exact reviewed authority"):
        plan.validate_compatible_device_profile({"cuda_visible_devices": "1"})


def test_input_authority_binds_exact_support_query_target_and_theta_rows() -> None:
    identity = _identity()
    authority = _input_authority()
    payload = authority.payload(identity=identity)
    assert len(payload["records"]) == 21
    first = payload["records"][0]
    assert first["support_trial_ids_by_budget"]["4"] != first["support_trial_ids_by_budget"]["10"]
    assert all(first["query_trial_ids_by_budget"][key] == first["chronological_trial_ids"][30:] for key in ("4", "10", "30"))
    assert all(
        max(first["chronological_trial_ids"].index(item) for item in first["support_trial_ids_by_budget"][key]) < 30
        for key in ("4", "10", "30")
    )
    assert first["target_last_bin_sha256_by_budget"]["30"]
    assert set(first["raw_m30_t4_axis_proof"]) >= {"budget", "raw_t4_sha256", "valid_mask_sha256"}
    assert first["raw_m30_t4_axis_proof"]["budget"] == 30
    assert "raw_t4_axis_proofs_by_budget" not in first
    corrupted = _record(plan.WITHIN, "sub-C_ses-CO-corrupt")
    bad = score.InputRecord(
        surface=corrupted.surface, session=corrupted.session, asset_id=corrupted.asset_id,
        frozen_path=corrupted.frozen_path, asset_bytes=corrupted.asset_bytes, asset_sha256=corrupted.asset_sha256,
        chronological_trial_ids=corrupted.chronological_trial_ids,
        support_trial_ids_by_budget=corrupted.support_trial_ids_by_budget,
        query_trial_ids_by_budget={
            **corrupted.query_trial_ids_by_budget,
            "4": (corrupted.chronological_trial_ids[29], *corrupted.query_trial_ids_by_budget["4"]),
        },
        neural_sha256=corrupted.neural_sha256, calibration_sha256=corrupted.calibration_sha256,
        target_last_bin_sha256_by_budget=corrupted.target_last_bin_sha256_by_budget,
        valid_last_bin_mask_sha256_by_budget=corrupted.valid_last_bin_mask_sha256_by_budget,
        valid_last_bin_count_by_budget=corrupted.valid_last_bin_count_by_budget,
        raw_m30_t4_axis_proof=corrupted.raw_m30_t4_axis_proof,
        theta_recovery_sha256=corrupted.theta_recovery_sha256,
    )
    with pytest.raises(score.ScoreError, match="causal support/query partition"):
        bad.payload()
    all_nonsupport = score.InputRecord(
        surface=corrupted.surface, session=corrupted.session, asset_id=corrupted.asset_id,
        frozen_path=corrupted.frozen_path, asset_bytes=corrupted.asset_bytes, asset_sha256=corrupted.asset_sha256,
        chronological_trial_ids=corrupted.chronological_trial_ids,
        support_trial_ids_by_budget=corrupted.support_trial_ids_by_budget,
        query_trial_ids_by_budget={
            **corrupted.query_trial_ids_by_budget,
            "4": tuple(item for item in corrupted.chronological_trial_ids if item not in set(corrupted.support_trial_ids_by_budget["4"])),
        },
        neural_sha256=corrupted.neural_sha256, calibration_sha256=corrupted.calibration_sha256,
        target_last_bin_sha256_by_budget=corrupted.target_last_bin_sha256_by_budget,
        valid_last_bin_mask_sha256_by_budget=corrupted.valid_last_bin_mask_sha256_by_budget,
        valid_last_bin_count_by_budget=corrupted.valid_last_bin_count_by_budget,
        raw_m30_t4_axis_proof=corrupted.raw_m30_t4_axis_proof,
        theta_recovery_sha256=corrupted.theta_recovery_sha256,
    )
    with pytest.raises(score.ScoreError, match="causal support/query partition"):
        all_nonsupport.payload()


def test_raw_m30_axis_is_the_only_raw_t4_helper_and_is_shared_by_all_budgets() -> None:
    """M4/M10 support fits cannot accidentally call chronological raw pools."""
    import numpy as np

    calls: list[int] = []

    class _UnitSide:
        MODULATION_EPS = 0.25
        # Deliberately irrelevant historical-looking values: the route must
        # neither inspect them nor need them to exist.
        raw_m4 = np.full((4, 4), -999.0, dtype=np.float32)
        raw_m10 = np.full((4, 4), 999.0, dtype=np.float32)

        @staticmethod
        def compute_unit_side_features_uncached(path, *, feature_group, pool_size, bin_size_ms,
                                                 window_size, trial_result_filter, signal_view):
            assert path == Path("/synthetic/held.nwb")
            assert (feature_group, bin_size_ms, window_size, trial_result_filter, signal_view) == (
                "t4", 20, 50, "R", "sua",
            )
            calls.append(pool_size)
            if pool_size != 30:
                raise AssertionError("M4/M10 raw T4 must not be requested")
            return np.array([
                [0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 0.4, 1.0],
                [0.0, 0.0, 0.6, 1.0], [0.0, 0.0, 0.7, 1.0],
            ], dtype=np.float32), types.SimpleNamespace(feature_group="t4", pool_size=30, signal_view="sua")

    runtime = object.__new__(physical.ReviewedCDMScoreRuntime)
    runtime.state = types.SimpleNamespace(
        closed=False,
        modules={
            "np": np,
            "unit_side": _UnitSide,
            "core": types.SimpleNamespace(channel_order_digest=lambda values: _sha(values.tobytes().hex())),
        },
    )
    record = types.SimpleNamespace(
        neural=np.zeros((50, 4), dtype=np.float32),
        channel_ids=np.arange(4, dtype=np.int64),
        source_unit_count=4,
    )
    valid_before, proof_before = runtime._raw_m30_t4_axis(
        snapshot=types.SimpleNamespace(path=Path("/synthetic/held.nwb")), session="synthetic", record=record,
    )
    _UnitSide.raw_m4[...] = 12345.0
    _UnitSide.raw_m10[...] = -12345.0
    delattr(_UnitSide, "raw_m4")
    delattr(_UnitSide, "raw_m10")
    valid_after, proof_after = runtime._raw_m30_t4_axis(
        snapshot=types.SimpleNamespace(path=Path("/synthetic/held.nwb")), session="synthetic", record=record,
    )
    assert calls == [30, 30]
    assert np.array_equal(valid_before, valid_after)
    assert proof_before.payload() == proof_after.payload()
    assert proof_before.payload()["budget"] == 30
    assert "_raw_t4_by_budget" not in inspect.getsource(physical.ReviewedCDMScoreRuntime._initial_memory)


def test_budget_initial_memory_cannot_consume_legacy_m4_m10_raw_t4_maps() -> None:
    """Exercise the real initializer with a poison legacy raw-T4 mapping."""
    import numpy as np
    from src.causal_dual_memory_cell_d_v1 import core

    session_id = "synthetic-raw-axis-isolation"
    channels = np.arange(8, dtype=np.int64)
    channel_sha = core.channel_order_digest(channels)
    ids = tuple(f"{session_id}:trial:{index}" for index in range(30))
    trials = {
        trial_id: physical.EvaluationTrial(
            trial_id, index,
            core.B3SInterpolatedSpikeCountTrial(
                activity=np.full((100, 8), float(index), dtype=np.float32), session_id=session_id,
                trial_id=trial_id, channel_order_sha256=channel_sha,
            ),
            None, None, None, np.empty(0, dtype=np.int64),
        )
        for index, trial_id in enumerate(ids)
    }
    valid = np.ones(8, dtype=np.bool_)
    raw_axis = physical.RawT4AxisProof(
        budget=30, raw_t4_sha256=_sha("raw-m30"), channel_order_sha256=_sha("channels-m30"),
        valid_mask_sha256=physical._array_sha(np, valid), source_unit_count=8, modulation_eps=1e-6,
    )
    prepared = physical.PreparedEvaluationSession(
        surface=plan.WITHIN, session=session_id, held=None, neural=None, behavior=None, channel_ids=channels,
        ordered_trial_ids=ids, trials_by_id=trials,
        support_trial_ids={4: ids[:4], 10: ids[:10], 30: ids}, query_trial_ids={4: (), 10: (), 30: ()},
        support_rates={
            trial_id: np.asarray([1.0 + (index % 8) + unit * 0.01 for unit in range(8)], dtype=np.float64)
            for index, trial_id in enumerate(ids)
        },
        support_direction_indices={trial_id: index % 8 for index, trial_id in enumerate(ids)},
        raw_m30_valid_mask=valid, raw_m30_axis_proof=raw_axis,
        theta_recovery={}, theta_recovery_sha256=_sha("theta"), input_record=None,
        input_record_payload={}, input_record_sha256=_sha("input"),
    )

    class _PoisonLegacyPools:
        def __getitem__(self, budget):
            raise AssertionError(f"legacy raw M{budget} pool must never enter initial memory")

    # Dataclasses here are deliberately not slots: installing the old field
    # proves that a future accidental regression cannot silently consume it.
    prepared.raw_t4_by_budget = _PoisonLegacyPools()  # type: ignore[attr-defined]
    runtime = object.__new__(physical.ReviewedCDMScoreRuntime)
    runtime.state = types.SimpleNamespace(closed=False, modules={"np": np, "core": core})
    m4, m4_evidence = runtime._initial_memory(session=prepared, budget=4)
    m10, m10_evidence = runtime._initial_memory(session=prepared, budget=10)
    assert m4.state.carrier.groups.digest == m4_evidence["group_assignment_sha256"]
    assert m10.state.carrier.groups.digest == m10_evidence["group_assignment_sha256"]
    assert m4_evidence["raw_m30_t4_used_for_m4_or_m10"] is False
    assert m10_evidence["raw_m30_t4_used_for_m4_or_m10"] is False
    assert m4_evidence["group_valid_mask_sha256"] == raw_axis.payload()["valid_mask_sha256"]
    assert m10_evidence["group_valid_mask_sha256"] == raw_axis.payload()["valid_mask_sha256"]


def test_cell_evidence_rejects_model_mutation_fifo_drift_and_target_binding_forgery() -> None:
    identity = _identity()
    authority = _input_authority()
    payload = authority.payload(identity=identity)
    cells = _cells(authority, budget=30)
    summary = score.summarize_budget(cells, budget=30, input_payload=payload)
    assert score.budget_gate(summary, budget=30)["external_gate_pass"] is True
    row = _session_score(authority.records[0], budget=30, system=plan.SYSTEM_CDMD, r2=0.3)
    mutated = score.SessionScore(**{**row.__dict__, "model_state_after_sha256": _sha("mutated")})
    with pytest.raises(score.ScoreError, match="model mutated"):
        mutated.payload(budget=30, system=plan.SYSTEM_CDMD)
    fifo = score.SessionScore(**{**row.__dict__, "activity_fifo_capacity": 1})
    with pytest.raises(score.ScoreError, match="activity capacity"):
        fifo.payload(budget=30, system=plan.SYSTEM_CDMD)
    forged = score.SessionScore(**{**row.__dict__, "target_last_bin_sha256": _sha("wrong-target")})
    bad_cell = score.CellEvidence(plan.WITHIN, 30, plan.SYSTEM_CDMD, _sha("input-authority"),
                                  plan.SEALED_CELL_D_SWA_SHA256, (forged,) + cells[1].sessions[1:], {})
    with pytest.raises(score.ScoreError, match="input/target/mask/count"):
        bad_cell.payload(input_payload=payload)


def test_summary_rejects_system_input_or_budget_group_mismatch() -> None:
    identity = _identity()
    authority = _input_authority()
    input_payload = authority.payload(identity=identity)
    cells = list(_cells(authority, budget=10))
    candidate = cells[1]
    changed = score.SessionScore(**{**candidate.sessions[0].__dict__, "group_assignment_sha256": _sha("wrong-budget-group-map")})
    cells[1] = score.CellEvidence(
        candidate.surface, candidate.budget, candidate.system, candidate.input_authority_sha256,
        candidate.model_swa_sha256, (changed,) + candidate.sessions[1:], candidate.resources,
    )
    with pytest.raises(score.ScoreError, match="same-input/initial-state"):
        score.summarize_budget(cells, budget=10, input_payload=input_payload)


def test_gate_boundaries_are_fixed_and_within_never_rescues_external() -> None:
    authority = _input_authority()
    payload = authority.payload(identity=_identity())
    stopped = score.summarize_budget(_cells(authority, budget=30, delta=-0.0100001), budget=30, input_payload=payload)
    assert score.budget_gate(stopped, budget=30)["external_gate_pass"] is False
    assert score.terminal_verdict({30: score.budget_gate(stopped, budget=30)}) == "STOP_M30_SAFETY"
    m10 = score.budget_gate(score.summarize_budget(_cells(authority, budget=10, delta=0.03), budget=10, input_payload=payload), budget=10)
    assert m10["external_gate_pass"] is True


def _complete_score_receipt(
    authority: score.InputAuthority, *, identity: plan.ScoreIdentity,
) -> tuple[dict[str, object], dict[str, object], str, dict[int, Mapping[str, object]]]:
    """Build a canonical all-budget synthetic score receipt for validators."""
    input_payload = authority.payload(identity=identity)
    input_sha = score._digest(input_payload)
    summaries: dict[str, object] = {}
    gates: dict[int, Mapping[str, object]] = {}
    for budget in plan.BUDGETS:
        summary = score.summarize_budget(
            _cells(authority, budget=budget, input_authority_sha256=input_sha),
            budget=budget, input_payload=input_payload,
        )
        summaries[str(budget)] = summary
        gates[budget] = score.budget_gate(summary, budget=budget)
    receipt = {
        "schema": "causal_dual_memory_cell_d_matched_score_v1",
        "identity": identity.payload(),
        "input_authority_sha256": input_sha,
        "budget_summaries": summaries,
        "budget_gates": {str(key): dict(value) for key, value in sorted(gates.items(), reverse=True)},
        "m10_m4_policy": "M4_runs_when_M10_stage_reached__M10_gate_never_tunes_or_rescues",
        "target_optimizer_backward_update": 0,
    }
    return receipt, input_payload, input_sha, gates


def test_input_authority_and_score_terminal_rebuild_reject_forged_assets_gates_and_verdict() -> None:
    identity = _identity()
    fixed = _authority()
    authority = _input_authority()
    receipt, input_payload, input_sha, gates = _complete_score_receipt(authority, identity=identity)
    assert score.validate_input_authority_payload(
        input_payload, identity=identity, evaluation_authority=fixed,
    ) == input_payload
    assert score.validate_score_payload(
        receipt, identity=identity, input_payload=input_payload, input_authority_sha256=input_sha,
        evaluation_authority=fixed,
    ) == receipt

    bad_asset = json.loads(json.dumps(input_payload))
    bad_asset["records"][0]["asset_id"] = "different-uuid-with-same-byte-claims"
    with pytest.raises(score.ScoreError, match="immutable manifest/ledger asset row"):
        score.validate_input_authority_payload(bad_asset, identity=identity, evaluation_authority=fixed)

    bad_gate = json.loads(json.dumps(receipt))
    bad_gate["budget_gates"]["30"]["external_gate_pass"] = False
    with pytest.raises(score.ScoreError, match="predeclared reconstruction"):
        score.validate_score_payload(
            bad_gate, identity=identity, input_payload=input_payload, input_authority_sha256=input_sha,
            evaluation_authority=fixed,
        )

    attempt_sha = _sha("attempt")
    score_sha = score._digest(receipt)
    terminal = score._terminal_payload(
        identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
        score_sha256=score_sha, gates=gates, verdict=score.terminal_verdict(gates),
    )
    assert score.validate_terminal_payload(
        terminal, identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
        score_payload=receipt, score_sha256=score_sha,
    ) == terminal
    bad_terminal = dict(terminal)
    bad_terminal["verdict"] = "STOP_M30_SAFETY"
    with pytest.raises(score.ScoreError, match="graph/verdict/closure"):
        score.validate_terminal_payload(
            bad_terminal, identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
            score_payload=receipt, score_sha256=score_sha,
        )


def test_receipt_boundaries_reject_target_state_use_and_direct_authority_reserve() -> None:
    row = _session_score(_input_authority().records[0], budget=4, system=plan.SYSTEM_CDMD, r2=0.3)
    with pytest.raises(score.ScoreError, match="no-dropout/no-target-update"):
        score.SessionScore(**{**row.__dict__, "target_label_state_uses": 1}).payload(
            budget=4, system=plan.SYSTEM_CDMD,
        )
    with pytest.raises(score.ScoreError, match="no-dropout/no-target-update"):
        score.SessionScore(**{**row.__dict__, "nonfinite_prediction_count": 1}).payload(
            budget=4, system=plan.SYSTEM_CDMD,
        )
    with pytest.raises(score.ScoreError, match="publication capability"):
        score.reserve_authority_artifact(Path("/tmp"), object())


def test_governing_query_authority_rejects_invalid_window_before_any_state_use() -> None:
    import numpy as np

    behavior = np.asarray([
        [0.0, 1.0], [2.0, 3.0], [-1.0, -1.0], [4.0, 5.0], [6.0, 7.0],
    ], dtype=np.float32)
    trials = {
        "one": physical.EvaluationTrial("one", 0, None, None, None, None, np.asarray([0, 2], dtype=np.int64)),
        "two": physical.EvaluationTrial("two", 1, None, None, None, None, np.asarray([4], dtype=np.int64)),
    }
    with pytest.raises(physical.PhysicalCDMDScoreError, match="invalid governed"):
        physical.ReviewedCDMScoreRuntime._governing_target_mask_authority(
            np, behavior=behavior, query_trial_ids=("one", "two"), trials_by_id=trials,
        )


class _Artifact:
    topology = score.SCORE_TOPOLOGY

    def __init__(self, *, fail_group: bool = False) -> None:
        self.items: dict[str, bytes] = {}
        self.fail_group = fail_group

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        if name in self.items:
            raise RuntimeError("collision")
        body = score._json(payload)
        self.items[name] = body
        return hashlib.sha256(body).hexdigest()

    def publish_group(self, bodies: Mapping[str, bytes], *, post_publish=None):
        if self.fail_group:
            raise RuntimeError("atomic group failure")
        if any(name in self.items for name in bodies):
            raise RuntimeError("collision")
        digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
        if post_publish is not None:
            post_publish(bodies, digests)
        self.items.update(bodies)
        return digests

    def reload_json(self, name: str, expected_sha256: str | None = None):
        return json.loads(self.items[name])

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        return self.items[name]

    def has_name(self, name: str) -> bool:
        return name in self.items


class _TempPairArtifact:
    """Tiny real-filesystem authority pair fixture; never a canonical root."""

    topology = score.AUTHORITY_TOPOLOGY

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True)

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        if name not in self.topology:
            raise RuntimeError("topology")
        body = score._json(payload)
        digest = hashlib.sha256(body).hexdigest()
        body_path = self.directory / name
        sidecar_path = self.directory / f"{name}.sha256"
        if body_path.exists() or sidecar_path.exists():
            raise RuntimeError("collision")
        body_path.write_bytes(body)
        sidecar_path.write_bytes(f"{digest}  {name}\n".encode("ascii"))
        os.chmod(body_path, 0o444)
        os.chmod(sidecar_path, 0o444)
        return digest

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        body = (self.directory / name).read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError("digest")
        assert (self.directory / f"{name}.sha256").read_bytes() == f"{digest}  {name}\n".encode("ascii")
        return body

    def reload_json(self, name: str, expected_sha256: str | None = None):
        return json.loads(self.reload_pair(name, expected_sha256))

    def publish_group(self, bodies, *, post_publish=None):  # pragma: no cover - authority test needs pairs only
        raise RuntimeError("not used")

    def has_name(self, name: str) -> bool:
        return (self.directory / name).exists()


class _MockRuntime:
    def __init__(self, authority: score.InputAuthority) -> None:
        self.authority = authority
        self.calls: list[str] = []

    def preflight(self, *, root: Path, identity: plan.ScoreIdentity):
        self.calls.append("preflight")
        return {"target_paths_resolved": False, "target_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

    def prepare(self, *, root: Path, identity: plan.ScoreIdentity):
        self.calls.append("prepare")
        return self

    def materialize_inputs(self, runtime, *, identity: plan.ScoreIdentity, evaluation_authority: score.FixedEvaluationAuthority):
        self.calls.append("materialize")
        return self.authority

    def score_budget(self, runtime, *, budget: int, input_authority_sha256: str, identity: plan.ScoreIdentity):
        self.calls.append(f"budget{budget}")
        return _cells(self.authority, budget=budget, input_authority_sha256=input_authority_sha256)

    def revalidate(self, runtime, *, root: Path, identity: plan.ScoreIdentity):
        self.calls.append("revalidate")

    def failure_progress(self, runtime):
        return {
            "within_assets_opened": False,
            "external_assets_opened": False,
            "checkpoint_opened": False,
            "cuda_initialized": False,
            "full_system_forward_count": 0,
            "group_forward_count": 0,
        }

    def close(self, runtime) -> None:
        self.calls.append("close")


def test_mock_lifecycle_attempt_precedes_target_and_atomic_failure_is_honest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    identity = _identity()
    binding = _gate_binding()
    fixed = _authority()
    preflight = score.build_target_free_preflight(root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed)
    pre_sha = hashlib.sha256(score._json(preflight)).hexdigest()
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = hashlib.sha256(score._json(authorization)).hexdigest()
    root_cap = score._issue_root_publication_capability()
    capability = score.issue_execution_capability(
        durable_preflight_sha256=pre_sha, durable_authorization_sha256=auth_sha,
        identity=identity, root_capability=root_cap,
    )
    closure = plan.ImplementationClosure({
        path: (plan.WORKORDER_SHA256 if path == plan.WORKORDER_RELATIVE else _sha(path))
        for path in plan.IMPLEMENTATION_PATHS
    })
    monkeypatch.setattr(score.plan, "implementation_closure", lambda root: closure)
    monkeypatch.setattr(score, "validate_completed_source_gate", lambda root: binding)
    monkeypatch.setattr(
        score, "validate_preflight_against_fixed_authorities",
        lambda root, payload, *, identity: score.validate_target_free_preflight(payload, identity=identity),
    )
    runtime = _MockRuntime(_input_authority())
    artifact = _Artifact()
    result = score.run_authorized_score_lifecycle(
        tmp_path, identity=identity, capability=capability, backend=runtime, artifact=artifact,
        official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
        preflight=preflight, authorization=authorization,
    )
    assert result["verdict"] == "HOLD_M4" or result["verdict"] == "PASS"
    assert tuple(runtime.calls[:3]) == ("preflight", "prepare", "materialize")
    assert "attempt.json" in artifact.items and "score.json" in artifact.items and "terminal.json" in artifact.items
    assert "failure.json" not in artifact.items
    failing = _Artifact(fail_group=True)
    runtime2 = _MockRuntime(_input_authority())
    with pytest.raises(RuntimeError, match="atomic group failure"):
        score.run_authorized_score_lifecycle(
            tmp_path, identity=identity, capability=capability, backend=runtime2, artifact=failing,
            official_preflight_sha256=pre_sha, root_authorization_sha256=auth_sha,
            preflight=preflight, authorization=authorization,
        )
    assert "attempt.json" in failing.items and "failure.json" in failing.items
    assert "score.json" not in failing.items and "terminal.json" not in failing.items
    failure = json.loads(failing.items["failure.json"])
    assert score.validate_failure_payload(
        failure, identity=identity, attempt_sha256=hashlib.sha256(failing.items["attempt.json"]).hexdigest(),
        input_authority_sha256=hashlib.sha256(failing.items["input_authority.json"]).hexdigest(),
    ) == failure
    forged_failure = dict(failure)
    forged_failure["target_paths_resolved_or_opened"] = True
    with pytest.raises(score.ScoreError, match="target-access/progress"):
        score.validate_failure_payload(
            forged_failure, identity=identity, attempt_sha256=hashlib.sha256(failing.items["attempt.json"]).hexdigest(),
            input_authority_sha256=hashlib.sha256(failing.items["input_authority.json"]).hexdigest(),
        )


def test_real_temporary_authority_publish_reload_and_durable_capability_are_root_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exercise real 0444 pair publication/reload without a canonical root."""
    identity = _identity()
    binding = _gate_binding()
    fixed = _authority()
    preflight = score.build_target_free_preflight(
        root=tmp_path, identity=identity, source_gate=binding, fixed_authority=fixed,
    )
    directory = tmp_path / plan.AUTHORITY_ROOT_RELATIVE
    artifact = _TempPairArtifact(directory)
    root_capability = score._issue_root_publication_capability()
    seen_roots: list[Path] = []

    def checked(actual_root: Path, payload: Mapping[str, object], *, identity: plan.ScoreIdentity):
        seen_roots.append(Path(actual_root).absolute())
        if Path(actual_root).absolute() != tmp_path.absolute():
            raise score.ScoreError("wrong reviewed authority root")
        return score.validate_target_free_preflight(payload, identity=identity)

    closure = plan.ImplementationClosure({
        path: (plan.WORKORDER_SHA256 if path == plan.WORKORDER_RELATIVE else _sha(path))
        for path in plan.IMPLEMENTATION_PATHS
    })
    monkeypatch.setattr(score.plan, "implementation_closure", lambda root: closure)
    monkeypatch.setattr(score, "validate_preflight_against_fixed_authorities", checked)
    monkeypatch.setattr(score, "validate_completed_source_gate", lambda root: binding)

    mismatched_gate = score.SourceGateBinding(
        directory_device=9, directory_inode=binding.directory_inode,
        body_sha256s=binding.body_sha256s, terminal_status=binding.terminal_status,
        source_gate_closure_sha256=binding.source_gate_closure_sha256,
        strict_source_roster=binding.strict_source_roster,
    )
    mismatched_preflight = score.build_target_free_preflight(
        root=tmp_path, identity=identity, source_gate=mismatched_gate, fixed_authority=fixed,
    )
    with pytest.raises(score.ScoreError, match="completed source-gate binding"):
        score.publish_target_free_preflight(
            tmp_path, _TempPairArtifact(tmp_path / "mismatch-authority"), root_capability,
            mismatched_preflight, identity=identity,
        )

    pre_sha = score.publish_target_free_preflight(
        tmp_path, artifact, root_capability, preflight, identity=identity,
    )
    authorization = score.build_root_authorization(official_preflight_sha256=pre_sha, preflight=preflight)
    auth_sha = score.publish_root_authorization(
        tmp_path, artifact, root_capability, authorization, identity=identity,
    )
    loaded_preflight, loaded_authorization, loaded_pre_sha, loaded_auth_sha = score.load_durable_authority(
        tmp_path, identity=identity,
    )
    assert (loaded_preflight, loaded_authorization, loaded_pre_sha, loaded_auth_sha) == (
        preflight, authorization, pre_sha, auth_sha,
    )
    capability = score.issue_durable_execution_capability(
        tmp_path, identity=identity, root_capability=root_capability,
        environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    assert capability.official_preflight_sha256 == pre_sha
    assert seen_roots and all(item == tmp_path.absolute() for item in seen_roots)
    with pytest.raises(score.ScoreError, match="wrong reviewed authority root"):
        score.publish_root_authorization(
            tmp_path / "wrong", artifact, root_capability, authorization, identity=identity,
        )


def test_physical_backend_is_a_narrow_deferred_seam_without_torch_or_target_open() -> None:
    identity = _identity()
    backend = physical.build_reviewed_physical_backend(
        root=Path.cwd(), selected_device_profile=plan.COMPATIBLE_DEVICE_PROFILES["gpu1"],
    )
    assert isinstance(backend, physical.PhysicalCDMDMatchedScoreBackend)
    assert backend.preflight(root=Path.cwd(), identity=identity) == {
        "target_paths_resolved": False, "target_opened": False, "checkpoint_opened": False,
        "cuda_initialized": False, "source_gate_reloaded": False, "normalizer_refit": False,
    }
    assert "torch" not in sys.modules
    source = inspect.getsource(physical.ReviewedCDMScoreRuntime._parse_session)
    assert "exclude_calibration_trials_from_windows=True" in source
    assert "tuple(ordered[30:])" in source


def test_causal_m30_capacity_zero_keeps_activity_exactly_static_but_can_update_carrier() -> None:
    """CPU seam: M30 never takes the historical ``[-0:]`` activity path."""
    import numpy as np
    from src.causal_dual_memory_cell_d_v1 import core

    session = "synthetic-evaluation-session"
    channels = np.arange(8, dtype=np.int64)
    channel_sha = core.channel_order_digest(channels)
    support = tuple(
        core.B3SInterpolatedSpikeCountTrial(
            activity=np.full((100, 8), float(index), dtype=np.float32), session_id=session,
            trial_id=f"support-{index}", channel_order_sha256=channel_sha,
        ) for index in range(30)
    )
    directions = np.asarray([index % 8 for index in range(30)], dtype=np.int64)
    rates = np.asarray([[1.0 + direction + unit * 0.01 for unit in range(8)] for direction in directions], dtype=np.float64)
    initial = core.fit_carriers_from_trial_table(
        rates, directions, mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
    )
    config = core.CDMDConfig(support_budget_m=30, active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL)
    carrier = core.CarrierMemory.from_support_trials(
        initial_raw_t4=initial, channel_ids=channels, support_trial_rates=rates,
        support_direction_indices=directions, config=config, valid_mask=np.ones(8, dtype=np.bool_),
    )
    memory = core.CausalDualMemory(
        activity=core.ActivityMemory.initialize(support, channel_ids=channels, fifo_capacity=0), carrier=carrier,
    )
    activity_before, carrier_before = memory.state.activity.digest, memory.state.carrier.digest
    query = core.B3SInterpolatedSpikeCountTrial(
        activity=np.ones((100, 8), dtype=np.float32), session_id=session, trial_id="query", channel_order_sha256=channel_sha,
    )
    native = core.NativeRewardedTrialSpikeCounts(
        counts=np.full((5, 8), 2, dtype=np.int64), session_id=session, trial_id="query",
        channel_order_sha256=channel_sha, rewarded_interval_start_bin=300, rewarded_interval_stop_bin=305,
    )
    validity = core.VelocityValidityEvidence(
        valid_mask=np.ones(4, dtype=np.bool_), session_id=session, trial_id="query",
        prediction_interval_start_bin=300, prediction_interval_stop_bin=304,
    )
    predictions = tuple(
        core.CompletedVelocityPrediction(
            velocity=np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (4, 1)), validity=validity,
        ) for _ in range(4)
    )
    outcome = memory.commit(memory.observe_completed_trial(
        b3s_trial_activity=query, carrier_trial_counts=native, complementary_predictions=predictions,
    ))
    assert memory.state.activity.digest == activity_before
    assert memory.state.activity.query_count == 0
    # A strict rank/condition gate can honestly reject a particular synthetic
    # pseudo trial; either way, activity remains exact.  If accepted, M30's
    # carrier-only update is demonstrably distinct from the frozen activity.
    if outcome.committed:
        assert memory.state.carrier.digest != carrier_before
    else:
        assert memory.state.carrier.digest == carrier_before


def test_public_cli_bootstraps_cleanly_and_never_imports_torch(tmp_path: Path) -> None:
    script = Path("tfpd_exploration/scripts/run_causal_dual_memory_cell_d_matched_score_v1.py").absolute()
    code = (
        "import importlib.util, sys; "
        f"s=importlib.util.spec_from_file_location('candidate', {str(script)!r}); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "m.main([]); print('TORCH=' + str('torch' in sys.modules))"
    )
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTHONPATH": str(Path.cwd() / "tfpd_exploration")}
    completed = subprocess.run([sys.executable, "-c", code], cwd=Path.cwd(), env=environment,
                               text=True, capture_output=True, check=True)
    assert '"no_torch_import":true' in completed.stdout
    assert "TORCH=False" in completed.stdout
    denied = subprocess.run([sys.executable, str(script), "--execute"], cwd=Path.cwd(), env=environment,
                            text=True, capture_output=True)
    assert denied.returncode != 0 and "both --execute" in denied.stderr


def test_importing_torch_for_cpu_fact_does_not_initialize_cuda() -> None:
    import torch

    assert torch.cuda.is_initialized() is False
