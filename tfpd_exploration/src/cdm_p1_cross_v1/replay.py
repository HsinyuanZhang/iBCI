"""Part A driver: the F00/F10/F01/F11 inference-only factorial.

Work order ``docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md`` section 2.
The frozen laws are ORCHESTRATED, never reimplemented:

* ``p2physical._runtime_for`` / ``prepare`` / ``materialize_inputs`` -- the
  sealed parser, chronology, normalizer and Cell-D SWA checkpoint, with the
  sealed fixed evaluation authority, so every cell shares ONE materialized
  session set and ONE identical trial order;
* ``stage_o_replay.rollout_o0`` -- the frozen-T4 cells (F00/F10) are the
  Stage-O O0 arm VERBATIM, which makes the F00 bit-anchor against the sealed
  Stage-P P0 / sealed activity-only rows a no-op proof of this route's loop;
* ``stage_p_replay.rollout_p`` with ``ROW_SPECS['P1']`` and the SEALED
  stage-P selection -- the P1 online cells (F01/F11) are the promoted Stage-P
  P1 law VERBATIM, hyperparameters never re-selected;
* ``weights.swap_runtime_weights`` -- the C1 weight arm, by the strict-load +
  state-digest proof pattern of ``src/cal_aug_v1/deployment``.

Per weight arm the frozen-T4 cell runs first (its M30 raw stream is the
arm's no-op reference), then the P1 cell.  The sealed arm runs first so the
two bit-anchors (F00 vs sealed P0, F01 vs sealed P1) fire before any C1
compute.  The model state digest is verified at every phase boundary and the
sealed model is restored and re-verified after the C1 arm.
"""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from src.support_anchored_t4_stage_o_v1 import replay as stage_o_replay
from src.support_anchored_t4_stage_p_v1 import gate as stage_p_gate
from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from . import gates, plan, weights


class CrossReplayError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossReplayError(message)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_receipt(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sidecar = Path(path).with_name(Path(path).name + ".sha256")
    _require(sidecar.exists(), f"missing sidecar: {sidecar}")
    _require(
        sidecar.read_text(encoding="ascii").strip() == f"{digest}  {Path(path).name}",
        f"sidecar drift for {Path(path).name}",
    )
    return digest


# ---------------------------------------------------------------------------
# Sealed bindings (CPU-only; no data, model or CUDA access).
# ---------------------------------------------------------------------------


def verify_stage_p_go_anchor(base: Path) -> dict[str, Any]:
    """Re-read the sealed Stage-P terminal receipt and check the GO anchor."""
    path = Path(base).absolute() / plan.STAGE_P_GO_ANCHOR["terminal_relative"]
    terminal = _read_json(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    decision = str(terminal.get("decision"))
    driving = str(terminal.get("driving_cell"))
    m4 = float(terminal["gate_summary"]["m4_external_p1_minus_p0"])  # type: ignore[index]
    matches = (
        decision == plan.STAGE_P_GO_ANCHOR["decision"]
        and driving == plan.STAGE_P_GO_ANCHOR["driving_cell"]
        and m4 == float(plan.STAGE_P_GO_ANCHOR["m4_external_p1_minus_p0"])
    )
    _require(matches, "the sealed Stage-P GO anchor no longer matches the work-order binding")
    return {
        "terminal_relative": plan.STAGE_P_GO_ANCHOR["terminal_relative"],
        "terminal_sha256": digest,
        "decision": decision,
        "driving_cell": driving,
        "m4_external_p1_minus_p0": m4,
        "law": plan.STAGE_P_GO_ANCHOR["law"],
        "verified": True,
    }


def verify_sealed_selection_from_terminal(base: Path) -> dict[str, Any]:
    """The m4/m10 halves of the sealed P1 selection (terminal receipt only)."""
    terminal = _read_json(
        Path(base).absolute() / plan.STAGE_P_GO_ANCHOR["terminal_relative"]
    )
    summary = terminal["hyperparameter_selection_summary"]
    checked: dict[str, Any] = {}
    for budget in (4, 10):
        sealed = summary[f"m{budget}"]
        pinned = dict(plan.SEALED_P1_HYPERPARAMETERS[budget])
        _require(
            sealed == pinned,
            f"the sealed Stage-P m{budget} P1 selection drifted from the Part-A pin",
        )
        checked[f"m{budget}"] = {"sealed": sealed, "matches_pin": True}
    return {
        "source": "terminal.json hyperparameter_selection_summary (m4/m10)",
        "checked": checked,
        "values_match_plan": True,
    }


def verify_c1_bindings(base: Path) -> dict[str, Any]:
    """The sealed C1 SWA artifact and its sealed deployment binding."""
    base = Path(base).absolute()
    deployment = _read_json(base / plan.C1_DEPLOYMENT_TERMINAL_RELATIVE)
    _require(
        str(deployment.get("status")) == plan.C1_DEPLOYMENT_STATUS,
        "the sealed cal_aug_v1 deployment terminal status drifted",
    )
    bindings = deployment["arms"]["c1"]["bindings"]
    _require(
        str(bindings.get("artifact_sha256")) == plan.C1_SWA_SHA256
        and str(bindings.get("arm_state_sha256")) == plan.C1_ARM_STATE_SHA256
        and bindings.get("strict_load") is True,
        "the sealed cal_aug_v1 deployment C1 binding drifted from the Part-A pin",
    )
    train = _read_json(base / plan.C1_TRAIN_TERMINAL_RELATIVE)
    _require(
        str(train.get("status")) == plan.C1_TRAIN_TERMINAL_STATUS,
        "the sealed C1 prefix-cycle training terminal status drifted",
    )
    artifact = weights.artifact_sha256(base / plan.C1_SWA_RELATIVE)
    _require(artifact == plan.C1_SWA_SHA256, "the sealed C1 SWA artifact hash drifted")
    return {
        "artifact_relative": plan.C1_SWA_RELATIVE,
        "artifact_sha256": artifact,
        "deployment_terminal_relative": plan.C1_DEPLOYMENT_TERMINAL_RELATIVE,
        "deployment_arm_state_sha256": str(bindings.get("arm_state_sha256")),
        "strict_load": True,
        "train_terminal_status": str(train.get("status")),
        "verified": True,
    }


_SEALED_ROW_FIELDS = (
    "house_raw_r2",
    "prediction_sha256_raw",
    "matrix_r2",
    "filtered_prediction_sha256",
    "n_windows",
    "initial_carrier_sha256",
    "final_carrier_sha256",
    "committed_evidence_rows",
    "carrier_transitions_committed",
)


def load_sealed_stage_p_view(base: Path) -> dict[str, Any]:
    """The compact sealed Stage-P P0/P1 session rows + the m30 selection law.

    Parses the sealed Stage-P replay receipt ONCE (it is ~0.5 GB) and keeps
    only the anchor-relevant fields, so the caller can drop the rest.
    """
    path = Path(base).absolute() / plan.STAGE_P_RESULT_ROOT_RELATIVE / "replay.json"
    payload = _read_json(path)
    matrix = payload["matrix"]
    rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {"P0": {}, "P1": {}}
    for budget_key, surfaces in matrix.items():
        for surface, cells in surfaces.items():
            for row_id in ("P0", "P1"):
                target = rows[row_id].setdefault(budget_key, {}).setdefault(surface, {})
                for session_row in cells[row_id]["sessions"]:
                    session = str(session_row["session"])
                    _require(session not in target, "sealed stage-P row topology drift")
                    target[session] = {
                        field: session_row.get(field) for field in _SEALED_ROW_FIELDS
                    }
    m30 = payload["hyperparameter_selection"]["m30"]
    pinned_m30 = dict(plan.SEALED_P1_HYPERPARAMETERS[30])
    _require(
        float(m30["alpha_M"]) == pinned_m30["alpha_M"]
        and float(m30["rho_M"]) == pinned_m30["rho_M"]
        and m30["c_M"] is None
        and dict(m30["thresholds"]) == dict(pinned_m30["thresholds"]),
        "the sealed Stage-P m30 no-op law drifted from the Part-A pin",
    )
    expected_cells = {
        f"m{budget}": {
            surface: {
                "P0": plan.WITHIN_SESSION_COUNT if surface == "within" else plan.EXTERNAL_SESSION_COUNT,
                "P1": plan.WITHIN_SESSION_COUNT if surface == "within" else plan.EXTERNAL_SESSION_COUNT,
            }
            for surface in plan.SURFACES
        }
        for budget in plan.BUDGETS
    }
    for budget_key, surfaces in expected_cells.items():
        for surface, counts in surfaces.items():
            for row_id, count in counts.items():
                _require(
                    len(rows[row_id][budget_key][surface]) == count,
                    f"sealed stage-P {row_id} row count drift at {budget_key} {surface}",
                )
    return {
        "replay_relative": f"{plan.STAGE_P_RESULT_ROOT_RELATIVE}/replay.json",
        "rows": rows,
        "m30_selection": {
            "alpha_M": float(m30["alpha_M"]), "rho_M": float(m30["rho_M"]),
            "c_M": None, "thresholds": dict(m30["thresholds"]),
            "matches_pin": True,
        },
    }


# ---------------------------------------------------------------------------
# Cell laws.
# ---------------------------------------------------------------------------


def cell_hyperparameters(budget: int) -> stage_p_replay.CellHyperparameters:
    """The sealed Stage-P P1 selection for one budget (never re-selected)."""
    pinned = plan.SEALED_P1_HYPERPARAMETERS[int(budget)]
    return stage_p_replay.CellHyperparameters(
        rho_M=float(pinned["rho_M"]),
        alpha_M=float(pinned["alpha_M"]),
        c_M=(None if pinned["c_M"] is None else float(pinned["c_M"])),
        thresholds=stage_p_gate.GateThresholds(
            tau_d=float(pinned["thresholds"]["tau_d_rad"]),
            r_max=int(pinned["thresholds"]["r_max_repetition_per_direction"]),
            d_min=int(pinned["thresholds"]["d_min_distinct_directions"]),
            max_mass_relative=float(
                pinned["thresholds"]["max_pseudo_mass_relative_to_support_rows"]
            ),
        ),
    )


def run_frozen_t4_rollout(runtime: Any, *, session: Any, budget: int) -> dict[str, Any]:
    """F00/F10: the Stage-O O0 arm VERBATIM under the current weight arm."""
    return stage_o_replay.rollout_o0(runtime, session=session, budget=budget)


def run_p1_rollout(
    runtime: Any, *, session: Any, budget: int, hp: stage_p_replay.CellHyperparameters,
) -> dict[str, Any]:
    """F01/F11: the sealed Stage-P P1 law VERBATIM under the current weight arm."""
    return stage_p_replay.rollout_p(
        runtime, session=session, budget=budget,
        spec=stage_p_replay.ROW_SPECS["P1"], hp=hp,
    )


def build_session_row(
    runtime: Any,
    *,
    cell: str,
    session: Any,
    budget: int,
    targets: Any,
    masks: Any,
    rollout: Mapping[str, Any],
    hp: Optional[stage_p_replay.CellHyperparameters],
    model_state_sha256: str,
) -> dict[str, Any]:
    """Assemble one cell's session row through the Stage-P row law."""
    carrier_law = plan.CARRIER_LAW_BY_CELL[cell]
    weights_arm = str(plan.CELLS[cell]["weights"])
    estimator = (
        "frozen support T4 (activity-only CDM; the Stage-O O0 arm) under the "
        f"{weights_arm} weight arm"
        if carrier_law == "frozen_t4"
        else f"{rollout['estimator']} under the {weights_arm} weight arm"
    )
    if carrier_law == "frozen_t4":
        row = stage_p_replay._session_row(
            runtime, row_id=cell, budget=budget, session=session,
            targets=targets, masks=masks,
            rollout={
                "per_trial_filtered": rollout["per_trial_filtered"],
                "per_trial_raw": rollout["per_trial_raw"],
                "initial": rollout["initial"],
                "leakage_flags": dict(rollout["leakage_flags"]),
                "estimator": estimator,
                "per_trial_receipts": rollout["per_trial_receipts"],
                "transitions": rollout["transitions"],
            },
            hp=None,
        )
        raw_row = rollout["rows"]["O0_raw"]
        governing = rollout["rows"]["O0"]
        _require(
            row["house_raw_r2"] == float(raw_row["house_raw_r2"])
            and row["prediction_sha256_raw"] == str(raw_row["prediction_sha256_raw"])
            and row["matrix_r2"] == float(governing["matrix_r2"]),
            f"{cell} row disagrees with its own O0 rollout payload",
        )
    else:
        row = stage_p_replay._session_row(
            runtime, row_id=cell, budget=budget, session=session,
            targets=targets, masks=masks, rollout=rollout, hp=hp, governing=True,
        )
    row["schema"] = f"{plan.SCHEMA}_row_session_v1"
    row["cell"] = cell
    row["weights"] = weights_arm
    row["carrier_law"] = carrier_law
    row["model_state_sha256"] = str(model_state_sha256)
    row["leakage_flags"] = dict(plan.LEAKAGE_FLAGS_BY_CELL[cell])
    return row


# ---------------------------------------------------------------------------
# Anchors.
# ---------------------------------------------------------------------------


def anchor_row_vs_sealed_stage_p(
    *, row: Mapping[str, Any], sealed_row: Mapping[str, Any], label: str,
) -> dict[str, Any]:
    fields = (
        "house_raw_r2", "prediction_sha256_raw", "matrix_r2",
        "filtered_prediction_sha256", "n_windows",
    )
    matches = {
        field: row.get(field) == sealed_row.get(field) for field in fields
    }
    exact = all(bool(value) for value in matches.values())
    return {
        "label": label,
        "field_matches": matches,
        "exact_match": bool(exact),
        "replayed_matrix_r2": float(row["matrix_r2"]),
        "sealed_matrix_r2": float(sealed_row["matrix_r2"]),
        "replayed_prediction_sha256_raw": str(row["prediction_sha256_raw"]),
        "sealed_prediction_sha256_raw": str(sealed_row["prediction_sha256_raw"]),
    }


def anchor_initial_carrier(
    *, row: Mapping[str, Any], baseline_row: Mapping[str, Any], label: str,
) -> dict[str, Any]:
    fields = ("initial_carrier_sha256", "initial_activity_sha256", "group_assignment_sha256")
    matches = {
        field: row.get(field) == baseline_row.get(field) for field in fields
    }
    return {
        "label": label,
        "field_matches": matches,
        "exact_match": all(bool(value) for value in matches.values()),
    }


def verify_m30_noop(
    *, rollout: Mapping[str, Any], reference_raw: Any, cell: str, session: str,
) -> None:
    """The exact no-op law at M30 against the SAME arm's frozen-T4 raw stream."""
    _require(
        len(rollout["per_trial_raw"]) == len(reference_raw),
        f"M30 no-op trial-count drift for {cell} in {session}",
    )
    for left, right in zip(rollout["per_trial_raw"], reference_raw):
        _require(
            np.array_equal(np.asarray(left), np.asarray(right)),
            f"M30 no-op law broken by {cell} in {session}",
        )
    for receipt in rollout["receipts"]:
        _require(
            receipt["cb"] == receipt["ca"],
            f"M30 carrier moved under {cell} in {session}",
        )
        _require(
            not receipt["acc"],
            f"M30 carrier transition committed under {cell} in {session}",
        )


# ---------------------------------------------------------------------------
# The factorial driver.
# ---------------------------------------------------------------------------


def run_cross_replay(base: Path, *, gpu_index: int, output_root: Path) -> Mapping[str, object]:
    """Run the four cells once each over the frozen activity-only runtime."""
    base = Path(base).absolute()
    output = Path(output_root)
    _require(gpu_index == plan.GPU_INDEX, "Part A is bound to GPU 1 (the work-order device law)")
    _require(not (output / "terminal.json").exists(), "the Part-A terminal receipt already exists")
    _require(not (output / "replay.json").exists(),
             "the Part-A replay receipt already exists; run --stage terminal, never a second replay")
    _require((output / "attempt.json").exists(), "reserve the Part-A attempt first")
    attempt_sha = _verify_receipt(output / "attempt.json")
    attempt = _read_json(output / "attempt.json")
    predecessors = plan.predecessor_sha256s(base)
    _require(
        predecessors == attempt["predecessor_sha256s"],
        "an immutable predecessor drifted between attempt and replay",
    )
    owned = plan.owned_sha256s(base)
    _require(owned == attempt["owned_sha256s"],
             "an owned Part-A module drifted between attempt and replay")
    go_anchor = verify_stage_p_go_anchor(base)
    selection_binding = verify_sealed_selection_from_terminal(base)
    c1_binding = verify_c1_bindings(base)
    sealed_view = load_sealed_stage_p_view(base)
    sealed_activity_cells = stage_o_replay._sealed_activity_cells(base)

    started = time.monotonic()
    from src.causal_dual_memory_cell_d_score_v1 import score as v1score
    from src.learned_gate_p2prime_v1 import physical as p2physical

    environment = p2physical.validate_environment(gpu_index=gpu_index)
    runtime, meta, identity = p2physical._runtime_for(base, gpu_index=gpu_index)
    try:
        runtime.prepare(identity=identity)
        fixed = v1score.derive_fixed_evaluation_authority(base)
        runtime.materialize_inputs(identity=identity, authority=fixed)
        state = runtime._require_state()
        arm = state.modules["arm_common"]
        sealed_digest = arm.state_sha256(state.model)
        ordered = {
            surface: p2physical._ordered_session_keys(runtime, surface)
            for surface in plan.SURFACES
        }
        _require(
            len(ordered["within"]) == plan.WITHIN_SESSION_COUNT
            and len(ordered["external"]) == plan.EXTERNAL_SESSION_COUNT,
            "the frozen surfaces must carry exactly six within and fifteen external sessions",
        )
        hp_by_budget = {budget: cell_hyperparameters(budget) for budget in plan.BUDGETS}

        matrix: dict[str, Any] = {}
        anchors_activity: dict[str, Any] = {}
        anchors_stage_p: dict[str, Any] = {}
        anchors_initial: dict[str, Any] = {}
        baseline_rows: dict[tuple[int, str, str], dict[str, Any]] = {}
        m30_reference_raw: dict[tuple[str, str, str], list[np.ndarray]] = {}
        swap_receipt: Optional[dict[str, Any]] = None
        restore_receipt: Optional[dict[str, Any]] = None
        causality_all = True
        trust_region_no_committed_drift = True
        m30_noop_all = True
        sealed_model_holder: dict[str, Any] = {}

        for weights_arm, cells in plan.CELL_BY_ARM.items():
            if weights_arm == "c1":
                swap = weights.swap_runtime_weights(
                    runtime,
                    swa_path=base / plan.C1_SWA_RELATIVE,
                    expected_artifact_sha256=plan.C1_SWA_SHA256,
                    expected_state_sha256=plan.C1_ARM_STATE_SHA256,
                )
                sealed_model_holder["sealed_model"] = swap.pop("sealed_model")
                swap_receipt = swap
                arm_digest = str(swap["arm_state_sha256"])
            else:
                checked = weights.verify_model_digest(
                    runtime, expected=sealed_digest, label="sealed_arm_start",
                )
                arm_digest = str(checked["model_state_sha256"])
            for cell in cells:
                carrier_law = plan.CARRIER_LAW_BY_CELL[cell]
                for budget in plan.BUDGETS:
                    hp = hp_by_budget[budget] if carrier_law == "p1_online" else None
                    for surface in plan.SURFACES:
                        session_rows: list[dict[str, Any]] = []
                        for key in ordered[surface]:
                            session = state.sessions[key]
                            targets, masks, _sst = runtime._session_target_views(session, budget)
                            if carrier_law == "frozen_t4":
                                rollout = run_frozen_t4_rollout(
                                    runtime, session=session, budget=budget,
                                )
                            else:
                                rollout = run_p1_rollout(
                                    runtime, session=session, budget=budget, hp=hp,
                                )
                            row = build_session_row(
                                runtime, cell=cell, session=session, budget=budget,
                                targets=targets, masks=masks, rollout=rollout, hp=hp,
                                model_state_sha256=arm_digest,
                            )
                            tag = f"{session.session}:m{budget}:{surface}"
                            if cell == "F00":
                                anchors_activity[tag] = stage_o_replay.anchor_o0_vs_sealed(
                                    row=row,
                                    sealed_row=sealed_activity_cells[(int(budget), surface)][session.session],
                                    label=f"F00 vs sealed activity-only m{budget} {surface} {session.session}",
                                )
                                anchors_stage_p[f"{tag}:F00"] = anchor_row_vs_sealed_stage_p(
                                    row=row,
                                    sealed_row=sealed_view["rows"]["P0"][f"m{budget}"][surface][session.session],
                                    label=f"F00 vs sealed Stage-P P0 m{budget} {surface} {session.session}",
                                )
                                baseline_rows[(int(budget), surface, session.session)] = row
                            if cell == "F01":
                                anchors_stage_p[f"{tag}:F01"] = anchor_row_vs_sealed_stage_p(
                                    row=row,
                                    sealed_row=sealed_view["rows"]["P1"][f"m{budget}"][surface][session.session],
                                    label=f"F01 vs sealed Stage-P P1 m{budget} {surface} {session.session}",
                                )
                            if cell != "F00":
                                anchors_initial[f"{tag}:{cell}"] = anchor_initial_carrier(
                                    row=row,
                                    baseline_row=baseline_rows[(int(budget), surface, session.session)],
                                    label=f"{cell} initial carrier invariance vs F00 "
                                          f"m{budget} {surface} {session.session}",
                                )
                            if carrier_law == "frozen_t4" and budget == 30:
                                m30_reference_raw[(weights_arm, surface, session.session)] = (
                                    list(rollout["per_trial_raw"])
                                )
                            if carrier_law == "p1_online":
                                if budget == 30:
                                    reference = m30_reference_raw.get(
                                        (weights_arm, surface, session.session)
                                    )
                                    _require(
                                        reference is not None,
                                        f"the {weights_arm}-arm frozen-T4 M30 pass must precede "
                                        f"{cell} in {session.session}",
                                    )
                                    verify_m30_noop(
                                        rollout=rollout, reference_raw=reference,
                                        cell=cell, session=session.session,
                                    )
                                    row["m30_noop_verified"] = True
                                for receipt in rollout["receipts"]:
                                    if receipt.get("acc") and receipt.get("d2") is not None:
                                        if hp is not None and hp.c_M is not None:
                                            if float(receipt["d2"]) > float(hp.c_M):
                                                trust_region_no_committed_drift = False
                            causality_all = causality_all and bool(row["causality_state_chain_verified"])
                            _require(
                                row["causality_state_chain_verified"],
                                f"causality receipt chain broken for {cell} {session.session} m{budget}",
                            )
                            session_rows.append(row)
                            _require(
                                (time.monotonic() - started) <= plan.HARD_TIMEOUT_SECONDS,
                                "the Part-A hard timeout fired during the factorial grid",
                            )
                        per_session = {
                            str(item["session"]): float(item["matrix_r2"]) for item in session_rows
                        }
                        matrix.setdefault(f"m{budget}", {}).setdefault(surface, {})[cell] = {
                            "mean_r2": float(sum(per_session.values()) / len(per_session)),
                            "per_session": per_session,
                            "sessions": [dict(item) for item in session_rows],
                            "full_per_trial_state_receipts": True,
                            "weights": weights_arm,
                            "carrier_law": carrier_law,
                        }
            if weights_arm == "c1":
                restore_receipt = weights.restore_runtime_weights(
                    runtime,
                    sealed_model=sealed_model_holder["sealed_model"],
                    expected_sealed_state_sha256=sealed_digest,
                )
                weights.verify_model_digest(
                    runtime, expected=sealed_digest, label="sealed_arm_restored",
                )
            else:
                weights.verify_model_digest(
                    runtime, expected=sealed_digest, label="sealed_arm_end",
                )

        resources = dict(runtime._resources())
    finally:
        runtime.close()

    f00_stage_p_exact = all(
        bool(item["exact_match"]) for key, item in anchors_stage_p.items() if key.endswith(":F00")
    )
    f01_stage_p_exact = all(
        bool(item["exact_match"]) for key, item in anchors_stage_p.items() if key.endswith(":F01")
    )
    payload = {
        "schema": f"{plan.SCHEMA}_replay_v1",
        "stage": "f00_f10_f01_f11_inference_only_factorial",
        "status": "REPLAY_COMPLETE",
        "attempt_sha256": attempt_sha,
        "identity_sha256": meta["identity_sha256"],
        "environment": environment,
        "gpu_index": int(gpu_index),
        "budgets": list(plan.BUDGETS),
        "surfaces": list(plan.SURFACES),
        "cells": list(plan.CELL_ORDER),
        "weight_arms": {arm: dict(spec) for arm, spec in plan.WEIGHT_ARMS.items()},
        "carrier_law_by_cell": dict(plan.CARRIER_LAW_BY_CELL),
        "hyperparameters": plan.hyperparameters_payload(),
        "hyperparameter_binding": {
            **selection_binding,
            "m30": sealed_view["m30_selection"],
            "re_selection": plan.HYPERPARAMETER_LAW["re_selection"],
        },
        "matrix": matrix,
        "anchors": {
            "stage_p_go": go_anchor,
            "c1_weight_binding": c1_binding,
            "f00_vs_sealed_activity_only": anchors_activity,
            "f00_vs_sealed_activity_only_all_exact": all(
                bool(item["exact_match"]) for item in anchors_activity.values()
            ),
            "f00_f01_vs_sealed_stage_p": anchors_stage_p,
            "f00_vs_sealed_stage_p_all_exact": bool(f00_stage_p_exact),
            "f01_vs_sealed_stage_p_all_exact": bool(f01_stage_p_exact),
            "initial_carrier_invariance": anchors_initial,
            "initial_carrier_invariance_all_exact": all(
                bool(item["exact_match"]) for item in anchors_initial.values()
            ),
            "m30_noop_all_cells": bool(m30_noop_all),
        },
        "weight_swap": {
            "law": dict(plan.WEIGHT_SWAP_LAW),
            "c1_swap": swap_receipt,
            "sealed_restore": restore_receipt,
        },
        "model_state_digests": {
            "sealed_arm_sha256": sealed_digest,
            "c1_arm_sha256": plan.C1_ARM_STATE_SHA256,
            "sealed_model_restored_and_verified": bool(
                restore_receipt is not None
                and restore_receipt["restored_state_sha256"] == sealed_digest
            ),
        },
        "causality_state_chains_all_rows": bool(causality_all),
        "trust_region_no_committed_drift": bool(trust_region_no_committed_drift),
        "resources": resources,
        "wall_seconds": float(time.monotonic() - started),
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "model_or_checkpoint_updated": False,
        "decoder_training": False,
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "normalizer_update_calls": 0,
    }
    _require(
        payload["anchors"]["f00_vs_sealed_activity_only_all_exact"],
        "the F00 anchor failed at least one sealed activity-only session row",
    )
    _require(f00_stage_p_exact, "the F00 anchor failed at least one sealed Stage-P P0 session row")
    _require(f01_stage_p_exact, "the F01 anchor failed at least one sealed Stage-P P1 session row")
    _require(
        payload["anchors"]["initial_carrier_invariance_all_exact"],
        "the initial carrier/activity/group digests differ across weight arms",
    )
    _require(payload["anchors"]["m30_noop_all_cells"], "the M30 exact no-op law failed")
    _require(
        payload["model_state_digests"]["sealed_model_restored_and_verified"],
        "the sealed model was not restored bit-exactly after the C1 arm",
    )
    return payload


# ---------------------------------------------------------------------------
# Terminal composition (receipt-only; no data, model or CUDA access).
# ---------------------------------------------------------------------------


def build_terminal(*, replay_payload: Mapping[str, object]) -> Mapping[str, object]:
    from src.tfpd_lane.matched_scorer import paired_session_stats

    matrix = replay_payload["matrix"]
    _require(isinstance(matrix, Mapping), "terminal needs the replay matrix")
    view = {
        budget_key: {
            surface: {
                cell: {
                    "mean_r2": float(entry["mean_r2"]),
                    "per_session": {
                        str(session): float(value)
                        for session, value in entry["per_session"].items()
                    },
                }
                for cell, entry in cells.items()
            }
            for surface, cells in surfaces.items()
        }
        for budget_key, surfaces in matrix.items()
    }
    anchors = replay_payload["anchors"]
    swap = replay_payload["weight_swap"]
    digests = replay_payload["model_state_digests"]
    safety = {
        "zero_target_updates": bool(
            replay_payload.get("target_optimizer_backward_update") == 0
            and replay_payload.get("target_backward_calls") == 0
            and replay_payload.get("target_parameter_update_calls") == 0
            and replay_payload.get("normalizer_update_calls") == 0
            and replay_payload.get("decoder_training") is False
            and replay_payload.get("model_or_checkpoint_updated") is False
        ),
        "f00_bit_anchor": bool(
            anchors["f00_vs_sealed_activity_only_all_exact"]
            and anchors["f00_vs_sealed_stage_p_all_exact"]
        ),
        "f01_bit_anchor": bool(anchors["f01_vs_sealed_stage_p_all_exact"]),
        "causality_state_chains": bool(replay_payload["causality_state_chains_all_rows"]),
        "weight_swap_binding_verified": bool(
            swap["c1_swap"] is not None
            and swap["c1_swap"].get("strict_load") is True
            and swap["c1_swap"].get("arm_state_sha256") == plan.C1_ARM_STATE_SHA256
            and swap["c1_swap"].get("artifact_sha256") == plan.C1_SWA_SHA256
        ),
        "sealed_model_digest_restored": bool(digests["sealed_model_restored_and_verified"]),
        "initial_carrier_invariance": bool(anchors["initial_carrier_invariance_all_exact"]),
        "m30_exact_noop_per_arm": bool(anchors["m30_noop_all_cells"]),
        "hyperparameters_sealed_selection_only": bool(
            replay_payload["hyperparameter_binding"]["values_match_plan"] is True
            and replay_payload["hyperparameter_binding"]["m30"]["matches_pin"] is True
        ),
        "trust_region_no_committed_drift": bool(
            replay_payload["trust_region_no_committed_drift"]
        ),
    }
    gate = gates.evaluate_cross_gate(view, safety=safety)
    stops = gates.stop_conditions(gate)
    contrasts = gates.paired_contrast_tables(view)
    paired: dict[str, object] = {}
    for key, delta in contrasts.items():
        ordered_deltas = [
            float(delta["per_session"][session]) for session in sorted(delta["per_session"])
        ]
        paired[key] = paired_session_stats(ordered_deltas)
    equal_session_means = {
        budget_key: {
            surface: {cell: float(entry["mean_r2"]) for cell, entry in cells.items()}
            for surface, cells in surfaces.items()
        }
        for budget_key, surfaces in view.items()
    }
    return {
        "gate": gate,
        "stop_conditions": stops,
        "paired_contrasts": paired,
        "equal_session_means": equal_session_means,
        "paired_delta_means": {
            key: float(delta["equal_session_mean_delta"]) for key, delta in contrasts.items()
        },
        "paired_positive_sessions": {
            key: int(delta["positive_sessions"]) for key, delta in contrasts.items()
        },
        "hyperparameter_binding": replay_payload["hyperparameter_binding"],
        "stage_p_go_anchor": anchors["stage_p_go"],
        "c1_weight_binding": anchors["c1_weight_binding"],
    }
