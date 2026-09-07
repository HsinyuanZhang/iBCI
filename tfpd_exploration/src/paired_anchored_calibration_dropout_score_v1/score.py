"""Session-outer matched-score composition over reviewed CAL-AUG primitives."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

from . import plan


class ScoreError(RuntimeError):
    pass


def _require(value: bool, message: str) -> None:
    if not value:
        raise ScoreError(message)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class InputRecord:
    surface: str
    session: str
    budget: int
    regime: str
    input_record_sha256: str
    target_sha256: str
    valid_mask_sha256: str
    calibration_activity_sha256: str
    selected_support_sha256: str
    raw_t4_sha256: str
    normalized_t4_sha256: str
    neural_sha256: str
    query_window_starts_sha256: str
    calibration_m30_sha256: str
    forward_authority_sha256: str


@dataclass(frozen=True)
class ForwardEvidence:
    """Complete, closed evidence emitted by one reviewed static forward.

    The scorer deliberately does not accept an open-ended dictionary here:
    an omitted target-update field or an invented affirmative boolean is a
    contract failure, not a value that can be silently ignored.
    """
    r2: float
    prediction_sha256: str
    identity_sha256: str
    repeated_identity_sha256: str
    repeated_prediction_sha256: str
    sentinel_coordinates: tuple[int, ...]
    sentinel_prediction_sha256: str
    repeated_sentinel_prediction_sha256: str
    n_windows: int
    valid_last_bin_count: int
    model_state_before_sha256: str
    model_state_after_sha256: str
    model_swa_sha256: str
    strict_load: bool
    eval_no_dropout_no_grad: bool
    repeated_forward_equal: bool
    identity_repeat_equal: bool
    finite_prediction: bool
    target_optimizer_steps: int
    target_backward_calls: int
    target_update_calls: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ForwardEvidence":
        expected = set(cls.__dataclass_fields__)
        actual = set(value)
        _require(actual == expected, "forward evidence schema")
        evidence = cls(**dict(value))
        _require(float(evidence.r2) == float(evidence.r2), "non-finite R2")
        for field in ("prediction_sha256", "identity_sha256", "repeated_identity_sha256", "repeated_prediction_sha256",
                      "sentinel_prediction_sha256", "repeated_sentinel_prediction_sha256",
                      "model_state_before_sha256", "model_state_after_sha256", "model_swa_sha256"):
            digest = getattr(evidence, field)
            _require(isinstance(digest, str) and len(digest) == 64 and
                     all(ch in "0123456789abcdef" for ch in digest), f"forward evidence {field}")
        _require(bool(evidence.sentinel_coordinates) and all(isinstance(x, int) and x >= 0 for x in evidence.sentinel_coordinates),
                 "sentinel coordinates")
        _require(evidence.n_windows > 0 and 0 < evidence.valid_last_bin_count <= evidence.n_windows,
                 "last-bin validity")
        _require(evidence.strict_load is True and evidence.eval_no_dropout_no_grad is True and
                 evidence.repeated_forward_equal is True and evidence.identity_repeat_equal is True and evidence.finite_prediction is True,
                 "static forward proof")
        _require(evidence.model_state_before_sha256 == evidence.model_state_after_sha256,
                 "model state drift")
        _require((evidence.target_optimizer_steps, evidence.target_backward_calls, evidence.target_update_calls) == (0, 0, 0),
                 "target update evidence")
        _require(evidence.prediction_sha256 == evidence.repeated_prediction_sha256 and evidence.identity_sha256 == evidence.repeated_identity_sha256 and
                 evidence.sentinel_prediction_sha256 == evidence.repeated_sentinel_prediction_sha256,
                 "repeat-forward digest drift")
        return evidence


def authority_record(inputs: Any, *, budget: int, regime: str, tensor_digest: Callable[[Any], str]) -> InputRecord:
    """Derive one regime/budget authority from a single P4 materialization."""
    _require(budget in plan.BUDGET_ORDER and regime in plan.REGIME_ORDER, "row regime/budget")
    selected_m30 = tuple(inputs.selected_by_budget[30])
    _require(selected_m30 == tuple(range(30)), "chronological M30 selection")
    calibration_m30 = inputs.calib[list(inputs.selected_by_budget[30])]
    if regime == "honest_total":
        activity = inputs.calib[list(inputs.selected_by_budget[budget])]
    else:
        # Keep this literal selection rather than relying on the materializer's
        # current prefix layout: the diagnostic must prove it uses selected M30.
        activity = calibration_m30
    activity_sha = tensor_digest(activity)
    selected_sha = str(inputs.selected_sha_by_budget[budget])
    raw_t4_sha = str(inputs.ridge_fit_by_budget[budget]["raw_t4_sha256"])
    normalized_t4_sha = str(inputs.side_sha_by_budget[budget])
    body = {
        "surface": inputs.surface, "session": inputs.session, "budget": int(budget), "regime": regime,
        "target": inputs.target_sha256, "valid": inputs.valid_mask_sha256,
        "activity": activity_sha, "selected": selected_sha, "raw_t4": raw_t4_sha,
        "normalized_t4": normalized_t4_sha,
        "neural": str(inputs.neural_sha256),
        "query_window_starts": str(inputs.query_window_starts_sha256),
        "calibration_m30": tensor_digest(calibration_m30),
    }
    return InputRecord(str(inputs.surface), str(inputs.session), int(budget), regime, _digest(body),
                       str(inputs.target_sha256), str(inputs.valid_mask_sha256), activity_sha,
                       selected_sha, raw_t4_sha, normalized_t4_sha, str(inputs.neural_sha256),
                       str(inputs.query_window_starts_sha256), tensor_digest(calibration_m30), _digest(body))


def materialize_authority(runtime: Any, *, materialize_session: Callable[[Any, str, str], Any],
                          tensor_digest: Callable[[Any], str]) -> tuple[dict[tuple[str, str, int, str], tuple[Any, InputRecord]], dict]:
    """Materialize each target session once, then expose 3×2 immutable rows."""
    authority: dict[tuple[str, str, int, str], tuple[Any, InputRecord]] = {}
    rosters: dict[str, tuple[str, ...]] = {}
    for surface in plan.SURFACE_ORDER:
        roster = tuple(runtime.within_roster if surface == "within" else runtime.external_roster)
        _require(len(roster) == plan.ROSTER_COUNTS[surface] and len(set(roster)) == len(roster), "sealed roster cardinality")
        rosters[surface] = roster
        for session in roster:
            # Exactly one call for a surface/session; no system sees a parser.
            inputs = materialize_session(runtime, surface, session)
            for budget in plan.BUDGET_ORDER:
                for regime in plan.REGIME_ORDER:
                    record = authority_record(inputs, budget=budget, regime=regime, tensor_digest=tensor_digest)
                    authority[(surface, session, budget, regime)] = (inputs, record)
    _require(len(authority) == 21 * 3 * 2, "input authority cardinality")
    records = [authority[(surface, session, budget, regime)][1].__dict__
               for surface in plan.SURFACE_ORDER for budget in plan.BUDGET_ORDER for regime in plan.REGIME_ORDER
               for session in rosters[surface]]
    return authority, {"rosters": rosters, "records": records, "records_sha256": _digest(records)}


def governed_last_bin_r2(prediction: Any, inputs: Any, *, session_r2: Callable[[Any, Any], float]) -> tuple[float, int, str]:
    """The required [49] variance-weighted house metric; no hidden label use."""
    import torch

    last = prediction[:, 49, :].contiguous()
    target = torch.from_numpy(inputs.last_targets.copy())
    valid = torch.from_numpy(inputs.last_valid_mask.copy())
    n_valid = int(valid.sum().item())
    _require(last.ndim == target.ndim == 2 and valid.ndim == 1 and target.shape[0] == valid.shape[0], "governed target/mask shape")
    _require(n_valid > 0 and tuple(last.shape) == tuple(target.shape), "governed last-bin shape/mask")
    _require(bool(torch.isfinite(last).all().item()) and bool(torch.isfinite(target).all().item()), "non-finite prediction/target")
    digest = hashlib.sha256(prediction.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
    return float(session_r2(last[valid], target[valid])), n_valid, digest


def assert_same_input(rows: Sequence[Mapping[str, Any]]) -> None:
    """Every six-system cell must share exactly the same input authority."""
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["surface"]), str(row["session"]), int(row["budget"]), str(row["regime"]))
        grouped.setdefault(key, []).append(row)
    for key, cell in grouped.items():
        _require(tuple(row["system"] for row in cell) == plan.SYSTEM_ORDER, f"system order {key}")
        fields = ("input_record_sha256", "target_sha256", "valid_mask_sha256", "calibration_activity_sha256",
                  "selected_support_sha256", "raw_t4_sha256", "normalized_t4_sha256", "neural_sha256",
                  "query_window_starts_sha256", "calibration_m30_sha256", "forward_authority_sha256")
        for field in fields:
            _require(len({row[field] for row in cell}) == 1, f"same-input drift {key} {field}")


def canonicalize_rows(rows: Sequence[Mapping[str, Any]], rosters: Mapping[str, Sequence[str]]) -> list[dict]:
    """Return and validate the fixed receipt order declared by the work order.

    Computation may remain session-outer for materialization economy.  Receipt
    order is independently canonicalized as surface → M30/M10/M4 → regime →
    roster → P0/P1/P2/T0/C1/SD, so a later execution cannot silently reorder a
    matched contrast.
    """
    expected = [
        (surface, session, budget, regime, system)
        for surface in plan.SURFACE_ORDER
        for budget in plan.BUDGET_ORDER
        for regime in plan.REGIME_ORDER
        for session in rosters[surface]
        for system in plan.SYSTEM_ORDER
    ]
    required_fields = {"system", "surface", "session", "budget", "regime"} | set(InputRecord.__dataclass_fields__) | set(ForwardEvidence.__dataclass_fields__)
    _require(all(set(row) == required_fields for row in rows), "exact score-row schema")
    for row in rows:
        ForwardEvidence.from_mapping({field: row[field] for field in ForwardEvidence.__dataclass_fields__})
    keys_in_rows = [(str(row["surface"]), str(row["session"]), int(row["budget"]), str(row["regime"]), str(row["system"])) for row in rows]
    indexed = {key: dict(row) for key, row in zip(keys_in_rows, rows, strict=True)}
    _require(len(indexed) == len(rows) == plan.EXPECTED_ROW_COUNT, "row cardinality/uniqueness")
    _require(set(indexed) == set(expected), "canonical score-row topology")
    ordered = [indexed[key] for key in expected]
    assert_same_input(ordered)
    return ordered


def score_session_outer(authority: Mapping[tuple[str, str, int, str], tuple[Any, InputRecord]], *,
                        rosters: Mapping[str, Sequence[str]],
                        system_forward: Callable[[str, Any, InputRecord], Mapping[str, Any]]) -> list[dict]:
    """Generic execution seam: one materialized session feeds all systems/regimes.

    ``system_forward`` is responsible for the reviewed strict SWA model swap
    and static P4 decode.  It receives no raw session locator, so it cannot
    rematerialize or substitute the target input authority.
    """
    rows: list[dict] = []
    for surface in plan.SURFACE_ORDER:
        for session in rosters[surface]:
            # Session outer: the authority was created by exactly one reviewed
            # materializer invocation before these six-system forwards.
            for budget in plan.BUDGET_ORDER:
                for regime in plan.REGIME_ORDER:
                    inputs, record = authority[(surface, session, budget, regime)]
                    for system in plan.SYSTEM_ORDER:
                        evidence = ForwardEvidence.from_mapping(system_forward(system, inputs, record))
                        rows.append({"system": system, "surface": surface, "session": session,
                                     "budget": budget, "regime": regime, **record.__dict__, **evidence.__dict__})
    return canonicalize_rows(rows, rosters)


def score_with_cal_aug_primitives(runtime: Any, *, swa_paths: Mapping[str, Any],
                                  pop_robust: Any, arm_common: Any, matched_scorer: Any) -> dict:
    """Deferred live adapter composed from reviewed deployment primitives.

    This is intentionally not invoked by dry mode.  It imports the existing
    CAL-AUG swap, P4 materializer/static decoder, and house metric only after
    an attempt has been committed by the lifecycle owner.  There is no copied
    DANDI parser or alternate scorer here.
    """
    import torch
    from src.cal_aug_v1 import deployment
    from src.calibration_gap_v1 import p4_stream_stats as p4

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "CPU scorer CUDA discipline")

    _require(tuple(swa_paths) == plan.SYSTEM_ORDER, "six-system SWA order")

    def tensor_digest(value: Any) -> str:
        return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

    authority, input_evidence = materialize_authority(
        runtime, materialize_session=p4.materialize_session, tensor_digest=tensor_digest
    )
    # Strict-load exactly one model per static system.  P4's static decoder
    # reads runtime._model, so the callback below selects only among these
    # already-loaded no-dropout eval models; no system rematerializes input.
    loaded: dict[str, tuple[Any, dict]] = {}
    for system in plan.SYSTEM_ORDER:
        swap = deployment._swap_runtime_model(runtime, pop_robust, arm_common, swa_paths[system])
        _require(swap["strict_load"] and swap["eval_mode"], "strict SWA swap evidence")
        loaded[system] = (runtime._model, swap)

    def forward(system: str, inputs: Any, record: InputRecord) -> Mapping[str, Any]:
        model, swap = loaded[system]
        runtime._model = model
        _require(all(parameter.device.type == "cpu" for parameter in model.parameters()), "CPU model placement")
        state_before = arm_common.state_sha256(model)
        calibration_m30 = inputs.calib[list(inputs.selected_by_budget[30])]
        _require(tuple(inputs.selected_by_budget[30]) == tuple(range(30)), "chronological M30 selection")
        if record.regime == "honest_total":
            activity = inputs.calib[list(inputs.selected_by_budget[record.budget])]
        else:
            activity = calibration_m30
        side = inputs.side_by_budget[record.budget]
        with pop_robust.dynamic_dropout_recorder() as dropout:
            prediction, _identity = p4._decode_static(runtime, inputs, activity, side)
        # A fixed repeated static forward makes forward purity an observed
        # fact rather than a model-eval assumption.  It uses the same authority
        # and does not perform backward/optimizer work.
        with pop_robust.dynamic_dropout_recorder() as repeated_dropout:
            repeated, _identity_repeat = p4._decode_static(runtime, inputs, activity, side)
        _require(prediction.device.type == repeated.device.type == "cpu" and not torch.cuda.is_initialized(), "CPU decode placement")
        _require(dropout["uniform_calls"] == 0 and not dropout["dropout_calls"], "deployment dropout active")
        _require(repeated_dropout["uniform_calls"] == 0 and not repeated_dropout["dropout_calls"], "repeated deployment dropout active")
        r2, n_valid, prediction_sha = governed_last_bin_r2(prediction, inputs, session_r2=matched_scorer.session_r2)
        _repeat_r2, _repeat_valid, repeated_prediction_sha = governed_last_bin_r2(repeated, inputs, session_r2=matched_scorer.session_r2)
        state_after = arm_common.state_sha256(model)
        identity_sha = tensor_digest(_identity[0] if isinstance(_identity, (tuple, list)) else _identity)
        identity_repeat_sha = tensor_digest(_identity_repeat[0] if isinstance(_identity_repeat, (tuple, list)) else _identity_repeat)
        sentinel = tuple(sorted({0, max(0, min(int(inputs.n_windows) - 1, 31))}))
        sentinel_prediction_sha = hashlib.sha256(prediction[list(sentinel)].detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        repeated_sentinel_sha = hashlib.sha256(repeated[list(sentinel)].detach().contiguous().cpu().numpy().tobytes()).hexdigest()
        return {
            "r2": r2, "prediction_sha256": prediction_sha, "identity_sha256": identity_sha, "repeated_identity_sha256": identity_repeat_sha,
            "repeated_prediction_sha256": repeated_prediction_sha, "sentinel_coordinates": sentinel,
            "sentinel_prediction_sha256": sentinel_prediction_sha,
            "repeated_sentinel_prediction_sha256": repeated_sentinel_sha, "n_windows": int(inputs.n_windows),
            "valid_last_bin_count": n_valid, "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after, "state_before_sha256": state_before,
            "state_after_sha256": state_after, "model_swa_sha256": swap["artifact_sha256"],
            "strict_load": True, "eval_no_dropout_no_grad": bool(not model.training), "finite_prediction": True,
            "repeated_forward_equal": bool(torch.equal(prediction, repeated)), "identity_repeat_equal": bool(identity_sha == identity_repeat_sha),
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        }

    rows = score_session_outer(authority, rosters=input_evidence["rosters"], system_forward=forward)
    assert_same_input(rows)
    return {"rows": rows, "input_authority": input_evidence,
            "system_swaps": {system: swap for system, (_model, swap) in loaded.items()}}


def _summary(values: Sequence[float]) -> dict:
    _require(bool(values), "empty summary")
    return {"equal_session_mean": float(mean(values)), "median": float(median(values)), "worst": float(min(values)), "n_sessions": len(values)}


def system_summaries(rows: Sequence[Mapping[str, Any]], *, bootstrap: Callable[[list[float], int], Mapping[str, Any]]) -> dict:
    """All fixed score cells with deterministic session-bootstrap evidence."""
    grouped: dict[str, list[float]] = {}
    roster_order: dict[str, list[str]] = {}
    for row in rows:
        key = f"{row['system']}:{row['surface']}:m{int(row['budget'])}:{row['regime']}"
        grouped.setdefault(key, []).append(float(row["r2"]))
        roster_order.setdefault(key, []).append(str(row["session"]))
    expected_cells = len(plan.SYSTEM_ORDER) * len(plan.SURFACE_ORDER) * len(plan.BUDGET_ORDER) * len(plan.REGIME_ORDER)
    _require(len(grouped) == expected_cells, "summary cell topology")
    return {key: {**_summary(values), "roster": roster_order[key],
                  "bootstrap": dict(bootstrap(list(values), plan.BOOTSTRAP_SEED))}
            for key, values in grouped.items()}


def paired_contrast(rows: Sequence[Mapping[str, Any]], *, left: str, right: str,
                    bootstrap: Callable[[list[float], int], Mapping[str, Any]]) -> dict:
    cells: dict[tuple[str, int, str], dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        if row["system"] not in {left, right}:
            continue
        cells.setdefault((row["surface"], int(row["budget"]), row["regime"]), {}).setdefault(row["system"], []).append(row)
    result = {}
    for key, systems in cells.items():
        lhs, rhs = systems.get(left, []), systems.get(right, [])
        _require(len(lhs) == len(rhs) and lhs and [row["session"] for row in lhs] == [row["session"] for row in rhs], "paired roster")
        deltas = [float(a["r2"]) - float(b["r2"]) for a, b in zip(lhs, rhs, strict=True)]
        stats = dict(bootstrap(deltas, plan.BOOTSTRAP_SEED))
        result[f"{key[0]}:m{key[1]}:{key[2]}"] = {
            "left": left, "right": right, "surface": key[0], "budget": key[1], "regime": key[2],
            "per_session": [{"session": a["session"], "delta_r2": d} for a, d in zip(lhs, deltas, strict=True)],
            "equal_session_mean_delta": float(mean(deltas)), "median_delta": float(median(deltas)),
            "positive_sessions": int(sum(delta > 0 for delta in deltas)), "worst_paired_delta": float(min(deltas)),
            "bootstrap": stats,
        }
    return result


def decision_gates(contrasts: Mapping[str, Mapping[str, Any]], *, rows: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """Exact predeclared gates; activity-isolation never serves as a rescue."""
    def cell(left: str, right: str, surface: str, budget: int) -> Mapping[str, Any]:
        return contrasts[f"{left}-{right}"][f"{surface}:m{budget}:honest_total"]
    input_state_parity = True if rows is None else all(
        row["system"] != "P0" or (row["strict_load"] is True and row["eval_no_dropout_no_grad"] is True and
        row["repeated_forward_equal"] is True and row["identity_repeat_equal"] is True and
        row["model_state_before_sha256"] == row["model_state_after_sha256"] and
        (row["target_optimizer_steps"], row["target_backward_calls"], row["target_update_calls"]) == (0, 0, 0))
        for row in rows)
    p0_valid = bool(input_state_parity and all(cell("P0", "T0", surface, 30)["equal_session_mean_delta"] >= plan.P0_M30_FLOOR
                   for surface in plan.SURFACE_ORDER))
    output = {"p0_valid": p0_valid, "p0_input_state_parity_pass": input_state_parity,
              "primary_regime": "honest_total", "activity_isolation_cannot_rescue": True}
    for arm, budget in (("P1", 4), ("P2", 10)):
        p0 = cell(arm, "P0", "external", budget)
        p0_m30_external = cell(arm, "P0", "external", 30)
        p0_m30_within = cell(arm, "P0", "within", 30)
        c1 = cell(arm, "C1", "external", budget)
        c1_m30 = cell(arm, "C1", "external", 30)
        primary = bool(p0_valid and p0["equal_session_mean_delta"] >= plan.PRIMARY_DELTA
                       and p0["positive_sessions"] >= plan.PRIMARY_POSITIVE
                       and p0_m30_external["equal_session_mean_delta"] >= plan.P0_M30_FLOOR
                       and p0_m30_within["equal_session_mean_delta"] >= plan.P0_M30_FLOOR
                       and p0["worst_paired_delta"] >= plan.WORST_DELTA_FLOOR)
        incremental = bool(primary and c1["equal_session_mean_delta"] >= plan.INCREMENTAL_DELTA
                           and c1["positive_sessions"] >= plan.INCREMENTAL_POSITIVE
                           and c1_m30["equal_session_mean_delta"] >= plan.INCREMENTAL_DELTA)
        output[arm] = {"budget": budget, "primary_gate": primary, "incremental_pairing_gate": incremental}
    return output


def assemble_score_result(rows: Sequence[Mapping[str, Any]], *, bootstrap: Callable[[list[float], int], Mapping[str, Any]]) -> dict:
    """Close all predeclared score readouts from canonical rows only.

    Terminal publication calls this pure reducer again; it never trusts a
    callback-provided aggregate, contrast, or interpretation gate.
    """
    pairs = (("P1", "P0"), ("P2", "P0"), ("P1", "C1"), ("P2", "C1"),
             ("P0", "T0"), ("P0", "SD"), ("C1", "T0"))
    contrasts = {f"{left}-{right}": paired_contrast(rows, left=left, right=right, bootstrap=bootstrap)
                 for left, right in pairs}
    return {"rows": list(rows), "summaries": system_summaries(rows, bootstrap=bootstrap),
            "contrasts": contrasts, "gates": decision_gates(contrasts, rows=rows)}
