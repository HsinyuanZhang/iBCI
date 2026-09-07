"""CPU-only budget sweep: does carrier↔token redundancy decay at low support M?"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3

from . import data as fold_data
from . import plan
from . import receipts as fold_receipts
from . import stage0 as fold_stage0
from .full_query import load_student, verify_pilot_arm_anchor
from .stage1 import ensure_streaming_path
from .token_probe import (
    _array_digest,
    _jsonable,
    _load_supports,
    loso_probe,
    raw_stats_features,
)


class BudgetProbeError(RuntimeError):
    """Fail closed for the fold-local budget-sweep identity-token probe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BudgetProbeError(message)


RELIABILITY_SCALAR = "weight_flattened_pearson"
RELIABILITY_IMPLEMENTATION = (
    "tfpd_exploration.src.m1_emg_syn3_fcm_v1.syn3.trial_stratified_split_half"
)
RELIABILITY_REASON = (
    "Frozen trial_stratified_split_half already exposes the flattened-weight Pearson "
    "of even vs odd unit-ridge weights; that is the reliability of the 3-D carrier "
    "weights the probe predicts, before source normalization. Intercept Pearson is "
    "recorded but is not the ceiling for weights_r2."
)


def _require_budget(budget: object) -> int:
    _require(type(budget) is int, "budget must be int")
    _require(1 <= int(budget) <= int(plan.SUPPORT_TRIALS), "budget out of range")
    return int(budget)


def redundancy_fraction(weights_r2: float, reliability: float | None) -> float | None:
    if reliability is None:
        return None
    value = float(reliability)
    if value <= 0.0:
        return None
    return float(weights_r2) / value


def carrier_reliability(
    scores: np.ndarray,
    rates: np.ndarray,
    *,
    trial_ids: np.ndarray,
) -> float | None:
    split = rsyn3.trial_stratified_split_half(scores, rates, trial_ids=trial_ids)
    value = split.get(RELIABILITY_SCALAR)
    if value is None:
        return None
    return float(value)


def _encode_session_at_budget(record, basis, budget: int) -> dict[str, np.ndarray | float | None | dict]:
    emg_b, rates_b, ids_b = fold_stage0._mask_budget(record, budget)
    rsyn3.require_support_bins(ids_b, budget=budget)
    _require(int(np.max(ids_b)) < int(plan.SUPPORT_TRIALS), "support boundary crossed into query")
    scores = rsyn3.project_basis(emg_b, basis)
    weights, intercepts = rsyn3.fit_all_units(scores, rates_b)
    raw = rsyn3.carrier_from_encoding(weights, intercepts)
    split = rsyn3.trial_stratified_split_half(scores, rates_b, trial_ids=ids_b)
    reliability = split.get(RELIABILITY_SCALAR)
    return {
        "raw": raw,
        "scores": scores,
        "rates": rates_b,
        "trial_ids": ids_b,
        "split_half": split,
        "reliability": None if reliability is None else float(reliability),
    }


def _fold0_scope_and_nmf(repo_root: Path) -> tuple[dict[str, Any], Any]:
    repo_root = Path(repo_root)
    plan.verify_bound_documents(repo_root)
    rectify.assert_frozen_law()
    loaded = fold_data.load_fold_scope(fold=0)
    isolation = dict(loaded["isolation"])
    _require(isolation["target_query_values_read"] is False, "target query leak")
    target = loaded["target"]
    _require(target.session == plan.FOLD0_TARGET_SESSION, "fold-0 target drift")
    source_emg = np.concatenate(
        [rectify.relu_nonnegative_projection(record.emg) for record in loaded["sources"].values()],
        axis=0,
    )
    nmf = rsyn3.fit_source_nmf(source_emg)
    return loaded, nmf


def build_budget_carrier_bank(
    repo_root: Path,
    *,
    budget: object,
    loaded: dict[str, Any] | None = None,
    nmf: Any | None = None,
) -> dict[str, Any]:
    """Source-frozen rSyn3 carriers at support budget M. Target query values unread."""
    budget = _require_budget(budget)
    repo_root = Path(repo_root)
    if loaded is None or nmf is None:
        loaded, nmf = _fold0_scope_and_nmf(repo_root)
    isolation = dict(loaded["isolation"])
    _require(isolation["target_query_values_read"] is False, "target query leak")
    sources = loaded["sources"]
    target = loaded["target"]
    _require(target.session == plan.FOLD0_TARGET_SESSION, "fold-0 target drift")
    encoded_sources = {
        name: _encode_session_at_budget(record, nmf, budget) for name, record in sources.items()
    }
    source_raw = [encoded_sources[name]["raw"] for name in sources]
    norm_mean, norm_scale = rsyn3.source_normalizer(list(source_raw))
    encoded_target = _encode_session_at_budget(target, nmf, budget)
    if budget == plan.SUPPORT_TRIALS:
        _require(
            rsyn3.array_digest(encoded_target["raw"]) == plan.FOLD0_M10_RAW_CARRIER_DIGEST,
            "fold-0 target M10 raw carrier digest drift vs sealed Stage-0",
        )
    encoded = dict(encoded_sources)
    encoded[target.session] = encoded_target
    raw: dict[str, np.ndarray] = {}
    normalized: dict[str, dict[str, np.ndarray]] = {}
    reliability_by_session: dict[str, float | None] = {}
    split_half: dict[str, object] = {}
    for name, item in encoded.items():
        carrier = np.asarray(item["raw"], dtype=np.float64)
        raw[name] = carrier
        syn = rsyn3.normalize_carriers(carrier, norm_mean, norm_scale)
        zero = np.zeros_like(syn)
        normalized[name] = {"rSyn3": syn, "Zero4": zero}
        _require(np.array_equal(zero, np.zeros(syn.shape)), "Zero4 is not exact zeros")
        reliability_by_session[name] = item["reliability"]
        split_half[name] = item["split_half"]
    values = [reliability_by_session[name] for name in plan.SESSIONS]
    _require(all(name in reliability_by_session for name in plan.SESSIONS), "session reliability missing")
    if any(value is None for value in values):
        reliability_mean: float | None = None
    else:
        reliability_mean = float(np.mean(np.asarray(values, dtype=np.float64)))
    return {
        "fold": 0,
        "budget": int(budget),
        "target_session": target.session,
        "source_sessions": list(plan.FOLD0_SOURCE_SESSIONS),
        "isolation": isolation,
        "normalizer_mean": norm_mean,
        "normalizer_scale": norm_scale,
        "raw": raw,
        "normalized": normalized,
        "reliability_scalar": RELIABILITY_SCALAR,
        "reliability_by_session": reliability_by_session,
        "reliability_mean": reliability_mean,
        "split_half": split_half,
        "ls4_enabled_in_stage1_pilot": False,
        "query_values_read": False,
        "target_query_values_read": False,
    }


def budget_row(
    probe: Mapping,
    *,
    budget: int,
    reliability_mean: float | None,
    reliability_by_session: Mapping[str, float | None],
) -> dict[str, object]:
    pooled = probe["pooled"]
    r2 = float(pooled["weights_r2"]["observed"])
    null_mean = float(pooled["weights_r2"]["null_mean"])
    advantage = float(pooled["weights_r2"]["advantage"])
    return {
        "budget": int(budget),
        "weights_r2": r2,
        "advantage": advantage,
        "null_mean": null_mean,
        "carrier_reliability": reliability_mean,
        "redundancy_fraction": redundancy_fraction(r2, reliability_mean),
        "positive_sessions": int(pooled["positive_sessions"]),
        "carrier_reliability_by_session": {
            name: reliability_by_session[name] for name in plan.SESSIONS
        },
        "encoder_trained_at_support_trials": int(plan.SUPPORT_TRIALS),
        "tokens_evaluated_off_training_budget": int(budget) != int(plan.SUPPORT_TRIALS),
    }


def _is_reliable(value: object, floor: float) -> bool:
    if value is None:
        return False
    return float(value) > float(floor)


def apply_read_rule(rows, rule: Mapping) -> str:
    _require(isinstance(rows, (list, tuple)) and isinstance(rule, Mapping), "read-rule inputs")
    _require(len(rows) >= 1, "read-rule has no budget rows")
    observed = {int(row["budget"]): row for row in rows}
    _require(set(observed) == set(plan.BUDGET_PROBE_BUDGETS), "read-rule budget set drift")
    persists = rule["REDUNDANCY_PERSISTS"]
    breaks = rule["REDUNDANCY_BREAKS_AT_LOW_BUDGET"]
    inconclusive = rule["INCONCLUSIVE_UNRELIABLE_TARGET"]
    ordered = [observed[int(budget)] for budget in plan.BUDGET_PROBE_BUDGETS]
    if all(
        float(row["advantage"]) >= float(persists["min_advantage"])
        and row["redundancy_fraction"] is not None
        and float(row["redundancy_fraction"]) >= float(persists["min_redundancy_fraction"])
        for row in ordered
    ):
        return "REDUNDANCY_PERSISTS"
    m10 = observed[int(plan.SUPPORT_TRIALS)]
    m10_holds = (
        float(m10["weights_r2"]) >= float(breaks["m10_min_weights_r2"])
        and float(m10["advantage"]) >= float(breaks["m10_min_advantage"])
    )
    floor = float(breaks["reliability_floor"])
    reliable = [row for row in ordered if _is_reliable(row["carrier_reliability"], floor)]
    if m10_holds and reliable:
        smallest = min(reliable, key=lambda row: int(row["budget"]))
        fraction = smallest["redundancy_fraction"]
        if (
            fraction is not None
            and float(fraction) <= float(breaks["max_redundancy_fraction_at_smallest_reliable"])
        ):
            return "REDUNDANCY_BREAKS_AT_LOW_BUDGET"
    below_floor = float(inconclusive["reliability_floor"])
    below = [row for row in ordered if int(row["budget"]) < int(inconclusive["budget_below"])]
    if not any(_is_reliable(row["carrier_reliability"], below_floor) for row in below):
        return "INCONCLUSIVE_UNRELIABLE_TARGET"
    return "PARTIAL_DECAY"


def _sealed_token_probe_reference(repo_root: Path) -> dict[str, object]:
    path = Path(repo_root) / plan.TOKEN_PROBE_ROOT_RELATIVE / "summary.json"
    _require(path.is_file(), "sealed token_probe_v1/summary.json missing")
    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes)
    _require(isinstance(payload, dict), "sealed token_probe summary is not an object")
    zfix = payload["pooled_table"]["Z-Fix"]
    return {
        "relative": f"{plan.TOKEN_PROBE_ROOT_RELATIVE}/summary.json",
        "sha256": plan.sha256_bytes(payload_bytes),
        "verdict": payload.get("verdict"),
        "weights_r2": float(zfix["weights_r2"]),
        "advantage": float(zfix["weights_r2_advantage"]),
        "positive_sessions": int(zfix["positive_sessions_weights_r2"]),
        "formal_benchmark_verdict": False,
    }


def _tokens_receipt(arm: str, tokens: Mapping[str, np.ndarray], anchor: Mapping, *, budget: int) -> dict:
    return {
        "schema": "m1_emg_rsyn3_fold_local_budget_probe_tokens_v1",
        "arm": arm,
        "budget": int(budget),
        "checkpoint_sha256": anchor["checkpoint_sha256"],
        "encoder_trained_at_support_trials": int(plan.SUPPORT_TRIALS),
        "tokens_evaluated_off_training_budget": int(budget) != int(plan.SUPPORT_TRIALS),
        "sessions": {
            name: {"sha256": _array_digest(array), "shape": list(array.shape)}
            for name, array in tokens.items()
        },
    }


def _probe_receipt(arm: str, probe: Mapping, *, features: str, budget: int) -> dict:
    return {
        "schema": "m1_emg_rsyn3_fold_local_budget_probe_result_v1",
        "arm": arm,
        "budget": int(budget),
        "features": features,
        "direction_exclusion": "||w|| <= eps on the direction target in use",
        **dict(probe),
        "direction_normalized_uses_normalized_weights": True,
        "direction_raw_uses_raw_weights": True,
        "encoder_trained_at_support_trials": int(plan.SUPPORT_TRIALS),
        "tokens_evaluated_off_training_budget": int(budget) != int(plan.SUPPORT_TRIALS),
    }


def _compute_budget_tokens(student, supports: Mapping[str, np.ndarray], *, budget: int, torch_module):
    tokens: dict[str, np.ndarray] = {}
    with torch_module.no_grad():
        for name, support in supports.items():
            sliced = np.ascontiguousarray(support[:budget], dtype=np.float32)
            _require(sliced.shape[0] == int(budget), f"{name} support budget slice")
            _require(
                sliced.shape[1] == int(plan.TRIAL_LENGTH),
                f"{name} trial length {sliced.shape}",
            )
            identity = student.compute_identity(
                torch_module.as_tensor(sliced[None, ...], dtype=torch_module.float32),
                side_features=None,
            )
            array = np.ascontiguousarray(identity.detach().cpu().numpy(), dtype=np.float32)
            _require(array.ndim == 3 and array.shape[0] == 1, f"{name} identity batch dim")
            token = np.ascontiguousarray(array[0], dtype=np.float32)
            _require(
                token.shape == (sliced.shape[-1], int(plan.WINDOW_SIZE)),
                f"{name} token shape {token.shape}",
            )
            tokens[name] = token
    return tokens


def _assert_m10_matches_fold0_bank(repo_root: Path, m10_bank: Mapping) -> None:
    from . import carrier_bank

    reference = carrier_bank.build_fold0_carrier_bank(repo_root)
    for name in plan.SESSIONS:
        left = np.asarray(m10_bank["normalized"][name]["rSyn3"])
        right = np.asarray(reference["normalized"][name]["rSyn3"])
        _require(np.array_equal(left, right), f"{name} M10 normalized rSyn3 differs from fold0 bank")
        _require(
            np.array_equal(np.asarray(m10_bank["raw"][name]), np.asarray(reference["raw"][name])),
            f"{name} M10 raw carrier differs from fold0 bank",
        )


def execute(repo_root: Path) -> tuple[dict[str, str], str | None, str | None]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    repo_root = Path(repo_root)
    plan.verify_bound_documents(repo_root)
    fold_stage0.verify_sealed_roots(repo_root)
    fold_receipts.refuse_sealed_roots(repo_root / plan.BUDGET_PROBE_ROOT_RELATIVE)
    arm = plan.BUDGET_PROBE_ARM
    _require(arm == "Z-Fix", "budget probe primary arm must be Z-Fix")
    anchor = verify_pilot_arm_anchor(repo_root, arm)
    sealed_token_probe = _sealed_token_probe_reference(repo_root)
    captured_verdict: list[str] = ["UNSET"]
    captured_m10_r2: list[float | None] = [None]

    def launch_builder() -> dict[str, object]:
        import torch

        _require(torch.cuda.is_available() is False, "budget probe requires CPU; CUDA is available")
        return {
            "schema": "m1_emg_rsyn3_fold_local_budget_probe_launch_v1",
            "arm": arm,
            "baseline": plan.TOKEN_PROBE_BASELINE,
            "budgets": list(plan.BUDGET_PROBE_BUDGETS),
            "ridge_lambda": plan.TOKEN_PROBE_RIDGE_LAMBDA,
            "null_permutations": plan.TOKEN_PROBE_NULL_PERMUTATIONS,
            "null_seed": plan.TOKEN_PROBE_NULL_SEED,
            "direction_eps": plan.TOKEN_PROBE_DIRECTION_EPS,
            "reliability_floor": plan.BUDGET_PROBE_RELIABILITY_FLOOR,
            "reliability_scalar": RELIABILITY_SCALAR,
            "reliability_implementation": RELIABILITY_IMPLEMENTATION,
            "reliability_reimplemented": False,
            "read_rule": _jsonable(plan.BUDGET_PROBE_READ_RULE),
            "sessions": list(plan.SESSIONS),
            "support_trials": [0, int(plan.SUPPORT_TRIALS)],
            "encoder_trained_at_support_trials": int(plan.SUPPORT_TRIALS),
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_is_available": bool(torch.cuda.is_available()),
            "target_optimizer_steps": 0,
            "target_query_values_read": False,
            "formal_benchmark_verdict": False,
            "anchor": {
                "checkpoint_sha256": anchor["checkpoint_sha256"],
                "checkpoint_relative": anchor["checkpoint_relative"],
            },
            "sealed_token_probe_v1": sealed_token_probe,
        }

    def body_publisher(artifact) -> dict[str, str]:
        import torch

        _require(torch.cuda.is_available() is False, "budget probe requires CPU; CUDA is available")
        ensure_streaming_path(repo_root)
        loaded, nmf = _fold0_scope_and_nmf(repo_root)
        banks: dict[int, dict[str, Any]] = {}
        for budget in plan.BUDGET_PROBE_BUDGETS:
            banks[int(budget)] = build_budget_carrier_bank(
                repo_root, budget=int(budget), loaded=loaded, nmf=nmf,
            )
            _require(
                banks[int(budget)]["isolation"]["target_query_values_read"] is False,
                "target query leak",
            )
        m10_bank = banks[int(plan.SUPPORT_TRIALS)]
        _assert_m10_matches_fold0_bank(repo_root, m10_bank)
        supports = _load_supports(repo_root, m10_bank)
        support_receipt = {
            name: {"sha256": _array_digest(array), "shape": list(array.shape)}
            for name, array in supports.items()
        }
        lit = load_student(repo_root, repo_root / str(anchor["checkpoint_relative"]), torch)
        lit.eval()
        student = lit.student
        shas: dict[str, str] = {}
        shas["anchor.json"] = artifact.publish_json(
            "anchor.json",
            _jsonable({
                "schema": "m1_emg_rsyn3_fold_local_budget_probe_anchor_v1",
                "arm": arm,
                "pilot": {
                    "checkpoint_sha256": anchor["checkpoint_sha256"],
                    "checkpoint_relative": anchor["checkpoint_relative"],
                    "terminal_sha256": anchor["terminal_sha256"],
                    "sealed_query": anchor["sealed_query"],
                    "sealed_static_r2": anchor["sealed_static_r2"],
                    "sealed_cdm_a_r2": anchor["sealed_cdm_a_r2"],
                },
                "sealed_token_probe_v1": sealed_token_probe,
                "cpu_only": True,
                "formal_benchmark_verdict": False,
                "target_query_values_read": False,
            }),
        )
        table: dict[str, dict[int, dict[str, object]]] = {arm: {}, plan.TOKEN_PROBE_BASELINE: {}}
        reliability_body: dict[str, object] = {
            "schema": "m1_emg_rsyn3_fold_local_budget_probe_reliability_v1",
            "scalar_field": RELIABILITY_SCALAR,
            "scalar_reason": RELIABILITY_REASON,
            "implementation": RELIABILITY_IMPLEMENTATION,
            "reimplemented": False,
            "pooling": "unweighted_mean_over_sessions",
            "sessions": list(plan.SESSIONS),
            "by_budget": {},
        }
        for budget in plan.BUDGET_PROBE_BUDGETS:
            bank = banks[int(budget)]
            session_order = tuple(supports)
            sliced = {name: array[:budget] for name, array in supports.items()}
            for name, array in sliced.items():
                _require(array.shape[0] == int(budget), f"{name} sliced support trials")
            tokens = _compute_budget_tokens(student, supports, budget=int(budget), torch_module=torch)
            carriers = {
                name: np.asarray(bank["normalized"][name]["rSyn3"], dtype=np.float64)
                for name in session_order
            }
            direction_by_session = {
                name: np.ascontiguousarray(np.asarray(bank["raw"][name], dtype=np.float64)[:, :3])
                for name in session_order
            }
            for name in session_order:
                _require(carriers[name].shape[0] == supports[name].shape[-1], f"{name} carrier/unit mismatch")
                _require(carriers[name].shape == (supports[name].shape[-1], 4), f"{name} carrier shape")
                _require(
                    direction_by_session[name].shape == (carriers[name].shape[0], 3),
                    f"{name} raw direction shape",
                )
            probe_kwargs = {
                "ridge_lambda": plan.TOKEN_PROBE_RIDGE_LAMBDA,
                "null_permutations": plan.TOKEN_PROBE_NULL_PERMUTATIONS,
                "null_seed": plan.TOKEN_PROBE_NULL_SEED,
                "eps": plan.TOKEN_PROBE_DIRECTION_EPS,
                "direction_by_session": direction_by_session,
            }
            token_features = {name: tokens[name] for name in session_order}
            zfix_probe = loso_probe(token_features, carriers, **probe_kwargs)
            stats_features = {
                name: raw_stats_features(sliced[name]) for name in session_order
            }
            stats_probe = loso_probe(stats_features, carriers, **probe_kwargs)
            shas[f"tokens_m{budget}.json"] = artifact.publish_json(
                f"tokens_m{budget}.json",
                _jsonable(_tokens_receipt(arm, tokens, anchor, budget=int(budget))),
            )
            shas[f"probe_m{budget}.json"] = artifact.publish_json(
                f"probe_m{budget}.json",
                _jsonable(_probe_receipt(arm, zfix_probe, features="identity_token", budget=int(budget))),
            )
            shas[f"probe_raw_stats_m{budget}.json"] = artifact.publish_json(
                f"probe_raw_stats_m{budget}.json",
                _jsonable(
                    _probe_receipt(
                        plan.TOKEN_PROBE_BASELINE, stats_probe, features="raw_stats", budget=int(budget),
                    )
                ),
            )
            reliability_mean = bank["reliability_mean"]
            reliability_by_session = bank["reliability_by_session"]
            table[arm][int(budget)] = budget_row(
                zfix_probe,
                budget=int(budget),
                reliability_mean=reliability_mean,
                reliability_by_session=reliability_by_session,
            )
            table[plan.TOKEN_PROBE_BASELINE][int(budget)] = budget_row(
                stats_probe,
                budget=int(budget),
                reliability_mean=reliability_mean,
                reliability_by_session=reliability_by_session,
            )
            reliability_body["by_budget"][str(int(budget))] = {
                "budget": int(budget),
                "by_session": reliability_by_session,
                "unweighted_mean": reliability_mean,
                "split_half": bank["split_half"],
            }
            if int(budget) == int(plan.SUPPORT_TRIALS):
                captured_m10_r2[0] = float(table[arm][int(budget)]["weights_r2"])
        del lit
        shas["reliability.json"] = artifact.publish_json(
            "reliability.json", _jsonable(reliability_body),
        )
        zfix_rows = [table[arm][int(budget)] for budget in plan.BUDGET_PROBE_BUDGETS]
        verdict = apply_read_rule(zfix_rows, plan.BUDGET_PROBE_READ_RULE)
        captured_verdict[0] = verdict
        summary = {
            "schema": "m1_emg_rsyn3_fold_local_budget_probe_summary_v1",
            "arm": arm,
            "baseline": plan.TOKEN_PROBE_BASELINE,
            "budgets": list(plan.BUDGET_PROBE_BUDGETS),
            "table": table,
            "rows": {
                arm: zfix_rows,
                plan.TOKEN_PROBE_BASELINE: [
                    table[plan.TOKEN_PROBE_BASELINE][int(budget)]
                    for budget in plan.BUDGET_PROBE_BUDGETS
                ],
            },
            "verdict": verdict,
            "read_rule": _jsonable(plan.BUDGET_PROBE_READ_RULE),
            "reliability_scalar": RELIABILITY_SCALAR,
            "reliability_reason": RELIABILITY_REASON,
            "reliability_implementation": RELIABILITY_IMPLEMENTATION,
            "reliability_reimplemented": False,
            "encoder_trained_at_support_trials": int(plan.SUPPORT_TRIALS),
            "tokens_evaluated_off_training_budget": {
                str(int(budget)): int(budget) != int(plan.SUPPORT_TRIALS)
                for budget in plan.BUDGET_PROBE_BUDGETS
            },
            "formal_benchmark_verdict": False,
            "cpu_only": True,
            "target_optimizer_steps": 0,
            "target_query_values_read": False,
            "support_receipt": support_receipt,
            "sealed_token_probe_v1": sealed_token_probe,
            "m10_cross_check": {
                "sealed_weights_r2": sealed_token_probe["weights_r2"],
                "observed_weights_r2": captured_m10_r2[0],
                "tolerance": 1.0e-6,
            },
        }
        shas["summary.json"] = artifact.publish_json("summary.json", _jsonable(summary))
        observed_m10 = captured_m10_r2[0]
        _require(observed_m10 is not None, "M10 Z-Fix weights_r2 missing")
        expected_m10 = float(sealed_token_probe["weights_r2"])
        _require(
            abs(float(observed_m10) - expected_m10) <= 1.0e-6,
            f"M10 weights_r2 drifted: {observed_m10} vs sealed token_probe_v1 {expected_m10}",
        )
        return shas

    def terminal_builder(shas: Mapping[str, str]) -> dict[str, object]:
        _require(captured_verdict[0] != "UNSET", "verdict was not recorded")
        return {
            "schema": "m1_emg_rsyn3_fold_local_budget_probe_terminal_v1",
            "status": "COMPLETE",
            "verdict": captured_verdict[0],
            "bodies": dict(shas),
            "cpu_only": True,
            "formal_benchmark_verdict": False,
            "target_query_values_read": False,
            "target_optimizer_steps": 0,
        }

    return fold_receipts.run_stage0(
        repo_root / plan.RESULT_ROOT_RELATIVE,
        attempt_payload={
            "schema": "m1_emg_rsyn3_fold_local_budget_probe_attempt_v1",
            "arm": arm,
            "budgets": list(plan.BUDGET_PROBE_BUDGETS),
            "fold": 0,
            "cpu_only": True,
            "target_query_values_read": False,
        },
        launch_builder=launch_builder,
        body_publisher=body_publisher,
        terminal_builder=terminal_builder,
        relative="budget_probe_v1",
    )
