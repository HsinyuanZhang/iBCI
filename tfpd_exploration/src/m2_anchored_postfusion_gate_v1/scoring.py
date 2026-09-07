"""Post-attempt APFG target scoring over already materialized locked inputs.

This module owns no loader.  The caller must supply the single locked 13-row
materialization used by the historical POOLED authority; this prevents an
accidental second DataModule/model construction while evaluating APFG.
"""
from __future__ import annotations

from typing import Any, Mapping

from . import plan
from .adapter import exact_positive_zero
from .laws import LAWS, validate_rows
from .source_replay import material_from_g00m_runtime_record, rollout_apfg_raw_pool


class ScoringError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ScoringError(message)


def _state_sha(module: Any) -> str:
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1.physical import _student_state_sha
    return str(_student_state_sha(module))


def _row(*, system: str, law: str, record: Mapping[str, Any], result: Mapping[str, Any], state: str) -> dict[str, object]:
    return {"system": system, "memory_law": law, "surface": str(record["surface"]),
            "session": str(record["session"]), "budget": 4, "input_authority_key": str(record["key"]),
            "r2": float(result["r2"]), "prediction_sha256": str(result["governed_prediction_sha256"]),
            "target_sha256": str(result["governed_target_sha256"]),
            "query_starts_sha256": str(result["governed_query_starts_sha256"]),
            "window_count": int(result["window_count"]), "initial_members": 4,
            "final_members": int(result["final_members"]), "commits": int(result["commits"]),
            "evictions": int(result["evictions"]), "causal_trace_sha256": str(result["causal_trace_sha256"]),
            "model_state_before_sha256": state, "model_state_after_sha256": state,
            "parameter_updates": 0, "target_updates": 0, "activity_authority": plan.ACTIVITY_AUTHORITY,
            "branch_cache_hits": int(result["branch_cache_hits"]), "branch_cache_misses": int(result["branch_cache_misses"])}


def _strict_zero_match(row: Mapping[str, object], comparator: Mapping[str, object]) -> None:
    _require(int(row["window_count"]) == int(comparator["window_count"]), "APFG zero POOLED window mismatch")
    _require(str(row["prediction_sha256"]) == str(comparator["prediction_sha256"]), "APFG zero POOLED prediction mismatch")
    _require(str(row["target_sha256"]) == str(comparator["target_sha256"]), "APFG zero POOLED target mismatch")
    _require(str(row["query_starts_sha256"]) == str(comparator["query_starts_sha256"]), "APFG zero POOLED starts mismatch")
    _require(float(row["r2"]) == float(comparator["r2"]), "APFG zero POOLED R2 mismatch")


def score_locked_records(*, torch: Any, zero_module: Any, learned_module: Any,
                         records: Mapping[str, Mapping[str, Any]], pooled_comparators: Mapping[str, Mapping[str, object]],
                         device: Any, decode_batch_size: int) -> list[dict[str, object]]:
    """Score exact 52 APFG rows without target adaptation or model mutation."""
    _require(len(records) == 13 and len(pooled_comparators) == 13, "APFG locked record/comparator topology drift")
    zero_adapter = zero_module.student.id_encoder
    learned_adapter = learned_module.student.id_encoder
    _require(exact_positive_zero(zero_adapter.alpha) and not zero_adapter._alpha_training_enabled,
             "APFG zero scorer must be exact positive-zero native path")
    _require(not learned_adapter._alpha_training_enabled, "APFG learned scorer must remain eval")
    for module in (zero_module, learned_module):
        _require(not module.training and all(not item.requires_grad for item in module.parameters()),
                 "APFG target scorer must be fully frozen/eval")
    result: list[dict[str, object]] = []
    cache: dict[str, tuple[Any, Any]] = {}
    for surface in ("external_post30_local", "within_post30"):
        selected = sorted((item for item in records.values() if item.get("surface") == surface), key=lambda item: str(item["session"]))
        expected = 6 if surface == "external_post30_local" else 7
        _require(len(selected) == expected, "APFG target surface roster drift")
        for record in selected:
            material = material_from_g00m_runtime_record(record)
            dataset = record["_runtime"]["dataset"]
            for law in LAWS:
                before_zero = _state_sha(zero_module)
                zero = rollout_apfg_raw_pool(torch=torch, model=zero_module, material=material, dataset=dataset,
                                             law=law, device=device, batch_size=decode_batch_size, branch_cache=cache)
                after_zero = _state_sha(zero_module)
                _require(before_zero == after_zero, "APFG zero target scoring mutated selected checkpoint")
                zero_row = _row(system="APFG-ZERO", law=law, record=record, result=zero, state=before_zero)
                if law == "FIXED30":
                    comparator = pooled_comparators.get(str(record["key"]))
                    _require(isinstance(comparator, Mapping), "APFG sealed POOLED comparator missing")
                    _strict_zero_match(zero_row, comparator)
                result.append(zero_row)
                before_learned = _state_sha(learned_module)
                learned = rollout_apfg_raw_pool(torch=torch, model=learned_module, material=material, dataset=dataset,
                                                law=law, device=device, batch_size=decode_batch_size, branch_cache=cache)
                after_learned = _state_sha(learned_module)
                _require(before_learned == after_learned, "APFG learned target scoring mutated refit model")
                result.append(_row(system="APFG-LEARNED", law=law, record=record, result=learned, state=before_learned))
    validate_rows(result)
    return result


__all__ = ("ScoringError", "score_locked_records")
