"""Pure 13-session / 78-row AJPF scorer receipt topology and gates."""
from __future__ import annotations
import math
import random
import statistics
import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence
from . import plan

class ScoreError(ValueError): pass
def _need(ok: bool, message: str) -> None:
    if not ok: raise ScoreError(message)
def _hex(value: object) -> bool: return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def _array_witness(value: Any) -> str:
    """Typed array digest: raw selected carrier cannot be confused with M30 audit."""
    import numpy as np
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)}, sort_keys=True).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def selected_support4_carrier_lineage(*, record: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the scored carrier witness from the one materialized record."""
    import numpy as np
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import source_replay
    material = source_replay.material_from_g00m_runtime_record(record)
    raw_carrier = np.ascontiguousarray(np.asarray(material.selected_support4_carrier_hz, dtype=np.float64))
    selected = np.asarray(material.support_indices, dtype=np.int64)
    return {"selected_indices": [int(value) for value in selected],
            "selected_order_sha256": _array_witness(selected),
            "raw_float64_sha256": _array_witness(raw_carrier),
            "counts_per_bin_float32_sha256": _array_witness(np.asarray(raw_carrier * 0.020, dtype=np.float32)),
            "normalized_side_sha256": _array_witness(np.asarray(material.normalized_side, dtype=np.float32)),
            "side_normalizer_sha256": str(material.normalizer_sha256),
            "query_activity_sha256": str(record["query_activity_sha256"]),
            "raw_m30_audit_sha256": _array_witness(np.asarray(material.raw_m30_hz_audit, dtype=np.float64))}

def validate_rows(rows: Sequence[dict[str,Any]], sessions: Sequence[str]) -> None:
    ordered_sessions=tuple(sorted(sessions)); _need(len(ordered_sessions)==13 and len(set(ordered_sessions))==13,"AJPF scorer session roster drift")
    expected=[(session,arm,law) for session in ordered_sessions for arm in plan.ARM_ORDER for law in ("FIXED30","UNCAPPED")]
    observed=[(row.get("session"),row.get("arm"),row.get("law")) for row in rows]
    _need(observed==expected,"AJPF scorer exact 78-row canonical order drift")
    for row in rows:
        _need(_hex(row.get("prediction_sha256")) and _hex(row.get("target_sha256")) and _hex(row.get("starts_sha256")),"AJPF scorer digest drift")
        _need(row.get("state_before_sha256")==row.get("state_after_sha256") and row.get("parameter_updates")==0 and row.get("target_updates")==0,"AJPF scorer mutation drift")
        _need(isinstance(row.get("r2"),float) and math.isfinite(row["r2"]),"AJPF scorer R2 drift")
        lineage = row.get("selected_support4_carrier_lineage")
        _need(isinstance(lineage, Mapping) and set(lineage) == {
            "selected_indices", "selected_order_sha256", "raw_float64_sha256", "counts_per_bin_float32_sha256",
            "normalized_side_sha256", "side_normalizer_sha256", "query_activity_sha256", "raw_m30_audit_sha256"},
            "AJPF selected-support4 carrier lineage drift")
        _need(isinstance(lineage["selected_indices"], list) and len(lineage["selected_indices"]) == plan.SUPPORT_COUNT
              and all(_hex(lineage[key]) for key in lineage if key != "selected_indices"),
              "AJPF selected-support4 carrier lineage witness drift")
    grouped=defaultdict(list)
    for row in rows: grouped[(row["session"],row["law"])].append(row)
    _need(all(len(values)==3 and len({v["target_sha256"] for v in values})==1 and len({v["starts_sha256"] for v in values})==1
              and len({json.dumps(v["selected_support4_carrier_lineage"], sort_keys=True) for v in values}) == 1
              for values in grouped.values()),"AJPF same-input scorer drift")


def bootstrap_ci(values: Sequence[float], *, seed: int = plan.SEED, resamples: int = 10_000) -> dict[str, float]:
    """Fixed-seed ordinary session bootstrap over equal-session means."""
    _need(len(values) > 0 and resamples == 10_000, "AJPF bootstrap law drift")
    numeric = [float(value) for value in values]
    _need(all(math.isfinite(value) for value in numeric), "AJPF bootstrap nonfinite input")
    generator = random.Random(seed)
    samples = sorted(sum(numeric[generator.randrange(len(numeric))] for _ in numeric) / len(numeric)
                     for _ in range(resamples))
    return {"seed": seed, "resamples": resamples, "low": float(samples[249]), "high": float(samples[9749])}


def _summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = [float(value) for value in values]
    return {"session_count": len(ordered), "equal_session_mean": float(statistics.fmean(ordered)),
            "median": float(statistics.median(ordered)), "positive_count": sum(value > 0.0 for value in ordered),
            "worst_session": float(min(ordered)), "bootstrap_95_ci": bootstrap_ci(ordered)}


def build_preregistered_gates(*, rows: Sequence[dict[str, Any]], historical_pooled: Mapping[str, Mapping[str, Any]],
                               external_sessions: Sequence[str], within_sessions: Sequence[str]) -> dict[str, Any]:
    """Compute only the two frozen J-R1/UNCAPPED contrasts and their gates."""
    validate_rows(rows, tuple(sorted(tuple(external_sessions) + tuple(within_sessions))))
    keyed = {(str(row["session"]), str(row["arm"]), str(row["law"])): row for row in rows}
    _need(len(historical_pooled) == 13, "AJPF historical POOLED roster drift")
    result: dict[str, Any] = {"schema": "m2_anchored_joint_postfusion_v1_preregistered_gates",
                              "primary": "J-R1/UNCAPPED", "contrasts": {}}
    for surface, sessions in (("external", tuple(external_sessions)), ("within", tuple(within_sessions))):
        method = []
        practical = []
        for session in sessions:
            r1 = float(keyed[(session, "J-R1", "UNCAPPED")]["r2"])
            native = float(keyed[(session, "J-NATIVE", "UNCAPPED")]["r2"])
            pooled = historical_pooled.get(session)
            _need(isinstance(pooled, Mapping) and math.isfinite(float(pooled.get("r2"))), "AJPF POOLED comparison absent")
            method.append(r1 - native)
            practical.append(r1 - float(pooled["r2"]))
        result["contrasts"][surface] = {"method_delta": {"per_session": dict(zip(sessions, method, strict=True)), **_summary(method)},
                                         "practical_delta": {"per_session": dict(zip(sessions, practical, strict=True)), **_summary(practical)}}
    external = result["contrasts"]["external"]
    def passed(summary: Mapping[str, Any]) -> bool:
        return (float(summary["equal_session_mean"]) >= 0.010 and int(summary["positive_count"]) >= 4
                and float(summary["worst_session"]) >= -0.015)
    result["paper_success"] = passed(external["method_delta"]) and passed(external["practical_delta"])
    result["no_target_selection_or_update"] = True
    return result


def score_whole_stack_78_cpu(*, torch: Any, modules: Mapping[str, Any], materialized: Mapping[str, Any]) -> list[dict[str, Any]]:
    """AJPF's generalized strict scorer, over one 13-session materialization.

    Unlike the historical PF scorer, each post-fusion identity is evaluated on
    the complete ordered raw activity stack at every decode.  This is also the
    only scorer path used for J-NATIVE: all six rows therefore share exactly
    the same support, selected-support4 carrier, starts and targets.
    """
    import os
    import hashlib
    import numpy as np
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import source_replay
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as old_physical
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ScoreError("AJPF whole-stack score must be a fresh CPU-only process")
    if tuple(modules) != plan.ARM_ORDER:
        raise ScoreError("AJPF strict scorer arm topology drift")
    records = materialized.get("records")
    if not isinstance(records, Mapping) or len(records) != 13:
        raise ScoreError("AJPF scorer materialized roster drift")
    ordered_records = sorted(records.values(), key=lambda record: str(record["session"]))
    # A lexical order is stable and makes the 78-row receipt independent of
    # old surface-specific screen ordering.
    rows: list[dict[str, Any]] = []
    for record in ordered_records:
        material = source_replay.material_from_g00m_runtime_record(record)
        runtime = record["_runtime"]
        dataset = runtime["dataset"]
        neural = np.asarray(dataset.neural_data[record["session"]], dtype=np.float32)
        targets_all = np.asarray(dataset.covariate_data[record["session"]], dtype=np.float32)
        side = torch.from_numpy(np.ascontiguousarray(material.normalized_side)).unsqueeze(0)
        lineage = selected_support4_carrier_lineage(record=record)
        for arm in plan.ARM_ORDER:
            model = modules[arm]
            if model.training or any(parameter.requires_grad for parameter in model.parameters()):
                raise ScoreError(f"AJPF scorer {arm} is not frozen/eval")
            adapter = model.student.id_encoder
            before = old_physical._student_state_sha(model)
            for law in ("FIXED30", "UNCAPPED"):
                pool = source_replay.OrderedRawActivityPool(
                    support_trial_ids=material.support_indices,
                    support_activities=[material.activities[index] for index in material.support_indices],
                    capacity=30 if law == "FIXED30" else None)
                predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []; starts_out: list[np.ndarray] = []
                trace: list[tuple[int, tuple[int, ...]]] = []
                with torch.inference_mode():
                    for query in material.query_rows:
                        starts = np.asarray(query["metric_starts"], dtype=np.int64)
                        if starts.size:
                            windows = np.ascontiguousarray(neural[starts[:, None] + np.arange(50)[None, :]], dtype=np.float32)
                            calibration = torch.from_numpy(np.ascontiguousarray(pool.stack())).unsqueeze(0)
                            identity = adapter.forward_batch(calibration, side_features=side)
                            pieces = []
                            for offset in range(0, len(windows), 1024):
                                output = model.student.decode_with_identity(torch.from_numpy(windows[offset:offset + 1024]), identity)
                                pieces.append(output[:, -1, :].detach().cpu().numpy().astype(np.float32, copy=False) / 5.0)
                            predictions.append(np.ascontiguousarray(np.concatenate(pieces), dtype=np.float32))
                            targets.append(np.ascontiguousarray(targets_all[starts + 49], dtype=np.float32))
                            starts_out.append(starts)
                        pool.commit_completed(trial_id=int(query["position"]), activity=np.asarray(query["activity"], dtype=np.float32))
                        trace.append((int(query["position"]), pool.member_trial_ids))
                prediction = np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32)
                target = np.ascontiguousarray(np.concatenate(targets), dtype=np.float32)
                starts = np.ascontiguousarray(np.concatenate(starts_out), dtype=np.int64)
                after = old_physical._student_state_sha(model)
                if before != after:
                    raise ScoreError(f"AJPF scorer mutated {arm}")
                if str(record["query_starts_sha256"]) != old_physical._governed_array_sha256(starts) or str(record["target_sha256"]) != old_physical._governed_array_sha256(target):
                    raise ScoreError("AJPF scorer escaped shared starts/targets")
                rows.append({"session": str(record["session"]), "surface": str(record["surface"]), "arm": arm, "law": law,
                             "r2": float(old_physical._r2(target, prediction)),
                             "prediction_sha256": old_physical._governed_array_sha256(prediction),
                             "target_sha256": old_physical._governed_array_sha256(target),
                             "starts_sha256": old_physical._governed_array_sha256(starts),
                             "window_count": int(len(starts)), "state_before_sha256": before, "state_after_sha256": after,
                             "parameter_updates": 0, "target_updates": 0, "support_never_evicted": True,
                             "causal_trace_sha256": hashlib.sha256(repr(trace).encode("utf-8")).hexdigest(),
                             "selected_support4_carrier_lineage": lineage})
    validate_rows(rows, tuple(record["session"] for record in ordered_records))
    return rows
