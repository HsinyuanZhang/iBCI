"""Whole-activity-pool Post-Fusion scorer.

The inherited V1 literal scorer is retained only for the PF-MEAN SHA control.
This module changes the residual arms at one precise point: it asks the public
trained adapter for an identity over the complete ordered activity stack at
each causal decode-before-commit state.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np

from . import plan
from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as v1_physical


class PhysicalError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise PhysicalError(message)


def _array_sha(value: np.ndarray) -> str:
    return v1_physical._array_sha256(np.ascontiguousarray(value))


class OrderedActivityPool:
    """Causal support-preserving activity FIFO, not an identity mean pool."""

    def __init__(self, support_activities: Sequence[np.ndarray], *, capacity: int | None) -> None:
        _require(len(support_activities) == plan.BUDGET, "operator pool needs exact D-opt-k4 support")
        _require(capacity is None or int(capacity) == plan.POOL_CAPACITY, "activity pool capacity drift")
        self.capacity = capacity
        self.support_count = len(support_activities)
        self._support = [np.ascontiguousarray(item, dtype=np.float32) for item in support_activities]
        self._members: deque[np.ndarray] = deque(self._support)
        self.completed_count = 0
        self.evictions = 0
        self._support_sha = [_array_sha(item) for item in self._support]

    @property
    def pool_count(self) -> int:
        return len(self._members)

    def stack(self) -> np.ndarray:
        values = list(self._members)
        _require(values and all(value.shape == self._support[0].shape for value in values), "activity stack shape drift")
        self.assert_support_retained()
        return np.ascontiguousarray(np.stack(values), dtype=np.float32)

    def assert_support_retained(self) -> list[str]:
        values = list(self._members)
        _require(len(values) >= self.support_count, "support prefix shortened")
        actual: list[str] = []
        for index, (expected, expected_sha) in enumerate(zip(self._support, self._support_sha, strict=True)):
            observed = values[index]
            _require(observed is expected and _array_sha(observed) == expected_sha,
                     f"support activity {index} evicted/replaced/mutated")
            actual.append(expected_sha)
        return actual

    def commit(self, activity: np.ndarray) -> dict[str, int]:
        value = np.ascontiguousarray(activity, dtype=np.float32)
        _require(value.shape == self._support[0].shape, "completed activity geometry drift")
        before = self.pool_count
        before_evictions = self.evictions
        self._members.append(value)
        if self.capacity is not None and self.pool_count > self.capacity:
            _require(self.pool_count > self.support_count, "activity FIFO would evict support")
            del self._members[self.support_count]
            self.evictions += 1
        self.completed_count += 1
        self.assert_support_retained()
        return {"members_before": before, "members_after": self.pool_count,
                "evictions_before": before_evictions, "evictions_after": self.evictions}


def whole_pool_identity(*, adapter: Any, torch: Any, activities: np.ndarray, side_tensor: Any,
                        expected_channels: int = 96) -> Any:
    """The training operator: one public-adapter call over `[1,M,100,96]`."""
    array = np.ascontiguousarray(activities, dtype=np.float32)
    _require(array.ndim == 3 and array.shape[1:] == (100, int(expected_channels)) and array.shape[0] >= plan.BUDGET,
             "whole activity stack must be [M,100,96] with M>=4")
    with torch.inference_mode():
        identity = adapter.forward_batch(torch.from_numpy(array).unsqueeze(0), side_features=side_tensor)
    _require(tuple(identity.shape) == (1, int(expected_channels), 50) and bool(torch.isfinite(identity).all().item()),
             "whole-pool trained identity shape/finite drift")
    return identity


def singleton_mean_identity(*, adapter: Any, torch: Any, activities: np.ndarray, side_tensor: Any,
                            expected_channels: int = 96) -> Any:
    """Audit-only legacy singleton decomposition; never the corrected residual scorer."""
    values = [v1_physical.trained_trial_identity(adapter=adapter, torch=torch, activity=item,
                                                  side_tensor=side_tensor, expected_channels=expected_channels)
              for item in np.asarray(activities, dtype=np.float32)]
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total / len(values)


def _canonical_records(materialized: Mapping[str, Any], selected: Sequence[str] | None) -> list[Mapping[str, Any]]:
    records = materialized.get("records")
    _require(isinstance(records, Mapping) and len(records) == 13, "13 locked input records absent")
    selected_set = None if selected is None else set(selected)
    if selected_set is not None:
        _require(selected_set and selected_set <= set(records), "selected record key drift")
    out: list[Mapping[str, Any]] = []
    for surface in plan.SURFACES:
        rows = sorted((record for record in records.values() if record.get("surface") == surface),
                      key=lambda record: str(record["session"]))
        _require(len(rows) == plan.ROSTER_SIZES[surface], "input surface roster drift")
        out.extend(record for record in rows if selected_set is None or str(record["key"]) in selected_set)
    return out


def _corrected_record_rows(*, prepared: Mapping[str, Any], record: Mapping[str, Any], arms: Sequence[str],
                           expected_channels: int) -> list[dict[str, Any]]:
    import torch
    modules = prepared["modules"]; adapters = prepared["adapters"]
    v1_physical._validate_runtime_record(record)
    runtime = record["_runtime"]; support = runtime["support"]; query_rows = tuple(runtime["query_rows"])
    dataset = runtime["dataset"]; session = str(record["session"])
    neural = np.asarray(dataset.neural_data[session], dtype=np.float32)
    targets_all = np.asarray(dataset.covariate_data[session], dtype=np.float32)
    selected = np.asarray(support["selected"], dtype=np.int64)
    support_activity = [np.ascontiguousarray(np.asarray(support["activities"][index], dtype=np.float32)) for index in selected]
    out: list[dict[str, Any]] = []
    for arm in arms:
        _require(arm in ("PF-R1", "PF-R50"), "whole-pool correction applies only residual arms")
        model = modules[arm]; adapter = adapters[arm]
        _require(adapter is model.student.id_encoder and not model.training
                 and all(not parameter.requires_grad for parameter in model.parameters()), f"{arm}: frozen adapter drift")
        state_before = v1_physical._student_state_sha(model)
        side = v1_physical._side_tensor_for_record(torch=torch, record=record)
        pools = {"FIXED30": OrderedActivityPool(support_activity, capacity=plan.POOL_CAPACITY),
                 "UNCAPPED": OrderedActivityPool(support_activity, capacity=None)}
        state: dict[str, dict[str, list[Any]]] = {law: {"pred": [], "target": [], "starts": [], "trace": [], "identity": []}
                                                   for law in plan.LAWS}
        support_order_sha = plan.sha256_bytes(plan.canonical_json([_array_sha(value) for value in support_activity]))
        for query in query_rows:
            starts = np.asarray(query["metric_starts"], dtype=np.int64)
            values: dict[str, np.ndarray] = {}
            if starts.size:
                indices = starts[:, None] + np.arange(50, dtype=np.int64)[None, :]
                windows = np.ascontiguousarray(neural[indices], dtype=np.float32)
                target = np.ascontiguousarray(targets_all[starts + 49], dtype=np.float32)
                for law in plan.LAWS:
                    identity = whole_pool_identity(adapter=adapter, torch=torch, activities=pools[law].stack(),
                                                   side_tensor=side, expected_channels=expected_channels)
                    prediction = v1_physical._decode_identity_cpu(torch=torch, model=model, neural_windows=windows,
                                                                    identity=identity, batch_size=plan.CPU_DECODE_BATCH_SIZE)
                    _require(prediction.shape == target.shape and np.isfinite(prediction).all(), "corrected decode drift")
                    values[law] = prediction
                    state[law]["pred"].append(prediction); state[law]["target"].append(target); state[law]["starts"].append(starts)
                    state[law]["identity"].append(_array_sha(identity.detach().cpu().numpy()))
                if pools["FIXED30"].evictions == 0:
                    _require(np.array_equal(values["FIXED30"], values["UNCAPPED"]),
                             "laws differ before first corrected FIFO eviction")
            for law in plan.LAWS:
                pool = pools[law]
                transaction = pool.commit(np.asarray(query["activity"], dtype=np.float32))
                state[law]["trace"].append({"trial_id": str(query["trial_id"]), "position": int(query["position"]),
                    "activity_sha256": _array_sha(np.asarray(query["activity"], dtype=np.float32)),
                    "metric_starts_sha256": _array_sha(starts), **transaction,
                    "support_activity_order_sha256": support_order_sha})
        for law in plan.LAWS:
            _require(pools[law].assert_support_retained() == [_array_sha(value) for value in support_activity],
                     "support activity receipt drift")
            prediction = np.ascontiguousarray(np.concatenate(state[law]["pred"]), dtype=np.float32)
            target = np.ascontiguousarray(np.concatenate(state[law]["target"]), dtype=np.float32)
            starts = np.ascontiguousarray(np.concatenate(state[law]["starts"]), dtype=np.int64)
            _require(record["query_starts_sha256"] == v1_physical._governed_array_sha256(starts)
                     and record["target_sha256"] == v1_physical._governed_array_sha256(target), "corrected row escaped V2 input")
            state_after = v1_physical._student_state_sha(model)
            _require(state_after == state_before, "corrected scorer mutated frozen model")
            trace = state[law]["trace"]
            out.append({"arm": arm, "memory_law": law, "surface": record["surface"], "session": record["session"],
                "budget": plan.BUDGET, "input_authority_key": record["key"], "r2": v1_physical._r2(target, prediction),
                "prediction_sha256": v1_physical._governed_array_sha256(prediction),
                "target_sha256": v1_physical._governed_array_sha256(target), "window_count": int(starts.size),
                "initial_members": plan.BUDGET, "final_members": pools[law].pool_count,
                "commits": pools[law].completed_count, "evictions": pools[law].evictions,
                "causal_trace_sha256": plan.sha256_bytes(plan.canonical_json(trace)),
                "query_identity_trace_sha256": plan.sha256_bytes(plan.canonical_json(state[law]["identity"])),
                "support_never_evicted": True, "support_retained_object_order_exact": True,
                "support_activity_order_sha256": support_order_sha, "operator": "trained_adapter_whole_ordered_activity_stack",
                "model_state_before_sha256": state_before, "model_state_after_sha256": state_after,
                "parameter_updates": 0, "target_updates": 0, "pooled_comparator_key": record["key"]})
        fixed, uncapped = out[-2], out[-1]
        if fixed["evictions"] == 0:
            _require(fixed["prediction_sha256"] == uncapped["prediction_sha256"] and fixed["r2"] == uncapped["r2"],
                     "corrected no-eviction law parity drift")
    return out


def assert_pfmean_control(rows: Sequence[Mapping[str, Any]], v2_control: Mapping[str, Mapping[str, Any]]) -> None:
    controls = [row for row in rows if row.get("arm") == "PF-MEAN"]
    _require(len(controls) == 26 and len(v2_control) == 26, "PF-MEAN control cardinality drift")
    for row in controls:
        key = f"{row['surface']}|{row['session']}|{row['memory_law']}"
        expected = v2_control.get(key)
        _require(isinstance(expected, Mapping), f"V2 PF-MEAN control missing: {key}")
        _require(row.get("prediction_sha256") == expected.get("prediction_sha256")
                 and row.get("target_sha256") == expected.get("target_sha256")
                 and row.get("r2") == expected.get("r2")
                 and row.get("window_count") == expected.get("window_count"),
                 f"PF-MEAN exact V2 control mismatch: {key}")


def score_operator_corrected_78_rows(*, prepared: Mapping[str, Any], materialized: Mapping[str, Any],
                                     v2_control: Mapping[str, Mapping[str, Any]],
                                     record_keys: Sequence[str] | None = None,
                                     expected_channels: int = 96) -> list[dict[str, Any]]:
    """Return canonical V2-compatible rows with residual arms whole-pool corrected."""
    import os
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "operator correction is CPU-only")
    # The inherited literal scorer deliberately remains PF-MEAN's arithmetic
    # authority after float32 FIFO eviction.  It also supplies the exact V2
    # row topology from which only residual rows are substituted.
    legacy = v1_physical.score_78_rows_from_materialized(prepared=prepared, materialized=materialized,
                                                          expected_channels=expected_channels, record_keys=record_keys)
    mean = [row for row in legacy if row["arm"] == "PF-MEAN"]
    assert_pfmean_control(mean, v2_control)
    corrected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in _canonical_records(materialized, record_keys):
        for row in _corrected_record_rows(prepared=prepared, record=record, arms=("PF-R1", "PF-R50"),
                                          expected_channels=expected_channels):
            corrected[(row["surface"], row["session"], row["arm"], row["memory_law"])] = row
    output: list[dict[str, Any]] = []
    for row in legacy:
        key = (row["surface"], row["session"], row["arm"], row["memory_law"])
        output.append(corrected.get(key, dict(row)))
    _require(len(output) == len(legacy) and len(corrected) == 2 * 2 * len(_canonical_records(materialized, record_keys)),
             "operator-corrected row topology drift")
    return output

