"""CPU-only synthetic and real-Cell-D gates for the pure-speed evaluator path."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
for _candidate in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration/src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.causal_dual_memory_cell_d_speed_v1 import accelerator, plan  # noqa: E402


def _digest_tensor(value: Any) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _host_rng_snapshot(torch: Any) -> tuple[object, tuple[object, ...], Any]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _restore_host_rng(torch: Any, state: tuple[object, tuple[object, ...], Any]) -> None:
    python_state, numpy_state, torch_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def _same_host_rng(torch: Any, left: tuple[object, tuple[object, ...], Any], right: tuple[object, tuple[object, ...], Any]) -> bool:
    left_python, left_numpy, left_torch = left
    right_python, right_numpy, right_torch = right
    return bool(
        left_python == right_python
        and left_numpy[0] == right_numpy[0]
        and np.array_equal(left_numpy[1], right_numpy[1])
        and left_numpy[2:] == right_numpy[2:]
        and torch.equal(left_torch, right_torch)
    )


def _real_cell_d(torch: Any) -> Any:
    """Build/materialize the real graph while restoring construction RNG effects."""
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    snapshot = _host_rng_snapshot(torch)
    try:
        model = build_population_robustness_model(seed=42, cell="D").eval()
        with torch.no_grad():
            model(
                torch.zeros((1, 50, 8), dtype=torch.float32),
                calib_trials=torch.zeros((1, 3, 100, 8), dtype=torch.float32),
                side_features=torch.zeros((1, 8, 4), dtype=torch.float32),
            )
        return model
    finally:
        _restore_host_rng(torch, snapshot)


def _eager_b128(model: Any, torch: Any, *, neural: Any, stack: Any, side: Any) -> Any:
    chunks: list[Any] = []
    with torch.no_grad():
        for start in range(0, int(neural.shape[0]), plan.LOGICAL_EVAL_BATCH_SIZE):
            stop = min(start + plan.LOGICAL_EVAL_BATCH_SIZE, int(neural.shape[0]))
            calibration = stack.unsqueeze(0).expand(stop - start, -1, -1, -1)
            side_batch = side.unsqueeze(0).expand(stop - start, -1, -1)
            value, _identity = model(neural[start:stop], calib_trials=calibration, side_features=side_batch)
            chunks.append(value.detach().cpu().contiguous())
    return torch.cat(chunks, dim=0)


def _run_cached(
    model: Any, torch: Any, *, neural: Any, stack: Any, side: Any,
    total_queries: int, query_index: int, path: str, held_mask: Any | None = None,
) -> tuple[accelerator.IdentityCacheAccelerator, accelerator.AcceleratedForwardResult]:
    engine = accelerator.IdentityCacheAccelerator()
    engine.begin_session(total_query_trials=total_queries)
    engine.begin_query(query_index, query_trial_id=f"synthetic-query-{query_index}")
    result = engine.forward(
        model=model, neural_windows=neural, activity_stack=stack, normalized_t4=side,
        path=path, held_mask=held_mask,
    )
    assert result.prediction.device.type == "cpu"
    assert torch.cuda.is_initialized() is False
    return engine, result


def test_static_plan_closure_and_dry_cli_are_torch_free() -> None:
    assert "torch" not in sys.modules
    dry = plan.dry_plan()
    assert dry["logical_eval_batch_size"] == 128
    assert dry["numeric_batch_variants"]["selected"] is None
    assert dry["numeric_batch_variants"]["candidates"] == [1024, 2048]
    assert dry["numeric_batch_variants"]["not_selectable_in_pure_speed_successor"] is True
    closure = plan.implementation_closure(ROOT).payload()
    paths = [row["path"] for row in closure["paths"]]
    assert len(paths) == len(plan.IMPLEMENTATION_PATHS)
    assert plan.OPTIMIZATION_NOTES_RELATIVE in paths
    assert plan.WORKORDER_RELATIVE in paths
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_score_v8/physical.py" in paths
    assert "tfpd_exploration/src/precision_aware_causal_dual_memory_cell_d_score_v1/physical.py" in paths
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py" in paths
    assert closure["paths"][-1]["path"] == "tfpd_exploration/tests/test_causal_dual_memory_cell_d_speed_v1.py"
    assert plan.validate_implementation_closure(closure) == closure
    forged = copy.deepcopy(closure)
    forged["paths"][1]["sha256"] = "0" * 64
    with pytest.raises(plan.CDMDSpeedPlanError, match="canonical|workorder"):
        plan.validate_implementation_closure(forged)
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_speed_v1.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--dry"], cwd=ROOT, check=True, text=True, capture_output=True,
        env={
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"],
        },
    )
    payload = json.loads(completed.stdout)
    assert payload["public_cli_imports_torch"] is False
    assert payload["public_cli_executes"] is False
    assert payload["numeric_batch_variants"]["selected"] is None
    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from src.causal_dual_memory_cell_d_speed_v1 import plan; "
            "print('TORCH_PRESENT='+str('torch' in sys.modules)); print(plan.CELL)",
        ],
        cwd=ROOT, check=True, text=True, capture_output=True,
        env={
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": f"{ROOT / 'tfpd_exploration'}:{ROOT / 'tfpd_exploration/src'}", "PATH": os.environ["PATH"],
        },
    )
    assert "TORCH_PRESENT=False" in probe.stdout


def test_real_cpu_cell_d_cached_identity_is_bitwise_eager_b128_and_rng_safe() -> None:
    import torch
    from src.tfpd_lane import pop_robust

    assert torch.cuda.is_initialized() is False
    model = _real_cell_d(torch)
    assert model.id_encoder.__class__.__name__ == "SideFeatureEarlyPoolEncoder"
    assert not hasattr(model.id_encoder, "forward_batch_with_gate")
    torch.manual_seed(811)
    neural = torch.randn((257, 50, 8), dtype=torch.float32)
    stack = torch.randn((10, 100, 8), dtype=torch.float32)
    side = torch.randn((8, 4), dtype=torch.float32)
    host_before = _host_rng_snapshot(torch)
    eager = _eager_b128(model, torch, neural=neural, stack=stack, side=side)
    host_after_eager = _host_rng_snapshot(torch)
    _restore_host_rng(torch, host_before)
    with pop_robust.dynamic_dropout_recorder() as recorder:
        engine, cached = _run_cached(
            model, torch, neural=neural, stack=stack, side=side,
            total_queries=3, query_index=0, path="full",
        )
    host_after_cached = _host_rng_snapshot(torch)
    assert torch.equal(eager, cached.prediction)
    assert _digest_tensor(eager) == cached.prediction_sha256 == _digest_tensor(cached.prediction)
    assert cached.logical_chunk_count == 3
    assert cached.actual_model_forward_count == 4  # first exact-state chunk repeats once
    assert cached.identity_encoder_forward_count == 1
    assert recorder["uniform_calls"] == 0 and recorder["dropout_calls"] == []
    assert _same_host_rng(torch, host_after_eager, host_before)
    assert _same_host_rng(torch, host_after_cached, host_before)
    proof = engine.payload()
    assert proof["identity_cache_entry_count"] == 1
    assert proof["actual_model_forward_count_by_path"] == {"full": 4}
    coordinate = proof["repeat_audit"]["coordinates"][0]
    assert coordinate["reasons"] == ["canonical_full_first_state"]
    assert coordinate["query_trial_id"] == "synthetic-query-0"
    assert coordinate["group"] is None
    assert coordinate["outputs_bitwise_equal"] is True
    assert len(coordinate["repeated_prediction_sha256"]) == 64
    # Sealed Cell-D has no held-group trajectory and may therefore finalize
    # after its canonical full proof alone.  The dynamic CDM-D path cannot.
    engine.repeat_audit.require_complete_coverage(require_held_group_0=False)
    with pytest.raises(accelerator.IdentityAccelerationError, match="held_group_0"):
        engine.repeat_audit.require_complete_coverage(require_held_group_0=True)


def test_real_cpu_variable_prefix_held_mask_uses_cached_identity_and_keeps_digest() -> None:
    import torch
    from src.causal_dual_memory_cell_d_v1 import source_execute_physical as source_physical

    assert torch.cuda.is_initialized() is False
    model = _real_cell_d(torch)
    torch.manual_seed(812)
    full_neural = np.ascontiguousarray(torch.randn((129, 50, 8)).numpy(), dtype=np.float32)
    full_stack = np.ascontiguousarray(torch.randn((4, 100, 8)).numpy(), dtype=np.float32)
    full_side = np.ascontiguousarray(torch.randn((8, 4)).numpy(), dtype=np.float32)
    held = np.asarray((True, False, False, True, False, False, True, False), dtype=np.bool_)
    sliced = source_physical.physically_slice_variable_prefix_held_units(
        full_neural, full_stack, full_side, held, expected_prefix_length=4,
        full_prefix_activity_sha256=source_physical._variable_prefix_array_digest(full_stack),
    )
    neural = torch.as_tensor(sliced.neural_windows, dtype=torch.float32)
    stack = torch.as_tensor(sliced.b3s_activity_stack, dtype=torch.float32)
    side = torch.as_tensor(sliced.normalized_t4, dtype=torch.float32)
    eager = _eager_b128(model, torch, neural=neural, stack=stack, side=side)
    engine, cached = _run_cached(
        model, torch, neural=neural, stack=stack, side=side,
        total_queries=5, query_index=2, path="held_group_3", held_mask=held,
    )
    assert torch.equal(eager, cached.prediction)
    assert _digest_tensor(eager) == cached.prediction_sha256
    evidence = engine.payload()
    # Only the canonical full path and the one held_group_0 mid-session probe
    # are duplicated.  First use of held_group_3 is deliberately one-pass.
    assert evidence["repeat_audit"]["coordinates"] == []
    assert cached.actual_model_forward_count == 2


def test_b128_logical_partition_and_incremental_digest_are_exact() -> None:
    import torch

    model = _real_cell_d(torch)
    torch.manual_seed(813)
    neural = torch.randn((257, 50, 7), dtype=torch.float32)
    stack = torch.randn((4, 100, 7), dtype=torch.float32)
    side = torch.randn((7, 4), dtype=torch.float32)
    eager = _eager_b128(model, torch, neural=neural, stack=stack, side=side)
    incremental = hashlib.sha256()
    for start in range(0, int(eager.shape[0]), plan.LOGICAL_EVAL_BATCH_SIZE):
        incremental.update(eager[start:start + plan.LOGICAL_EVAL_BATCH_SIZE].contiguous().numpy().tobytes())
    engine, cached = _run_cached(
        model, torch, neural=neural, stack=stack, side=side,
        total_queries=4, query_index=1, path="full",
    )
    assert torch.equal(eager, cached.prediction)
    assert incremental.hexdigest() == cached.prediction_sha256 == _digest_tensor(eager)
    assert engine.payload()["logical_chunk_count_by_path"] == {"full": 3}
    with pytest.raises(accelerator.IdentityAccelerationError, match="B128"):
        accelerator.IdentityCacheAccelerator(logical_batch_size=1024)


class _TinyEncoder:
    pass


class _TinyCachedIdentityModel:
    """Minimal CPU model with exact direct-identity semantics and call counters."""

    decoder_mode = "coupled"

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.id_encoder = _TinyEncoder()
        self.compute_calls = 0
        self.forward_calls = 0

    def compute_identity(self, calib_trials: Any, *, side_features: Any) -> Any:
        self.compute_calls += 1
        return calib_trials.mean(dim=(1, 2)).unsqueeze(-1) + side_features

    def __call__(self, neural: Any, *, identity: Any) -> tuple[Any, Any]:
        self.forward_calls += 1
        output = neural + identity[:, :, : neural.shape[1]].permute(0, 2, 1)
        return output, identity


def test_repeat_coordinates_are_fixed_and_forward_accounting_is_honest() -> None:
    import torch

    model = _TinyCachedIdentityModel(torch)
    neural = torch.zeros((129, 4, 3), dtype=torch.float32)
    stack_a = torch.ones((2, 5, 3), dtype=torch.float32)
    stack_b = torch.full((2, 5, 3), 2.0, dtype=torch.float32)
    side = torch.zeros((3, 4), dtype=torch.float32)
    engine = accelerator.IdentityCacheAccelerator()
    engine.begin_session(total_query_trials=5)
    for index, stack in enumerate((stack_a, stack_a, stack_a, stack_b, stack_b)):
        engine.begin_query(index, query_trial_id=f"trial-{index}")
        engine.forward(
            model=model, neural_windows=neural, activity_stack=stack, normalized_t4=side, path="full",
        )
        if index == 0:
            for group, held_mask in enumerate(((True, False, False), (False, True, False), (False, False, True)), 1):
                engine.forward(
                    model=model, neural_windows=neural, activity_stack=stack, normalized_t4=side,
                    path=f"held_group_{group}", held_mask=torch.tensor(held_mask, dtype=torch.bool),
                )
        if index == 2:
            engine.forward(
                model=model, neural_windows=neural, activity_stack=stack, normalized_t4=side,
                path="held_group_0", held_mask=torch.tensor((True, False, False), dtype=torch.bool),
            )
    payload = engine.payload()
    coordinates = payload["repeat_audit"]["coordinates"]
    assert [(item["path"], item["query_index"], item["query_trial_id"], item["chunk_start"], item["reasons"])
            for item in coordinates] == [
        ("full", 0, "trial-0", 0, ["canonical_full_first_state"]),
        ("held_group_0", 2, "trial-2", 0, ["held_group_0_fixed_mid_session"]),
        ("full", 3, "trial-3", 0, ["canonical_full_first_state"]),
    ]
    assert all(item["outputs_bitwise_equal"] is True for item in coordinates)
    assert all(len(item["cache_key_sha256"]) == 64 and len(item["repeated_prediction_sha256"]) == 64
               for item in coordinates)
    # Five canonical B129 paths have ten logical chunks.  Every noncanonical
    # held group gets no duplicate on its first cache miss; only the two full
    # state proofs and single predeclared held_group_0 mid-session proof add a
    # call.
    assert payload["logical_chunk_count_by_path"] == {
        "full": 10, "held_group_0": 2, "held_group_1": 2,
        "held_group_2": 2, "held_group_3": 2,
    }
    assert payload["actual_model_forward_count_by_path"] == {
        "full": 12, "held_group_0": 3, "held_group_1": 2,
        "held_group_2": 2, "held_group_3": 2,
    }
    assert payload["identity_encoder_forward_count"] == 6
    assert model.compute_calls == 6 and model.forward_calls == 21
    engine.repeat_audit.require_complete_coverage(require_held_group_0=True)


def test_gate_bearing_identity_encoder_hard_fails_before_any_direct_forward() -> None:
    import torch

    class _GateEncoder:
        def forward_batch_with_gate(self) -> None:
            return None

    model = _TinyCachedIdentityModel(torch)
    model.id_encoder = _GateEncoder()
    engine = accelerator.IdentityCacheAccelerator()
    engine.begin_session(total_query_trials=1)
    engine.begin_query(0, query_trial_id="trial-0")
    with pytest.raises(accelerator.IdentityAccelerationError, match="forward_batch_with_gate"):
        engine.forward(
            model=model, neural_windows=torch.zeros((1, 4, 3)),
            activity_stack=torch.zeros((1, 5, 3)), normalized_t4=torch.zeros((3, 4)), path="full",
        )
    assert model.compute_calls == 0 and model.forward_calls == 0


def test_speed_runtime_mro_preserves_v8_precision_transition_hooks_without_shared_edits() -> None:
    import torch
    from src.causal_dual_memory_cell_d_score_v8 import physical as v8physical
    from src.precision_aware_causal_dual_memory_cell_d_score_v1 import physical as precisionphysical
    from src.causal_dual_memory_cell_d_speed_v1 import physical

    assert torch.cuda.is_initialized() is False
    assert issubclass(physical.SpeedV8ReviewedCDMScoreRuntime, v8physical.V8ReviewedCDMScoreRuntime)
    assert issubclass(physical.SpeedPrecisionV2ReviewedCDMScoreRuntime, precisionphysical.PrecisionV2ReviewedCDMScoreRuntime)
    assert physical.SpeedPrecisionV2ReviewedCDMScoreRuntime._initial_memory is precisionphysical.PrecisionV2ReviewedCDMScoreRuntime._initial_memory
    assert physical.SpeedPrecisionV2ReviewedCDMScoreRuntime._commit_completed_transition is precisionphysical.PrecisionV2ReviewedCDMScoreRuntime._commit_completed_transition
    assert physical.SpeedV8ReviewedCDMScoreRuntime._source_physical_helper_module is v8physical.V8ReviewedCDMScoreRuntime._source_physical_helper_module
    assert physical.SpeedV8ReviewedCDMScoreRuntime._forward_full is physical.SpeedAcceleratedRuntimeMixin._forward_full
    assert physical.SpeedV8ReviewedCDMScoreRuntime._group_predictions is physical.SpeedAcceleratedRuntimeMixin._group_predictions
    assert "reserve_score_artifact" not in vars(physical)
    assert "execute_reviewed_physical_score" not in vars(physical)
