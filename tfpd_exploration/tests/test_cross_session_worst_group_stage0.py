"""Focused CPU/synthetic gates for Cross-Session Worst-Group Stage 0."""
from __future__ import annotations

import gc
from dataclasses import replace
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tfpd_exploration") not in sys.path:
    sys.path.insert(0, str(ROOT / "tfpd_exploration"))

from src.cross_session_worst_group_v1 import core  # noqa: E402
from src.cross_session_worst_group_v1 import plan  # noqa: E402


def _source_labels(session: str, *, scale: float = 1.0) -> core.SourceOnlyFinalBinLabels:
    values = np.zeros((12, plan.M1_RAW_BEHAVIOR_OUTPUTS), dtype=np.float32)
    for index in range(values.shape[0]):
        values[index, index % plan.M1_RAW_BEHAVIOR_OUTPUTS] = scale * float(index + 1)
        values[index, (index + 3) % plan.M1_RAW_BEHAVIOR_OUTPUTS] = -0.25 * scale * float(index)
    values[0] = 0.0  # deterministic inactive/argmax-lowest-index case
    valid = np.asarray([True, True, False, True, True, True, True, True, True, True, True, True])
    return core.SourceOnlyFinalBinLabels(session, values, valid)


def _authority(sessions: tuple[str, ...] = plan.HELD_IN_SOURCE_SESSIONS[:3]) -> tuple[core.SourceStratumAuthority, dict[str, core.SourceOnlyFinalBinLabels]]:
    labels = {session: _source_labels(session, scale=1.0 + index) for index, session in enumerate(sessions)}
    return core.fit_source_stratum_authority(labels, source_sessions=sessions), labels


def _row(
    session: str, index: int, stratum: core.TaskStratum, *, units: int = plan.M1_UNIT_COUNT,
) -> core.SourceEpisodeRow:
    # Each row has its own B3S calibration tensor and a distinguishable value;
    # a later concatenation test asserts these values remain row-local.
    return core.SourceEpisodeRow(
        session_id=session,
        sample_index=index,
        sample_id=f"{session}:sample:{index}",
        stratum=stratum,
        model_inputs={
            "x": np.full((plan.M1_WINDOW_SIZE, units), float(index + 1), dtype=np.float32),
            "calib_trialized_neural_features": np.full(
                (plan.M1_CALIBRATION_TRIALS, plan.M1_B3S_MAX_TRIAL_LENGTH, units),
                float(1000 + index),
                dtype=np.float32,
            ),
        },
        raw_final_target=np.full((plan.M1_RAW_BEHAVIOR_OUTPUTS,), float(index % 5), dtype=np.float32),
        final_bin_valid=True,
    )


def _pools(
    sessions: tuple[str, ...] = plan.HELD_IN_SOURCE_SESSIONS[:3], *, units: int = plan.M1_UNIT_COUNT,
) -> dict[str, core.SessionStratumPool]:
    authority_sha = "a" * 64
    strata = (
        core.TaskStratum(False, 0, 0),
        core.TaskStratum(True, 1, 1),
        core.TaskStratum(True, 2, 2),
        core.TaskStratum(True, 3, 3),
    )
    result: dict[str, core.SessionStratumPool] = {}
    for session_index, session in enumerate(sessions):
        rows = tuple(
            _row(session, session_index * 100 + stratum_index * 10 + replicate, stratum, units=units)
            for stratum_index, stratum in enumerate(strata)
            for replicate in range(4)
        )
        result[session] = core.SessionStratumPool(session, authority_sha, rows)
    return result


class _SyntheticM1:
    """One callable graph stand-in; Stage 0 does not define a replacement model."""

    def __init__(self) -> None:
        import torch

        self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.calls = 0

    def __call__(self, *, x, calib_trialized_neural_features):
        import torch

        self.calls += 1
        assert x.shape == (plan.TOTAL_BATCH_SIZE, plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT)
        assert calib_trialized_neural_features.shape == (
            plan.TOTAL_BATCH_SIZE, plan.M1_CALIBRATION_TRIALS,
            plan.M1_B3S_MAX_TRIAL_LENGTH, plan.M1_UNIT_COUNT,
        )
        signal = x.mean(dim=(1, 2), keepdim=True)
        signal = signal + calib_trialized_neural_features[:, 0].mean(dim=(1, 2), keepdim=True) * 0.0
        raw = (signal * self.scale).expand(-1, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS)
        return raw


def _episode(*, step: int = 0, units: int = plan.M1_UNIT_COUNT) -> core.BalancedEpisode:
    sessions = plan.HELD_IN_SOURCE_SESSIONS[:3]
    return core.build_balanced_episode(_pools(sessions, units=units), quota=plan.outer_fold_episode_quota(sessions, step_index=step))


def test_static_workorder_graph_contract_and_explicit_closure() -> None:
    assert plan.WORKORDER_SHA256 == "225fccec7588e28c25d1e4c3240066eb896fcd32c48e11236aaa8d45f8cb491d"
    assert plan.M1_GRAPH_CONTRACT.live_parameters_after_lazy1024 == 15_007_496
    assert plan.M1_GRAPH_CONTRACT.window_size == 100
    assert plan.M1_GRAPH_CONTRACT.raw_behavior_outputs == 16
    assert plan.M1_GRAPH_CONTRACT.unit_count == 64
    assert plan.M1_GRAPH_CONTRACT.calibration_trials == 10
    assert plan.M1_GRAPH_CONTRACT.b3s_max_trial_length == 1024
    assert plan.M1_CALIBRATION_SHAPE_PER_ROW == (10, 1024, 64)
    assert plan.M1_GRAPH_CONTRACT.forward_input_keys == (
        "x", "calib_trialized_neural_features",
    )
    assert plan.M1_HELD_IN_UNIT_ID_SHAPE == (64,)
    assert plan.dry_plan()["m1_b3s_max_trial_length"] == 1024
    assert plan.dry_plan()["m1_calibration_shape_per_row"] == [10, 1024, 64]
    assert plan.MIXED_SESSION_TRAINING_STEP_CONTRACT.baseline_training_step_policy == "rejects_mixed_session_batches"
    assert plan.M1_SOURCE_DECODER_RECIPE.payload() == {
        "calibration_budget": "M10",
        "calibration_trials": 10,
        "b3s_max_trial_length": 1024,
        "calibration_shape_per_row": [10, 1024, 64],
        "epoch_budget": 20,
        "optimizer": "Adam",
        "adam_lr": 1.0e-5,
        "adam_weight_decay": 0.0,
        "lr_schedule": "None",
        "final_bin_only_raw_output_mse": True,
        "total_batch_size": 32,
        "hidden_inner_grid_or_sweep_forbidden": True,
    }
    closure = plan.execution_closure_payload(ROOT)
    assert [row["path"] for row in closure["paths"]] == [
        plan.WORKORDER_RELATIVE,
        plan.M1_MODEL_CONFIG_RELATIVE,
        *plan.M1_DATA_FOLD_CONFIG_RELATIVES,
        "streaming_calibration_exp/src/__init__.py",
        "streaming_calibration_exp/src/models/__init__.py",
        plan.M1_FALCON_FORWARD_INTERFACE_RELATIVE,
        "streaming_calibration_exp/src/models/components/__init__.py",
        plan.M1_SPINT_MODEL_RELATIVE,
        *plan.M1_SOURCE_DECODER_RUNTIME_IMPORT_RELATIVES,
        "tfpd_exploration/src/__init__.py",
        "tfpd_exploration/src/cross_session_worst_group_v1/__init__.py",
        "tfpd_exploration/src/cross_session_worst_group_v1/plan.py",
        "tfpd_exploration/src/cross_session_worst_group_v1/core.py",
        "tfpd_exploration/scripts/run_cross_session_worst_group_stage0.py",
        "tfpd_exploration/tests/test_cross_session_worst_group_stage0.py",
    ]
    assert len(closure["closure_sha256"]) == 64
    falcon_source = (ROOT / plan.M1_FALCON_FORWARD_INTERFACE_RELATIVE).read_text(encoding="utf-8")
    assert "def forward(" in falcon_source
    assert "x: torch.Tensor" in falcon_source
    assert "calib_trialized_neural_features: torch.Tensor | None = None" in falcon_source
    assert "All samples in the batch should belong to the same session" in falcon_source
    model_config = (ROOT / plan.M1_MODEL_CONFIG_RELATIVE).read_text(encoding="utf-8")
    assert "_target_: src.models.components.spint.SpintModel" in model_config
    for relative in plan.M1_DATA_FOLD_CONFIG_RELATIVES:
        assert "max_trial_length: 1024" in (ROOT / relative).read_text(encoding="utf-8")
    source_decoder = (ROOT / plan.M1_SOURCE_DECODER_DATAMODULE_RELATIVE).read_text(encoding="utf-8")
    assert "M1SourceOnlyDecoderFold0DataModule" in source_decoder
    assert "M1_SOURCE_ONLY_FOLDS" in source_decoder


def _closure_bound_spint_model_type():
    """Load the exact closure-bound production component without `src` aliasing."""
    module_path = ROOT / plan.M1_SPINT_MODEL_RELATIVE
    module_name = "_cswg_stage0_closure_bound_spint"
    prior = sys.modules.pop(module_name, None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        assert Path(module.__file__).resolve() == module_path.resolve()
        return module.SpintModel
    finally:
        sys.modules.pop(module_name, None)
        if prior is not None:
            sys.modules[module_name] = prior


def test_closure_bound_m1_spint_cpu_materialization_requires_b3s_t1024() -> None:
    """Mandatory real CPU graph proof; W=100 is never a calibration-time axis."""
    import torch

    assert torch.cuda.is_initialized() is False
    SpintModel = _closure_bound_spint_model_type()
    kwargs = {
        "model_dim": 1024,
        "num_covariates": plan.M1_RAW_BEHAVIOR_OUTPUTS,
        "window_size": plan.M1_WINDOW_SIZE,
        "num_heads": 64,
        "num_layers": 1,
        "num_id_layers": 3,
        "use_learnable_id": True,
        "learnable_id_type": "mlp",
        "learnable_rep": True,
        "dropout_rate": 0.0,
        "dynamic_dropout": True,
        "dynamic_dropout_low": 0.0,
        "dynamic_dropout_high": 1.0,
        "tf_drop_rate": 0.1,
        "readin_layer_type": "mlp",
    }
    x = torch.zeros((1, plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT), dtype=torch.float32)
    calibrated = torch.zeros((1, *plan.M1_CALIBRATION_SHAPE_PER_ROW), dtype=torch.float32)
    model = SpintModel(**kwargs).eval()
    with torch.no_grad():
        output = model(x, calib_trialized_neural_features=calibrated)
    assert output.shape == (1, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS)
    assert tuple(model.fc_id_in[0].weight.shape) == (1024, 1024)
    live_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert live_count == plan.M1_LIVE_PARAMETERS_AFTER_LAZY1024 == 15_007_496
    plan.validate_materialized_baseline_model(
        live_parameter_count=live_count,
        topology=plan.M1_BASELINE_INFERENCE_TOPOLOGY,
    )
    del model, output, calibrated
    gc.collect()

    wrong_model = SpintModel(**kwargs).eval()
    wrong_calibrated = torch.zeros(
        (1, plan.M1_CALIBRATION_TRIALS, plan.M1_WINDOW_SIZE, plan.M1_UNIT_COUNT), dtype=torch.float32,
    )
    with torch.no_grad():
        wrong_output = wrong_model(x, calib_trialized_neural_features=wrong_calibrated)
    assert wrong_output.shape == (1, plan.M1_WINDOW_SIZE, plan.M1_RAW_BEHAVIOR_OUTPUTS)
    assert tuple(wrong_model.fc_id_in[0].weight.shape) == (1024, 100)
    wrong_count = sum(parameter.numel() for parameter in wrong_model.parameters() if parameter.requires_grad)
    assert wrong_count == 14_061_320
    with pytest.raises(plan.CSWGPlanError, match="live-parameter"):
        plan.validate_materialized_baseline_model(
            live_parameter_count=wrong_count,
            topology=plan.M1_BASELINE_INFERENCE_TOPOLOGY,
        )
    with pytest.raises(plan.CSWGPlanError, match="baseline M1 graph"):
        replace(plan.M1_GRAPH_CONTRACT, b3s_max_trial_length=plan.M1_WINDOW_SIZE)
    assert torch.cuda.is_initialized() is False


def test_source_only_quantile_fit_is_typed_and_rejects_target_or_caller_boundaries() -> None:
    sessions = plan.HELD_IN_SOURCE_SESSIONS[:3]
    authority, labels = _authority(sessions)
    assert authority.source_sessions == sessions
    assert authority.payload()["target_labels_used"] is False
    assert authority.source_label_digests == {session: labels[session].digest for session in sessions}
    with pytest.raises(core.CSWGCoreError, match="rejects target"):
        core.fit_source_stratum_authority({session: np.zeros((2, 16)) for session in sessions}, source_sessions=sessions)
    with pytest.raises(core.CSWGCoreError, match="authority drift"):
        core.SourceStratumAuthority(
            sessions, 4, (1.0, 2.0, 3.0), {session: "a" * 64 for session in sessions}, _seal=None,
        )
    target_like = core.SourceOnlyFinalBinLabels(plan.HELD_IN_SOURCE_SESSIONS[3], np.ones((3, 16)), np.ones(3, dtype=bool))
    with pytest.raises(core.CSWGCoreError, match="mapping order/session"):
        core.fit_source_stratum_authority({**labels, target_like.session_id: target_like}, source_sessions=sessions)
    run_spec, _erm = plan.build_outer_fold_specs(
        outer_target_session="20120928", initialization_seed=42, lambda_=0.4, tau=0.8,
    )
    run_labels = {session: _source_labels(session) for session in run_spec.source_sessions}
    bound = core.fit_run_spec_source_stratum_authority(run_spec, run_labels)
    assert bound.source_sessions == run_spec.source_sessions
    # A caller cannot use the typed held-in outer-target table as if it were a
    # source table merely by changing map order or omitting one actual source.
    bad_labels = dict(run_labels)
    del bad_labels[run_spec.source_sessions[0]]
    bad_labels["20120928"] = target_like
    with pytest.raises(core.CSWGCoreError, match="target/caller"):
        core.fit_run_spec_source_stratum_authority(run_spec, bad_labels)


def test_stratum_assignment_is_deterministic_active_norm_quantile_and_dominant_coordinate() -> None:
    sessions = plan.HELD_IN_SOURCE_SESSIONS[:3]
    authority, labels = _authority(sessions)
    first = core.assign_source_task_strata(labels[sessions[0]], authority)
    second = core.assign_source_task_strata(labels[sessions[0]], authority)
    assert first == second
    assert first.sample_indices == tuple(labels[sessions[0]].valid_indices.tolist())
    assert first.strata[0].active_flag is False
    assert first.strata[0].dominant_coordinate == 0
    assert all(0 <= item.raw_output_norm_quantile < 4 for item in first.strata)
    altered = replace(labels[sessions[0]], raw_final_outputs=np.ones((12, 16), dtype=np.float32))
    with pytest.raises(core.CSWGCoreError, match="exact frozen source authority"):
        core.assign_source_task_strata(altered, authority)


def test_assignment_builds_typed_pool_in_valid_source_order() -> None:
    sessions = plan.HELD_IN_SOURCE_SESSIONS[:3]
    authority, labels = _authority(sessions)
    assigned = core.assign_source_task_strata(labels[sessions[0]], authority)
    rows = {
        index: _row(sessions[0], index, stratum)
        for index, stratum in zip(assigned.sample_indices, assigned.strata, strict=True)
    }
    pool = core.build_session_stratum_pool(assigned, rows_by_sample_index=rows)
    assert tuple(row.sample_index for row in pool.rows) == assigned.sample_indices
    mismatched = dict(rows)
    first_index = assigned.sample_indices[0]
    mismatched[first_index] = _row(sessions[0], first_index, core.TaskStratum(True, 1, 1))
    with pytest.raises(core.CSWGCoreError, match="row/assigned"):
        core.build_session_stratum_pool(assigned, rows_by_sample_index=mismatched)


def test_rotating_total_b32_quotas_and_equal_represented_task_strata() -> None:
    sessions = plan.HELD_IN_SOURCE_SESSIONS[:3]
    assert [plan.outer_fold_episode_quota(sessions, step_index=index).counts for index in range(3)] == [
        (10, 11, 11), (11, 10, 11), (11, 11, 10),
    ]
    episode = _episode(step=2)
    assert tuple(len(batch.rows) for batch in episode.microbatches) == (11, 11, 10)
    assert sum(len(batch.rows) for batch in episode.microbatches) == 32
    for batch in episode.microbatches:
        counts = [sum(row.stratum == stratum for row in batch.rows) for stratum in episode.represented_strata]
        assert min(counts) > 0 and max(counts) - min(counts) <= 1
    all_four = plan.all_four_source_episode_quota(plan.HELD_IN_SOURCE_SESSIONS, step_index=99)
    assert all_four.counts == (8, 8, 8, 8)
    all_four_episode = core.build_balanced_episode(_pools(plan.HELD_IN_SOURCE_SESSIONS), quota=all_four)
    assert tuple(len(batch.rows) for batch in all_four_episode.microbatches) == (8, 8, 8, 8)
    all_four_compatibility = core.derive_concat_compatibility_authority(all_four_episode)
    all_four_model = _SyntheticM1()
    all_four_ownership = core.run_one_concatenated_forward(
        all_four_model,
        core.concatenate_episode_for_one_forward(all_four_episode, compatibility=all_four_compatibility),
    )
    assert all_four_model.calls == all_four_ownership.forward_call_count == 1
    assert tuple(item.session_id for item in all_four_ownership.concatenated.session_slices) == plan.HELD_IN_SOURCE_SESSIONS
    with pytest.raises(plan.CSWGPlanError, match="exact ordered"):
        plan.all_four_source_episode_quota(tuple(reversed(plan.HELD_IN_SOURCE_SESSIONS)), step_index=0)


def test_missing_stratum_fails_closed_without_undeclared_fallback() -> None:
    sessions = plan.HELD_IN_SOURCE_SESSIONS[:3]
    pools = _pools(sessions)
    reduced = pools[sessions[1]]
    pools[sessions[1]] = core.SessionStratumPool(
        reduced.session_id, reduced.authority_sha256,
        tuple(row for row in reduced.rows if row.stratum != reduced.strata[-1]),
    )
    with pytest.raises(core.CSWGCoreError, match="missing task stratum"):
        core.build_balanced_episode(pools, quota=plan.outer_fold_episode_quota(sessions, step_index=0))


def test_exact_concat_compatibility_one_forward_and_per_row_calibration_ownership() -> None:
    episode = _episode(step=1)
    compatibility = core.derive_concat_compatibility_authority(episode)
    assert compatibility.payload()["unit_padding_masks_or_shape_coercion_forbidden"] is True
    concatenated = core.concatenate_episode_for_one_forward(episode, compatibility=compatibility)
    model = _SyntheticM1()
    ownership = core.run_one_concatenated_forward(model, concatenated)
    assert model.calls == ownership.forward_call_count == 1
    assert ownership.raw_outputs.shape == (32, 100, 16)
    assert all(row["calibration_tensor_owned_per_row"] is True for row in concatenated.row_ownership)
    for index, row in enumerate(episode.all_rows):
        assert float(concatenated.model_inputs["calib_trialized_neural_features"][index, 0, 0, 0]) == float(
            row.model_inputs["calib_trialized_neural_features"][0, 0, 0],
        )
        assert concatenated.row_ownership[index]["session_id"] == row.session_id
    assert set(concatenated.model_inputs) == set(plan.M1_FORWARD_INPUT_KEYS)
    # There is no legal padding/mask path for different session unit shapes:
    # the typed row fails before a mixed forward can be assembled.
    with pytest.raises(core.CSWGCoreError, match="exact Falcon-M1"):
        _row(plan.HELD_IN_SOURCE_SESSIONS[2], 999, episode.represented_strata[0], units=63)
    exact_row = episode.all_rows[0]
    with pytest.raises(core.CSWGCoreError, match="exact Falcon-M1"):
        replace(
            exact_row,
            model_inputs={
                **dict(exact_row.model_inputs),
                "side_features": np.zeros((plan.M1_UNIT_COUNT, 4), dtype=np.float32),
            },
        )
    with pytest.raises(core.CSWGCoreError, match="exact Falcon-M1"):
        replace(
            exact_row,
            model_inputs={"x": exact_row.model_inputs["x"]},
        )


def test_route_owned_training_step_binds_exact_outer_fold_sources_and_current_objective() -> None:
    episode = _episode(step=1)
    compatibility = core.derive_concat_compatibility_authority(episode)
    cswg, _erm = plan.build_outer_fold_specs(
        outer_target_session="20120928", initialization_seed=42, lambda_=0.4, tau=0.7,
    )
    model = _SyntheticM1()
    route_step = core.RouteOwnedMixedSessionTrainingStep(
        model=model, compatibility=compatibility, run_spec=cswg,
    )
    objective = route_step.run(episode, lambda_=0.4, tau=0.7)
    assert model.calls == 1
    assert tuple(row.session_id for row in objective.session_losses) == tuple(sorted(episode.session_ids))
    with pytest.raises(core.CSWGCoreError, match="run-spec/target/objective"):
        route_step.run(episode, lambda_=0.0, tau=0.7)
    wrong_fold, _wrong_erm = plan.build_outer_fold_specs(
        outer_target_session="20120924", initialization_seed=42, lambda_=0.4, tau=0.7,
    )
    with pytest.raises(core.CSWGCoreError, match="construction"):
        core.RouteOwnedMixedSessionTrainingStep(
            model=_SyntheticM1(), compatibility=compatibility, run_spec=wrong_fold,
        )


def test_per_session_final_bin_raw_mse_and_complete_objective_match_manual_formula() -> None:
    import torch

    episode = _episode()
    compatibility = core.derive_concat_compatibility_authority(episode)
    ownership = core.run_one_concatenated_forward(_SyntheticM1(), core.concatenate_episode_for_one_forward(episode, compatibility=compatibility))
    losses = core.compute_current_session_losses(ownership)
    objective = core.complete_centered_smooth_worst_group_objective(losses, ownership=ownership, lambda_=0.5, tau=0.7)
    values = torch.stack([item.value for item in sorted(losses, key=lambda item: item.session_id)])
    manual_mean = values.mean()
    manual_robust = 0.7 * (torch.logsumexp((values - manual_mean) / 0.7, dim=0) - np.log(len(values)))
    assert torch.allclose(objective.mean_loss, manual_mean)
    assert torch.allclose(objective.centered_robust_term, manual_robust)
    assert torch.allclose(objective.loss, manual_mean + 0.5 * manual_robust)
    assert objective.payload()["complete_centered_smooth_worst_group"] is True


def test_lambda_zero_equal_losses_monotonicity_and_gradients_every_session_loss() -> None:
    import torch

    episode = _episode()
    compatibility = core.derive_concat_compatibility_authority(episode)
    ownership = core.run_one_concatenated_forward(_SyntheticM1(), core.concatenate_episode_for_one_forward(episode, compatibility=compatibility))
    current = core.compute_current_session_losses(ownership)
    zero = core.complete_centered_smooth_worst_group_objective(current, ownership=ownership, lambda_=0.0, tau=0.5)
    assert torch.equal(zero.loss, zero.mean_loss)
    live = core.complete_centered_smooth_worst_group_objective(current, ownership=ownership, lambda_=0.7, tau=0.5)
    live_grads = torch.autograd.grad(live.loss, [row.value for row in current], retain_graph=True)
    assert len(live_grads) == 3
    assert all(gradient is not None and torch.isfinite(gradient) and float(gradient) > 0.0 for gradient in live_grads)
    equal_value = torch.tensor(2.0, requires_grad=True)
    equal_rows = tuple(replace(row, value=equal_value * 1.0) for row in current)
    equal = core.complete_centered_smooth_worst_group_objective(equal_rows, ownership=ownership, lambda_=1.0, tau=0.5)
    assert float(equal.centered_robust_term.detach()) == 0.0
    lower = tuple(replace(row, value=torch.tensor(float(index + 1), requires_grad=True)) for index, row in enumerate(current))
    higher_values = list(lower)
    higher_values[-1] = replace(higher_values[-1], value=torch.tensor(10.0, requires_grad=True))
    lower_obj = core.complete_centered_smooth_worst_group_objective(lower, ownership=ownership, lambda_=1.0, tau=0.5)
    higher_obj = core.complete_centered_smooth_worst_group_objective(tuple(higher_values), ownership=ownership, lambda_=1.0, tau=0.5)
    assert float(higher_obj.loss.detach()) >= float(lower_obj.loss.detach())
    grads = torch.autograd.grad(lower_obj.loss, [row.value for row in lower])
    assert all(gradient is not None and float(gradient) >= 0.0 for gradient in grads)


def test_single_session_detached_stale_and_session_permutation_loss_tables_are_handled_exactly() -> None:
    episode = _episode()
    compatibility = core.derive_concat_compatibility_authority(episode)
    concatenated = core.concatenate_episode_for_one_forward(episode, compatibility=compatibility)
    ownership = core.run_one_concatenated_forward(_SyntheticM1(), concatenated)
    losses = core.compute_current_session_losses(ownership)
    forward_again = core.run_one_concatenated_forward(_SyntheticM1(), concatenated)
    with pytest.raises(core.CSWGCoreError, match="single-session"):
        core.complete_centered_smooth_worst_group_objective(losses[:1], ownership=ownership, lambda_=0.1, tau=1.0)
    with pytest.raises(core.CSWGCoreError, match="differentiable"):
        core.CurrentSessionLoss(
            losses[0].session_id, losses[0].value.detach(), ownership,
            losses[0].row_count, True, core._CURRENT_LOSS_SEAL,
        )
    stale = tuple(replace(row, ownership=forward_again) for row in losses)
    with pytest.raises(core.CSWGCoreError, match="stale"):
        core.complete_centered_smooth_worst_group_objective(stale, ownership=ownership, lambda_=0.1, tau=1.0)
    first = core.complete_centered_smooth_worst_group_objective(losses, ownership=ownership, lambda_=0.4, tau=0.9)
    second = core.complete_centered_smooth_worst_group_objective(tuple(reversed(losses)), ownership=ownership, lambda_=0.4, tau=0.9)
    assert float(first.loss.detach()) == pytest.approx(float(second.loss.detach()))


def test_duplicate_value_windows_cannot_change_equal_session_weight() -> None:
    import torch

    episode = _episode()
    compatibility = core.derive_concat_compatibility_authority(episode)
    ownership = core.run_one_concatenated_forward(_SyntheticM1(), core.concatenate_episode_for_one_forward(episode, compatibility=compatibility))
    template = core.compute_current_session_losses(ownership)
    values = (torch.tensor(1.0, requires_grad=True), torch.tensor(3.0, requires_grad=True), torch.tensor(5.0, requires_grad=True))
    ordinary = tuple(replace(row, value=value, row_count=1) for row, value in zip(template, values, strict=True))
    duplicated = tuple(replace(row, value=value, row_count=100) for row, value in zip(template, values, strict=True))
    one = core.complete_centered_smooth_worst_group_objective(ordinary, ownership=ownership, lambda_=0.0, tau=1.0)
    two = core.complete_centered_smooth_worst_group_objective(duplicated, ownership=ownership, lambda_=0.0, tau=1.0)
    assert float(one.mean_loss.detach()) == float(two.mean_loss.detach()) == 3.0


def test_final_bin_validity_and_raw_output_shape_are_enforced() -> None:
    import torch

    episode = _episode()
    compatibility = core.derive_concat_compatibility_authority(episode)
    concatenated = core.concatenate_episode_for_one_forward(episode, compatibility=compatibility)
    invalid = replace(concatenated, final_bin_valid=torch.zeros(plan.TOTAL_BATCH_SIZE, dtype=torch.bool))
    ownership = core.run_one_concatenated_forward(_SyntheticM1(), invalid)
    with pytest.raises(core.CSWGCoreError, match="valid final-bin"):
        core.compute_current_session_losses(ownership)

    class WrongShape:
        def __call__(self, **kwargs):
            return torch.zeros((32, 99, 16), requires_grad=True)

    with pytest.raises(core.CSWGCoreError, match="W100"):
        core.run_one_concatenated_forward(WrongShape(), concatenated)


def test_stage0_owns_no_model_parameters_or_inference_module_and_future_graph_gate_is_exact() -> None:
    assert core.stage0_trainable_parameters() == ()
    assert core.stage0_inference_modules() == ()
    source = inspect.getsource(core)
    assert "torch.nn.Module" not in source
    plan.validate_materialized_baseline_model(live_parameter_count=15_007_496, topology=plan.M1_BASELINE_INFERENCE_TOPOLOGY)
    with pytest.raises(plan.CSWGPlanError, match="parameter"):
        plan.validate_materialized_baseline_model(live_parameter_count=15_007_495, topology=plan.M1_BASELINE_INFERENCE_TOPOLOGY)
    with pytest.raises(plan.CSWGPlanError, match="topology"):
        plan.validate_materialized_baseline_model(live_parameter_count=15_007_496, topology="forked")


def test_matched_same_fold_erm_boundary_forbids_outer_target_trained_historical_checkpoint() -> None:
    cswg, erm = plan.build_outer_fold_specs(
        outer_target_session="20120924", initialization_seed=42, checkpoint_swa_rule="frozen_swa",
        lambda_=0.7, tau=1.0,
    )
    plan.validate_matched_same_fold_pair(cswg, erm)
    assert cswg.source_sessions == erm.source_sessions == ("20120926", "20120927", "20120928")
    assert erm.lambda_ == 0.0
    with pytest.raises(plan.CSWGPlanError, match="checkpoint"):
        replace(erm, checkpoint_training_sessions=(*erm.source_sessions, "20120924"))
    with pytest.raises(plan.CSWGPlanError, match="source-decoder recipe"):
        replace(erm, epoch_budget=11)


def test_all_four_held_in_loso_folds_bind_fold3_recipe_and_no_hidden_grid() -> None:
    pairs = plan.build_all_outer_fold_spec_pairs(
        initialization_seed=42, lambda_=0.4, tau=0.75, checkpoint_swa_rule="inherited_checkpoint_swa",
    )
    assert len(pairs) == 4
    assert tuple(cswg.outer_target_session for cswg, _erm in pairs) == plan.HELD_IN_SOURCE_SESSIONS
    fold3_cswg, fold3_erm = pairs[3]
    assert fold3_cswg.outer_target_session == "20120928"
    assert fold3_cswg.source_sessions == fold3_erm.source_sessions == (
        "20120924", "20120926", "20120927",
    )
    for cswg, erm in pairs:
        plan.validate_matched_same_fold_pair(cswg, erm)
        assert cswg.epoch_budget == erm.epoch_budget == 20
        assert cswg.optimizer_literal == erm.optimizer_literal == "Adam"
        assert cswg.adam_lr == erm.adam_lr == 1.0e-5
        assert cswg.adam_weight_decay == erm.adam_weight_decay == 0.0
        assert cswg.lr_schedule_literal == erm.lr_schedule_literal == "None"
        assert cswg.calibration_budget == erm.calibration_budget == "M10"
        assert cswg.hidden_inner_grid_or_sweep_forbidden is erm.hidden_inner_grid_or_sweep_forbidden is True
    with pytest.raises(plan.CSWGPlanError, match="source-decoder recipe"):
        replace(fold3_cswg, epoch_budget=21)
    dry = plan.dry_plan()
    assert dry["future_four_fold_matched_erm_plus_cswg_training_runs"] == 8
    assert dry["historical_baseline_fold_wall_minutes"] == 68.8
    assert dry["historical_reference_minutes_for_eight_runs_if_each_matches_baseline"] == pytest.approx(550.4)
    assert dry["actual_cswg_runtime_requires_measured_physical_receipts"] is True
    assert dry["hidden_inner_grid_or_sweep_forbidden"] is True


def test_static_cli_is_inert_no_torch_no_write_no_launch() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_cross_session_worst_group_stage0.py"
    dry = subprocess.run([sys.executable, "-I", str(script), "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["workorder_sha256"] == plan.WORKORDER_SHA256
    assert payload["opens_source_or_target"] is payload["loads_checkpoint"] is payload["initializes_cuda"] is False
    assert payload["future_four_fold_matched_erm_plus_cswg_training_runs"] == 8
    assert payload["fixed_m1_source_decoder_recipe"]["b3s_max_trial_length"] == 1024
    assert payload["fixed_m1_source_decoder_recipe"]["calibration_shape_per_row"] == [10, 1024, 64]
    assert payload["historical_reference_minutes_for_eight_runs_if_each_matches_baseline"] == 550.4
    assert payload["actual_cswg_runtime_requires_measured_physical_receipts"] is True
    assert "import torch" not in script.read_text(encoding="utf-8")
    blocked = subprocess.run([sys.executable, "-I", str(script), "--execute"], cwd=ROOT, text=True, capture_output=True)
    assert blocked.returncode != 0
    assert "CPU/synthetic-only" in blocked.stderr


def test_focused_synthetic_torch_path_never_initializes_cuda() -> None:
    """The core may use CPU Torch tensors, but Stage 0 must not touch CUDA."""
    import torch

    assert torch.cuda.is_initialized() is False
