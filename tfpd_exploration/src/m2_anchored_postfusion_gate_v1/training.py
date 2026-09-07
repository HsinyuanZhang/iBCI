"""APFG alpha-only source optimisation over POOLED/G00m-linear activity.

All imports that can construct a model/data stack are inside post-attempt
functions.  The training path does not call Lightning ``model_step``: that
would reintroduce sealed teacher/distillation losses and invalidate the
one-scalar attribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import plan
from .adapter import AnchoredPostFusionGate, alpha_only_adam, freeze_and_install
from .runtime import FrozenIdentityCacheKey, RuntimeContractError, task_only_scaled_last_bin_mse
from .source_replay import SourceCoordinate, SourceSessionMaterial, pool_for_controller_coordinate
from .source_replay import canonical_m30_source_batches, rollout_apfg_raw_pool
from .selection import EpochValidation, lexical_source_split, select_earliest_best


class TrainingError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise TrainingError(message)


@dataclass
class PreparedAPFG:
    """One strict-loaded selected-T4 model and its sole trainable scalar."""
    module: Any
    datamodule: Any
    adapter: AnchoredPostFusionGate
    optimizer: Any
    device: Any
    pit_prepare_calls: int
    evidence: Mapping[str, object]


def validate_cuda_launch_attestation(*, torch: Any, device: str, launch_attestation: Mapping[str, object]) -> None:
    """Require the exact already-initialized GPU0 launch context.

    ``driver.launch`` owns CUDA initialization.  Prepare must not initialize a
    second or differently visible context after attempt publication.
    """
    _require(device == "cuda:0" and torch.cuda.is_available() and torch.cuda.is_initialized(),
             "APFG prepare requires initialized launch-attested CUDA0")
    _require(torch.cuda.current_device() == 0 and torch.cuda.device_count() == 1,
             "APFG prepare logical CUDA topology drift")
    _require(launch_attestation.get("logical_device") == 0 and launch_attestation.get("physical_device") == 0
             and launch_attestation.get("cuda_visible_devices") == "0"
             and launch_attestation.get("cuda_initialized") is True,
             "APFG prepare launch attestation drift")


def prepare_selected_t4_apfg_after_attempt(*, repo_root: Any, device: str = "cuda:0",
                                            launch_attestation: Mapping[str, object] | None = None) -> PreparedAPFG:
    """Single PIT prepare, strict selected-T4 load, then install/freeze APFG.

    This function is intentionally not called by the inert CLI or no-data
    tests.  It is the only permitted production construction sequence.
    """
    import torch
    from tfpd_exploration.src.pit_m2_v1 import trainer as pit_trainer
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as pooled_physical

    _require(launch_attestation is not None, "APFG prepare requires launch attestation")
    validate_cuda_launch_attestation(torch=torch, device=device, launch_attestation=launch_attestation)
    runner = pit_trainer.PitM2ArmedRunner(repo_root, "t0m", device=device)
    runner.prepare(attach_operator=False)
    base = runner._litmodule
    _require(base is not None and runner._datamodule is not None, "APFG PIT source stack is absent")
    # Reuse the audited selected-T4 strict loader.  It receives the one
    # already prepared PIT module and never creates a second DataModule.
    module, selected_evidence = pooled_physical._strict_sealed_pooled_clone(repo_root=repo_root, base_module=base)
    adapter = freeze_and_install(module.student)
    module.to(torch.device(device))
    module.eval()
    for parameter in module.parameters():
        if parameter is not adapter.alpha:
            _require(not parameter.requires_grad, "APFG inherited parameter became trainable after device move")
    _require(not module.training and not adapter.native.training, "APFG inherited branch must remain eval")
    optimizer = alpha_only_adam(adapter)
    adapter.set_alpha_training(True)
    _require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["params"] == [adapter.alpha],
             "APFG Adam topology drift")
    return PreparedAPFG(module=module, datamodule=runner._datamodule, adapter=adapter, optimizer=optimizer,
                        device=torch.device(device), pit_prepare_calls=1,
                        evidence={"selected_t4": dict(selected_evidence), "strict_load_before_adapter": True,
                                  "activity_authority": plan.ACTIVITY_AUTHORITY,
                                  "teacher_forward_calls": 0, "inherited_eval": True,
                                  "dropout_calls": 0, "optimizer_parameter_names": ["id_encoder.alpha"]})


def _side_tensor(*, torch: Any, material: SourceSessionMaterial, dataset: Any, device: Any) -> Any:
    mean = np.asarray(dataset.side_feature_mean, dtype=np.float32)
    std = np.asarray(dataset.side_feature_std, dtype=np.float32)
    model_count = np.ascontiguousarray(material.selected_support4_carrier_hz * np.float64(0.020), dtype=np.float32)
    check = np.ascontiguousarray((model_count - mean) / std, dtype=np.float32)
    _require(np.array_equal(check, material.normalized_side) and np.isfinite(check).all(),
             "APFG selected T4 normalization authority drift")
    side = material.normalized_side
    return torch.from_numpy(side).unsqueeze(0).to(device)


def _branch_key(*, coordinate: SourceCoordinate, state: Any) -> FrozenIdentityCacheKey:
    key = FrozenIdentityCacheKey(session_id=coordinate.session, pool_state_sha256=state.identity_digest(),
                                 ordered_activity_sha256=state.ordered_activity_sha256,
                                 selected_support4_carrier_hz_sha256=state.selected_support4_carrier_hz_sha256,
                                 normalized_side_sha256=state.normalized_side_sha256,
                                 normalizer_sha256=state.normalizer_sha256,
                                 query_trial_index=coordinate.query_trial_index,
                                 requested_pool_size=state.requested_size)
    key.validate()
    return key


def cached_frozen_branches(*, prepared: PreparedAPFG, material: SourceSessionMaterial,
                           coordinate: SourceCoordinate, state: Any, cache: dict[FrozenIdentityCacheKey, tuple[Any, Any]]) -> tuple[Any, Any]:
    """Compute frozen native/post identities once, detached, under full authority."""
    import torch
    key = _branch_key(coordinate=coordinate, state=state)
    cached = cache.get(key)
    if cached is not None:
        return cached
    members = np.asarray(state.member_trial_ids, dtype=np.int64)
    activity = np.ascontiguousarray(material.activities[members], dtype=np.float32)
    calibration = torch.from_numpy(activity).unsqueeze(0).to(prepared.device)
    side = _side_tensor(torch=torch, material=material, dataset=prepared.datamodule.train_dataset, device=prepared.device)
    with torch.no_grad():
        native = prepared.adapter.native.forward_batch(calibration, side_features=side).detach()
        post = prepared.adapter._postfusion_identity(calibration, side).detach()
    _require(tuple(native.shape) == tuple(post.shape) and bool(torch.isfinite(native).all()) and bool(torch.isfinite(post).all()),
             "APFG frozen branch identity topology/finite drift")
    cache[key] = (native, post)
    return cache[key]


def expand_identity_for_windows(*, identity: Any, batch_size: int, channels: int, window_bins: int) -> Any:
    """Expand one same-pool identity over a homogeneous supervised batch."""
    _require(tuple(identity.shape) == (1, int(channels), int(window_bins)),
             "APFG cached identity must be [1,N,W] before batch expansion")
    _require(int(batch_size) >= 1, "APFG supervised batch is empty")
    expanded = identity.expand(int(batch_size), -1, -1)
    _require(tuple(expanded.shape) == (int(batch_size), int(channels), int(window_bins)),
             "APFG expanded identity [B,N,W] topology drift")
    return expanded


def alpha_task_step(*, prepared: PreparedAPFG, material: SourceSessionMaterial,
                    coordinates: Sequence[SourceCoordinate], epoch_one_indexed: int,
                    canonical_batch_ordinal: int, cache: dict[FrozenIdentityCacheKey, tuple[Any, Any]]) -> float:
    """One homogeneous causal source batch; no teacher forward is reachable."""
    import torch
    _require(prepared.adapter._alpha_training_enabled and not prepared.module.training and not prepared.adapter.native.training,
             "APFG alpha training must preserve inherited eval/no-dropout state")
    _require(bool(coordinates) and len(coordinates) <= plan.SOURCE_BATCH_MAX_MEMBERS,
             "APFG source batch cardinality drift")
    anchor = coordinates[0]
    _require(all(item.session == anchor.session and item.query_trial_index == anchor.query_trial_index for item in coordinates),
             "APFG source batch crossed session/query trial")
    state = pool_for_controller_coordinate(material=material, coordinate=anchor,
                                           epoch_one_indexed=epoch_one_indexed,
                                           canonical_batch_ordinal=canonical_batch_ordinal)
    native, post = cached_frozen_branches(prepared=prepared, material=material, coordinate=anchor, state=state, cache=cache)
    one_identity = native + torch.tanh(prepared.adapter.alpha) * (post - native)
    starts = np.asarray([item.window_start for item in coordinates], dtype=np.int64)
    neural_all = np.asarray(prepared.datamodule.train_dataset.neural_data[anchor.session], dtype=np.float32)
    target_all = np.asarray(prepared.datamodule.train_dataset.covariate_data[anchor.session], dtype=np.float32)
    windows = np.ascontiguousarray(neural_all[starts[:, None] + np.arange(plan.WINDOW_BINS)[None, :]], dtype=np.float32)
    target = np.ascontiguousarray(target_all[starts + plan.WINDOW_BINS - 1], dtype=np.float32)
    neural = torch.from_numpy(windows).to(prepared.device)
    target_tensor = torch.from_numpy(target).to(prepared.device).unsqueeze(1).expand(-1, plan.WINDOW_BINS, -1)
    identity = expand_identity_for_windows(identity=one_identity, batch_size=len(coordinates),
                                           channels=int(neural.shape[2]), window_bins=plan.WINDOW_BINS)
    prepared.optimizer.zero_grad(set_to_none=True)
    loss = task_only_scaled_last_bin_mse(torch=torch, student=prepared.module.student, neural=neural,
                                         identity=identity, target=target_tensor)
    _require(bool(torch.isfinite(loss)), "APFG source task loss is nonfinite")
    loss.backward()
    _require(prepared.adapter.alpha.grad is not None and bool(torch.isfinite(prepared.adapter.alpha.grad)),
             "APFG scalar alpha is disconnected/nonfinite")
    for name, parameter in prepared.module.named_parameters():
        if name != "student.id_encoder.alpha":
            _require(parameter.grad is None, f"APFG inherited parameter received gradient: {name}")
    prepared.optimizer.step()
    _require(bool(torch.isfinite(prepared.adapter.alpha)), "APFG alpha became nonfinite after optimizer step")
    return float(loss.detach().cpu())


def _set_alpha(adapter: AnchoredPostFusionGate, value: float, *, training: bool) -> None:
    import torch
    with torch.no_grad():
        adapter.alpha.copy_(torch.tensor(value, dtype=torch.float32, device=adapter.alpha.device))
    adapter.set_alpha_training(training)


def _session_metric(*, prepared: PreparedAPFG, material: SourceSessionMaterial, law: str,
                    branch_cache: dict[str, tuple[Any, Any]] | None = None) -> float:
    result = rollout_apfg_raw_pool(torch=__import__("torch"), model=prepared.module, material=material,
                                   dataset=prepared.datamodule.train_dataset, law=law, device=prepared.device,
                                   batch_size=plan.TRAIN_BATCH_SIZE, branch_cache=branch_cache)
    return float(result["r2"])


def train_select_and_refit(*, prepared: PreparedAPFG, materials: Mapping[str, SourceSessionMaterial]) -> dict[str, object]:
    """Run exact 12-epoch lexical 5/2 selection and all-seven reset/refit.

    The function is post-attempt only.  It retains one prepared DataModule;
    refit clones only the frozen strict-loaded module and resets scalar alpha.
    """
    import copy
    sessions = tuple(sorted(materials))
    split = lexical_source_split(sessions)
    _require(set(sessions) == set(materials) and len(sessions) == plan.SOURCE_SESSION_COUNT,
             "APFG source material roster drift")
    batches = [(name, batch) for name in split["fit"] for batch in canonical_m30_source_batches(materials[name])]
    _require(bool(batches), "APFG lexical source fit has no canonical batches")
    cache: dict[FrozenIdentityCacheKey, tuple[Any, Any]] = {}
    validation_branch_cache: dict[str, tuple[Any, Any]] = {}
    zero_validation: dict[str, float] | None = None
    rows: list[EpochValidation] = []
    losses: list[dict[str, object]] = []
    for epoch in range(1, plan.EPOCHS + 1):
        prepared.adapter.set_alpha_training(True)
        values = [alpha_task_step(prepared=prepared, material=materials[name], coordinates=batch,
                                  epoch_one_indexed=epoch, canonical_batch_ordinal=ordinal, cache=cache)
                  for ordinal, (name, batch) in enumerate(batches)]
        _require(bool(values), "APFG epoch had no alpha steps")
        prepared.adapter.set_alpha_training(False)
        learned = {name: _session_metric(prepared=prepared, material=materials[name], law="UNCAPPED",
                                         branch_cache=validation_branch_cache)
                   for name in split["validation"]}
        if zero_validation is None:
            previous = float(prepared.adapter.alpha.detach().cpu())
            _set_alpha(prepared.adapter, 0.0, training=False)
            zero_validation = {name: _session_metric(prepared=prepared, material=materials[name], law="UNCAPPED",
                                                      branch_cache=validation_branch_cache)
                               for name in split["validation"]}
            _set_alpha(prepared.adapter, previous, training=True)
        zero = zero_validation
        rows.append(EpochValidation(epoch=epoch, per_session_r2=learned, zero_gate_per_session_r2=zero))
        losses.append({"epoch": epoch, "mean_task_only_last_bin_mse": float(sum(values) / len(values)),
                       "steps": len(values), "teacher_forward_calls": 0,
                       "alpha_after_epoch": float(prepared.adapter.alpha.detach().cpu())})
    selection = select_earliest_best(rows, split["validation"])
    # The full post30 deployment validation population is intentionally wider
    # than the M30-ready supervised training batch population.  FIXED30 is a
    # selected-epoch descriptive readout only; it cannot affect selection.
    selected_alpha = float(losses[int(selection["selected_epoch"]) - 1]["alpha_after_epoch"])
    _set_alpha(prepared.adapter, selected_alpha, training=False)
    selected_fixed_learned = {name: _session_metric(prepared=prepared, material=materials[name], law="FIXED30",
                                                     branch_cache=validation_branch_cache)
                              for name in split["validation"]}
    _set_alpha(prepared.adapter, 0.0, training=False)
    selected_fixed_zero = {name: _session_metric(prepared=prepared, material=materials[name], law="FIXED30",
                                                  branch_cache=validation_branch_cache)
                           for name in split["validation"]}
    _set_alpha(prepared.adapter, float(losses[-1]["alpha_after_epoch"]), training=True)
    # Fresh model clone and alpha reset; same DataModule/material authorities.
    refit_module = copy.deepcopy(prepared.module)
    refit_adapter = refit_module.student.id_encoder
    _require(isinstance(refit_adapter, AnchoredPostFusionGate), "APFG refit adapter clone drift")
    _set_alpha(refit_adapter, 0.0, training=True)
    refit = PreparedAPFG(module=refit_module, datamodule=prepared.datamodule, adapter=refit_adapter,
                         optimizer=alpha_only_adam(refit_adapter), device=prepared.device,
                         pit_prepare_calls=prepared.pit_prepare_calls, evidence=prepared.evidence)
    refit_cache: dict[FrozenIdentityCacheKey, tuple[Any, Any]] = {}
    all_batches = [(name, batch) for name in sessions for batch in canonical_m30_source_batches(materials[name])]
    for epoch in range(1, int(selection["selected_epoch"]) + 1):
        for ordinal, (name, batch) in enumerate(all_batches):
            alpha_task_step(prepared=refit, material=materials[name], coordinates=batch,
                            epoch_one_indexed=epoch, canonical_batch_ordinal=ordinal, cache=refit_cache)
    refit.adapter.set_alpha_training(False)
    return {"selection": selection, "selection_rows": [
                {"epoch": row.epoch, "uncapped_validation_r2": row.equal_session_mean(),
                 "zero_uncapped_validation_r2": row.zero_gate_mean(),
                 "learned_per_session_r2": dict(row.per_session_r2),
                 "zero_per_session_r2": dict(row.zero_gate_per_session_r2)} for row in rows],
            "training_epochs": losses, "fit_sessions": list(split["fit"]),
            "validation_sessions": list(split["validation"]), "refit_module": refit.module,
            "refit_adapter": refit.adapter, "refit_alpha": float(refit.adapter.alpha.detach().cpu()),
            "teacher_forward_calls": 0, "activity_authority": plan.ACTIVITY_AUTHORITY,
            "validation_branch_cache_states": len(validation_branch_cache),
            "zero_validation_reused_across_epochs": True,
            "fit_optimizer_updates": int(sum(item["steps"] for item in losses)),
            "refit_optimizer_updates": int(len(all_batches) * int(selection["selected_epoch"])),
            "source_batch_count_fit": len(batches), "source_batch_count_all7": len(all_batches),
            "training_population": {"name": "m30_ready_supervised_coordinates",
                                    "coordinate_count": int(sum(len(item.coordinates) for item in materials.values()))},
            "validation_population": {"name": "complete_post30_deployment_replay",
                                      "query_trial_count": int(sum(len(materials[name].query_rows) for name in split["validation"]))},
            "selected_epoch_fixed30_descriptive": {"epoch": int(selection["selected_epoch"]),
                "learned_per_session_r2": selected_fixed_learned, "zero_per_session_r2": selected_fixed_zero,
                "not_used_for_selection_or_safety_gate": True}}


__all__ = ("TrainingError", "PreparedAPFG", "validate_cuda_launch_attestation", "prepare_selected_t4_apfg_after_attempt",
           "cached_frozen_branches", "expand_identity_for_windows", "alpha_task_step", "train_select_and_refit")
