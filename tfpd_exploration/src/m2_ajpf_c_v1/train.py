"""Paired continual-law joint training for AJPF-C (two modules per law).

Reuses the reviewed AJPF V1 machinery by import: strict authorization
(``_authorize_joint_trainable``), two-group Adam (``_fresh_adam``), internal
unit-dropout reservation (``_disable_clone_internal_unit_dropout``), the
paired governing dropout draw (``dropout.governing_probability_and_mask``),
the first-call fc_in mask hook, the packed single-H2D resident batch, and the
task-only one-arm step (``one_joint_step``).  The only route-owned difference
is the pool law: residents are built from the boundary-free chunk stream, so
the training graph equals the continual deployment graph.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import chunk_law, plan, source_stream


class TrainError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainError(message)


class _ChunkStateShim:
    def __init__(self, requested_size: int):
        self.requested_size = int(requested_size)


def materialize_chunk_resident_for_law(*, torch, material, coordinates, law, joint_training):
    import numpy as np

    coordinate = coordinates[0]
    require(all(x.session == coordinate.session and x.state_index == coordinate.state_index
                for x in coordinates), "C-law mixed source group")
    commits = material.commits[law]
    stack = chunk_law.pool_rows_for_state(
        material.neural, material.seed_rows, material.support_count, commits,
        coordinate.state_index)
    starts = np.asarray([x.window_start for x in coordinates], dtype=np.int64)
    neural_all = material.neural
    target_all = material.targets
    cpu = (
        ("calibration", torch.from_numpy(np.ascontiguousarray(stack)).unsqueeze(0)),
        ("side", torch.from_numpy(np.ascontiguousarray(material.normalized_side)).unsqueeze(0)),
        ("windows", torch.from_numpy(np.ascontiguousarray(
            neural_all[starts[:, None] + np.arange(50)[None, :]], dtype=np.float32))),
        ("target", torch.from_numpy(np.ascontiguousarray(
            target_all[starts + 49], dtype=np.float32)).unsqueeze(1)),
    )
    values, receipt = joint_training._packed_single_h2d(torch=torch, tensors=cpu)
    return {"state": _ChunkStateShim(len(stack)), "coordinates": coordinates, **values, **receipt}


def prepare_base(repo_root: Path):
    """Strict-load the Selected-T4 checkpoint once (CPU) via the sealed loader."""
    import sys

    streaming_root = repo_root / "streaming_calibration_exp"
    for path in (repo_root, streaming_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import src.models.streaming_calibration_module  # noqa: F401 - pins `src`

    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import runner as ajpf_runner

    model, data_module, _task, metadata = load_frozen_model_and_data()
    require(metadata["checkpoint_sha256"] == plan.SELECTED_CHECKPOINT_SHA256, "checkpoint drift")
    state_sha = ajpf_runner._student_state_sha256(model.student)
    require(state_sha == plan.SELECTED_STUDENT_STATE_SHA256, f"selected student state drift: {state_sha}")
    return model, data_module, ajpf_runner


def prepare_arms(base_module, ajpf_runner, device: str, arms: tuple[str, ...]):
    import torch

    modules, optimizers, dropout_original = {}, {}, {}
    for arm in arms:
        module = copy.deepcopy(base_module)
        if arm.endswith("R1"):
            from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import adapter
            installed = adapter.install_after_strict_load(module.student, "J-R1")
            adapter.require_positive_zero(installed.alpha)
        module.to(torch.device(device))
        encoder_alpha, decoder = ajpf_runner._authorize_joint_trainable(module, arm)
        dropout_original[arm] = ajpf_runner._disable_clone_internal_unit_dropout(module)
        modules[arm] = module
        optimizers[arm] = ajpf_runner._fresh_adam(torch=torch, encoder_alpha=encoder_alpha, decoder=decoder)
    ids = [id(parameter) for module in modules.values() for parameter in module.parameters()]
    require(len(ids) == len(set(ids)), "C-law arms share Parameter objects")
    return modules, optimizers, dropout_original


def train_one_law(*, repo_root: Path, law: str, base_module, data_module, ajpf_runner,
                  device: str, result_root: Path, progress) -> dict[str, Any]:
    import torch

    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import dropout as ajpf_dropout
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import joint_training

    arms = plan.ARMS_PER_LAW[law]
    materials = [source_stream.build_session_material(data_module.train_dataset, session)
                 for session in sorted(data_module.train_dataset.calib_trialized_neural_features)]
    require(len(materials) == 7, "source session count drift")
    groups = source_stream.coordinate_stream(materials, law)
    total_coordinates = sum(len(group) for group in groups)
    progress(f"law={law} groups={len(groups)} coordinates={total_coordinates}")

    modules, optimizers, dropout_original = prepare_arms(base_module, ajpf_runner, device, arms)

    random.seed(plan.SEED)
    np.random.seed(plan.SEED)
    torch.manual_seed(plan.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(plan.SEED)

    material_by_session = {material.session: material for material in materials}
    epoch_losses = {arm: [] for arm in arms}
    sentinel_ok = {arm: {"enc_nonzero": False, "dec_nonzero": False} for arm in arms}
    rng_capture = lambda: (random.getstate(), np.random.get_state(),
                           torch.get_rng_state(),
                           torch.cuda.get_rng_state() if torch.cuda.is_available() else None)

    smoke_started = time.monotonic()
    wall_started = time.monotonic()
    step_count = 0
    for epoch in range(1, plan.EPOCHS + 1):
        running = {arm: [] for arm in arms}
        for ordinal, group in enumerate(groups):
            material = material_by_session[group[0].session]
            resident = materialize_chunk_resident_for_law(
                torch=torch, material=material, coordinates=group, law=law,
                joint_training=joint_training)
            origin = dropout_original[arms[0]]
            probability, mask, _digest = ajpf_dropout.governing_probability_and_mask(
                torch=torch, batch_size=len(group), units=96,
                dynamic=origin["dynamic_dropout"], dropout_rate=origin["dropout_rate"],
                dynamic_low=origin["dynamic_dropout_low"], dynamic_high=origin["dynamic_dropout_high"],
                device=resident["windows"].device)
            post_mask = rng_capture()
            is_sentinel = ordinal in plan.UPDATE_SENTINEL_ORDINALS or ordinal == len(groups) - 1
            for arm in arms:
                random.setstate(post_mask[0])
                np.random.set_state(post_mask[1])
                torch.set_rng_state(post_mask[2])
                if post_mask[3] is not None:
                    torch.cuda.set_rng_state(post_mask[3])
                hook = ajpf_dropout.FirstCallMaskHook(mask=mask)
                step = joint_training.one_joint_step(
                    torch=torch, module=modules[arm], optimizer=optimizers[arm],
                    material=None, coordinates=group, epoch=epoch, ordinal=ordinal,
                    mask=mask, hook=hook, resident=resident, update_sentinel=is_sentinel)
                running[arm].append(float(step["loss_tensor"].detach()))
                if is_sentinel:
                    updates = step["group_update"]
                    if bool(updates[0]["nonzero_tensor"]):
                        sentinel_ok[arm]["enc_nonzero"] = True
                    if bool(updates[1]["nonzero_tensor"]):
                        sentinel_ok[arm]["dec_nonzero"] = True
            step_count += 1
            if step_count == plan.SMOKE_STEPS:
                elapsed = time.monotonic() - smoke_started
                per_step = elapsed / plan.SMOKE_STEPS
                projection = per_step * len(groups) * plan.EPOCHS
                progress(f"law={law} smoke {plan.SMOKE_STEPS} groups in {elapsed:.1f}s "
                         f"-> projected {projection:.0f}s (cap {plan.WALL_CAP_SECONDS_PER_LAW})")
                require(projection <= plan.WALL_CAP_SECONDS_PER_LAW,
                        f"law={law} projected wall {projection:.0f}s exceeds cap")
            if (time.monotonic() - wall_started) > plan.WALL_CAP_SECONDS_PER_LAW:
                raise TrainError(f"law={law} wall cap exceeded")
        for arm in arms:
            epoch_losses[arm].append(float(np.mean(running[arm])))
            progress(f"law={law} epoch {epoch} mean loss {arm} {epoch_losses[arm][-1]:.6f}")

    for arm in arms:
        require(sentinel_ok[arm]["enc_nonzero"] and sentinel_ok[arm]["dec_nonzero"],
                f"law={law} arm {arm} never produced a nonzero sentinel update in a group")

    checkpoints = {}
    checkpoint_dir = result_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        module = modules[arm]
        module.eval()
        path = checkpoint_dir / f"{arm}_epoch12.pt"
        import io
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        body = buffer.getvalue()
        import hashlib
        digest = hashlib.sha256(body).hexdigest()
        path.write_bytes(body)
        # strict reload proof on a fresh clone
        fresh = copy.deepcopy(base_module)
        if arm.endswith("R1"):
            from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import adapter
            adapter.install_after_strict_load(fresh.student, "J-R1")
        fresh.load_state_dict(torch.load(io.BytesIO(body), map_location="cpu", weights_only=True), strict=True)
        state_sha = ajpf_runner._student_state_sha256(fresh.student)
        alpha = None
        if arm.endswith("R1"):
            with torch.no_grad():
                alpha = float(fresh.student.id_encoder.alpha.item())
        checkpoints[arm] = {"path": str(path.relative_to(repo_root)), "sha256": digest,
                            "student_state_sha256": state_sha, "alpha": alpha}
    return {
        "law": law,
        "groups_per_epoch": len(groups),
        "coordinates_per_epoch": total_coordinates,
        "epoch_mean_loss": epoch_losses,
        "sentinel_nonzero_updates": sentinel_ok,
        "wall_seconds": time.monotonic() - wall_started,
        "checkpoints": checkpoints,
        "alpha_initial_positive_zero": True,
    }
