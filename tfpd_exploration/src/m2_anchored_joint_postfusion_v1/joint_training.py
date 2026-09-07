"""Route-owned AJPF joint task step over the APFG causal source material."""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from . import plan, runtime

class JointTrainingError(RuntimeError): pass
def _need(ok: bool, message: str) -> None:
    if not ok: raise JointTrainingError(message)


UPDATE_SENTINEL_ORDINALS = (0, 11, plan.SOURCE_GROUPS_PER_EPOCH - 1)


def is_update_sentinel(*, epoch: int, ordinal: int) -> bool:
    _need(1 <= epoch <= plan.EPOCHS and 0 <= ordinal < plan.SOURCE_GROUPS_PER_EPOCH,
          "AJPF update-sentinel domain drift")
    return ordinal in UPDATE_SENTINEL_ORDINALS


def _async_finite(torch: Any, value: Any) -> Any:
    finite = torch.isfinite(value).all()
    # CUDA's async assertion records a device-side failure without an ordinary
    # per-parameter CPU synchronization.  CPU/unit-test tensors simply carry
    # the boolean to the epoch/sentinel boundary.
    if getattr(value, "is_cuda", False) and hasattr(torch, "_assert_async"):
        torch._assert_async(finite, "AJPF nonfinite materialized gradient")
    return finite


def _packed_single_h2d(*, torch: Any, tensors: Sequence[tuple[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make one pinned homogeneous CPU payload and issue exactly one H2D call.

    All four route inputs are float32.  Metadata is retained as a receipt
    witness; CPU slice equality proves packing has not altered values, while
    device views are aliases of the one transferred allocation.
    """
    _need(len(tensors) == 4 and all(getattr(value, "dtype", None) == torch.float32 for _, value in tensors),
          "AJPF packed H2D dtype/topology drift")
    flat_parts = [value.contiguous().reshape(-1) for _, value in tensors]
    flat_cpu = torch.cat(flat_parts).contiguous()
    # pin_memory is part of the production CUDA route.  It is intentionally
    # isolated so the no-CUDA unit test can substitute the host allocator.
    pinned = flat_cpu.pin_memory()
    flat_device = pinned.to("cuda:0", non_blocking=True)
    values: dict[str, Any] = {}; fields: list[dict[str, Any]] = []; offset = 0
    for (name, original), part in zip(tensors, flat_parts, strict=True):
        count = int(part.numel())
        _need(bool(torch.equal(flat_cpu.narrow(0, offset, count), part)), "AJPF packed CPU value drift")
        view = flat_device.narrow(0, offset, count).view(tuple(original.shape))
        _need(tuple(view.shape) == tuple(original.shape) and view.dtype == original.dtype,
              "AJPF packed device view shape/dtype drift")
        values[name] = view
        fields.append({"name": name, "byte_offset": offset * int(original.element_size()),
                       "shape": list(original.shape), "dtype": str(original.dtype), "nbytes": count * int(original.element_size())})
        offset += count
    return values, {"h2d_transfers": 1, "resident_batch_groups": 1, "packed_h2d": True,
                    "packed_byte_length": int(flat_cpu.numel() * flat_cpu.element_size()), "packed_fields": fields,
                    "cpu_pack_values_exact": True}

def materialize_resident_batch(*, torch: Any, module: Any, material: Any, coordinates: Sequence[Any],
                               epoch: int, ordinal: int) -> dict[str, Any]:
    """Construct and transfer one shared causal batch exactly once."""
    from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import source_replay
    _need(coordinates and len(coordinates) <= plan.BATCH_SIZE, "AJPF source group cardinality drift")
    coordinate = coordinates[0]
    _need(all(x.session == coordinate.session and x.query_trial_index == coordinate.query_trial_index for x in coordinates),
          "AJPF mixed source group")
    state = source_replay.pool_for_controller_coordinate(material=material, coordinate=coordinate,
        epoch_one_indexed=epoch, canonical_batch_ordinal=ordinal)
    members = np.asarray(state.member_trial_ids, dtype=np.int64)
    dataset = module._ajpf_dataset
    starts = np.asarray([x.window_start for x in coordinates], dtype=np.int64)
    neural_all = np.asarray(dataset.neural_data[coordinate.session], dtype=np.float32)
    target_all = np.asarray(dataset.covariate_data[coordinate.session], dtype=np.float32)
    cpu = (("calibration", torch.from_numpy(np.ascontiguousarray(material.activities[members], dtype=np.float32)).unsqueeze(0)),
           ("side", torch.from_numpy(np.ascontiguousarray(material.normalized_side, dtype=np.float32)).unsqueeze(0)),
           ("windows", torch.from_numpy(np.ascontiguousarray(neural_all[starts[:, None] + np.arange(50)[None, :]], dtype=np.float32))),
           ("target", torch.from_numpy(np.ascontiguousarray(target_all[starts + 49], dtype=np.float32)).unsqueeze(1)))
    values, receipt = _packed_single_h2d(torch=torch, tensors=cpu)
    return {"state": state, "coordinates": coordinates, **values, **receipt}


def one_joint_step(*, torch: Any, module: Any, optimizer: Any, material: Any, coordinates: Sequence[Any],
                   epoch: int, ordinal: int, mask: Any, hook: Any, resident: dict[str, Any] | None = None,
                   update_sentinel: bool | None = None, audit: bool = False) -> dict[str,Any]:
    """One arm, one causal state, one task-only update; called by paired runner."""
    resident = materialize_resident_batch(torch=torch, module=module, material=material, coordinates=coordinates,
                                         epoch=epoch, ordinal=ordinal) if resident is None else resident
    _need(resident.get("coordinates") is coordinates and resident.get("h2d_transfers") == 1,
          "AJPF resident batch ownership/transfer drift")
    state = resident["state"]; calibration = resident["calibration"]; side = resident["side"]
    windows = resident["windows"]; target = resident["target"]
    identity=module.student.id_encoder.forward_batch(calibration,side_features=side).expand(len(coordinates),-1,-1)
    sentinel = is_update_sentinel(epoch=epoch, ordinal=ordinal) if update_sentinel is None else bool(update_sentinel)
    before = ([[parameter.detach().clone() for parameter in group["params"]] for group in optimizer.param_groups]
              if sentinel else None)
    optimizer.zero_grad(set_to_none=True)
    hook.install(module.student.decoder.fc_in)
    try:
        prediction=module.student.decode_with_identity(windows,identity)
        loss=runtime.task_only_last_bin_mse(torch=torch,prediction=prediction,target=target,audit=audit)
        loss.backward()
        group_grad = []
        for group in optimizer.param_groups:
            gradients = [parameter.grad for parameter in group["params"] if parameter.grad is not None]
            finite = ([ _async_finite(torch, value) for value in gradients ] if gradients else [])
            finite_all = torch.stack(finite).all() if finite else torch.tensor(True, device=windows.device)
            nonzero = (torch.stack([torch.count_nonzero(value) != 0 for value in gradients]).any()
                       if gradients else torch.tensor(False, device=windows.device))
            group_grad.append({"materialized": len(gradients),
                               "finite_tensor": finite_all, "nonzero_tensor": nonzero})
        optimizer.step()
    finally:
        hook.remove()
    hook.validate(require_rep_call=True)
    updates = []
    if sentinel:
        _need(before is not None, "AJPF sentinel snapshot missing")
        for before_group, group in zip(before, optimizer.param_groups, strict=True):
            norm = sum((torch.linalg.vector_norm(parameter.detach() - initial)
                        for parameter, initial in zip(group["params"], before_group, strict=True)),
                       torch.zeros((), device=windows.device))
            updates.append({"norm_tensor": norm, "nonzero_tensor": norm != 0,
                            "finite_tensor": _async_finite(torch, norm)})
    else:
        updates = [{"norm_tensor": None, "nonzero_tensor": None, "finite_tensor": None} for _ in optimizer.param_groups]
    return {"forward_calls":1,"backward_calls":1,"adam_steps":1,"teacher_forward_calls":0,
            "internal_unit_dropout_draws":0,"loss_tensor":loss.detach(),"requested_m":state.requested_size,
            "hook_first_calls":hook.first_calls,"hook_rep_calls":hook.rep_calls,
            "group_gradient":group_grad,"group_update":updates,"uses_shared_resident": True,
            "resident_object_id": id(resident), "update_sentinel": sentinel}
