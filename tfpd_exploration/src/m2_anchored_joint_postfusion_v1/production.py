"""The sole AJPF production route, intentionally unreachable from the CLI."""
from __future__ import annotations

import copy
import hashlib
import io
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import lifecycle, plan, scorer


class ProductionError(RuntimeError):
    pass


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise ProductionError(message)


def _tensor_bool(value: Any) -> bool:
    """Synchronize only at a declared receipt/sentinel boundary."""
    return bool(value.detach().item()) if hasattr(value, "detach") else bool(value)


def _receipt_step(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Convert a sparse sentinel's device evidence into immutable JSON facts."""
    result: dict[str, Any] = {}
    for arm, row in rows.items():
        converted = {key: value for key, value in row.items()
                     if key not in {"loss_tensor", "group_gradient", "group_update", "resident_object_id"}}
        converted["loss"] = float(row["loss_tensor"].detach().cpu())
        converted["group_gradient"] = [{"materialized": int(group["materialized"]),
            "finite": _tensor_bool(group["finite_tensor"]), "nonzero": _tensor_bool(group["nonzero_tensor"])}
            for group in row["group_gradient"]]
        converted["group_update"] = [{"observed": group["nonzero_tensor"] is not None,
            "nonzero": (_tensor_bool(group["nonzero_tensor"]) if group["nonzero_tensor"] is not None else None),
            "finite": (_tensor_bool(group["finite_tensor"]) if group.get("finite_tensor") is not None else None),
            "norm": (float(group["norm_tensor"].detach().cpu()) if group["norm_tensor"] is not None else None)}
            for group in row["group_update"]]
        result[arm] = converted
    return result


def _rng_digest(state: tuple[Any, Any, Any, Any]) -> str:
    """Digest an RNG snapshot only at an explicitly declared receipt point."""
    import pickle
    digest = hashlib.sha256()
    digest.update(pickle.dumps(state[0], protocol=4)); digest.update(pickle.dumps(state[1], protocol=4))
    for value in state[2:]:
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _sparse_arm_evidence(*, torch: Any, module: Any, optimizer: Any, arm: str) -> dict[str, Any]:
    """CPU/digest work allowed solely on smoke or declared sentinels."""
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical
    buffer = io.BytesIO(); torch.save(optimizer.state_dict(), buffer)
    evidence: dict[str, Any] = {"student_state_sha256": physical._student_state_sha(module),
                                 "optimizer_state_sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}
    _need(len(evidence["student_state_sha256"]) == 64 and len(evidence["optimizer_state_sha256"]) == 64,
          "AJPF sparse state/optimizer digest drift")
    if arm == "J-R1":
        alpha = module.student.id_encoder.alpha
        evidence["alpha_value"] = float(alpha.detach().cpu())
        evidence["alpha_gradient"] = (float(alpha.grad.detach().cpu()) if alpha.grad is not None else None)
        _need(math.isfinite(evidence["alpha_value"]) and evidence["alpha_gradient"] is not None
              and math.isfinite(evidence["alpha_gradient"]), "AJPF sparse J-R1 alpha evidence nonfinite")
    return evidence


def _smoke_rates(elapsed_seconds: float) -> dict[str, float | int]:
    _need(elapsed_seconds > 0.0, "AJPF smoke elapsed-time drift")
    return {"steps_per_second": 12.0 / elapsed_seconds, "arm_forwards": 36,
            "arm_forwards_per_second": 36.0 / elapsed_seconds}


def _training_attempt(cap: lifecycle.Capability) -> dict[str, Any]:
    return {"schema": "m2_anchored_joint_postfusion_v1_attempt", "closure": cap.closure,
            "closure_sha256": cap.closure_sha256, "target_access": False,
            "checkpoint_or_data_opened": False, "cuda_initialized": False}


_FROZEN_COMMON_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                      "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                      "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                      "NUMEXPR_NUM_THREADS": "1"}


def validate_frozen_environment(*, repo_root: Path, score: bool) -> dict[str, str]:
    """Stdlib-only exact environment gate, safe before any artifact exists."""
    expected = {**_FROZEN_COMMON_ENV, "CUDA_VISIBLE_DEVICES": "" if score else "0", "PYTHONPATH": str(repo_root)}
    observed = {key: os.environ.get(key, "") for key in expected}
    _need(observed == expected, "AJPF frozen execution environment drift")
    return observed


def _created_root_identity(root: Path) -> tuple[int, int]:
    info = root.stat(follow_symlinks=False)
    _need(root.is_dir() and not root.is_symlink(), "AJPF named result root type drift")
    return int(info.st_dev), int(info.st_ino)


def _revalidate_capability_boundary(*, cap: lifecycle.Capability, repo_root: Path,
                                    root_identity: tuple[int, int]) -> None:
    """Repeat admission invariants before an immutable terminal/failure leaf."""
    parent = cap.root.parent.stat(follow_symlinks=False)
    _need((parent.st_dev, parent.st_ino) == (cap.parent_dev, cap.parent_ino), "AJPF root-parent identity drift")
    _need(_created_root_identity(cap.root) == root_identity, "AJPF named result root identity drift")
    _need(lifecycle.closure_map(repo_root) == cap.closure, "AJPF closure drift at terminal boundary")


def _cuda_launch_attestation(torch: Any) -> dict[str, Any]:
    """Verify only the exposed logical GPU0; GPU1 is deliberately unobserved."""
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
          "AJPF GPU launch environment drift")
    _need(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
          "AJPF requires exactly logical CUDA device 0")
    properties = torch.cuda.get_device_properties(0)
    uuid = getattr(properties, "uuid", None)
    raw_uuid = str(uuid); suffix = "ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
    _need(raw_uuid in {suffix, "GPU-" + suffix}, "AJPF GPU0 UUID drift")
    canonical_uuid = "GPU-" + suffix
    free, total = torch.cuda.mem_get_info(0)
    _need(int(free) >= 4 * 1024**3, "AJPF GPU0 launch free-memory gate")
    return {"logical_device": 0, "uuid": canonical_uuid, "raw_uuid": raw_uuid, "canonical_uuid": canonical_uuid,
            "free_bytes_before": int(free), "total_bytes": int(total),
            "cuda_visible_devices": "0"}


class _TrainRuntime:
    """One post-attempt source construction and paired three-arm loop."""

    def __init__(self, *, repo_root: Path, launch: Mapping[str, Any]) -> None:
        import torch
        from . import runner
        from tfpd_exploration.src.m2_anchored_postfusion_gate_v1 import source_replay
        self.torch = torch
        self.prepared = runner.prepare_joint_arms_after_attempt(
            repo_root=repo_root, device="cuda:0", launch_attestation=dict(launch))
        dataset = self.prepared.datamodule.train_dataset
        raw = self.prepared.datamodule.train_calib_heldin_sessions
        names = tuple(sorted(dataset.calib_trialized_neural_features))
        _need(len(names) == 7, "AJPF exact seven-source roster drift")
        # This is the only source materialization.  The three arms only see
        # references into this frozen materialized graph.
        self.materials = {name: source_replay.materialize_pooled_g00m_source_session(
            dataset=dataset, raw_sessions=raw, session=name) for name in names}
        self.summary = source_replay.source_authority_summary(tuple(self.materials.values()))
        _need(self.summary.get("activity_authority") == plan.ACTIVITY_AUTHORITY
              and self.summary.get("m30_causal_coordinate_count") == plan.SOURCE_COORDINATES
              and self.summary.get("ordered_coordinate_sha256") == plan.SOURCE_COORDINATE_SHA256
              and self.summary.get("ordered_batch_sha256") == plan.SOURCE_BATCH_SHA256,
              "AJPF live 91717-coordinate authority drift")
        self.batches = tuple((name, batch) for name in names
                             for batch in source_replay.canonical_m30_source_batches(self.materials[name]))
        _need(len(self.batches) == plan.SOURCE_GROUPS_PER_EPOCH, "AJPF live 3455-batch authority drift")
        for module in self.prepared.modules.values():
            module._ajpf_dataset = dataset

    def source_authority(self, predecessor: Mapping[str, Any]) -> dict[str, Any]:
        return {"schema": "m2_anchored_joint_postfusion_v1_source_authority", "summary": self.summary,
                "pit_prepare_calls": self.prepared.pit_prepare_calls, "ordinary_dataloader_calls": 0,
                "materialization_count": 1, "shared_batch_groups": len(self.batches),
                "apfg_predecessor": predecessor["apfg"], "pooled_predecessor": predecessor["pooled"]}

    def _common_mask(self, batch: Sequence[Any], *, audit: bool) -> tuple[float, Any, str | None, Any, Any]:
        from . import dropout
        decoder = self.prepared.modules["J-NATIVE"].student.decoder
        sample = batch[0]
        units = int(self.prepared.datamodule.train_dataset.neural_data[sample.session].shape[1])
        original = self.prepared.decoder_dropout_original["J-NATIVE"]
        pre_mask_state = _rng_state(self.torch)
        probability, mask, digest = dropout.governing_probability_and_mask(
            torch=self.torch, batch_size=len(batch), units=units, dynamic=original["dynamic_dropout"],
            dropout_rate=original["dropout_rate"], dynamic_low=original["dynamic_dropout_low"],
            dynamic_high=original["dynamic_dropout_high"], device="cuda:0", audit=audit)
        _need(decoder.training and not decoder.dynamic_dropout and decoder.dropout_rate == 0.0,
              "AJPF clone internal decoder dropout was not disabled")
        return probability, mask, digest, pre_mask_state, _rng_state(self.torch)

    def _jr1_first_step_parity(self, resident: Mapping[str, Any]) -> dict[str, Any]:
        """A real pre-update J-R1 +0 proof on the exact first training batch."""
        from . import adapter, runtime
        native = self.prepared.modules["J-NATIVE"]
        r1 = self.prepared.modules["J-R1"]
        gated = r1.student.id_encoder
        adapter.require_positive_zero(gated.alpha)
        calibration, side = resident["calibration"], resident["side"]
        windows, target = resident["windows"], resident["target"]
        state = _rng_state(self.torch)
        try:
            native_identity = native.student.id_encoder.forward_batch(calibration, side_features=side)
            r1_identity = gated.forward_batch(calibration, side_features=side)
            post = gated._post(calibration, side)
            residual = post - native_identity
            _need(self.torch.equal(native_identity, r1_identity)
                  and _tensor_bool(self.torch.isfinite(post).all()) and _tensor_bool(self.torch.isfinite(residual).all()),
                  "AJPF first-step J-R1 identity/post/residual parity drift")
            native_prediction = native.student.decode_with_identity(windows, native_identity.expand(len(resident["coordinates"]), -1, -1))
            _set_rng(self.torch, state)
            r1_prediction = r1.student.decode_with_identity(windows, r1_identity.expand(len(resident["coordinates"]), -1, -1))
            native_loss = runtime.task_only_last_bin_mse(torch=self.torch, prediction=native_prediction, target=target, audit=True)
            r1_loss = runtime.task_only_last_bin_mse(torch=self.torch, prediction=r1_prediction, target=target, audit=True)
            _need(self.torch.equal(native_prediction, r1_prediction) and self.torch.equal(native_loss, r1_loss),
                  "AJPF first-step J-R1 prediction/task-loss bitwise parity drift")
            return {"alpha_ieee_positive_zero": True, "identity_bitwise": True, "prediction_bitwise": True,
                    "task_loss_bitwise": True, "post_finite": True, "residual_finite": True}
        finally:
            _set_rng(self.torch, state)

    def run_and_publish(self, *, root: Path, published: list[str]) -> tuple[str, list[str]]:
        """Run once and publish smoke/epoch receipts at their natural boundaries."""
        _need(not hasattr(self, "_complete"), "AJPF training runtime reused")
        from . import dropout, joint_training
        smoke: list[dict[str, Any]] = []; epoch_sha: list[str] = []
        self.torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        smoke_preflight: dict[str, Any] | None = None
        for epoch in range(1, plan.EPOCHS + 1):
            first: dict[str, Any] | None = None; last: dict[str, Any] | None = None
            coverage = {arm: [{"finite": self.torch.tensor(True, device="cuda:0"),
                                "nonzero": self.torch.tensor(False, device="cuda:0"), "updates": []}
                               for _ in range(2)] for arm in plan.ARM_ORDER}
            loss_sum = {arm: self.torch.zeros((), device="cuda:0") for arm in plan.ARM_ORDER}
            for ordinal, (name, batch) in enumerate(self.batches):
                resident = joint_training.materialize_resident_batch(torch=self.torch,
                    module=self.prepared.modules["J-NATIVE"], material=self.materials[name], coordinates=batch,
                    epoch=epoch, ordinal=ordinal)
                audit = (epoch == 1 and ordinal < 12) or joint_training.is_update_sentinel(epoch=epoch, ordinal=ordinal)
                if epoch == 1 and ordinal == 0:
                    smoke_preflight = {"jr1_first_step_parity": self._jr1_first_step_parity(resident)}
                probability, mask, mask_digest, pre_mask_state, post_mask_state = self._common_mask(batch, audit=audit)
                arm_rows: dict[str, Any] = {}; arm_end_states = []
                for arm in plan.ARM_ORDER:
                    _set_rng(self.torch, post_mask_state)
                    hook = dropout.FirstCallMaskHook(mask)
                    row = joint_training.one_joint_step(torch=self.torch, module=self.prepared.modules[arm],
                        optimizer=self.prepared.optimizers[arm], material=self.materials[name], coordinates=batch,
                        epoch=epoch, ordinal=ordinal, mask=mask, hook=hook, resident=resident, audit=audit)
                    row.update({"governing_dropout_probability": probability, "governing_mask_sha256": mask_digest,
                                "mask_applications": 1, "decoder_training": True, "decoder_frozen": False})
                    if audit:
                        row.update({"rng_pre_mask_sha256": _rng_digest(pre_mask_state),
                                    "rng_post_mask_sha256": _rng_digest(post_mask_state),
                                    "rng_state_gets_per_group": 2,
                                    "rng_state_restores_per_group": 3,
                                    "sparse_evidence": _sparse_arm_evidence(torch=self.torch,
                                        module=self.prepared.modules[arm], optimizer=self.prepared.optimizers[arm], arm=arm)})
                    loss_sum[arm] = loss_sum[arm] + row["loss_tensor"]
                    for group_index in (0, 1):
                        coverage[arm][group_index]["finite"] = self.torch.logical_and(
                            coverage[arm][group_index]["finite"], row["group_gradient"][group_index]["finite_tensor"])
                        coverage[arm][group_index]["nonzero"] = self.torch.logical_or(
                            coverage[arm][group_index]["nonzero"], row["group_gradient"][group_index]["nonzero_tensor"])
                        update = row["group_update"][group_index]["nonzero_tensor"]
                        if update is not None:
                            coverage[arm][group_index]["updates"].append(update)
                    arm_rows[arm] = row
                    if audit:
                        arm_end_states.append(_rng_state(self.torch))
                        row["rng_arm_end_sha256"] = _rng_digest(arm_end_states[-1])
                if audit:
                    _need(all(_same_rng(state, arm_end_states[0]) for state in arm_end_states[1:]), "AJPF paired end RNG drift")
                _need(all(row["uses_shared_resident"] and row["resident_object_id"] == id(resident)
                          for row in arm_rows.values()), "AJPF shared resident transfer evidence drift")
                if ordinal == 0:
                    first = _receipt_step(arm_rows)
                if ordinal == plan.SOURCE_GROUPS_PER_EPOCH - 1:
                    last = _receipt_step(arm_rows)
                if epoch == 1 and ordinal < 12:
                    smoke.append(_receipt_step(arm_rows))
                if epoch == 1 and ordinal == 11:
                    elapsed12 = time.monotonic() - started
                    peak_allocated = int(self.torch.cuda.max_memory_allocated(0))
                    peak_reserved = int(self.torch.cuda.max_memory_reserved(0))
                    projected12 = elapsed12 / 12.0 * plan.SOURCE_GROUPS_PER_EPOCH * plan.EPOCHS
                    _need(peak_reserved <= 24 * 1024**3 and projected12 <= 7200.0,
                          "AJPF 12-step smoke resource gate")
                    smoke_preflight = {**(smoke_preflight or {}), "elapsed_seconds": elapsed12, **_smoke_rates(elapsed12),
                                       "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
                                       "projected_12_epoch_seconds": projected12}
                    smoke_body = {"groups": 12, "counted_in_epoch1": True, **smoke_preflight, "rows": smoke}
                    smoke_sha = lifecycle._pair(root, "smoke.json", smoke_body); published.append("smoke.json")
            for arm in plan.ARM_ORDER:
                encoder, decoder = coverage[arm]
                _need(_tensor_bool(encoder["finite"]) and _tensor_bool(decoder["finite"]) and _tensor_bool(encoder["nonzero"])
                      and _tensor_bool(decoder["nonzero"]) and bool(encoder["updates"]) and bool(decoder["updates"])
                      and any(_tensor_bool(item) for item in encoder["updates"]) and any(_tensor_bool(item) for item in decoder["updates"]),
                      "AJPF epoch gradient/update coverage drift")
            _need(first is not None and last is not None, "AJPF epoch receipt sentinel drift")
            epoch_body = {"schema": "m2_anchored_joint_postfusion_v1_epoch", "epoch": epoch,
                          "shared_groups": plan.SOURCE_GROUPS_PER_EPOCH, "first": first, "last": last,
                          "source_task_loss": {arm: float((loss_sum[arm] / plan.SOURCE_GROUPS_PER_EPOCH).detach().cpu())
                                               for arm in plan.ARM_ORDER},
                          "update_sentinel_ordinals": list(joint_training.UPDATE_SENTINEL_ORDINALS)}
            epoch_sha.append(lifecycle._pair(root, f"epoch_{epoch:02d}.json", epoch_body)); published.append(f"epoch_{epoch:02d}.json")
        _need(len(smoke) == 12 and smoke_preflight is not None and "smoke.json" in published,
              "AJPF same-iterator 12-step smoke drift")
        self._complete = True
        return smoke_sha, epoch_sha

    def checkpoint_bytes(self, arm: str) -> bytes:
        import io
        _need(getattr(self, "_complete", False), "AJPF checkpoints require completed fixed epoch12 run")
        buffer = io.BytesIO(); self.torch.save(self.prepared.modules[arm].state_dict(), buffer)
        return buffer.getvalue()


def _rng_state(torch: Any) -> tuple[Any, Any, Any, Any]:
    import random
    import numpy as np
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(0)


def _set_rng(torch: Any, state: tuple[Any, Any, Any, Any]) -> None:
    import random
    import numpy as np
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2]); torch.cuda.set_rng_state(state[3], 0)


def _same_rng(left: tuple[Any, Any, Any, Any], right: tuple[Any, Any, Any, Any]) -> bool:
    import numpy as np
    return (left[0] == right[0] and left[1][0] == right[1][0] and np.array_equal(left[1][1], right[1][1])
            and left[1][2:] == right[1][2:] and bool((left[2] == right[2]).all()) and bool((left[3] == right[3]).all()))


def _arm_proof_clone(*, prepared: Any, arm: str) -> Any:
    """Clone the matching trained arm; PF schemas cannot start from native."""
    _need(arm in plan.ARM_ORDER and set(prepared.modules) == set(plan.ARM_ORDER),
          "AJPF checkpoint proof arm topology drift")
    clone = copy.deepcopy(prepared.modules[arm])
    _need(clone is not prepared.modules[arm], "AJPF checkpoint proof clone alias")
    return clone


def _checkpoint_proof(*, repo_root: Path, arm: str, body: bytes, runtime: _TrainRuntime) -> dict[str, Any]:
    """Fresh strict-load proof and state-preserving repeat-eval finite check."""
    import io
    import torch
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical
    module = _arm_proof_clone(prepared=runtime.prepared, arm=arm).to("cpu")
    module.load_state_dict(torch.load(io.BytesIO(body), map_location="cpu", weights_only=True), strict=True)
    module.eval(); [parameter.requires_grad_(False) for parameter in module.parameters()]
    before = physical._student_state_sha(module)
    name, coordinate_batch = runtime.batches[0]
    material = runtime.materials[name]
    dataset = runtime.prepared.datamodule.train_dataset
    coordinate = coordinate_batch[0]
    import numpy as np
    calibration = torch.from_numpy(np.ascontiguousarray(material.activities[list(material.support_indices)], dtype=np.float32)).unsqueeze(0)
    side = torch.from_numpy(np.ascontiguousarray(material.normalized_side, dtype=np.float32)).unsqueeze(0)
    windows = torch.from_numpy(np.ascontiguousarray(
        dataset.neural_data[coordinate.session][np.arange(coordinate.window_start, coordinate.window_start + 50)], dtype=np.float32)).unsqueeze(0)
    with torch.inference_mode():
        identity = module.student.id_encoder.forward_batch(calibration, side_features=side)
        first = module.student.decode_with_identity(windows, identity)
        second = module.student.decode_with_identity(windows, identity)
    after = physical._student_state_sha(module)
    _need(before == after and bool(torch.isfinite(first).all()) and torch.equal(first, second),
          "AJPF checkpoint strict/eval-forward proof drift")
    return {"arm": arm, "strict_load": True, "student_state_sha256": before,
            "eval_repeated_forward_bitwise": True, "forward_finite": True, "state_unchanged": True}


def _seal_epoch12_checkpoints(*, root: Path, published: list[str], checkpoint_arms: list[str],
                               runtime: Any, repo_root: Path) -> dict[str, Any]:
    """Proof-first checkpoint transaction; never leaves an empty directory."""
    checkpoints: dict[str, Any] = {}
    checkpoint_dir_created = False
    for arm in plan.ARM_ORDER:
        body = runtime.checkpoint_bytes(arm)
        # The proof is deliberately before ``mkdir(checkpoints)``.  A first
        # proof failure therefore has exactly the receipt prefix plus failure.
        proof = _checkpoint_proof(repo_root=repo_root, arm=arm, body=body, runtime=runtime)
        if not checkpoint_dir_created:
            (root / "checkpoints").mkdir(mode=0o755)
            checkpoint_dir_created = True
        descriptor = lifecycle.publish_binary(root, f"checkpoints/{arm}_epoch12.pt", body)
        checkpoint_arms.append(arm); published.append(f"checkpoints/{arm}_epoch12.pt")
        descriptor["proof"] = proof
        checkpoints[arm] = descriptor
    return checkpoints


def execute_training(*, cap: lifecycle.Capability, repo_root: Path) -> dict[str, Any]:
    """Publish exactly the immutable training root; no user-supplied work seam."""
    environment = validate_frozen_environment(repo_root=repo_root, score=False)
    lifecycle.consume(cap, repo_root); root = cap.root; root.mkdir(mode=0o755); root_identity = _created_root_identity(root); published: list[str] = []; checkpoint_arms: list[str] = []; launch_attestation: Mapping[str, Any] | None = None; predecessors: Mapping[str, Any] | None = None
    try:
        _revalidate_capability_boundary(cap=cap, repo_root=repo_root, root_identity=root_identity)
        attempt = lifecycle._pair(root, "attempt.json", {**_training_attempt(cap), "frozen_environment": environment}); published.append("attempt.json")
        # Torch/CUDA is deliberately imported only after the immutable attempt.
        import torch
        launch_attestation = _cuda_launch_attestation(torch)
        _revalidate_capability_boundary(cap=cap, repo_root=repo_root, root_identity=root_identity)
        _need(validate_frozen_environment(repo_root=repo_root, score=False) == environment, "AJPF training launch environment drift")
        launch = lifecycle._pair(root, "launch.json", {"schema": "m2_anchored_joint_postfusion_v1_launch", **launch_attestation, "frozen_environment": environment,
                                  "target_access": False}); published.append("launch.json")
        # Descriptor admission is deliberately after attempt/launch.  It
        # binds the one source materialization to held APFG/POOLED authorities.
        from . import binding
        predecessors = {"apfg": binding.validate_apfg_source(repo_root / plan.APFG_SOURCE_ROOT_RELATIVE),
                        "pooled": binding.validate_pooled_score(repo_root / plan.POOLED_ROOT_RELATIVE)}
        runtime = _TrainRuntime(repo_root=repo_root, launch=launch_attestation)
        source = lifecycle._pair(root, "source_authority.json", runtime.source_authority(predecessors)); published.append("source_authority.json")
        smoke_sha, epoch_sha = runtime.run_and_publish(root=root, published=published)
        checkpoints = _seal_epoch12_checkpoints(root=root, published=published,
            checkpoint_arms=checkpoint_arms, runtime=runtime, repo_root=repo_root)
        manifest = lifecycle._pair(root, "manifest.json", {"schema": "m2_anchored_joint_postfusion_v1_manifest",
            "checkpoint": checkpoints, "epoch_sha256": epoch_sha, "smoke_sha256": smoke_sha,
            "selected_t4_strict_load": runtime.prepared.strict_load_evidence, "fixed_epoch": 12,
            "checkpoint_selection": "epoch12_only_no_metric_selection"}); published.append("manifest.json")
        _revalidate_capability_boundary(cap=cap, repo_root=repo_root, root_identity=root_identity)
        _need(validate_frozen_environment(repo_root=repo_root, score=False) == environment, "AJPF training terminal environment drift")
        final_gpu = _cuda_launch_attestation(torch)
        _need(final_gpu["logical_device"] == launch_attestation["logical_device"]
              and final_gpu["uuid"] == launch_attestation["uuid"], "AJPF GPU0 terminal attestation drift")
        _need(binding.validate_apfg_source(repo_root / plan.APFG_SOURCE_ROOT_RELATIVE) == predecessors["apfg"]
              and binding.validate_pooled_score(repo_root / plan.POOLED_ROOT_RELATIVE) == predecessors["pooled"],
              "AJPF predecessor authority changed before terminal")
        terminal = lifecycle._pair(root, "terminal.json", {"schema": "m2_anchored_joint_postfusion_v1_terminal",
            "attempt_sha256": attempt, "launch_sha256": launch, "source_authority_sha256": source,
            "smoke_sha256": smoke_sha, "manifest_sha256": manifest, "terminal_xor_failure": True}); published.append("terminal.json")
        lifecycle.validate_training_topology(root, terminal=True)
        return {"root": str(root), "terminal_sha256": terminal, "manifest_sha256": manifest, "checkpoint": checkpoints}
    except Exception as error:
        if root.exists() and "terminal.json" not in published:
            _revalidate_capability_boundary(cap=cap, repo_root=repo_root, root_identity=root_identity)
            _need(validate_frozen_environment(repo_root=repo_root, score=False) == environment, "AJPF training failure environment drift")
            if launch_attestation is not None:
                # A failure after CUDA admission still proves the same sole
                # visible GPU immediately before the immutable failure leaf.
                import torch
                final_gpu = _cuda_launch_attestation(torch)
                _need(final_gpu["uuid"] == launch_attestation["uuid"], "AJPF GPU0 failure attestation drift")
            if predecessors is not None:
                from . import binding
                _need(binding.validate_apfg_source(repo_root / plan.APFG_SOURCE_ROOT_RELATIVE) == predecessors["apfg"]
                      and binding.validate_pooled_score(repo_root / plan.POOLED_ROOT_RELATIVE) == predecessors["pooled"],
                      "AJPF predecessor authority changed before failure")
            lifecycle._pair(root, "failure.json", {"schema": "m2_anchored_joint_postfusion_v1_failure",
                "published_prefix": published, "exception_class": type(error).__name__,
                "exception_message": str(error)[:400], "target_access": False})
            lifecycle.validate_training_failure_topology(root, published=tuple(published), checkpoint_arms=tuple(checkpoint_arms))
        raise


def _strict_cpu_modules(*, repo_root: Path, checkpoint_bytes: Mapping[str, bytes]) -> tuple[dict[str, Any], Any]:
    """Rebuild one CPU PIT stack then strict-load each sealed epoch-12 arm."""
    import io
    import torch
    from . import adapter
    from tfpd_exploration.src.pit_m2_v1 import trainer as pit_trainer
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "AJPF score CPU environment drift")
    runner = pit_trainer.PitM2ArmedRunner(repo_root, "t0m", device="cpu"); runner.prepare(attach_operator=False)
    _need(runner._litmodule is not None and runner._datamodule is not None, "AJPF CPU PIT preparation drift")
    strict, evidence = physical._strict_sealed_pooled_clone(repo_root=repo_root, base_module=runner._litmodule)
    modules = {}
    for arm in plan.ARM_ORDER:
        module = copy.deepcopy(strict)
        if arm != "J-NATIVE": adapter.install_after_strict_load(module.student, arm)
        body = checkpoint_bytes.get(arm)
        _need(isinstance(body, bytes) and body, "AJPF held checkpoint bytes absent")
        module.load_state_dict(torch.load(io.BytesIO(body), map_location="cpu", weights_only=True), strict=True)
        module.eval(); [parameter.requires_grad_(False) for parameter in module.parameters()]
        modules[arm] = module
    return modules, {"datamodule": runner._datamodule, "base_module": runner._litmodule, "evidence": evidence}


def _revalidate_score_admissions(*, repo_root: Path, training_root: Path, training_witness: Mapping[str, Any],
                                 pooled_witness: Mapping[str, Any], historical_witness: Mapping[str, Any],
                                 binding_module: Any, historical_binding_module: Any) -> None:
    """Final/failure TOCTOU closure for all three score producers."""
    _need(lifecycle.validate_held_training_success_graph(root=training_root, repo_root=repo_root)["descriptor_witness"]
          == training_witness["descriptor_witness"], "AJPF held training producer changed before score publication")
    _need(binding_module.validate_pooled_score(repo_root / plan.POOLED_ROOT_RELATIVE) == pooled_witness,
          "AJPF POOLED descriptor changed before score publication")
    _need(historical_binding_module.validate_pooled_comparator_score(repo_root / plan.POOLED_ROOT_RELATIVE) == historical_witness,
          "AJPF historical POOLED descriptor changed before score publication")


def execute_cpu_score(*, cap: lifecycle.Capability, repo_root: Path, training_root: Path) -> dict[str, Any]:
    """Fresh-process CPU scorer.  It is internal and has no callback inputs."""
    environment = validate_frozen_environment(repo_root=repo_root, score=True)
    lifecycle.consume(cap, repo_root); root = cap.root; root.mkdir(mode=0o755); root_identity = _created_root_identity(root); published: list[str] = []
    training_witness: Mapping[str, Any] | None = None; predecessor: Mapping[str, Any] | None = None; held_pooled: Mapping[str, Any] | None = None
    try:
        _revalidate_capability_boundary(cap=cap, repo_root=repo_root, root_identity=root_identity)
        # Keep score admission stdlib-only.  The attempt is immutable before
        # importing Torch, reading the training root, or touching a target.
        attempt = lifecycle._pair(root, "attempt.json", {"schema": "m2_anchored_joint_postfusion_v1_score_attempt",
            "closure": cap.closure, "training_root": str(training_root), "target_access": False,
            "torch_imported": False, "cuda_initialized": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "frozen_environment": environment}); published.append("attempt.json")
        # The immutable score attempt is already published before any runtime
        # import or held producer access below.
        import torch
        _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "AJPF score requires fresh CPU process")
        _revalidate_capability_boundary(cap=cap, repo_root=repo_root, root_identity=root_identity)
        _need(validate_frozen_environment(repo_root=repo_root, score=True) == environment, "AJPF score launch environment drift")
        launch = lifecycle._pair(root, "launch.json", {"schema": "m2_anchored_joint_postfusion_v1_score_launch",
            "cuda_visible_devices": "", "cuda_initialized": False, "frozen_environment": environment}); published.append("launch.json")
        # Held descriptors are read only after score attempt publication.
        training_witness = lifecycle.validate_held_training_success_graph(root=training_root, repo_root=repo_root)
        from . import binding
        predecessor = binding.validate_pooled_score(repo_root / plan.POOLED_ROOT_RELATIVE)
        from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import binding as historical_binding
        held_pooled = historical_binding.validate_pooled_comparator_score(repo_root / plan.POOLED_ROOT_RELATIVE)
        modules, prepared = _strict_cpu_modules(repo_root=repo_root, checkpoint_bytes=training_witness["checkpoint_bytes"])
        producer = lifecycle._pair(root, "producer_authority.json", {"training_root_identity": training_witness["root_identity"],
            "training_terminal_sha256": training_witness["terminal_sha256"], "training_manifest_sha256": training_witness["manifest_sha256"],
            "training_checkpoint_sha256": {arm: training_witness["checkpoint"][arm]["sha256"] for arm in plan.ARM_ORDER},
            "pooled_score_sha256": predecessor["score_sha256"], "checkpoint_names": list(plan.ARMS)}); published.append("producer_authority.json")
        # Reuse the reviewed narrow one-time 13-session materializer.  It
        # creates a strict native 25d replay before any AJPF R2 is inspected.
        from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical as old_physical
        materialized = old_physical.materialize_13_inputs_and_pooled_comparators(
            prepared=prepared, pooled_comparator=None, repo_root=repo_root)
        records = materialized["records"]
        pooled = {str(record["session"]): held_pooled["comparators"][str(record["key"])] for record in records.values()}
        # Exact native sentinel against the sealed m4_activity_only rows is
        # asserted before new-arm rows are evaluated.
        for record in records.values():
            reference = record["pooled_comparator"]
            sealed = held_pooled["comparators"].get(str(record["key"]))
            _need(isinstance(sealed, Mapping) and reference["prediction_sha256"] == sealed["prediction_sha256"]
                  and reference["target_sha256"] == sealed["target_sha256"]
                  and reference["query_starts_sha256"] == sealed["query_starts_sha256"]
                  and reference["window_count"] == sealed["window_count"] and reference["r2"] == sealed["r2"],
                  "AJPF exact 25d J-NATIVE FIXED30 historical sentinel drift")
        carrier_lineage = {key: scorer.selected_support4_carrier_lineage(record=record) for key, record in records.items()}
        input_body = {"schema": "m2_anchored_joint_postfusion_v1_input_authority", "record_count": 13,
                      "records": {key: {field: value for field, value in record.items() if field != "_runtime"}
                                  for key, record in records.items()},
                      "selected_support4_carrier_lineage": carrier_lineage, "cuda_initialized": False}
        inputs = lifecycle._pair(root, "input_authority.json", input_body); published.append("input_authority.json")
        rows = scorer.score_whole_stack_78_cpu(torch=torch, modules=modules, materialized=materialized)
        _need(all(row["selected_support4_carrier_lineage"] == carrier_lineage[
                  next(key for key, record in records.items() if str(record["session"]) == row["session"])] for row in rows),
              "AJPF score/input selected-support4 lineage cross-bind drift")
        external = tuple(sorted(record["session"] for record in records.values() if record["surface"] == "external_post30_local"))
        within = tuple(sorted(record["session"] for record in records.values() if record["surface"] == "within_post30"))
        gates = scorer.build_preregistered_gates(rows=rows, historical_pooled=pooled, external_sessions=external, within_sessions=within)
        score = lifecycle._pair(root, "score.json", {"schema": "m2_anchored_joint_postfusion_v1_score", "row_count": 78,
            "rows": rows, "gates": gates, "native_fixed30_sentinel": {"passed": True, "session_count": 13},
            "input_authority_sha256": inputs}); published.append("score.json")
        _revalidate_capability_boundary(cap=cap, repo_root=repo_root, root_identity=root_identity)
        _need(validate_frozen_environment(repo_root=repo_root, score=True) == environment, "AJPF score terminal environment drift")
        _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
              "AJPF CPU scorer terminal CUDA attestation drift")
        _revalidate_score_admissions(repo_root=repo_root, training_root=training_root, training_witness=training_witness,
            pooled_witness=predecessor, historical_witness=held_pooled, binding_module=binding,
            historical_binding_module=historical_binding)
        terminal = lifecycle._pair(root, "terminal.json", {"schema": "m2_anchored_joint_postfusion_v1_score_terminal",
            "attempt_sha256": attempt, "launch_sha256": launch, "producer_authority_sha256": producer,
            "input_authority_sha256": inputs, "score_sha256": score, "terminal_xor_failure": True}); published.append("terminal.json")
        lifecycle.validate_score_topology(root, terminal=True)
        return {"root": str(root), "terminal_sha256": terminal}
    except Exception as error:
        if root.exists() and "terminal.json" not in published:
            _revalidate_capability_boundary(cap=cap, repo_root=repo_root, root_identity=root_identity)
            _need(validate_frozen_environment(repo_root=repo_root, score=True) == environment, "AJPF score failure environment drift")
            if training_witness is not None:
                from . import binding
                from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import binding as historical_binding
                _need(predecessor is not None and held_pooled is not None, "AJPF partial score predecessor admission")
                _revalidate_score_admissions(repo_root=repo_root, training_root=training_root, training_witness=training_witness,
                    pooled_witness=predecessor, historical_witness=held_pooled, binding_module=binding,
                    historical_binding_module=historical_binding)
            lifecycle._pair(root, "failure.json", {"schema": "m2_anchored_joint_postfusion_v1_score_failure",
                "published_prefix": published, "exception_class": type(error).__name__, "exception_message": str(error)[:400],
                "cuda_initialized": False})
            lifecycle.validate_score_failure_topology(root, published=tuple(published))
        raise


def _execute_synthetic_for_test(*, cap: lifecycle.Capability, repo_root: Path, runtime: Any) -> Mapping[str, Any]:
    """Private no-CUDA harness exercising the exact *training* publication order."""
    lifecycle.consume(cap, repo_root); root = cap.root; root.mkdir(mode=0o755)
    attempt=lifecycle._pair(root, "attempt.json", _training_attempt(cap)); launch=lifecycle._pair(root, "launch.json", {"gpu_profile": "synthetic"})
    source=lifecycle._pair(root, "source_authority.json", runtime.source_authority()); smoke=lifecycle._pair(root, "smoke.json", runtime.smoke())
    epochs = tuple(runtime.epochs()); _need(len(epochs) == plan.EPOCHS, "AJPF synthetic epoch law")
    epoch_sha = [lifecycle._pair(root, f"epoch_{index:02d}.json", body) for index, body in enumerate(epochs, 1)]
    (root / "checkpoints").mkdir(mode=0o755)
    checkpoints = {arm: {**lifecycle.publish_binary(root, f"checkpoints/{arm}_epoch12.pt", runtime.checkpoint_bytes(arm)),
                         "proof": {"arm": arm, "strict_load": True, "eval_repeated_forward_bitwise": True,
                                   "state_unchanged": True, "forward_finite": True}}
                   for arm in plan.ARM_ORDER}
    manifest = lifecycle._pair(root, "manifest.json", {"checkpoint": checkpoints, "epoch_sha256": epoch_sha,
        "smoke_sha256": smoke, "fixed_epoch": 12, "checkpoint_selection": "epoch12_only_no_metric_selection"})
    terminal = lifecycle._pair(root, "terminal.json", {"attempt_sha256":attempt,"launch_sha256":launch,"source_authority_sha256":source,
        "smoke_sha256":smoke,"manifest_sha256": manifest, "terminal_xor_failure": True})
    lifecycle.validate_training_topology(root, terminal=True)
    return {"root": str(root), "terminal_sha256": terminal}


def _run_cpu_score_child(repo_root_text: str) -> None:
    """Private subprocess entry: the only transition from GPU training to CPU score."""
    repo_root = Path(repo_root_text).absolute()
    training_root = repo_root / plan.TRAINING_ROOT_RELATIVE
    # Do not open/validate the held training root here: score attempt-first
    # requires its immutable attempt before any producer descriptor access.
    validate_frozen_environment(repo_root=repo_root, score=True)
    cap = lifecycle.issue_production_capability(repo_root=repo_root, root=repo_root / plan.SCORE_ROOT_RELATIVE)
    execute_cpu_score(cap=cap, repo_root=repo_root, training_root=training_root)
