"""Performance-first calibration-budget marginalization for sealed Cell D."""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src import cell_d_equal_session_v1 as equal_session


CELL = "CALIBRATION_BUDGET_MARGINALIZED_CELL_D_SEED42"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CALIBRATION_BUDGET_MARGINALIZED_CELL_D_20260823.md"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_marginalized_cell_d_seed42_v3"
SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_marginalized_cell_d_seed42_smoke_v3"
FAILED_V1_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_marginalized_cell_d_seed42_smoke_v1"
FAILED_V1_ATTEMPT_SHA256 = "10c984277310fff017ca1f642044249f5569923be9098aa1bcc46d032a44f128"
FAILED_V1_FAILURE_SHA256 = "e7410c025f2b9d95db1292472c98619117997bd5a2e7530ad5667933de4eaa7e"
FAILED_V2_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_marginalized_cell_d_seed42_smoke_v2"
FAILED_V2_ATTEMPT_SHA256 = "50d84170f5e8e1d95ecce61419f407c326b9d8f306ccecc5ef3b9467b9f17e04"
FAILED_V2_FAILURE_SHA256 = "cf9eb94acadd1fee0fe59217884d63efe86f4ba2452b47063df62ebdbc4ee653"
BUDGETS = tuple(range(4, 31))
BUDGET_COUNT = len(BUDGETS)
EPOCHS = 48
STEPS_PER_EPOCH = 33_925
CHECKPOINT_EPOCHS = (44, 45, 46, 47)
SEED = 42


class CBMError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CBMError(message)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return sha256_bytes(canonical_json_bytes({
        "dtype": str(tensor.dtype), "shape": list(tensor.shape),
        "bytes_sha256": sha256_bytes(tensor.numpy().tobytes()),
    }))


def budget_for_batch(*, epoch: int, session_index: int, cycle: int) -> int:
    """A per-session 27-cycle permutation, independent of host RNG state."""
    require(type(epoch) is int and 0 <= epoch < EPOCHS, "CBM epoch drift")
    require(type(session_index) is int and 0 <= session_index < 27, "CBM session index drift")
    require(type(cycle) is int and cycle >= 0, "CBM cycle drift")
    # Seven is coprime to 27.  Therefore each session sees all integer budgets
    # exactly once over every 27 successive exposure cycles.  The other terms
    # rotate the phase across sessions and logical epochs.
    return 4 + ((7 * cycle + 11 * session_index + 13 * epoch) % BUDGET_COUNT)


def budget_schedule_digest(*, epochs: int = EPOCHS, cycles: int = BUDGET_COUNT) -> str:
    rows = [
        [budget_for_batch(epoch=epoch, session_index=session, cycle=cycle)
         for cycle in range(cycles)]
        for epoch in range(epochs) for session in range(27)
    ]
    return sha256_bytes(canonical_json_bytes(rows))


@dataclass(frozen=True)
class SourceBudgetSideCache:
    roster: tuple[str, ...]
    side_by_session_budget: Mapping[str, Mapping[int, torch.Tensor]]
    evidence_by_session: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        require(len(self.roster) == 27 and len(set(self.roster)) == 27, "CBM strict roster drift")
        require(tuple(self.side_by_session_budget) == self.roster, "CBM side-cache roster/order drift")
        require(tuple(self.evidence_by_session) == self.roster, "CBM evidence roster/order drift")
        for session in self.roster:
            rows = self.side_by_session_budget[session]
            require(tuple(rows) == BUDGETS, f"{session}: CBM budget topology drift")
            unit_count: int | None = None
            for budget, side in rows.items():
                require(torch.is_tensor(side) and side.dtype == torch.float32 and side.device.type == "cpu",
                        f"{session}/M{budget}: CBM side tensor boundary drift")
                require(side.ndim == 2 and side.shape[1] == 4 and torch.isfinite(side).all().item(),
                        f"{session}/M{budget}: CBM side shape/nonfinite drift")
                unit_count = int(side.shape[0]) if unit_count is None else unit_count
                require(int(side.shape[0]) == unit_count, f"{session}: CBM unit axis drift")

    def side(self, *, session: str, budget: int, device: torch.device | str) -> torch.Tensor:
        require(session in self.side_by_session_budget and budget in BUDGETS, "CBM side lookup drift")
        return self.side_by_session_budget[session][budget].to(device=device, non_blocking=True)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "calibration_budget_marginalized_side_cache_v1",
            "roster": list(self.roster), "budgets": list(BUDGETS),
            "schedule_sha256": budget_schedule_digest(),
            "sessions": {session: dict(self.evidence_by_session[session]) for session in self.roster},
            "posterior_used": False, "ridge_used": False, "target_labels_used": False,
            "sealed_ordinary_ols_normalizer": True,
        }


def build_source_budget_side_cache(root: Path, runtime: Mapping[str, Any]) -> SourceBudgetSideCache:
    """Build all M4..M30 ordinary-OLS source carriers before any iterator."""
    import numpy as np
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import (
        _nearest_canonical_direction_index,
        _pool_trial_rate_matrix,
        _unit_tuning_features,
    )
    from src.posterior_marginalized_cell_d_v1 import plan

    roster = tuple(runtime["roster"])
    train_files = tuple(runtime["train_files"])
    dataset = runtime["dataset"]
    require(len(roster) == len(train_files) == 27, "CBM source topology drift")
    mean = torch.tensor(plan.SEALED_OLS_T4_MEAN_FLOAT32, dtype=torch.float32)
    std = torch.tensor(plan.SEALED_OLS_T4_STD_FLOAT32, dtype=torch.float32)
    cache: dict[str, dict[int, torch.Tensor]] = {}
    evidence: dict[str, dict[str, object]] = {}
    for session, path in zip(roster, train_files, strict=True):
        trials = list_datamodule_rewarded_trials(path, bin_size_ms=20, window_size=50, trial_result_filter="R")
        rates_units_trials, unit_count = _pool_trial_rate_matrix(path, trials)
        rates = np.ascontiguousarray(rates_units_trials.T, dtype=np.float64)
        # Reproduce the sealed datamodule's ordinary-T4 producer literally.
        # Do not import the posterior route's same-row cue recovery: that is a
        # second estimator treatment and makes the M30 control differ from the
        # Cell-D input.  The legacy producer passes the one audited non-finite,
        # non-None target_dir to this helper; numpy argmin maps it to index 0.
        prefix_trials = trials[:30]
        directions = np.array(
            [
                _nearest_canonical_direction_index(trial["target_dir"])
                if trial.get("target_dir") is not None
                else -1
                for trial in prefix_trials
            ],
            dtype=np.int64,
        )
        nonfinite_rows = [
            index for index, trial in enumerate(prefix_trials)
            if trial.get("target_dir") is not None and not np.isfinite(float(trial["target_dir"]))
        ]
        require(rates.shape[0] >= 30 and rates.shape[1] == unit_count, f"{session}: CBM source matrix drift")
        rows: dict[int, torch.Tensor] = {}
        raw_sha: dict[str, str] = {}
        for budget in BUDGETS:
            budget_directions = directions[:budget]
            present = sorted({int(index) for index in budget_directions if int(index) >= 0})
            require(len(present) >= 2, f"{session}/M{budget}: ordinary-T4 direction degeneracy")
            # Call the exact sealed ordinary-T4 per-unit producer.  The generic
            # D-optimal helper treats -1 as Python's last array index; the
            # datamodule correctly excludes missing directions from the fit.
            raw = np.stack([
                _unit_tuning_features(
                    rates[:budget, unit], budget_directions, present,
                )[0]
                for unit in range(unit_count)
            ]).astype(np.float32, copy=False)
            raw_tensor = torch.as_tensor(np.ascontiguousarray(raw, dtype=np.float32))
            normalized = ((raw_tensor - mean) / std).contiguous()
            require(tuple(normalized.shape) == (unit_count, 4), f"{session}/M{budget}: CBM carrier shape drift")
            rows[budget] = normalized
            raw_sha[str(budget)] = tensor_sha256(raw_tensor)
        ordinary = torch.as_tensor(dataset.sessions[session].side_features, dtype=torch.float32)
        require(torch.equal(rows[30], ordinary), f"{session}: CBM M30/datamodule T4 parity drift")
        cache[session] = rows
        evidence[session] = {
            "unit_count": unit_count, "raw_t4_sha256_by_budget": raw_sha,
            "normalized_t4_sha256_by_budget": {str(m): tensor_sha256(rows[m]) for m in BUDGETS},
            "m30_datamodule_bitwise_equal": True,
            "direction_semantics": "sealed_ordinary_t4_legacy_nearest_direction_no_recovery",
            "nonfinite_target_dir_prefix_positions": nonfinite_rows,
            "missing_target_dir_prefix_positions": [
                index for index, trial in enumerate(prefix_trials)
                if trial.get("target_dir") is None
            ],
            "shared_unit_tuning_producer": "mc_maze.unit_side_features._unit_tuning_features",
        }
    return SourceBudgetSideCache(roster=roster, side_by_session_budget=cache, evidence_by_session=evidence)


def dry_plan() -> dict[str, object]:
    return {
        "schema": "calibration_budget_marginalized_cell_d_plan_v2",
        "status": "DRY_NO_DATA_NO_CUDA_NO_GPU_NO_WRITE_NO_LAUNCH",
        "cell": CELL, "budgets": list(BUDGETS), "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH, "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
        "schedule_sha256": budget_schedule_digest(),
        "sole_treatment": "joint_source_training_prefix_budget_marginalization_for_B3S_and_ordinary_OLS_T4",
        "target_optimizer_steps": 0,
        "failed_v1_predecessor": {
            "attempt_sha256": FAILED_V1_ATTEMPT_SHA256,
            "failure_sha256": FAILED_V1_FAILURE_SHA256,
            "reason": "posterior_same_row_direction_recovery_broke_sealed_M30_parity",
        },
        "failed_v2_predecessor": {
            "attempt_sha256": FAILED_V2_ATTEMPT_SHA256,
            "failure_sha256": FAILED_V2_FAILURE_SHA256,
            "reason": "generic_doptimal_fit_included_missing_direction_index_minus_one",
        },
    }


def _publish_pair(directory: Path, name: str, body: bytes) -> str:
    digest = sha256_bytes(body)
    for leaf, content in ((name, body), (name + ".sha256", f"{digest}  {name}\n".encode("ascii"))):
        descriptor = os.open(directory / leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
        try:
            offset = 0
            while offset < len(content):
                written = os.write(descriptor, content[offset:])
                require(written > 0, f"short immutable CBM write: {leaf}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(directory / leaf, 0o444)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def _publish_json(directory: Path, name: str, value: Mapping[str, object]) -> str:
    return _publish_pair(directory, name, canonical_json_bytes(value))


def implementation_closure(root: Path) -> dict[str, object]:
    paths = (
        WORKORDER_RELATIVE,
        "tfpd_exploration/src/calibration_budget_marginalized_cell_d_v1.py",
        "tfpd_exploration/scripts/run_calibration_budget_marginalized_cell_d_seed42.py",
        "tfpd_exploration/tests/test_calibration_budget_marginalized_cell_d_v1.py",
        "tfpd_exploration/src/cell_d_equal_session_v1.py",
        "tfpd_exploration/src/tfpd_lane/pop_robust.py",
        "tfpd_exploration/src/tfpd_lane/arm_common.py",
        "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
        "sua_exploration/mc_maze/multisession_datamodule.py",
        "sua_exploration/mc_maze/unit_side_features.py",
    )
    hashes = {relative: sha256_bytes((Path(root) / relative).read_bytes()) for relative in paths}
    payload: dict[str, object] = {"schema": "calibration_budget_marginalized_closure_v2", "sha256_by_path": hashes}
    payload["closure_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def _validate_failed_v1_predecessor(root: Path) -> dict[str, object]:
    """Bind the immutable, pre-update v1 failure before reserving v2."""
    directory = root / FAILED_V1_ROOT_RELATIVE
    require(directory.is_dir() and not directory.is_symlink(), "CBM failed-v1 predecessor root drift")
    expected = {
        "attempt.json": FAILED_V1_ATTEMPT_SHA256,
        "failure.json": FAILED_V1_FAILURE_SHA256,
    }
    require({leaf.name for leaf in directory.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
    }, "CBM failed-v1 predecessor topology drift")
    payloads: dict[str, object] = {}
    for name, expected_sha in expected.items():
        body_path = directory / name
        sidecar_path = directory / f"{name}.sha256"
        require(body_path.is_file() and not body_path.is_symlink(), f"CBM failed-v1 {name} drift")
        require(sidecar_path.is_file() and not sidecar_path.is_symlink(),
                f"CBM failed-v1 {name} sidecar drift")
        require((body_path.stat().st_mode & 0o777) == 0o444, f"CBM failed-v1 {name} mode drift")
        require((sidecar_path.stat().st_mode & 0o777) == 0o444,
                f"CBM failed-v1 {name} sidecar mode drift")
        body = body_path.read_bytes()
        require(sha256_bytes(body) == expected_sha, f"CBM failed-v1 {name} body drift")
        require(sidecar_path.read_bytes() == f"{expected_sha}  {name}\n".encode("ascii"),
                f"CBM failed-v1 {name} sidecar content drift")
        payloads[name] = json.loads(body)
    attempt = payloads["attempt.json"]
    failure = payloads["failure.json"]
    require(isinstance(attempt, dict) and attempt.get("status") == "ATTEMPT_RESERVED",
            "CBM failed-v1 attempt semantics drift")
    require(isinstance(failure, dict) and failure.get("status") == "FAILED",
            "CBM failed-v1 failure semantics drift")
    require(failure.get("attempt_sha256") == FAILED_V1_ATTEMPT_SHA256,
            "CBM failed-v1 lineage drift")
    progress = failure.get("progress")
    require(isinstance(progress, dict) and progress.get("optimizer_steps_completed") == 0
            and progress.get("backward_calls") == 0 and progress.get("update_calls") == 0,
            "CBM failed-v1 was not a pre-update failure")
    return {
        "root": FAILED_V1_ROOT_RELATIVE,
        "attempt_sha256": FAILED_V1_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V1_FAILURE_SHA256,
        "status": "VALIDATED_PRE_UPDATE_FAILURE",
    }


def _validate_failed_v2_predecessor(root: Path) -> dict[str, object]:
    """Bind the second immutable, pre-update M30-parity failure."""
    directory = root / FAILED_V2_ROOT_RELATIVE
    require(directory.is_dir() and not directory.is_symlink(), "CBM failed-v2 predecessor root drift")
    expected = {"attempt.json": FAILED_V2_ATTEMPT_SHA256, "failure.json": FAILED_V2_FAILURE_SHA256}
    require({leaf.name for leaf in directory.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
    }, "CBM failed-v2 predecessor topology drift")
    payloads: dict[str, object] = {}
    for name, digest in expected.items():
        body_path = directory / name
        sidecar_path = directory / f"{name}.sha256"
        require(body_path.is_file() and not body_path.is_symlink(), f"CBM failed-v2 {name} drift")
        require(sidecar_path.is_file() and not sidecar_path.is_symlink(),
                f"CBM failed-v2 {name} sidecar drift")
        body = body_path.read_bytes()
        require(sha256_bytes(body) == digest, f"CBM failed-v2 {name} body drift")
        require(sidecar_path.read_bytes() == f"{digest}  {name}\n".encode("ascii"),
                f"CBM failed-v2 {name} sidecar content drift")
        payloads[name] = json.loads(body)
    attempt = payloads["attempt.json"]
    failure = payloads["failure.json"]
    require(isinstance(attempt, dict) and attempt.get("status") == "ATTEMPT_RESERVED",
            "CBM failed-v2 attempt semantics drift")
    require(isinstance(failure, dict) and failure.get("status") == "FAILED"
            and failure.get("attempt_sha256") == FAILED_V2_ATTEMPT_SHA256,
            "CBM failed-v2 failure lineage drift")
    progress = failure.get("progress")
    require(isinstance(progress, dict) and progress.get("optimizer_steps_completed") == 0
            and progress.get("backward_calls") == 0 and progress.get("update_calls") == 0,
            "CBM failed-v2 was not a pre-update failure")
    return {
        "root": FAILED_V2_ROOT_RELATIVE,
        "attempt_sha256": FAILED_V2_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V2_FAILURE_SHA256,
        "status": "VALIDATED_PRE_UPDATE_FAILURE",
    }


class CBMPhysicalBackend(equal_session.PhysicalEqualSessionBackend):
    """Equal-session backend with one changed batch treatment: joint prefix M."""

    def prepare(self, spec: Any, identity: Any, flags: Any) -> dict[str, Any]:
        runtime = super().prepare(spec, identity, flags)
        cache = build_source_budget_side_cache(self.root, runtime)
        runtime["cbm_cache"] = cache
        runtime["cbm_side_device"] = {
            session: {budget: side.to(runtime["device"]) for budget, side in cache.side_by_session_budget[session].items()}
            for session in cache.roster
        }
        runtime["cbm_session_index"] = {session: index for index, session in enumerate(cache.roster)}
        runtime["cbm_closure"] = implementation_closure(self.root)
        return runtime

    def source_authority(self, runtime: Mapping[str, Any], spec: Any, identity: Any, flags: Any) -> Mapping[str, object]:
        inherited = super().source_authority(runtime, spec, identity, flags)
        cache = runtime["cbm_cache"]
        require(isinstance(cache, SourceBudgetSideCache), "CBM physical cache type drift")
        return {
            "schema": "calibration_budget_marginalized_source_authority_v1",
            "cell": CELL, "source_only": True,
            "inherited_equal_session_source_authority": dict(inherited),
            "budget_side_cache": cache.payload(),
            "changed_training_field": "joint_calibration_prefix_and_ordinary_ols_t4_prefix",
            "target_opened": False, "within_opened": False, "external_opened": False,
            "formal_opened": False, "target_optimizer_steps": 0,
        }

    def train_step(self, runtime: Mapping[str, Any], *, global_step: int, expected_lr: float,
                   require_full_proof: bool, flags: Any) -> Any:
        torch_module = runtime["torch"]
        iterator = runtime.get("epoch_iterator")
        sampler = runtime.get("epoch_sampler")
        require(iterator is not None and isinstance(sampler, equal_session._OneIteratorEpochBatchSampler),
                "CBM train step lacks inherited one-iterator loader")
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise CBMError("CBM DataLoader exhausted before schedule") from error
        scheduled = sampler.consume_observed_batch()
        neural, behavior, calibration, sessions, ordinary_side = batch[:5]
        observed = tuple(sessions)
        require(len(observed) == equal_session.PUBLIC_SPEC.batch_size and set(observed) == {scheduled.session},
                "CBM loaded session/schedule mismatch")
        session_index = runtime["cbm_session_index"][scheduled.session]
        budget = budget_for_batch(epoch=scheduled.epoch, session_index=session_index, cycle=scheduled.cycle)
        selected_side = runtime["cbm_side_device"][scheduled.session][budget]
        neural = neural.to(runtime["device"])
        behavior = behavior.to(runtime["device"])
        calibration = calibration.to(runtime["device"])
        ordinary_side = ordinary_side.to(runtime["device"])
        require(calibration.ndim == 4 and calibration.shape[1] == 30,
                "CBM inherited calibration tensor is not M30")
        require(tuple(selected_side.shape) == tuple(ordinary_side.shape[1:]), "CBM selected side shape drift")
        if budget == 30:
            expected = selected_side.unsqueeze(0).expand_as(ordinary_side)
            require(torch_module.equal(expected, ordinary_side), "CBM M30 side differs from inherited batch")
        side = selected_side.unsqueeze(0).expand(ordinary_side.shape[0], -1, -1)
        calibration = calibration[:, :budget]
        optimizer = runtime["optimizer"]
        optimizer.param_groups[0]["lr"] = expected_lr
        optimizer.zero_grad(set_to_none=True)
        with runtime["pop_robust"].dynamic_dropout_recorder() as dropout_record:
            prediction, _identity = runtime["model"](neural, calib_trials=calibration, side_features=side)
        valid = (behavior != -1.0).all(dim=-1)
        diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
        loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
        require(bool(torch_module.isfinite(loss).item()), "CBM dense source loss nonfinite")
        loss.backward()
        flags.backward_calls += 1
        critical = equal_session._critical_gradient_proof(runtime["model"], torch_module) if require_full_proof else None
        optimizer.step()
        flags.update_calls += 1
        calls, sampled = dropout_record["dropout_calls"], dropout_record["sampled_p"]
        require(len(sampled) == 1 and len(calls) == 1, "CBM dynamic dropout call topology drift")
        call = calls[0]
        shape = call.get("shape")
        require(isinstance(shape, list) and len(shape) == 2 and shape[0] == equal_session.PUBLIC_SPEC.batch_size,
                "CBM dropout mask shape drift")
        total_mask = int(shape[0]) * int(shape[1])
        kept = int(round(float(call["retained_unit_fraction"]) * total_mask))
        p = float(sampled[0])
        if require_full_proof:
            finite_model, finite_optimizer = equal_session._finite_model_and_optimizer(
                runtime["model"], optimizer, torch_module,
            )
            require(finite_model and finite_optimizer and critical is not None, "CBM full proof failed")
            model_sha = runtime["arm_common"].state_sha256(runtime["model"])
            optimizer_sha = runtime["arm_common"].optimizer_sha256(optimizer)
        else:
            finite_model = finite_optimizer = model_sha = optimizer_sha = None
        return equal_session.StepOutcome(
            loss=float(loss.detach().item()), lr=expected_lr, dropout_p=p,
            kept=kept, dropped=total_mask - kept,
            all_zero_examples=int(call["all_zero_population_samples"]),
            population_examples=equal_session.PUBLIC_SPEC.batch_size,
            max_gain=0.0 if p == 1.0 else 1.0 / (1.0 - p),
            critical_gradients=critical, finite_model=finite_model, finite_optimizer=finite_optimizer,
            model_state_sha256=model_sha, optimizer_state_sha256=optimizer_sha,
            batch_evidence={
                "session": scheduled.session, "session_index": session_index,
                "dataset_indices": list(scheduled.dataset_indices), "cycle": scheduled.cycle,
                "epoch": scheduled.epoch, "global_step": global_step,
                "calibration_budget": budget,
                "b3s_trials": budget, "ordinary_ols_t4_trials": budget,
            },
        )


def _checkpoint_body(runtime: Mapping[str, Any], *, epoch: int, global_step: int,
                     closure: Mapping[str, object], source_authority_sha256: str) -> bytes:
    import io

    state_sha = runtime["arm_common"].state_sha256(runtime["model"])
    buffer = io.BytesIO()
    runtime["torch"].save({
        "schema": "calibration_budget_marginalized_checkpoint_v1", "cell": CELL,
        "epoch": epoch, "global_step": global_step,
        "state_dict": runtime["model"].state_dict(), "state_dict_sha256": state_sha,
        "closure_sha256": closure["closure_sha256"],
        "source_authority_sha256": source_authority_sha256,
    }, buffer)
    return buffer.getvalue()


def execute_reviewed_training(root: Path, *, source_smoke: bool = False) -> Mapping[str, object]:
    """Run the authorized source-only CBM cell on the frozen equal-session lane."""
    import time
    from src.tfpd_lane.matched_scorer import build_swa_final_four

    root = Path(root).absolute()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
            "CBM launch requires the reviewed physical GPU0 environment")
    failed_v1 = _validate_failed_v1_predecessor(root)
    failed_v2 = _validate_failed_v2_predecessor(root)
    result_relative = SMOKE_ROOT_RELATIVE if source_smoke else RESULT_ROOT_RELATIVE
    result_root = root / result_relative
    require(not result_root.exists() and not result_root.is_symlink(), "CBM result root is not fresh")
    result_root.mkdir(parents=False, mode=0o700)
    closure = implementation_closure(root)
    attempt = {
        "schema": "calibration_budget_marginalized_attempt_v1", "cell": CELL,
        "status": "ATTEMPT_RESERVED", "kind": "source_smoke" if source_smoke else "full_train",
        "closure": closure, "target_optimizer_steps": 0, "target_backward_calls": 0,
        "target_update_calls": 0, "within_opened": False, "external_opened": False, "formal_opened": False,
        "failed_v1_predecessor": failed_v1,
        "failed_v2_predecessor": failed_v2,
    }
    attempt_sha = _publish_json(result_root, "attempt.json", attempt)
    backend = CBMPhysicalBackend(root)
    runtime: Mapping[str, Any] | None = None
    flags = equal_session.LifecycleFlags()
    started = time.monotonic()
    try:
        spec = equal_session.SOURCE_SMOKE_SPEC if source_smoke else equal_session.FULL_TRAIN_SPEC
        identity = equal_session._build_run_identity(root)
        runtime = backend.prepare(spec, identity, flags)
        source_authority = dict(backend.source_authority(runtime, spec, identity, flags))
        source_authority_sha = _publish_json(result_root, "source_authority.json", source_authority)
        launch = {
            "schema": "calibration_budget_marginalized_launch_v1", "cell": CELL,
            "attempt_sha256": attempt_sha, "source_authority_sha256": source_authority_sha,
            "closure": closure, "spec": spec.payload(), "started_monotonic_seconds": started,
        }
        launch_sha = _publish_json(result_root, "launch.json", launch)
        epoch_shas: dict[str, str] = {}
        checkpoint_shas: dict[str, str] = {}
        global_step = 0
        for epoch in range(spec.epochs):
            schedule = dict(backend.begin_epoch(runtime, epoch, flags))
            outcomes: list[Any] = []
            budget_counts = {str(budget): 0 for budget in BUDGETS}
            for step in range(spec.steps_per_epoch):
                outcome = backend.train_step(
                    runtime, global_step=global_step, expected_lr=backend.lr_for_step(global_step),
                    require_full_proof=(step == spec.steps_per_epoch - 1), flags=flags,
                )
                outcomes.append(outcome)
                budget_counts[str(outcome.batch_evidence["calibration_budget"])] += 1
                global_step += 1
                flags.optimizer_steps_completed += 1
            end = dict(backend.end_epoch(runtime, epoch, outcomes, flags))
            losses = [row.loss for row in outcomes]
            last = outcomes[-1]
            epoch_payload = {
                "schema": "calibration_budget_marginalized_epoch_v1", "cell": CELL,
                "epoch": epoch, "steps": len(outcomes), "cumulative_optimizer_steps": global_step,
                "loss": {"mean": sum(losses) / len(losses), "min": min(losses), "max": max(losses)},
                "lr": {"first": outcomes[0].lr, "last": last.lr},
                "budget_counts": budget_counts, "schedule": schedule, "end_epoch": end,
                "proof": {"critical_gradients": last.critical_gradients,
                          "finite_model": last.finite_model, "finite_optimizer": last.finite_optimizer,
                          "model_state_sha256": last.model_state_sha256,
                          "optimizer_state_sha256": last.optimizer_state_sha256},
                "launch_sha256": launch_sha, "source_authority_sha256": source_authority_sha,
            }
            epoch_shas[str(epoch)] = _publish_json(result_root, f"epoch_{epoch:02d}.json", epoch_payload)
            if not source_smoke and epoch in CHECKPOINT_EPOCHS:
                body = _checkpoint_body(runtime, epoch=epoch, global_step=global_step,
                                        closure=closure, source_authority_sha256=source_authority_sha)
                checkpoint_shas[str(epoch)] = _publish_pair(result_root, f"checkpoint_epoch_{epoch:02d}.pt", body)
        swa_sha: str | None = None
        swa_manifest: Mapping[str, object] | None = None
        if not source_smoke:
            paths = [result_root / f"checkpoint_epoch_{epoch:02d}.pt" for epoch in CHECKPOINT_EPOCHS]
            swa_path = result_root / "swa_final4.pt"
            swa_manifest = build_swa_final_four(paths, swa_path)
            os.chmod(swa_path, 0o444)
            swa_body = swa_path.read_bytes()
            swa_sha = sha256_bytes(swa_body)
            sidecar = result_root / "swa_final4.pt.sha256"
            descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
            try:
                os.write(descriptor, f"{swa_sha}  swa_final4.pt\n".encode("ascii")); os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(sidecar, 0o444)
            _publish_json(result_root, "swa_manifest.json", {
                "schema": "calibration_budget_marginalized_swa_manifest_v1", "cell": CELL,
                "swa_sha256": swa_sha, "checkpoint_sha256": checkpoint_shas,
                "manifest": dict(swa_manifest), "closure": closure,
            })
        terminal = {
            "schema": "calibration_budget_marginalized_terminal_v1", "cell": CELL,
            "status": "TERMINAL", "kind": "source_smoke" if source_smoke else "full_train",
            "attempt_sha256": attempt_sha, "launch_sha256": launch_sha,
            "source_authority_sha256": source_authority_sha, "epoch_sha256": epoch_shas,
            "checkpoint_sha256": checkpoint_shas, "swa_sha256": swa_sha,
            "optimizer_steps": global_step, "epoch_count": spec.epochs,
            "closure": closure, "elapsed_seconds": time.monotonic() - started,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "within_opened": False, "external_opened": False, "formal_opened": False,
        }
        terminal_sha = _publish_json(result_root, "terminal.json", terminal)
        flags.terminal_published = True
        os.chmod(result_root, 0o555)
        return {"terminal_sha256": terminal_sha, "terminal": terminal}
    except BaseException as error:
        try:
            _publish_json(result_root, "failure.json", {
                "schema": "calibration_budget_marginalized_failure_v1", "cell": CELL,
                "status": "FAILED", "attempt_sha256": attempt_sha,
                "error_class": type(error).__name__,
                "error_sha256": sha256_bytes(f"{type(error).__name__}: {error}".encode()),
                "progress": flags.disclosure(), "target_optimizer_steps": 0,
            })
        finally:
            os.chmod(result_root, 0o555)
        raise
    finally:
        backend.close(runtime)
