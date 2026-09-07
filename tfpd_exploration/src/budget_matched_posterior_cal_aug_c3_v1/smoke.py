"""Combined non-performance GPU smoke for C3-Const and C3-Real."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from budget_matched_posterior_cal_aug_v1 import c2_smoke
from budget_matched_posterior_cal_aug_v1 import source_audit as audit_v1
from budget_matched_posterior_cal_aug_v1.training import (
    BudgetTaggedBatchSampler,
    tagged_batch_digest,
)

from .features import (
    BudgetMatchedReliabilityDataset,
    C3Arm,
    fit_source_reliability_normalizer,
    widen_cell_d_for_reliability,
)


SCHEMA = "budget_matched_posterior_cal_aug_c3_runtime_smoke_v1"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c3_runtime_smoke_v1"
)
WORKORDER_RELATIVE = (
    "tfpd_exploration/docs/WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_C3_RUNTIME_V1_20260830.md"
)
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
SMOKE_STEPS = 120
EXPECTED_COUNTS = {30: 40, 10: 40, 4: 40}

_INHERITED_RUNTIME = tuple(
    path for path in c2_smoke.IMPLEMENTATION_PATHS
    if "/tests/" not in path and "/docs/" not in path
)
EXECUTION_PATHS = tuple(dict.fromkeys((
    *_INHERITED_RUNTIME,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/__init__.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/features.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/smoke.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_smoke_v1.py",
)))
REVIEW_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_features_v1.py",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_smoke_v1.py",
)


class C3SmokeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C3SmokeError(message)


def _closure(repository_root: Path, paths: Sequence[str], schema: str) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for relative in paths:
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing/symlink: {relative}")
        rows[relative] = {"bytes": path.stat().st_size, "sha256": audit_v1._file_sha(path)}
    payload: dict[str, object] = {"schema": schema, "files": rows}
    payload["closure_sha256"] = audit_v1._json_sha(payload)
    return payload


def execution_closure(repository_root: Path) -> dict[str, object]:
    return _closure(repository_root, EXECUTION_PATHS, SCHEMA + "_execution_closure")


def review_closure(repository_root: Path) -> dict[str, object]:
    return _closure(repository_root, REVIEW_PATHS, SCHEMA + "_review_closure")


def _tensor_sha(tensor) -> str:
    return audit_v1._sha_bytes(
        tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    )


def _review_drift(start: Mapping[str, object], end: Mapping[str, object]) -> dict[str, object]:
    before = start["files"]
    after = end["files"]
    changed = [path for path in sorted(set(before) | set(after)) if before.get(path) != after.get(path)]
    return {
        "status": "NO_REVIEW_DRIFT" if not changed else "ACCEPTED_NON_NUMERIC_DRIFT",
        "changed_paths": changed,
        "start_sha256": start["closure_sha256"],
        "end_sha256": end["closure_sha256"],
        "numerical_acceptance_affected": False,
    }


def _validate_arm(
    *, stats: Mapping[str, object], rows, dropout: Mapping[str, object],
    parameters_finite: bool, optimizer_finite: bool, arm: C3Arm,
    q_weight_nonzero: int, q_moment_nonzero: int,
) -> dict[str, object]:
    consumed = list(rows[:SMOKE_STEPS])
    budgets = [int(row["budget"]) for row in consumed]
    counts = {budget: budgets.count(budget) for budget in (30, 10, 4)}
    checks = {
        "optimizer_steps_exact": stats.get("optimizer_steps") == SMOKE_STEPS,
        "visible_side_clean": stats.get("visible_side_violation_count") == 0,
        "loss_finite": stats.get("nonfinite_loss_steps") == 0,
        "gradients_finite": stats.get("nonfinite_grad_steps") == 0,
        "parameters_finite": bool(parameters_finite),
        "optimizer_finite": bool(optimizer_finite),
        "budget_cycle_exact": budgets == [(30, 10, 4)[index % 3] for index in range(SMOKE_STEPS)],
        "budget_counts_exact": counts == EXPECTED_COUNTS,
        "dropout_forwards_exact": dropout.get("n_forwards_with_sampled_p") == SMOKE_STEPS,
        "dropout_mask_calls_exact": dropout.get("n_recorded_unit_mask_calls") == SMOKE_STEPS,
        "side_branch_gradient_nonzero": not bool(stats.get("w_side_grad_exact_zero_all_steps", True)),
        "q_weight_semantics": (
            q_weight_nonzero == 0 if arm is C3Arm.CONSTANT else q_weight_nonzero > 0
        ),
        "q_optimizer_moment_semantics": (
            q_moment_nonzero == 0 if arm is C3Arm.CONSTANT else q_moment_nonzero > 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "budget_counts": counts,
        "schedule_sha256": tagged_batch_digest(consumed),
        "q_weight_nonzero": q_weight_nonzero,
        "q_optimizer_moment_nonzero": q_moment_nonzero,
    }


def execute_reviewed(
    repository_root: Path,
    *,
    device_text: str = "cuda:0",
    seed: int = 42,
    train_batch_size: int = 32,
    num_workers: int = 2,
    initial_state_relative: str = (
        "tfpd_exploration/results/admission_arms_v1/canonical_initial_state.pt"
    ),
) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "C3 smoke requires CUDA_VISIBLE_DEVICES=1")
    _require(seed == 42 and train_batch_size == 32, "C3 matched seed/batch drift")
    source = c2_smoke.validate_source_authority(repository_root)
    strict_closure = execution_closure(repository_root)
    advisory_closure = review_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "C3 smoke root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
        "source_terminal_sha256": c2_smoke.SOURCE_TERMINAL_SHA256,
        "execution_closure": strict_closure,
        "review_closure_start": advisory_closure,
        "arms": [C3Arm.CONSTANT.value, C3Arm.REAL.value],
        "smoke_steps_per_arm": SMOKE_STEPS,
        "target_data_opened": False,
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    stage = "post_attempt"
    arms_completed: list[str] = []
    try:
        import lightning.pytorch as pl
        import torch
        from torch.nn.parameter import UninitializedParameter
        from torch.utils.data import DataLoader, default_collate
        from mc_maze.multisession_datamodule import SessionBatchSampler
        from cal_aug_v1 import plan as cal_plan
        from cal_aug_v1 import receipts as cal_receipts

        device = torch.device(device_text)
        _require(device.type == "cuda" and torch.cuda.is_available(), "CUDA unavailable")
        gpu = cal_receipts.gpu_binding(device_text, require_uuid=GPU1_UUID)
        stack = cal_receipts.load_sealed_runner_stack(
            repository_root, repository_root / "tfpd_exploration"
        )
        arm_runner = stack["arm_runner"]
        arm_common = stack["arm_common"]
        pop_robust = stack["pop_robust"]
        from src.models.components.streaming_encoders import build_encoder

        class Args:
            pass

        args = Args()
        args.train_batch_size = train_batch_size
        args.num_workers = num_workers
        args.seed = seed
        stage = "source_materialization"
        dm, _ = arm_runner.build_datamodule(args)
        source_roster = tuple(dm.session_splits["train"])
        _require(len(source_roster) == 27, "strict source roster drift")
        features, feature_evidence = c2_smoke._materialize_features(
            repository_root=repository_root, dm=dm, authority=source["authority"]
        )
        q_normalizer = fit_source_reliability_normalizer(
            features, source_roster=source_roster
        )
        q_payload = q_normalizer.payload()
        q_payload["body_sha256"] = audit_v1._json_sha(q_payload)

        initial_path = repository_root / initial_state_relative
        initial_sidecar = Path(str(initial_path) + ".sha256")
        _require(initial_path.is_file() and initial_sidecar.is_file(), "initial state pair absent")
        initial_artifact_sha = audit_v1._file_sha(initial_path)
        _require(
            initial_artifact_sha == initial_sidecar.read_text().split()[0],
            "initial state sidecar drift",
        )
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
        results: dict[str, dict[str, object]] = {}
        initial_state_digests: list[str] = []
        initial_prediction_digests: list[str] = []
        run_started = time.perf_counter()
        for arm in (C3Arm.CONSTANT, C3Arm.REAL):
            stage = f"arm_{arm.value}"
            pl.seed_everything(seed, workers=True)
            model = pop_robust.build_population_robustness_model(seed=seed, cell="D")
            model.load_state_dict(initial_payload["state_dict"], strict=True)
            _require(
                arm_common.state_sha256(model) == initial_payload["state_sha256"],
                "strict C2 initial state drift",
            )
            widen_cell_d_for_reliability(model, build_encoder)
            widened_initial_sha = arm_common.state_sha256(model)
            initial_state_digests.append(widened_initial_sha)
            model.to(device)

            base_dataset = dm.train_dataset
            base_sampler = SessionBatchSampler(
                base_dataset, batch_size=train_batch_size, shuffle=True, seed=seed
            )
            tagged_sampler = BudgetTaggedBatchSampler(base_sampler)
            dataset = BudgetMatchedReliabilityDataset(
                base_dataset, features, q_normalizer, arm=arm
            )
            loader = DataLoader(
                dataset,
                batch_sampler=tagged_sampler,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
            sample_indices = base_sampler.batched_indices[0][:4]
            sample = default_collate([dataset[(index, 30, 0)] for index in sample_indices])
            neural, _behavior, calibration, _sessions, side = sample[:5]
            model.eval()
            with torch.no_grad():
                initial_prediction, _ = model(
                    neural.to(device), calib_trials=calibration.to(device),
                    side_features=side.to(device),
                )
            initial_prediction_sha = _tensor_sha(initial_prediction)
            initial_prediction_digests.append(initial_prediction_sha)

            model.train()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=arm_common.ADAM_CONSTRUCTOR["lr"],
                betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
                eps=arm_common.ADAM_CONSTRUCTOR["eps"],
                weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
                amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
            )
            lr_fn = lambda step: arm_common.lr_at_step(step, 48, 33_925)  # noqa: E731
            arm_started = time.perf_counter()
            with pop_robust.dynamic_dropout_recorder() as recorder:
                stats = arm_runner.train_epoch(
                    model, optimizer, loader, "t4", lr_fn, device, 0,
                    max_steps=SMOKE_STEPS,
                )
            dropout = pop_robust.summarize_dropout_record(recorder, 2)
            q_weight = model.id_encoder.post_pool[0].weight[:, -1]
            q_state = optimizer.state[model.id_encoder.post_pool[0].weight]
            q_moment = q_state["exp_avg"][:, -1]
            parameters_finite = all(
                bool(torch.isfinite(parameter.detach()).all().item())
                for parameter in model.parameters()
                if parameter.requires_grad
                and not isinstance(parameter, UninitializedParameter)
                and parameter.numel()
            )
            verdict = _validate_arm(
                stats=stats,
                rows=tagged_sampler.tagged_batches,
                dropout=dropout,
                parameters_finite=parameters_finite,
                optimizer_finite=arm_runner.optimizer_state_finite(optimizer),
                arm=arm,
                q_weight_nonzero=int(torch.count_nonzero(q_weight).item()),
                q_moment_nonzero=int(torch.count_nonzero(q_moment).item()),
            )
            _require(verdict["passed"], f"C3 {arm.value} smoke invariant failure: {verdict}")
            results[arm.value] = {
                "arm": arm.value,
                "initial_state_sha256": widened_initial_sha,
                "initial_prediction_sha256": initial_prediction_sha,
                "final_state_sha256": arm_common.state_sha256(model),
                "stats": dict(stats),
                "dropout": dropout,
                "verdict": verdict,
                "wall_seconds": time.perf_counter() - arm_started,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            arms_completed.append(arm.value)
            del optimizer, model, loader, dataset, tagged_sampler, base_sampler
            torch.cuda.empty_cache()

        _require(len(set(initial_state_digests)) == 1, "C3 arm initial state mismatch")
        _require(len(set(initial_prediction_digests)) == 1, "zero-init q did not preserve arm parity")
        final_execution = execution_closure(repository_root)
        _require(
            final_execution["closure_sha256"] == strict_closure["closure_sha256"],
            "C3 execution closure drift",
        )
        final_review = review_closure(repository_root)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C3_RUNTIME_SMOKE_PASSED__NON_PERFORMANCE",
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
            "execution_closure_sha256": strict_closure["closure_sha256"],
            "review_drift": _review_drift(advisory_closure, final_review),
            "gpu": gpu,
            "initial_state": {
                "relative": initial_state_relative,
                "artifact_sha256": initial_artifact_sha,
                "c2_state_sha256": initial_payload["state_sha256"],
                "c3_widened_state_sha256": initial_state_digests[0],
                "new_q_column_exact_zero": True,
            },
            "q_normalizer": q_payload,
            "feature_evidence": feature_evidence,
            "arms": results,
            "initial_prediction_parity": True,
            "wall_seconds": time.perf_counter() - run_started,
            "target_data_opened": False,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "attempt_sha256": attempt_sha,
            "terminal_sha256": terminal_sha,
            "execution_closure_sha256": strict_closure["closure_sha256"],
            "arms": list(results),
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C3_RUNTIME_SMOKE_FAILED",
            "attempt_sha256": attempt_sha,
            "stage": stage,
            "arms_completed": arms_completed,
            "error_class": type(error).__name__,
            "error_sha256": audit_v1._sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "traceback": traceback.format_exc(),
            "target_data_opened": False,
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
        "arms": [C3Arm.CONSTANT.value, C3Arm.REAL.value],
        "smoke_steps_per_arm": SMOKE_STEPS,
        "budget_cycle": [30, 10, 4],
        "execution_closure_path_count": len(EXECUTION_PATHS),
        "review_closure_path_count": len(REVIEW_PATHS),
    }
