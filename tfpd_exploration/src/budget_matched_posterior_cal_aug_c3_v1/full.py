"""Matched 48-epoch C3-Const/C3-Real full-training lifecycle."""

from __future__ import annotations

import json
import os
import stat
import time
import traceback
from pathlib import Path
from typing import Mapping, Sequence

from budget_matched_posterior_cal_aug_v1 import c2_smoke
from budget_matched_posterior_cal_aug_v1 import source_audit as audit_v1
from budget_matched_posterior_cal_aug_v1.training import (
    BudgetTaggedBatchSampler,
    tagged_batch_digest,
)

from . import smoke
from .features import (
    BudgetMatchedReliabilityDataset,
    C3Arm,
    fit_source_reliability_normalizer,
    widen_cell_d_for_reliability,
)


SCHEMA = "budget_matched_posterior_cal_aug_c3_full_v1"
RESULT_ROOTS = {
    C3Arm.CONSTANT: (
        "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c3_constant_full_v1"
    ),
    C3Arm.REAL: (
        "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c3_real_full_v1"
    ),
}
SMOKE_ATTEMPT_SHA256 = "292c1972a99b8b98b25ec5f47dcb07b9c3216098b49f362667c64c62fc77adf1"
SMOKE_TERMINAL_SHA256 = "f9dc00b09bf12d917415f1a78fb1a9eec780246aa76f680a33ec58575f0f98d9"
SMOKE_EXECUTION_CLOSURE_SHA256 = (
    "0269b725d19c48c05b4856ae2f41b11a0820649d3bc25ad3e2ca85cac9d06850"
)
WIDENED_INITIAL_STATE_SHA256 = (
    "a2fbb353caf17aac0e8934eb2d6bf3aefab2ee75530303d0c89b3917de8218cb"
)
EPOCHS = 48
STEPS_PER_EPOCH = 33_925
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
EXPECTED_TOTAL_COUNTS = {30: TOTAL_STEPS // 3, 10: TOTAL_STEPS // 3, 4: TOTAL_STEPS // 3}
FINAL_FOUR = (44, 45, 46, 47)

EXECUTION_PATHS = tuple(dict.fromkeys((
    *smoke.EXECUTION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_c3_v1/full.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c3_full_v1.py",
)))
REVIEW_PATHS = tuple(dict.fromkeys((
    *smoke.REVIEW_PATHS,
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c3_full_v1.py",
)))


class C3FullError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C3FullError(message)


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


def validate_smoke_predecessor(repository_root: Path) -> dict[str, object]:
    root = repository_root / smoke.RESULT_ROOT_RELATIVE
    _require(root.is_dir() and not root.is_symlink(), "C3 smoke root missing/symlink")
    expected = {"attempt.json", "attempt.json.sha256", "terminal.json", "terminal.json.sha256"}
    _require({entry.name for entry in root.iterdir()} == expected, "C3 smoke topology drift")
    for entry in root.iterdir():
        info = entry.lstat()
        _require(
            stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444,
            "C3 smoke leaf type/mode drift",
        )
    attempt = json.loads(
        audit_v1._read_exact_pair(root / "attempt.json", SMOKE_ATTEMPT_SHA256)
    )
    terminal = json.loads(
        audit_v1._read_exact_pair(root / "terminal.json", SMOKE_TERMINAL_SHA256)
    )
    _require(
        terminal.get("status") == "C3_RUNTIME_SMOKE_PASSED__NON_PERFORMANCE",
        "C3 smoke status drift",
    )
    _require(terminal.get("attempt_sha256") == SMOKE_ATTEMPT_SHA256, "C3 smoke link drift")
    _require(
        terminal.get("execution_closure_sha256") == SMOKE_EXECUTION_CLOSURE_SHA256,
        "C3 smoke execution closure drift",
    )
    _require(terminal.get("initial_prediction_parity") is True, "C3 smoke initial parity absent")
    _require(
        terminal.get("initial_state", {}).get("c3_widened_state_sha256")
        == WIDENED_INITIAL_STATE_SHA256,
        "C3 widened initial state drift",
    )
    for arm in (C3Arm.CONSTANT, C3Arm.REAL):
        row = terminal.get("arms", {}).get(arm.value, {})
        _require(row.get("verdict", {}).get("passed") is True, f"C3 {arm.value} smoke failed")
    _require(
        terminal["arms"][C3Arm.CONSTANT.value]["verdict"]["q_weight_nonzero"] == 0,
        "C3 constant q weight drift",
    )
    _require(
        terminal["arms"][C3Arm.REAL.value]["verdict"]["q_weight_nonzero"] > 0,
        "C3 real q did not learn in smoke",
    )
    return {
        "root_relative": smoke.RESULT_ROOT_RELATIVE,
        "attempt_sha256": SMOKE_ATTEMPT_SHA256,
        "terminal_sha256": SMOKE_TERMINAL_SHA256,
        "execution_closure_sha256": SMOKE_EXECUTION_CLOSURE_SHA256,
        "attempt": attempt,
        "terminal": terminal,
    }


def budget_counts_for_span(start_step: int, steps: int) -> dict[int, int]:
    _require(type(start_step) is int and start_step >= 0, "start step invalid")
    _require(type(steps) is int and steps >= 0, "span invalid")
    cycle = (30, 10, 4)
    return {
        budget: sum(cycle[(start_step + offset) % 3] == budget for offset in range(steps))
        for budget in cycle
    }


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


def execute_reviewed(
    repository_root: Path,
    *,
    arm: C3Arm | str,
    device_text: str = "cuda:0",
    seed: int = 42,
    train_batch_size: int = 32,
    num_workers: int = 4,
    timeout_seconds: int = 12 * 60 * 60,
    initial_state_relative: str = (
        "tfpd_exploration/results/admission_arms_v1/canonical_initial_state.pt"
    ),
) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    try:
        selected = C3Arm(arm)
    except ValueError as error:
        raise C3FullError(f"unknown C3 arm: {arm!r}") from error
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "C3 full requires CUDA_VISIBLE_DEVICES=1")
    _require(seed == 42 and train_batch_size == 32, "matched seed/batch drift")
    source = c2_smoke.validate_source_authority(repository_root)
    predecessor = validate_smoke_predecessor(repository_root)
    strict_closure = execution_closure(repository_root)
    advisory_closure = review_closure(repository_root)
    result_root = repository_root / RESULT_ROOTS[selected]
    _require(not result_root.exists() and not result_root.is_symlink(), "C3 full root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "arm": selected.value,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
        "smoke_predecessor": {
            key: predecessor[key] for key in (
                "root_relative", "attempt_sha256", "terminal_sha256",
                "execution_closure_sha256",
            )
        },
        "execution_closure": strict_closure,
        "review_closure_start": advisory_closure,
        "recipe": {
            "seed": seed,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "total_optimizer_steps": TOTAL_STEPS,
            "train_batch_size": train_batch_size,
            "budget_cycle": [30, 10, 4],
            "expected_total_budget_counts": EXPECTED_TOTAL_COUNTS,
            "final_four": list(FINAL_FOUR),
            "timeout_seconds": timeout_seconds,
        },
        "target_data_opened": False,
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    context = {"stage": "post_attempt", "global_step": 0, "epochs_completed": 0}
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
        gpu = cal_receipts.gpu_binding(device_text, require_uuid=smoke.GPU1_UUID)
        pinned = cal_receipts.verify_pinned_files(repository_root)
        sealed_predecessors = cal_receipts.verify_sealed_predecessors(repository_root)
        stack = cal_receipts.load_sealed_runner_stack(
            repository_root, repository_root / "tfpd_exploration"
        )
        arm_runner = stack["arm_runner"]
        arm_common = stack["arm_common"]
        matched_scorer = stack["matched_scorer"]
        pop_robust = stack["pop_robust"]
        from src.models.components.streaming_encoders import build_encoder

        class Args:
            pass

        args = Args()
        args.train_batch_size = train_batch_size
        args.num_workers = num_workers
        args.seed = seed
        context["stage"] = "source_materialization"
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
        _require(
            q_payload == predecessor["terminal"]["q_normalizer"],
            "C3 q normalizer changed since smoke",
        )

        initial_path = repository_root / initial_state_relative
        initial_sidecar = Path(str(initial_path) + ".sha256")
        _require(initial_path.is_file() and initial_sidecar.is_file(), "initial state pair absent")
        initial_artifact_sha = audit_v1._file_sha(initial_path)
        _require(
            initial_artifact_sha == initial_sidecar.read_text().split()[0],
            "initial state sidecar drift",
        )
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
        pl.seed_everything(seed, workers=True)
        model = pop_robust.build_population_robustness_model(seed=seed, cell="D")
        model.load_state_dict(initial_payload["state_dict"], strict=True)
        _require(
            arm_common.state_sha256(model) == initial_payload["state_sha256"],
            "strict C2 initial state drift",
        )
        widen_cell_d_for_reliability(model, build_encoder)
        widened_initial_sha = arm_common.state_sha256(model)
        _require(
            widened_initial_sha == WIDENED_INITIAL_STATE_SHA256,
            "C3 widened initial state no longer matches smoke",
        )
        model.to(device)

        base_dataset = dm.train_dataset
        base_sampler = SessionBatchSampler(
            base_dataset, batch_size=train_batch_size, shuffle=True, seed=seed
        )
        _require(len(base_sampler) == STEPS_PER_EPOCH, "steps-per-epoch drift")
        tagged_sampler = BudgetTaggedBatchSampler(base_sampler)
        dataset = BudgetMatchedReliabilityDataset(
            base_dataset, features, q_normalizer, arm=selected
        )
        loader = DataLoader(
            dataset,
            batch_sampler=tagged_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        deadline = cal_receipts.DeadlineGuard(timeout_seconds)
        guarded_loader = cal_receipts.DeadlineLoader(loader, deadline)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=arm_common.ADAM_CONSTRUCTOR["lr"],
            betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
            eps=arm_common.ADAM_CONSTRUCTOR["eps"],
            weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
            amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
        )
        lr_fn = lambda step: arm_common.lr_at_step(step, EPOCHS, STEPS_PER_EPOCH)  # noqa: E731
        launch = {
            "schema": SCHEMA + "_launch",
            "status": "C3_FULL_TRAINING_LAUNCHED",
            "arm": selected.value,
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
            "smoke_terminal_sha256": SMOKE_TERMINAL_SHA256,
            "execution_closure_sha256": strict_closure["closure_sha256"],
            "initial_state": {
                "relative": initial_state_relative,
                "artifact_sha256": initial_artifact_sha,
                "c2_state_sha256": initial_payload["state_sha256"],
                "c3_widened_state_sha256": widened_initial_sha,
                "new_q_column_exact_zero": True,
            },
            "q_normalizer": q_payload,
            "environment": {
                "device": device_text,
                "gpu": gpu,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python_no_user_site": bool(__import__("sys").flags.no_user_site),
            },
            "recipe": attempt["recipe"],
            "feature_evidence": feature_evidence,
            "pinned_sha256": pinned,
            "sealed_predecessors": sealed_predecessors,
            "target_data_opened": False,
            "target_optimizer_backward_update": 0,
        }
        launch_sha = audit_v1._publish_pair(result_root, "launch.json", launch)

        diagnostics = []
        checkpoints = []
        global_counts = {30: 0, 10: 0, 4: 0}
        run_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(EPOCHS):
            context["stage"] = f"epoch_{epoch:02d}"
            tagged_sampler.start_batch = context["global_step"]
            guarded_loader.steps_offset = context["global_step"]
            deadline.check(context["stage"], context["global_step"])
            epoch_started = time.perf_counter()
            with pop_robust.dynamic_dropout_recorder() as recorder:
                stats = arm_runner.train_epoch(
                    model, optimizer, guarded_loader, "t4", lr_fn, device,
                    context["global_step"], max_steps=None,
                )
            dropout = pop_robust.summarize_dropout_record(recorder, 2)
            _require(stats["optimizer_steps"] == STEPS_PER_EPOCH, "epoch step count drift")
            _require(len(tagged_sampler.tagged_batches) == STEPS_PER_EPOCH, "epoch sampler count drift")
            epoch_counts = budget_counts_for_span(context["global_step"], STEPS_PER_EPOCH)
            realized_counts = {
                budget: sum(row["budget"] == budget for row in tagged_sampler.tagged_batches)
                for budget in (30, 10, 4)
            }
            _require(realized_counts == epoch_counts, "realized budget counts drift")
            for budget in global_counts:
                global_counts[budget] += epoch_counts[budget]
            context["global_step"] += STEPS_PER_EPOCH
            parameters_finite = all(
                bool(torch.isfinite(parameter.detach()).all().item())
                for parameter in model.parameters()
                if parameter.requires_grad
                and not isinstance(parameter, UninitializedParameter)
                and parameter.numel()
            )
            optimizer_finite = arm_runner.optimizer_state_finite(optimizer)
            q_weight = model.id_encoder.post_pool[0].weight[:, -1]
            q_state = optimizer.state[model.id_encoder.post_pool[0].weight]
            q_moment = q_state["exp_avg"][:, -1]
            q_weight_nonzero = int(torch.count_nonzero(q_weight).item())
            q_moment_nonzero = int(torch.count_nonzero(q_moment).item())
            q_semantics = (
                q_weight_nonzero == 0 and q_moment_nonzero == 0
                if selected is C3Arm.CONSTANT
                else q_weight_nonzero > 0 and q_moment_nonzero > 0
            )
            checks = {
                "steps_exact": stats["optimizer_steps"] == STEPS_PER_EPOCH,
                "visible_side_clean": stats["visible_side_violation_count"] == 0,
                "loss_finite": stats["nonfinite_loss_steps"] == 0,
                "gradients_finite": stats["nonfinite_grad_steps"] == 0,
                "parameters_finite": parameters_finite,
                "optimizer_finite": optimizer_finite,
                "side_branch_gradient_nonzero": not stats["w_side_grad_exact_zero_all_steps"],
                "dropout_forwards_exact": dropout["n_forwards_with_sampled_p"] == STEPS_PER_EPOCH,
                "dropout_mask_calls_exact": dropout["n_recorded_unit_mask_calls"] == STEPS_PER_EPOCH,
                "q_semantics": q_semantics,
            }
            _require(all(checks.values()), f"epoch {epoch} invariant failure: {checks}")
            diag = {
                "schema": SCHEMA + "_epoch",
                "arm": selected.value,
                "epoch": epoch,
                "optimizer_steps": STEPS_PER_EPOCH,
                "optimizer_steps_total": context["global_step"],
                "budget_counts": epoch_counts,
                "budget_schedule_sha256": tagged_batch_digest(tagged_sampler.tagged_batches),
                "stats": dict(stats),
                "dropout_summary": dropout,
                "invariant_checks": checks,
                "q_weight_nonzero": q_weight_nonzero,
                "q_weight_norm": float(q_weight.norm().item()),
                "q_optimizer_moment_nonzero": q_moment_nonzero,
                "state_dict_sha256": arm_common.state_sha256(model),
                "optimizer_state_sha256": arm_common.optimizer_sha256(optimizer),
                "wall_seconds": time.perf_counter() - epoch_started,
                "checkpoint": None,
            }
            if epoch in FINAL_FOUR:
                path = result_root / f"checkpoint_epoch_{epoch:02d}.pt"
                torch.save({
                    "kind": SCHEMA + "_checkpoint",
                    "arm": selected.value,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "state_dict_sha256": diag["state_dict_sha256"],
                    "optimizer_state_sha256": diag["optimizer_state_sha256"],
                }, path)
                checkpoint_sha = arm_runner.seal_file(path)
                diag["checkpoint"] = {
                    "name": path.name,
                    "sha256": checkpoint_sha,
                    "state_dict_sha256": diag["state_dict_sha256"],
                }
                checkpoints.append({
                    "path": path,
                    "name": path.name,
                    "epoch": epoch,
                    "sha256": checkpoint_sha,
                    "state_dict_sha256": diag["state_dict_sha256"],
                })
            epoch_sha = audit_v1._publish_pair(result_root, f"epoch_{epoch:02d}.json", diag)
            diagnostics.append({
                "epoch": epoch,
                "body_sha256": epoch_sha,
                "loss": stats["train_loss_mean_per_step"],
                "state_dict_sha256": diag["state_dict_sha256"],
                "q_weight_norm": diag["q_weight_norm"],
                "checkpoint": diag["checkpoint"],
            })
            context["epochs_completed"] = epoch + 1
            print(json.dumps({
                "arm": selected.value,
                "epoch": epoch,
                "loss": stats["train_loss_mean_per_step"],
                "q_weight_norm": diag["q_weight_norm"],
                "steps_total": context["global_step"],
                "wall_seconds": diag["wall_seconds"],
            }), flush=True)

        _require(context["global_step"] == TOTAL_STEPS, "total optimizer-step drift")
        _require(global_counts == EXPECTED_TOTAL_COUNTS, "global budget balance drift")
        _require([row["epoch"] for row in checkpoints] == list(FINAL_FOUR), "final-four topology drift")
        swa_path = result_root / "swa_final4.pt"
        swa_manifest = matched_scorer.build_swa_final_four(
            [row["path"] for row in checkpoints], swa_path
        )
        swa_sha = arm_runner.seal_file(swa_path)
        swa_payload = torch.load(swa_path, map_location="cpu", weights_only=False)
        swa_model = pop_robust.build_population_robustness_model(seed=seed, cell="D")
        swa_model.load_state_dict(initial_payload["state_dict"], strict=True)
        widen_cell_d_for_reliability(swa_model, build_encoder)
        swa_model.load_state_dict(swa_payload["state_dict"], strict=True)
        swa_state_sha = arm_common.state_sha256(swa_model)
        sample_indices = base_sampler.batched_indices[0][:4]
        sample = default_collate([dataset[(index, 30, TOTAL_STEPS)] for index in sample_indices])
        neural, _behavior, calibration, _sessions, side = sample[:5]
        swa_model.to(device).eval()
        with torch.no_grad():
            prediction, _ = swa_model(
                neural.to(device), calib_trials=calibration.to(device),
                side_features=side.to(device),
            )
        strict_reload_finite = bool(torch.isfinite(prediction).all().item())
        _require(strict_reload_finite, "SWA strict reload finite forward failed")
        final_execution = execution_closure(repository_root)
        _require(
            final_execution["closure_sha256"] == strict_closure["closure_sha256"],
            "C3 execution closure drift",
        )
        final_review = review_closure(repository_root)
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C3_FULL_TRAINING_TERMINAL",
            "arm": selected.value,
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
            "smoke_terminal_sha256": SMOKE_TERMINAL_SHA256,
            "execution_closure_sha256": strict_closure["closure_sha256"],
            "review_drift": _review_drift(advisory_closure, final_review),
            "epoch_count": len(diagnostics),
            "optimizer_steps": context["global_step"],
            "budget_counts": global_counts,
            "epoch_receipts": diagnostics,
            "checkpoints": [
                {key: row[key] for key in ("name", "epoch", "sha256", "state_dict_sha256")}
                for row in checkpoints
            ],
            "swa": {
                "name": swa_path.name,
                "sha256": swa_sha,
                "state_dict_sha256": swa_state_sha,
                "window_epochs": list(FINAL_FOUR),
                "manifest": swa_manifest,
                "strict_reload_finite_forward": strict_reload_finite,
            },
            "resources": {
                "device": device_text,
                "gpu": gpu,
                "wall_seconds": time.perf_counter() - run_started,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
            "target_data_opened": False,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "arm": selected.value,
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "terminal_sha256": terminal_sha,
            "swa_sha256": swa_sha,
            "optimizer_steps": TOTAL_STEPS,
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C3_FULL_TRAINING_FAILED",
            "arm": selected.value,
            "attempt_sha256": attempt_sha,
            "stage": context["stage"],
            "global_step": context["global_step"],
            "epochs_completed": context["epochs_completed"],
            "error_class": type(error).__name__,
            "error_sha256": audit_v1._sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "traceback": traceback.format_exc(),
            "target_data_opened": False,
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan(arm: C3Arm | str) -> dict[str, object]:
    selected = C3Arm(arm)
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
        "arm": selected.value,
        "result_root_relative": RESULT_ROOTS[selected],
        "smoke_terminal_sha256": SMOKE_TERMINAL_SHA256,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_optimizer_steps": TOTAL_STEPS,
        "expected_total_budget_counts": EXPECTED_TOTAL_COUNTS,
        "train_batch_size": 32,
    }
