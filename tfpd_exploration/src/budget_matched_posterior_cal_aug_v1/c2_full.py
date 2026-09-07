"""Full 48-epoch C2 budget-matched posterior training lifecycle."""

from __future__ import annotations

import json
import os
import stat
import time
import traceback
from pathlib import Path
from typing import Mapping

from . import c2_smoke
from . import source_audit as audit_v1
from .training import BudgetMatchedPosteriorDataset, BudgetTaggedBatchSampler, tagged_batch_digest


SCHEMA = "budget_matched_posterior_cal_aug_c2_full_v1"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c2_full_v1"
)
SMOKE_ROOT_RELATIVE = c2_smoke.RESULT_ROOT_RELATIVE
SMOKE_ATTEMPT_SHA256 = "1287f4183d04da48a20264812749365b15866f2abce81440d4b89a8337b8ef75"
SMOKE_TERMINAL_SHA256 = "565327a916443539f3eb7ecced9a22cb5cf7a7794289c6b7ccc0987b0e6f537b"
EPOCHS = 48
STEPS_PER_EPOCH = 33_925
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
EXPECTED_TOTAL_COUNTS = {30: TOTAL_STEPS // 3, 10: TOTAL_STEPS // 3, 4: TOTAL_STEPS // 3}
FINAL_FOUR = (44, 45, 46, 47)

IMPLEMENTATION_PATHS = tuple(dict.fromkeys((
    *c2_smoke.IMPLEMENTATION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/c2_full.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c2_full_v1.py",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c2_full_v1.py",
)))


class C2FullError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2FullError(message)


def implementation_closure(repository_root: Path) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing/symlink: {relative}")
        rows[relative] = {"bytes": path.stat().st_size, "sha256": audit_v1._file_sha(path)}
    payload: dict[str, object] = {"schema": SCHEMA + "_closure", "files": rows}
    payload["closure_sha256"] = audit_v1._json_sha(payload)
    return payload


def validate_smoke_predecessor(repository_root: Path) -> dict[str, object]:
    root = repository_root / SMOKE_ROOT_RELATIVE
    _require(root.is_dir() and not root.is_symlink(), "C2 smoke root missing/symlink")
    expected = {"attempt.json", "attempt.json.sha256", "terminal.json", "terminal.json.sha256"}
    _require({entry.name for entry in root.iterdir()} == expected, "C2 smoke topology drift")
    for entry in root.iterdir():
        info = entry.lstat()
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444, "C2 smoke leaf mode/type drift")
    attempt = json.loads(audit_v1._read_exact_pair(root / "attempt.json", SMOKE_ATTEMPT_SHA256))
    terminal = json.loads(audit_v1._read_exact_pair(root / "terminal.json", SMOKE_TERMINAL_SHA256))
    _require(terminal.get("status") == "C2_RUNTIME_SMOKE_PASSED__NON_PERFORMANCE", "C2 smoke status drift")
    _require(terminal.get("attempt_sha256") == SMOKE_ATTEMPT_SHA256, "C2 smoke link drift")
    _require(terminal.get("source_authority_sha256") == c2_smoke.SOURCE_AUTHORITY_SHA256, "C2 source link drift")
    _require(terminal.get("verdict", {}).get("passed") is True, "C2 smoke verdict drift")
    _require(terminal.get("smoke_steps") == 120, "C2 smoke step count drift")
    return {
        "root_relative": SMOKE_ROOT_RELATIVE,
        "attempt_sha256": SMOKE_ATTEMPT_SHA256,
        "terminal_sha256": SMOKE_TERMINAL_SHA256,
        "closure_sha256": terminal["closure_sha256"],
        "steps_per_second": terminal["resources"]["steps_per_second"],
        "exact_leaf_count": 4,
        "attempt": attempt,
        "terminal": terminal,
    }


def budget_counts_for_span(start_step: int, steps: int) -> dict[int, int]:
    _require(type(start_step) is int and start_step >= 0, "start step invalid")
    _require(type(steps) is int and steps >= 0, "span invalid")
    sequence = (30, 10, 4)
    return {
        budget: sum(sequence[(start_step + offset) % 3] == budget for offset in range(steps))
        for budget in sequence
    }


def _parameters_finite(model, torch, UninitializedParameter) -> bool:
    return all(
        bool(torch.isfinite(parameter.detach()).all().item())
        for parameter in model.parameters()
        if parameter.requires_grad
        and not isinstance(parameter, UninitializedParameter)
        and parameter.numel()
    )


def execute_reviewed(
    repository_root: Path,
    *,
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
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "C2 full requires CUDA_VISIBLE_DEVICES=0")
    _require(train_batch_size == 32 and seed == 42, "matched recipe batch/seed drift")
    source = c2_smoke.validate_source_authority(repository_root)
    smoke = validate_smoke_predecessor(repository_root)
    closure = implementation_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "C2 full root is not fresh")
    result_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    result_root.mkdir(mode=0o700)
    attempt = {
        "schema": SCHEMA + "_attempt",
        "status": "ATTEMPT_PUBLISHED",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_authority": {key: source[key] for key in (
            "root_relative", "attempt_sha256", "source_authority_sha256",
            "terminal_sha256", "closure_sha256",
        )},
        "smoke_predecessor": {key: smoke[key] for key in (
            "root_relative", "attempt_sha256", "terminal_sha256",
            "closure_sha256", "steps_per_second", "exact_leaf_count",
        )},
        "closure": closure,
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
        "data_boundary": {
            "source_opened_before_attempt": False,
            "within_opened": False, "external_opened": False, "formal_opened": False,
        },
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
        gpu = cal_receipts.gpu_binding(device_text, cal_plan.BOUND_GPU_UUID)
        pinned = cal_receipts.verify_pinned_files(repository_root)
        sealed_predecessors = cal_receipts.verify_sealed_predecessors(repository_root)
        stack = cal_receipts.load_sealed_runner_stack(
            repository_root, repository_root / "tfpd_exploration"
        )
        arm_runner = stack["arm_runner"]
        arm_common = stack["arm_common"]
        matched_scorer = stack["matched_scorer"]
        pop_robust = stack["pop_robust"]

        class Args:
            pass

        args = Args()
        args.train_batch_size = train_batch_size
        args.num_workers = num_workers
        args.seed = seed
        context["stage"] = "source_materialization"
        dm, _a2 = arm_runner.build_datamodule(args)
        _require(len(dm.session_splits["train"]) == 27, "strict source roster drift")
        features, feature_evidence = c2_smoke._materialize_features(
            repository_root=repository_root, dm=dm, authority=source["authority"]
        )

        initial_path = repository_root / initial_state_relative
        initial_sidecar = Path(str(initial_path) + ".sha256")
        _require(initial_path.is_file() and initial_sidecar.is_file(), "initial state pair absent")
        initial_sha = audit_v1._file_sha(initial_path)
        _require(initial_sha == initial_sidecar.read_text().split()[0], "initial state sidecar drift")
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
        pl.seed_everything(seed, workers=True)
        model = pop_robust.build_population_robustness_model(seed=seed, cell="D")
        model.load_state_dict(initial_payload["state_dict"], strict=True)
        initial_state_sha = arm_common.state_sha256(model)
        _require(initial_state_sha == initial_payload["state_sha256"], "initial model state drift")
        model.to(device)

        base_dataset = dm.train_dataset
        base_sampler = SessionBatchSampler(
            base_dataset, batch_size=train_batch_size, shuffle=True, seed=seed
        )
        _require(len(base_sampler) == STEPS_PER_EPOCH, "steps-per-epoch drift")
        tagged_sampler = BudgetTaggedBatchSampler(base_sampler)
        dataset = BudgetMatchedPosteriorDataset(base_dataset, features)
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
        lr_fn = lambda step: arm_common.lr_at_step(  # noqa: E731
            step, EPOCHS, STEPS_PER_EPOCH
        )
        launch = {
            "schema": SCHEMA + "_launch",
            "status": "FULL_TRAINING_LAUNCHED",
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
            "smoke_terminal_sha256": SMOKE_TERMINAL_SHA256,
            "closure_sha256": closure["closure_sha256"],
            "initial_state": {
                "relative": initial_state_relative,
                "artifact_sha256": initial_sha,
                "state_sha256": initial_state_sha,
                "strict_load": True,
            },
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
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
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
            _require(len(tagged_sampler.tagged_batches) == STEPS_PER_EPOCH, "epoch sampler cardinality drift")
            epoch_counts = budget_counts_for_span(context["global_step"], STEPS_PER_EPOCH)
            realized_counts = {
                budget: sum(row["budget"] == budget for row in tagged_sampler.tagged_batches)
                for budget in (30, 10, 4)
            }
            _require(realized_counts == epoch_counts, "realized epoch budget counts drift")
            for budget in global_counts:
                global_counts[budget] += epoch_counts[budget]
            context["global_step"] += STEPS_PER_EPOCH
            parameters_finite = _parameters_finite(model, torch, UninitializedParameter)
            optimizer_finite = arm_runner.optimizer_state_finite(optimizer)
            invariant_checks = {
                "steps_exact": stats["optimizer_steps"] == STEPS_PER_EPOCH,
                "visible_side_clean": stats["visible_side_violation_count"] == 0,
                "loss_finite": stats["nonfinite_loss_steps"] == 0,
                "gradients_finite": stats["nonfinite_grad_steps"] == 0,
                "parameters_finite": parameters_finite,
                "optimizer_finite": optimizer_finite,
                "side_branch_gradient_nonzero": not stats["w_side_grad_exact_zero_all_steps"],
                "dropout_forwards_exact": dropout["n_forwards_with_sampled_p"] == STEPS_PER_EPOCH,
                "dropout_mask_calls_exact": dropout["n_recorded_unit_mask_calls"] == STEPS_PER_EPOCH,
            }
            _require(all(invariant_checks.values()), f"epoch {epoch} invariant failure: {invariant_checks}")
            diag = {
                "schema": SCHEMA + "_epoch",
                "epoch": epoch,
                "optimizer_steps": STEPS_PER_EPOCH,
                "optimizer_steps_total": context["global_step"],
                "budget_counts": epoch_counts,
                "budget_schedule_sha256": tagged_batch_digest(tagged_sampler.tagged_batches),
                "budget_first": tagged_sampler.tagged_batches[0]["budget"],
                "budget_last": tagged_sampler.tagged_batches[-1]["budget"],
                "stats": dict(stats),
                "dropout_summary": dropout,
                "invariant_checks": invariant_checks,
                "state_dict_sha256": arm_common.state_sha256(model),
                "optimizer_state_sha256": arm_common.optimizer_sha256(optimizer),
                "wall_seconds": time.perf_counter() - epoch_started,
                "checkpoint": None,
            }
            if epoch in FINAL_FOUR:
                path = result_root / f"checkpoint_epoch_{epoch:02d}.pt"
                torch.save({
                    "kind": SCHEMA + "_checkpoint",
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
                "checkpoint": diag["checkpoint"],
            })
            context["epochs_completed"] = epoch + 1
            print(json.dumps({
                "epoch": epoch,
                "loss": stats["train_loss_mean_per_step"],
                "steps_total": context["global_step"],
                "budget_counts_total": global_counts,
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
        swa_model.load_state_dict(swa_payload["state_dict"], strict=True)
        swa_state_sha = arm_common.state_sha256(swa_model)
        sample_indices = base_sampler.batched_indices[0][:4]
        sample = default_collate([dataset[(index, 30, TOTAL_STEPS)] for index in sample_indices])
        neural, _behavior, calibration, _sessions, side = sample[:5]
        swa_model.to(device).eval()
        with torch.no_grad():
            prediction, _identity = swa_model(
                neural.to(device),
                calib_trials=calibration.to(device),
                side_features=side.to(device),
            )
        strict_reload_finite = bool(torch.isfinite(prediction).all().item())
        _require(strict_reload_finite, "SWA strict reload finite forward failed")
        final_closure = implementation_closure(repository_root)
        _require(final_closure["closure_sha256"] == closure["closure_sha256"], "C2 full closure drift")
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": "C2_FULL_TRAINING_TERMINAL",
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
            "smoke_terminal_sha256": SMOKE_TERMINAL_SHA256,
            "closure_sha256": closure["closure_sha256"],
            "epoch_count": len(diagnostics),
            "optimizer_steps": context["global_step"],
            "budget_counts": global_counts,
            "epoch_receipts": diagnostics,
            "checkpoints": [
                {key: row[key] for key in (
                    "name", "epoch", "sha256", "state_dict_sha256"
                )}
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
                "optimizer_steps_per_second": TOTAL_STEPS / (time.perf_counter() - run_started),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
            "target_optimizer_backward_update": 0,
        }
        terminal_sha = audit_v1._publish_pair(result_root, "terminal.json", terminal)
        os.chmod(result_root, 0o555)
        return {
            "status": terminal["status"],
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "terminal_sha256": terminal_sha,
            "swa_sha256": swa_sha,
            "optimizer_steps": TOTAL_STEPS,
            "budget_counts": global_counts,
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_FULL_TRAINING_FAILED",
            "attempt_sha256": attempt_sha,
            "stage": context["stage"],
            "global_step": context["global_step"],
            "epochs_completed": context["epochs_completed"],
            "error_class": type(error).__name__,
            "error_sha256": audit_v1._sha_bytes(f"{type(error).__name__}: {error}".encode()),
            "traceback": traceback.format_exc(),
            "target_optimizer_backward_update": 0,
        })
        os.chmod(result_root, 0o555)
        raise


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan",
        "status": "DRY_NO_DATA_NO_GPU_NO_WRITE",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "smoke_terminal_sha256": SMOKE_TERMINAL_SHA256,
        "source_authority_sha256": c2_smoke.SOURCE_AUTHORITY_SHA256,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_optimizer_steps": TOTAL_STEPS,
        "expected_total_budget_counts": EXPECTED_TOTAL_COUNTS,
        "train_batch_size": 32,
    }

