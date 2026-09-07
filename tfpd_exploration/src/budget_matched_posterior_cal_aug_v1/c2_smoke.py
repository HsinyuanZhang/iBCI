"""Bounded GPU runtime smoke for C2 budget-matched posterior training."""

from __future__ import annotations

import json
import os
import stat
import time
import traceback
from pathlib import Path
from typing import Mapping

import numpy as np

from . import source_audit as audit_v1
from . import source_audit_v3 as audit_v3
from .posterior import angular_reliability, array_sha256, fit_posterior_mean
from .training import (
    BudgetMatchedPosteriorDataset,
    BudgetTaggedBatchSampler,
    build_session_posterior_features,
    posterior_normalizer_from_payload,
    source_prior_from_payload,
    tagged_batch_digest,
)


SCHEMA = "budget_matched_posterior_cal_aug_c2_smoke_v1"
RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c2_smoke_v1"
)
SOURCE_ROOT_RELATIVE = audit_v3.RESULT_ROOT_RELATIVE
SOURCE_ATTEMPT_SHA256 = "155a38a6d65f73ea2653122aeb3531073f3b8a14cb9906964a5da45e33d827e0"
SOURCE_AUTHORITY_SHA256 = "92f898aa828225b4805f2b23cc5f419f6c0f48cb6aaf54819ea884ab951a0791"
SOURCE_TERMINAL_SHA256 = "7aba814933071f91bbb0fbc17eb3271324999d8eede0d5cef50841d201766420"
SMOKE_STEPS = 120
EXPECTED_COUNTS = {30: 40, 10: 40, 4: 40}

IMPLEMENTATION_PATHS = tuple(dict.fromkeys((
    *audit_v3.IMPLEMENTATION_PATHS,
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/training.py",
    "tfpd_exploration/src/budget_matched_posterior_cal_aug_v1/c2_smoke.py",
    "tfpd_exploration/scripts/run_budget_matched_posterior_cal_aug_c2_smoke_v1.py",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_training_v1.py",
    "tfpd_exploration/tests/test_budget_matched_posterior_cal_aug_c2_smoke_v1.py",
    "tfpd_exploration/src/cal_aug_v1/plan.py",
    "tfpd_exploration/src/cal_aug_v1/receipts.py",
    "tfpd_exploration/scripts/run_cal_aug_cell_v1.py",
    "tfpd_exploration/scripts/run_admission_arm.py",
    "tfpd_exploration/scripts/run_pop_robust_cell.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
)))


class C2SmokeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2SmokeError(message)


def implementation_closure(repository_root: Path) -> dict[str, object]:
    rows: dict[str, dict[str, object]] = {}
    for relative in IMPLEMENTATION_PATHS:
        path = repository_root / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing/symlink: {relative}")
        rows[relative] = {"bytes": path.stat().st_size, "sha256": audit_v1._file_sha(path)}
    payload: dict[str, object] = {"schema": SCHEMA + "_closure", "files": rows}
    payload["closure_sha256"] = audit_v1._json_sha(payload)
    return payload


def validate_source_authority(repository_root: Path) -> dict[str, object]:
    root = repository_root / SOURCE_ROOT_RELATIVE
    _require(root.is_dir() and not root.is_symlink(), "V3 source authority root missing/symlink")
    expected = {
        "attempt.json", "attempt.json.sha256", "source_authority.json",
        "source_authority.json.sha256", "terminal.json", "terminal.json.sha256",
    }
    _require({entry.name for entry in root.iterdir()} == expected, "V3 source topology drift")
    for entry in root.iterdir():
        info = entry.lstat()
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444, "V3 source leaf type/mode drift")
    attempt = json.loads(audit_v1._read_exact_pair(root / "attempt.json", SOURCE_ATTEMPT_SHA256))
    authority = json.loads(
        audit_v1._read_exact_pair(root / "source_authority.json", SOURCE_AUTHORITY_SHA256)
    )
    terminal = json.loads(
        audit_v1._read_exact_pair(root / "terminal.json", SOURCE_TERMINAL_SHA256)
    )
    _require(authority.get("status") == "SOURCE_POSTERIOR_AUTHORITY_V3_PASSED", "V3 authority status drift")
    _require(authority.get("attempt_sha256") == SOURCE_ATTEMPT_SHA256, "V3 authority attempt link drift")
    _require(terminal.get("status") == "SOURCE_POSTERIOR_AUTHORITY_V3_PASSED", "V3 terminal status drift")
    _require(terminal.get("source_authority_sha256") == SOURCE_AUTHORITY_SHA256, "V3 terminal authority link drift")
    _require(terminal.get("gpu_smoke_authorized") is True, "V3 did not authorize GPU smoke")
    return {
        "root_relative": SOURCE_ROOT_RELATIVE,
        "attempt_sha256": SOURCE_ATTEMPT_SHA256,
        "source_authority_sha256": SOURCE_AUTHORITY_SHA256,
        "terminal_sha256": SOURCE_TERMINAL_SHA256,
        "closure_sha256": terminal["closure_sha256"],
        "attempt": attempt,
        "authority": authority,
        "terminal": terminal,
    }


def _session_expected(authority: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = authority.get("posterior_rows")
    _require(isinstance(rows, list) and len(rows) == 27, "V3 posterior row topology drift")
    table = {str(row["session"]): row for row in rows}
    _require(len(table) == 27, "V3 posterior session duplication")
    return table


def _materialize_features(
    *, repository_root: Path, dm, authority: Mapping[str, object]
) -> tuple[dict, list[dict[str, object]]]:
    from mc_maze.multisession_datamodule import (
        list_datamodule_rewarded_trials,
        session_name_from_path,
    )
    from mc_maze.unit_side_features import _pool_trial_rate_matrix

    prior = source_prior_from_payload(authority["source_prior"])
    normalizer = posterior_normalizer_from_payload(authority["posterior_normalizer"])
    expected = _session_expected(authority)
    features = {}
    evidence = []
    for nwb_path in dm.session_files["train"]:
        path = Path(nwb_path)
        session = session_name_from_path(path)
        _require(session in expected and session in dm.train_dataset.sessions, f"unauthorized source session: {session}")
        trials = list_datamodule_rewarded_trials(
            path, bin_size_ms=20, window_size=50, trial_result_filter="R"
        )[:30]
        _require(len(trials) == 30, f"{session}: physical support cardinality drift")
        theta = np.asarray(
            [np.nan if row.get("target_dir") is None else float(row["target_dir"]) for row in trials],
            dtype=np.float64,
        )
        rates, units = _pool_trial_rate_matrix(path, trials)
        feature = build_session_posterior_features(
            session=session,
            trial_rates=rates,
            theta_radians=theta,
            source_prior=prior,
            posterior_normalizer=normalizer,
        )
        record = dm.train_dataset.sessions[session]
        _require(feature.unit_count == units == record.side_features.shape[0], f"{session}: unit axis drift")
        row_evidence = {"session": session, "budgets": {}}
        for budget in (4, 10, 30):
            expected_budget = expected[session]["budgets"][str(budget)]
            mask = np.isfinite(theta[:budget])
            fit = fit_posterior_mean(
                rates[:, :budget][:, mask], theta[:budget][mask], prior_variance=prior.variance
            )
            q_sha = array_sha256(angular_reliability(fit))
            _require(fit.raw_t4_sha256 == expected_budget["raw_t4_sha256"], f"{session} M{budget}: raw posterior drift")
            _require(q_sha == expected_budget["angular_reliability_sha256"], f"{session} M{budget}: q drift")
            row_evidence["budgets"][str(budget)] = {
                "physical_trial_count": budget,
                "usable_direction_count": int(mask.sum()),
                "raw_t4_sha256": fit.raw_t4_sha256,
                "angular_reliability_sha256": q_sha,
                "normalized_side_sha256": feature.normalized_side_sha256_by_budget[budget],
            }
        features[session] = feature
        evidence.append(row_evidence)
    _require(len(features) == 27, "materialized source roster is not strict-27")
    return features, evidence


def validate_smoke_result(
    *, stats: Mapping[str, object], realized_rows, dropout_summary: Mapping[str, object],
    parameters_finite: bool, optimizer_finite: bool,
) -> dict[str, object]:
    consumed = list(realized_rows[:SMOKE_STEPS])
    budgets = [int(row["budget"]) for row in consumed]
    counts = {budget: budgets.count(budget) for budget in (30, 10, 4)}
    checks = {
        "optimizer_steps_exact": stats.get("optimizer_steps") == SMOKE_STEPS,
        "visible_side_clean": stats.get("visible_side_violation_count") == 0,
        "loss_finite": stats.get("nonfinite_loss_steps") == 0,
        "gradients_finite": stats.get("nonfinite_grad_steps") == 0,
        "parameters_finite": bool(parameters_finite),
        "optimizer_finite": bool(optimizer_finite),
        "budget_prefix_exact": budgets == [
            (30, 10, 4)[index % 3] for index in range(SMOKE_STEPS)
        ],
        "budget_counts_exact": counts == EXPECTED_COUNTS,
        "dropout_forwards_exact": dropout_summary.get("n_forwards_with_sampled_p") == SMOKE_STEPS,
        "dropout_mask_calls_exact": dropout_summary.get("n_recorded_unit_mask_calls") == SMOKE_STEPS,
        "side_branch_gradient_nonzero": not bool(stats.get("w_side_grad_exact_zero_all_steps", True)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "budget_counts": counts,
        "consumed_schedule_sha256": tagged_batch_digest(consumed),
    }


def execute_reviewed(
    repository_root: Path,
    *, device_text: str = "cuda:0",
    initial_state_relative: str = (
        "tfpd_exploration/results/admission_arms_v1/canonical_initial_state.pt"
    ),
    seed: int = 42,
    train_batch_size: int = 32,
    num_workers: int = 4,
) -> Mapping[str, object]:
    repository_root = repository_root.resolve()
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "C2 smoke requires exact CUDA_VISIBLE_DEVICES=0")
    source = validate_source_authority(repository_root)
    closure = implementation_closure(repository_root)
    result_root = repository_root / RESULT_ROOT_RELATIVE
    _require(not result_root.exists() and not result_root.is_symlink(), "C2 smoke root is not fresh")
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
        "closure": closure,
        "smoke_steps": SMOKE_STEPS,
        "budget_cycle": [30, 10, 4],
        "environment_contract": {
            "cuda_visible_devices": "0", "device": device_text,
            "python_no_user_site": bool(__import__("sys").flags.no_user_site),
        },
        "data_boundary": {
            "source_opened_before_attempt": False,
            "within_opened": False, "external_opened": False, "formal_opened": False,
        },
        "target_optimizer_backward_update": 0,
    }
    attempt_sha = audit_v1._publish_pair(result_root, "attempt.json", attempt)
    try:
        import lightning.pytorch as pl
        import torch
        from torch.nn.parameter import UninitializedParameter
        from torch.utils.data import DataLoader
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
        pop_robust = stack["pop_robust"]

        class Args:
            pass

        args = Args()
        args.train_batch_size = train_batch_size
        args.num_workers = num_workers
        args.seed = seed
        dm, _a2 = arm_runner.build_datamodule(args)
        _require(len(dm.session_splits["train"]) == 27, "strict source roster drift")
        features, feature_evidence = _materialize_features(
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
        state_before = arm_common.state_sha256(model)
        _require(state_before == initial_payload["state_sha256"], "initial model state drift")
        model.to(device)

        base_dataset = dm.train_dataset
        base_sampler = SessionBatchSampler(
            base_dataset, batch_size=train_batch_size, shuffle=True, seed=seed
        )
        tagged_sampler = BudgetTaggedBatchSampler(base_sampler)
        dataset = BudgetMatchedPosteriorDataset(base_dataset, features)
        loader = DataLoader(
            dataset,
            batch_sampler=tagged_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=arm_common.ADAM_CONSTRUCTOR["lr"],
            betas=tuple(arm_common.ADAM_CONSTRUCTOR["betas"]),
            eps=arm_common.ADAM_CONSTRUCTOR["eps"],
            weight_decay=arm_common.ADAM_CONSTRUCTOR["weight_decay"],
            amsgrad=arm_common.ADAM_CONSTRUCTOR["amsgrad"],
        )
        lr_fn = lambda step: arm_common.lr_at_step(  # noqa: E731
            step, cal_plan.EPOCHS, cal_plan.STEPS_PER_EPOCH
        )
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with pop_robust.dynamic_dropout_recorder() as recorder:
            stats = arm_runner.train_epoch(
                model, optimizer, loader, "t4", lr_fn, device, 0,
                max_steps=SMOKE_STEPS,
            )
        wall_seconds = time.perf_counter() - started
        dropout_summary = pop_robust.summarize_dropout_record(recorder, 2)
        parameters_finite = all(
            bool(torch.isfinite(parameter.detach()).all().item())
            for parameter in model.parameters()
            if parameter.requires_grad
            and not isinstance(parameter, UninitializedParameter)
            and parameter.numel()
        )
        optimizer_finite = arm_runner.optimizer_state_finite(optimizer)
        verdict = validate_smoke_result(
            stats=stats,
            realized_rows=tagged_sampler.tagged_batches,
            dropout_summary=dropout_summary,
            parameters_finite=parameters_finite,
            optimizer_finite=optimizer_finite,
        )
        _require(len(tagged_sampler.tagged_batches) >= SMOKE_STEPS, "sampler did not realize 120 batches")
        final_closure = implementation_closure(repository_root)
        _require(final_closure["closure_sha256"] == closure["closure_sha256"], "C2 smoke closure drift")
        terminal = {
            "schema": SCHEMA + "_terminal",
            "status": (
                "C2_RUNTIME_SMOKE_PASSED__NON_PERFORMANCE"
                if verdict["passed"] else "C2_RUNTIME_SMOKE_FAILED"
            ),
            "attempt_sha256": attempt_sha,
            "source_authority_sha256": SOURCE_AUTHORITY_SHA256,
            "closure_sha256": closure["closure_sha256"],
            "smoke_steps": SMOKE_STEPS,
            "stats": dict(stats),
            "dropout_summary": dropout_summary,
            "verdict": verdict,
            "feature_evidence": feature_evidence,
            "model": {
                "initial_artifact_sha256": initial_sha,
                "state_before_sha256": state_before,
                "state_after_sha256": arm_common.state_sha256(model),
                "parameters_finite": parameters_finite,
                "optimizer_state_finite": optimizer_finite,
            },
            "resources": {
                "device": device_text,
                "gpu": gpu,
                "wall_seconds": wall_seconds,
                "steps_per_second": SMOKE_STEPS / wall_seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
            "pinned_sha256": pinned,
            "sealed_predecessors": sealed_predecessors,
            "scientific_disclosure": (
                "runtime/no-harm smoke only; it does not estimate C2 R2 or select a checkpoint"
            ),
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
            "terminal_sha256": terminal_sha,
            "steps_per_second": terminal["resources"]["steps_per_second"],
            "peak_cuda_reserved_bytes": terminal["resources"]["peak_cuda_reserved_bytes"],
        }
    except BaseException as error:
        audit_v1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure",
            "status": "C2_RUNTIME_SMOKE_FAILED",
            "attempt_sha256": attempt_sha,
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
        "source_authority_sha256": SOURCE_AUTHORITY_SHA256,
        "smoke_steps": SMOKE_STEPS,
        "budget_cycle": [30, 10, 4],
        "performance_claim": False,
    }
