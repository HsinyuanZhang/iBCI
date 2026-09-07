"""Inference-only normalized ridge-T4 lambda curve at M4/M10."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import calibration_budget_comparators_v1 as phase1


SCHEMA = "ridge_t4_lambda_curve_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/ridge_t4_lambda_curve_v1"
BUDGETS = (4, 10)
LAMBDAS = (0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
REGIMES = (("label_limited_m30_activity", 30), ("total_calibration_limited", None))


class LambdaCurveError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LambdaCurveError(message)


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return phase1._summary(rows)


def _select_within(cells: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for budget in BUDGETS:
        for regime, _count in REGIMES:
            within = [cell for cell in cells if cell["surface"] == "within"
                      and cell["budget"] == budget and cell["regime"] == regime]
            require(len(within) == len(LAMBDAS), "ridge within selector topology drift")
            winner = max(within, key=lambda cell: (float(cell["summary"]["equal_session_mean_r2"]),
                                                    -float(cell["normalized_lambda"])))
            external = [cell for cell in cells if cell["surface"] == "external"
                        and cell["budget"] == budget and cell["regime"] == regime
                        and cell["normalized_lambda"] == winner["normalized_lambda"]]
            require(len(external) == 1, "ridge external locked-row topology drift")
            selected.append({
                "budget": budget, "regime": regime,
                "selected_by": "within_equal_session_mean_only",
                "normalized_lambda": winner["normalized_lambda"],
                "within_summary": winner["summary"], "external_summary": external[0]["summary"],
                "external_was_not_used_for_selection": True,
            })
    return selected


def execute_reviewed(root: Path) -> Mapping[str, object]:
    import numpy as np
    import torch
    from mc_maze import d_optimal_calibration_design as design
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix
    from src import low_cost_calibration_v1 as lowcost
    from src.posterior_marginalized_cell_d_v1 import plan

    root = Path(root).absolute()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
            and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "ridge curve launch environment drift")
    require(os.environ.get("SUBC_DATA_ROOT") == str(root / "sua_exploration/data/dandi_000688/sub-C"), "SUBC root drift")
    require(os.environ.get("SUBM_DATA_ROOT") == str(root / "sua_exploration/data/dandi_000688/sub-M"), "SUBM root drift")
    result_root = root / RESULT_ROOT_RELATIVE
    require(not result_root.exists() and not result_root.is_symlink(), "ridge curve root is not fresh")
    phase1_body = (root / phase1.RESULT_ROOT_RELATIVE / "receipt.json").read_bytes()
    phase1_receipt = json.loads(phase1_body)
    require(phase1_receipt.get("status") == "PHASE1_COMPARATORS_COMPLETE", "Phase-1 authority incomplete")
    phase1_fixed_side = {
        (str(cell["surface"]), int(cell["budget"]), str(row["session"])):
            str(row["normalized_side_sha256"])
        for cell in phase1_receipt["cells"]
        if cell["system"] == "cell_d_ridge_t4_fixed_0p1"
        and cell["regime"] == "label_limited_m30_activity"
        and int(cell["budget"]) in BUDGETS
        for row in cell["sessions"]
    }
    require(len(phase1_fixed_side) == 21 * len(BUDGETS), "Phase-1 fixed-ridge curve authority drift")
    result_root.mkdir(parents=False, mode=0o700)
    attempt_sha = phase1._publish_pair(result_root, "attempt.json", {
        "schema": SCHEMA + "_attempt", "status": "ATTEMPT_RESERVED",
        "phase1_receipt_sha256": phase1.sha256(phase1_body), "budgets": list(BUDGETS),
        "normalized_lambda_grid": list(LAMBDAS), "formal_opened": False, "target_optimizer_steps": 0,
    })
    backend: Any | None = None
    try:
        profile = plan.validate_compatible_device_profile(plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
        backend, _identity, _authority = lowcost._build_reviewed_backend(root, profile)
        runtime = backend.base.runtime
        model = backend.sealed_model()
        state_before = runtime.state_digest(model)
        mean = torch.tensor(plan.SEALED_OLS_T4_MEAN_FLOAT32, dtype=torch.float32, device="cuda:0")
        std = torch.tensor(plan.SEALED_OLS_T4_STD_FLOAT32, dtype=torch.float32, device="cuda:0")
        rows: dict[tuple[str, int, str, float], list[dict[str, object]]] = {}
        for surface in ("within", "external"):
            for session, prepared in backend.sessions(surface).items():
                private = prepared.opaque
                snapshot = private.held_asset.private_snapshot()
                try:
                    trials = list_datamodule_rewarded_trials(snapshot.path, bin_size_ms=20, window_size=50,
                                                             trial_result_filter="R")
                    theta_all = np.asarray([item["target_dir"] for item in trials], dtype=np.float64)
                    directions = design.direction_indices_from_thetas(theta_all)
                    rates_ut, unit_count = _pool_trial_rate_matrix(snapshot.path, trials)
                    snapshot.reverify()
                finally:
                    snapshot.close()
                require(unit_count == private.neural.shape[1], f"{session}: ridge curve unit drift")
                rates = np.ascontiguousarray(rates_ut.T, dtype=np.float64)
                for budget in BUDGETS:
                    canonical_theta = np.asarray(
                        [design.CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions[:budget]],
                        dtype=np.float64,
                    )
                    for value in LAMBDAS:
                        raw, evidence = phase1.fit_ridge_t4(
                            rates[:budget], canonical_theta, normalized_lambda=value,
                        )
                        side = ((torch.as_tensor(raw, dtype=torch.float32, device="cuda:0") - mean) / std).unsqueeze(0)
                        side_sha = phase1.array_sha256(side.cpu().numpy())
                        if value == phase1.RIDGE_T4_FIXED_LAMBDA:
                            require(side_sha == phase1_fixed_side[(surface, budget, session)],
                                    f"{session}/M{budget}: Phase-1 lambda-0.1 side parity drift")
                        for regime, fixed_count in REGIMES:
                            calibration_count = budget if fixed_count is None else fixed_count
                            r2, prediction_sha, _ = phase1._forward_neural_model(
                                runtime=runtime, model=model, prepared=prepared, side=side,
                                calibration_trials=calibration_count,
                            )
                            rows.setdefault((surface, budget, regime, value), []).append({
                                "session": session, "r2": r2, "n_windows": prepared.n_windows,
                                "prediction_sha256": prediction_sha,
                                "normalized_side_sha256": side_sha,
                                "fit": evidence, "b3s_calibration_trials": calibration_count,
                            })
        require(runtime.state_digest(model) == state_before, "ridge curve model state changed")
        cells = [{
            "surface": surface, "budget": budget, "system": "cell_d_ridge_t4_fixed_curve",
            "regime": regime, "normalized_lambda": value, "sessions": session_rows,
            "summary": _summary(session_rows),
        } for (surface, budget, regime, value), session_rows in rows.items()]
        require(len(cells) == 2 * len(BUDGETS) * len(REGIMES) * len(LAMBDAS), "ridge curve cell topology drift")
        receipt = {
            "schema": SCHEMA + "_receipt", "status": "RIDGE_T4_LAMBDA_CURVE_COMPLETE",
            "attempt_sha256": attempt_sha, "phase1_receipt_sha256": phase1.sha256(phase1_body),
            "cells": cells, "within_selected_external_locked": _select_within(cells),
            "boundaries": {"external_not_used_for_selection": True, "formal_opened": False,
                           "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0},
        }
        receipt_sha = phase1._publish_pair(result_root, "receipt.json", receipt)
        terminal_sha = phase1._publish_pair(result_root, "terminal.json", {
            "schema": SCHEMA + "_terminal", "status": "TERMINAL", "attempt_sha256": attempt_sha,
            "receipt_sha256": receipt_sha, "formal_opened": False,
        })
        os.chmod(result_root, 0o555)
        return {"receipt_sha256": receipt_sha, "terminal_sha256": terminal_sha, "receipt": receipt}
    except BaseException as error:
        phase1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure", "status": "RIDGE_T4_LAMBDA_CURVE_FAILED",
            "attempt_sha256": attempt_sha, "error_class": type(error).__name__,
            "error_sha256": phase1.sha256(f"{type(error).__name__}: {error}".encode()),
            "formal_opened": False, "target_optimizer_steps": 0,
        })
        raise
    finally:
        if backend is not None:
            backend.close()


def dry_plan() -> dict[str, object]:
    return {"schema": SCHEMA + "_plan", "status": "DRY_NO_DATA_NO_GPU_NO_WRITE_NO_SCORE",
            "budgets": list(BUDGETS), "normalized_lambda_grid": list(LAMBDAS),
            "regimes": [item[0] for item in REGIMES], "external_not_used_for_selection": True,
            "formal_opened": False, "target_optimizer_steps": 0, "result_root": RESULT_ROOT_RELATIVE}
