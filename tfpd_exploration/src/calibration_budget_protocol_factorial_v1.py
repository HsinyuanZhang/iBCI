"""Causal M4/M10 support x estimator x B3S-view factorial on sealed Cell D."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import calibration_budget_comparators_v1 as compare


SCHEMA = "calibration_budget_protocol_factorial_v2"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_protocol_factorial_v2"
FAILED_V1_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_protocol_factorial_v1"
FAILED_V1_ATTEMPT_SHA256 = "818fa9d929e1c165573eaa66ea4405e5bc4a899e622f0e9669420fa02065c0be"
FAILED_V1_FAILURE_SHA256 = "4436fdaed05b14056f5ee8cbf84cfc7fca775796482a7f0f8ee48ce07cd627c7"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_M4_M10_PROTOCOL_FACTORIAL_20260824.md"
BUDGETS = (4, 10)
SUPPORTS = ("chronological", "doptimal_first30")
ESTIMATORS = ("ols", "ridge_fixed_0p1")
REGIMES = ("label_limited_m30_activity", "total_selected_calibration")


class FactorialError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FactorialError(message)


def dry_plan() -> dict[str, object]:
    return {
        "schema": SCHEMA + "_plan", "status": "DRY_NO_DATA_NO_GPU_NO_WRITE_NO_SCORE",
        "budgets": list(BUDGETS), "supports": list(SUPPORTS),
        "estimators": list(ESTIMATORS), "regimes": list(REGIMES),
        "target_optimizer_steps": 0, "query_after_trial30": True,
        "failed_v1_predecessor": {
            "attempt_sha256": FAILED_V1_ATTEMPT_SHA256,
            "failure_sha256": FAILED_V1_FAILURE_SHA256,
            "reason": "raw_OLS_float64_was_not_cast_to_CellD_float32",
        },
    }


def normalize_raw_t4(raw: Any, *, mean: Any, std: Any, torch_module: Any) -> Any:
    import numpy as np

    tensor = torch_module.as_tensor(
        np.ascontiguousarray(raw, dtype=np.float32),
        device=mean.device, dtype=torch_module.float32,
    )
    require(tensor.ndim == 2 and tensor.shape[1] == 4, "factorial raw T4 shape drift")
    result = ((tensor - mean) / std).contiguous().unsqueeze(0)
    require(result.dtype == torch_module.float32 and torch_module.isfinite(result).all().item(),
            "factorial normalized T4 dtype/nonfinite drift")
    return result


def _forward(
    *, runtime: Any, model: Any, prepared: Any, side: Any,
    calibration_indices: Sequence[int],
) -> tuple[float, str]:
    import numpy as np
    import torch
    from src.posterior_marginalized_cell_d_v1 import matched_score_physical as v1p
    from src.tfpd_lane import pop_robust

    private = prepared.opaque
    selected = tuple(int(index) for index in calibration_indices)
    require(selected and len(selected) <= 30 and len(set(selected)) == len(selected),
            "factorial calibration selection drift")
    require(min(selected) >= 0 and max(selected) < 30, "factorial support crosses query boundary")
    state_before = runtime.state_digest(model)
    predictions: list[Any] = []
    targets: list[Any] = []
    digest = hashlib.sha256()
    starts = tuple(int(item) for item in private.starts.tolist())
    for offset in range(0, len(starts), compare.EVAL_BATCH_SIZE):
        chunk = starts[offset:offset + compare.EVAL_BATCH_SIZE]
        neural = torch.from_numpy(np.stack([private.neural[start:start + 50] for start in chunk])).to("cuda:0")
        behavior = torch.from_numpy(np.stack([private.behavior[start:start + 50] for start in chunk])).to("cuda:0")
        calibration = torch.from_numpy(np.ascontiguousarray(private.calibration[list(selected)])).to("cuda:0")
        calibration = calibration.unsqueeze(0).expand(len(chunk), -1, -1, -1)
        expanded_side = side.expand(len(chunk), -1, -1)
        with pop_robust.dynamic_dropout_recorder() as recorder:
            with torch.no_grad():
                output, _identity = model(neural, calib_trials=calibration, side_features=expanded_side)
        require(recorder["uniform_calls"] == 0 and recorder["dropout_calls"] == [],
                "factorial eval dropout became active")
        require(tuple(output.shape) == (len(chunk), 50, 2) and torch.isfinite(output).all().item(),
                "factorial output drift")
        cpu = output.detach().cpu().contiguous()
        digest.update(cpu.numpy().tobytes())
        predictions.append(cpu[:, 49, :])
        targets.append(behavior.detach().cpu().contiguous()[:, 49, :])
    prediction = torch.cat(predictions).contiguous()
    target = torch.cat(targets).contiguous()
    result = v1p.ForwardResult(
        input_token_sha256=prepared.input_token_sha256,
        prediction_sha256=digest.hexdigest(), predictions=None, targets=None, valid_mask=None,
        output_shape=(prepared.n_windows, 50, 2), governed_bin=49,
        last_bin_predictions=prediction, last_bin_targets=target,
        last_bin_valid_mask=torch.ones(prediction.shape[0], dtype=torch.bool),
        b3s_m30_recomputed=(selected == tuple(range(30))),
    )
    score = float(runtime.score_result(result, session=prepared))
    require(runtime.state_digest(model) == state_before, "factorial model state changed")
    return score, digest.hexdigest()


def _closure(root: Path) -> dict[str, object]:
    paths = (
        WORKORDER_RELATIVE,
        "tfpd_exploration/src/calibration_budget_protocol_factorial_v1.py",
        "tfpd_exploration/tests/test_calibration_budget_protocol_factorial_v1.py",
        "tfpd_exploration/src/calibration_budget_comparators_v1.py",
        "tfpd_exploration/src/low_cost_calibration_v1.py",
        "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    )
    rows = {path: compare.sha256((root / path).read_bytes()) for path in paths}
    payload: dict[str, object] = {"schema": SCHEMA + "_closure", "sha256_by_path": rows}
    payload["closure_sha256"] = compare.sha256(compare.json_bytes(payload))
    return payload


def _paired_contrasts(cells: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    import numpy as np

    result: list[dict[str, object]] = []
    for surface in ("within", "external"):
        for budget in BUDGETS:
            bases = [
                cell for cell in cells
                if cell["surface"] == surface and cell["budget"] == budget
                and cell["support"] == "chronological" and cell["estimator"] == "ols"
                and cell["regime"] == "label_limited_m30_activity"
            ]
            require(len(bases) == 1, "factorial paired base drift")
            base = {str(row["session"]): float(row["r2"]) for row in bases[0]["sessions"]}
            for cell in cells:
                if cell["surface"] != surface or cell["budget"] != budget or cell is bases[0]:
                    continue
                other = {str(row["session"]): float(row["r2"]) for row in cell["sessions"]}
                require(tuple(other) == tuple(base), "factorial paired roster/order drift")
                delta = np.asarray([other[name] - base[name] for name in base], dtype=np.float64)
                result.append({
                    "surface": surface, "budget": budget,
                    "contrast": (
                        f"{cell['support']}__{cell['estimator']}__{cell['regime']}"
                        "__minus_chronological_ols_label_limited"
                    ),
                    "mean_delta_r2": float(delta.mean()),
                    "median_delta_r2": float(np.median(delta)),
                    "positive_sessions": int((delta > 0).sum()),
                    "session_count": int(delta.size),
                    "bootstrap_95": compare.bootstrap_delta(delta.tolist()),
                    "per_session": [
                        {"session": name, "delta_r2": float(value)}
                        for name, value in zip(base, delta, strict=True)
                    ],
                })
    return result


def _validate_failed_v1(root: Path) -> dict[str, object]:
    directory = root / FAILED_V1_ROOT_RELATIVE
    require(directory.is_dir() and not directory.is_symlink(), "factorial failed-v1 root drift")
    require({leaf.name for leaf in directory.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
    }, "factorial failed-v1 topology drift")
    payloads: dict[str, object] = {}
    for name, digest in {
        "attempt.json": FAILED_V1_ATTEMPT_SHA256,
        "failure.json": FAILED_V1_FAILURE_SHA256,
    }.items():
        body = (directory / name).read_bytes()
        require(compare.sha256(body) == digest, f"factorial failed-v1 {name} body drift")
        require((directory / f"{name}.sha256").read_bytes()
                == f"{digest}  {name}\n".encode("ascii"),
                f"factorial failed-v1 {name} sidecar drift")
        payloads[name] = json.loads(body)
    attempt = payloads["attempt.json"]
    failure = payloads["failure.json"]
    require(isinstance(attempt, dict) and attempt.get("status") == "ATTEMPT_RESERVED",
            "factorial failed-v1 attempt semantics drift")
    require(isinstance(failure, dict) and failure.get("status") == "FAILED"
            and failure.get("attempt_sha256") == FAILED_V1_ATTEMPT_SHA256
            and failure.get("target_optimizer_steps") == 0,
            "factorial failed-v1 failure lineage drift")
    return {
        "root": FAILED_V1_ROOT_RELATIVE, "attempt_sha256": FAILED_V1_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V1_FAILURE_SHA256,
        "status": "VALIDATED_NO_UPDATE_FAILURE",
    }


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
            and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID", "factorial GPU environment drift")
    require(os.environ.get("SUBC_DATA_ROOT") == str(root / "sua_exploration/data/dandi_000688/sub-C"),
            "factorial SUBC root drift")
    require(os.environ.get("SUBM_DATA_ROOT") == str(root / "sua_exploration/data/dandi_000688/sub-M"),
            "factorial SUBM root drift")
    result_root = root / RESULT_ROOT_RELATIVE
    failed_v1 = _validate_failed_v1(root)
    require(not result_root.exists() and not result_root.is_symlink(), "factorial result root is not fresh")
    result_root.mkdir(parents=False, mode=0o700)
    closure = _closure(root)
    phase1_body = (root / compare.RESULT_ROOT_RELATIVE / "receipt.json").read_bytes()
    attempt_sha = compare._publish_pair(result_root, "attempt.json", {
        "schema": SCHEMA + "_attempt", "status": "ATTEMPT_RESERVED", "closure": closure,
        "phase1_receipt_sha256": compare.sha256(phase1_body), "target_optimizer_steps": 0,
        "formal_opened": False, "failed_v1_predecessor": failed_v1,
    })
    backend: Any | None = None
    try:
        profile = plan.validate_compatible_device_profile(plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
        backend, _identity, _authority = lowcost._build_reviewed_backend(root, profile)
        runtime = backend.base.runtime
        model = backend.sealed_model()
        state_before = runtime.state_digest(model)
        mean = torch.tensor(plan.SEALED_OLS_T4_MEAN_FLOAT32, device="cuda:0", dtype=torch.float32)
        std = torch.tensor(plan.SEALED_OLS_T4_STD_FLOAT32, device="cuda:0", dtype=torch.float32)
        rows_by_cell: dict[tuple[str, int, str, str, str], list[dict[str, object]]] = {}
        for surface in ("within", "external"):
            for session, prepared in backend.sessions(surface).items():
                private = prepared.opaque
                snapshot = private.held_asset.private_snapshot()
                try:
                    trials = list_datamodule_rewarded_trials(
                        snapshot.path, bin_size_ms=20, window_size=50, trial_result_filter="R",
                    )
                    theta = np.asarray([row["target_dir"] for row in trials], dtype=np.float64)
                    rates_units_trials, unit_count = _pool_trial_rate_matrix(snapshot.path, trials)
                    rates = np.ascontiguousarray(rates_units_trials.T, dtype=np.float64)
                    snapshot.reverify()
                finally:
                    snapshot.close()
                require(unit_count == private.neural.shape[1] and rates.shape[0] >= 30,
                        f"{session}: factorial source topology drift")
                direction = design.direction_indices_from_thetas(theta)
                for budget in BUDGETS:
                    selections = {
                        "chronological": np.arange(budget, dtype=np.int64),
                        "doptimal_first30": np.sort(
                            design.greedy_forward_d_optimal_indices(theta[:30], budget)
                        ).astype(np.int64),
                    }
                    for support, selected in selections.items():
                        raw_ols = design.fit_carriers_from_selected_trials(rates, direction, selected)
                        canonical_theta = np.asarray(
                            [design.CANONICAL_DIRECTIONS_RAD[int(direction[index])] for index in selected],
                            dtype=np.float64,
                        )
                        raw_ridge, fit = compare.fit_ridge_t4(
                            rates[selected], canonical_theta,
                            normalized_lambda=compare.RIDGE_T4_FIXED_LAMBDA,
                        )
                        sides = {
                            "ols": normalize_raw_t4(raw_ols, mean=mean, std=std, torch_module=torch),
                            "ridge_fixed_0p1": normalize_raw_t4(
                                raw_ridge, mean=mean, std=std, torch_module=torch,
                            ),
                        }
                        for estimator, side in sides.items():
                            for regime in REGIMES:
                                calibration_indices = (
                                    tuple(range(30)) if regime == "label_limited_m30_activity"
                                    else tuple(int(index) for index in selected.tolist())
                                )
                                score, prediction_sha = _forward(
                                    runtime=runtime, model=model, prepared=prepared, side=side,
                                    calibration_indices=calibration_indices,
                                )
                                key = (surface, budget, support, estimator, regime)
                                rows_by_cell.setdefault(key, []).append({
                                    "session": session, "r2": score, "n_windows": prepared.n_windows,
                                    "prediction_sha256": prediction_sha,
                                    "selected_indices": selected.tolist(),
                                    "selected_indices_sha256": compare.array_sha256(selected),
                                    "selected_trials_precede_query": bool(int(selected.max()) < 30),
                                    "b3s_calibration_indices": list(calibration_indices),
                                    "ridge_fit": fit if estimator == "ridge_fixed_0p1" else None,
                                })
        require(runtime.state_digest(model) == state_before, "factorial final model state drift")
        cells = [
            {"surface": surface, "budget": budget, "support": support,
             "estimator": estimator, "regime": regime, "sessions": rows,
             "summary": compare._summary(rows)}
            for (surface, budget, support, estimator, regime), rows in rows_by_cell.items()
        ]
        require(len(cells) == 32, f"factorial cell topology drift: {len(cells)}")
        receipt = {
            "schema": SCHEMA + "_receipt", "status": "PROTOCOL_FACTORIAL_COMPLETE",
            "attempt_sha256": attempt_sha, "closure": closure, "cells": cells,
            "phase1_receipt_sha256": compare.sha256(phase1_body),
            "paired_contrasts": _paired_contrasts(cells),
            "boundaries": {"formal_opened": False, "target_optimizer_steps": 0,
                           "target_backward_calls": 0, "target_update_calls": 0,
                           "all_selected_trials_precede_fixed_query": True},
        }
        receipt_sha = compare._publish_pair(result_root, "receipt.json", receipt)
        terminal_sha = compare._publish_pair(result_root, "terminal.json", {
            "schema": SCHEMA + "_terminal", "status": "TERMINAL",
            "attempt_sha256": attempt_sha, "receipt_sha256": receipt_sha,
            "formal_opened": False, "target_optimizer_steps": 0,
        })
        os.chmod(result_root, 0o555)
        return {"receipt_sha256": receipt_sha, "terminal_sha256": terminal_sha, "receipt": receipt}
    except BaseException as error:
        compare._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure", "status": "FAILED",
            "attempt_sha256": attempt_sha, "error_class": type(error).__name__,
            "error_sha256": compare.sha256(f"{type(error).__name__}: {error}".encode()),
            "formal_opened": False, "target_optimizer_steps": 0,
        })
        os.chmod(result_root, 0o555)
        raise
    finally:
        if backend is not None:
            backend.close()
