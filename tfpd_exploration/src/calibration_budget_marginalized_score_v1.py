"""Matched M4/M10/M30 score for the CBM-D Phase-2 system."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import calibration_budget_comparators_v1 as phase1
from src import calibration_budget_marginalized_cell_d_v1 as cbm


SCHEMA = "calibration_budget_marginalized_score_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_marginalized_score_v2"
FAILED_V1_ROOT_RELATIVE = "tfpd_exploration/results/calibration_budget_marginalized_score_v1"
FAILED_V1_ATTEMPT_SHA256 = "6711a285671ed729559c82602bfd0839f431232f50b0e93106455c6443bb5259"
FAILED_V1_FAILURE_SHA256 = "1d24b363656f2f91c0c217ce0c6d3af775aea816e8470976350fa2e2e9a9b63d"


class CBMScoreError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CBMScoreError(message)


def _read_pair(path: Path) -> tuple[bytes, str]:
    body = path.read_bytes()
    digest = phase1.sha256(body)
    sidecar = path.with_name(path.name + ".sha256").read_bytes()
    require(sidecar == f"{digest}  {path.name}\n".encode("ascii"), f"immutable pair drift: {path}")
    return body, digest


def _validate_failed_v1(root: Path) -> dict[str, object]:
    failed_root = root / FAILED_V1_ROOT_RELATIVE
    require(failed_root.is_dir() and not failed_root.is_symlink(), "CBM-score V1 root drift")
    require({path.name for path in failed_root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
    }, "CBM-score V1 failed topology drift")
    attempt_body, attempt_sha = _read_pair(failed_root / "attempt.json")
    failure_body, failure_sha = _read_pair(failed_root / "failure.json")
    require(attempt_sha == FAILED_V1_ATTEMPT_SHA256 and failure_sha == FAILED_V1_FAILURE_SHA256,
            "CBM-score V1 predecessor SHA drift")
    attempt = json.loads(attempt_body)
    failure = json.loads(failure_body)
    require(attempt.get("status") == "ATTEMPT_RESERVED"
            and failure.get("status") == "CBM_SCORE_FAILED"
            and failure.get("attempt_sha256") == attempt_sha
            and failure.get("error_class") == "CBMScoreError"
            and failure.get("formal_opened") is False
            and failure.get("target_optimizer_steps") == 0,
            "CBM-score V1 predecessor semantics drift")
    return {
        "root": FAILED_V1_ROOT_RELATIVE,
        "attempt_sha256": attempt_sha,
        "failure_sha256": failure_sha,
        "reason": "terminal aggregation estimator-name mismatch after complete inference",
        "scientific_result_published": False,
        "target_optimizer_steps": 0,
    }


def _load_cbm_model(root: Path, runtime: Any) -> tuple[Any, dict[str, object]]:
    import torch
    from torch.nn.parameter import UninitializedParameter

    train_root = root / cbm.RESULT_ROOT_RELATIVE
    terminal_body, terminal_sha = _read_pair(train_root / "terminal.json")
    terminal = json.loads(terminal_body)
    require(terminal.get("status") == "TERMINAL" and terminal.get("cell") == cbm.CELL,
            "CBM training terminal drift")
    swa_body, swa_sha = _read_pair(train_root / "swa_final4.pt")
    require(terminal.get("swa_sha256") == swa_sha, "CBM terminal/SWA binding drift")
    with torch.serialization.safe_globals([UninitializedParameter]):
        payload = torch.load(io.BytesIO(swa_body), map_location="cpu", weights_only=True)
    require(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping), "CBM SWA payload drift")
    runtime_modules = runtime._load_runtime()
    model = runtime_modules["pop_robust"].build_population_robustness_model(seed=42, cell="D")
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to("cuda:0").eval()
    require(not model.training and all(parameter.grad is None for parameter in model.parameters()), "CBM score eval boundary drift")
    return model, {
        "terminal_sha256": terminal_sha, "swa_sha256": swa_sha,
        "state_sha256": runtime.state_digest(model),
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return phase1._summary(rows)


def _contrasts(cbm_cells: Sequence[Mapping[str, object]], phase1_cells: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    import numpy as np

    results: list[dict[str, object]] = []
    for cell in cbm_cells:
        own = {str(row["session"]): float(row["r2"]) for row in cell["sessions"]}
        for base in phase1_cells:
            if base["surface"] != cell["surface"] or int(base["budget"]) != int(cell["budget"]):
                continue
            other = {str(row["session"]): float(row["r2"]) for row in base["sessions"]}
            require(tuple(own) == tuple(other), "CBM/Phase-1 paired roster drift")
            delta = np.asarray([own[name] - other[name] for name in own], dtype=np.float64)
            results.append({
                "surface": cell["surface"], "budget": cell["budget"],
                "cbm_regime": cell["regime"], "cbm_estimator": cell["estimator"],
                "cbm_support": cell["support"],
                "baseline_system": base["system"], "baseline_regime": base["regime"],
                "mean_delta_r2": float(delta.mean()), "median_delta_r2": float(np.median(delta)),
                "positive_sessions": int((delta > 0).sum()), "session_count": int(delta.size),
                "bootstrap_95": phase1.bootstrap_delta(delta.tolist()),
                "per_session": [{"session": name, "delta_r2": float(value)} for name, value in zip(own, delta)],
            })
    return results


def _gate(contrasts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    def one(surface: str, budget: int, cbm_regime: str, base_regime: str) -> Mapping[str, object]:
        rows = [row for row in contrasts if row["surface"] == surface and row["budget"] == budget
                and row["cbm_regime"] == cbm_regime and row["cbm_estimator"] == "ordinary_ols"
                and row["cbm_support"] == "chronological"
                and row["baseline_system"] == "cell_d_ols"
                and row["baseline_regime"] == base_regime]
        require(len(rows) == 1, "CBM gate contrast topology drift")
        return rows[0]

    m4 = one("external", 4, "total_calibration_limited", "total_calibration_limited")
    m10 = one("external", 10, "total_calibration_limited", "total_calibration_limited")
    m30 = one("external", 30, "total_calibration_limited", "total_calibration_limited")
    checks = {
        "external_m4_mean_delta_ge_0p05": float(m4["mean_delta_r2"]) >= 0.05,
        "external_m4_breadth_ge_10_of_15": int(m4["positive_sessions"]) >= 10,
        "external_m10_nonnegative": float(m10["mean_delta_r2"]) >= 0.0,
        "external_m30_not_below_minus_0p03": float(m30["mean_delta_r2"]) >= -0.03,
    }
    return {"decision": "PASS" if all(checks.values()) else "STOP", "checks": checks,
            "external_total_delta": {"M4": m4, "M10": m10, "M30": m30}}


def _matched_training_effects(
    cbm_cells: Sequence[Mapping[str, object]],
    factorial_cells: Sequence[Mapping[str, object]],
    phase1_cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare CBM with the exact predeclared deployment recipe at each M."""
    import numpy as np

    recipe = {
        4: ("doptimal_first30", "fixed_ridge_0p1"),
        10: ("chronological", "fixed_ridge_0p1"),
        30: ("chronological", "fixed_ridge_0p1"),
    }
    result: list[dict[str, object]] = []
    for surface in ("within", "external"):
        for budget, (support, estimator) in recipe.items():
            own = [cell for cell in cbm_cells if cell["surface"] == surface and cell["budget"] == budget
                   and cell["support"] == support and cell["estimator"] == estimator
                   and cell["regime"] == "total_calibration_limited"]
            require(len(own) == 1, "CBM matched-training own-cell topology drift")
            if budget in (4, 10):
                factorial_estimator = (
                    "ridge_fixed_0p1" if estimator == "fixed_ridge_0p1" else "ols"
                )
                base = [cell for cell in factorial_cells if cell["surface"] == surface
                        and cell["budget"] == budget and cell["support"] == support
                        and cell["estimator"] == factorial_estimator
                        and cell["regime"] == "total_selected_calibration"]
                authority = "protocol_factorial_v2"
            else:
                base = [cell for cell in phase1_cells if cell["surface"] == surface
                        and cell["budget"] == budget
                        and cell["system"] == "cell_d_ridge_t4_fixed_0p1"
                        and cell["regime"] == "label_limited_m30_activity"]
                authority = "phase1_fixed_ridge_m30"
            require(len(base) == 1, "CBM matched-training baseline topology drift")
            own_map = {str(row["session"]): float(row["r2"]) for row in own[0]["sessions"]}
            base_map = {str(row["session"]): float(row["r2"]) for row in base[0]["sessions"]}
            require(tuple(own_map) == tuple(base_map), "CBM matched-training roster drift")
            delta = np.asarray([own_map[name] - base_map[name] for name in own_map], dtype=np.float64)
            result.append({
                "surface": surface, "budget": budget, "support": support, "estimator": estimator,
                "regime": "total_calibration_limited", "baseline_authority": authority,
                "mean_delta_r2": float(delta.mean()), "median_delta_r2": float(np.median(delta)),
                "positive_sessions": int((delta > 0).sum()), "session_count": int(delta.size),
                "bootstrap_95": phase1.bootstrap_delta(delta.tolist()),
                "per_session": [{"session": name, "delta_r2": float(value)}
                                for name, value in zip(own_map, delta)],
            })
    return result


def _short_budget_system_contrasts(
    cbm_cells: Sequence[Mapping[str, object]],
    baseline_cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Descriptive paired comparisons to the sealed SPINT-B0/Arm-A curves.

    These are deliberately not used as treatment-attribution gates: unlike the
    recipe-matched Cell-D contrasts, the historical systems have different
    trained weights and identity mechanisms.  They complete the performance
    table while preserving that causal distinction in the receipt.
    """
    import numpy as np

    results: list[dict[str, object]] = []
    for cell in cbm_cells:
        if cell["regime"] != "total_calibration_limited":
            continue
        own = {str(row["session"]): float(row["r2"]) for row in cell["sessions"]}
        matches = [
            base for base in baseline_cells
            if base["surface"] == cell["surface"] and int(base["budget"]) == int(cell["budget"])
        ]
        require(len(matches) == 2, "short-budget baseline topology drift")
        for base in matches:
            other = {str(row["session"]): float(row["r2"]) for row in base["sessions"]}
            require(tuple(own) == tuple(other), "CBM/short-budget paired roster drift")
            delta = np.asarray([own[name] - other[name] for name in own], dtype=np.float64)
            results.append({
                "comparison_role": "descriptive_system_comparison_not_treatment_attribution",
                "surface": cell["surface"], "budget": cell["budget"],
                "cbm_regime": cell["regime"], "cbm_estimator": cell["estimator"],
                "cbm_support": cell["support"], "baseline_system": base["system"],
                "mean_delta_r2": float(delta.mean()), "median_delta_r2": float(np.median(delta)),
                "positive_sessions": int((delta > 0).sum()), "session_count": int(delta.size),
                "bootstrap_95": phase1.bootstrap_delta(delta.tolist()),
                "per_session": [{"session": name, "delta_r2": float(value)}
                                for name, value in zip(own, delta)],
            })
    return results


def _performance_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    external = {int(row["budget"]): row for row in rows if row["surface"] == "external"}
    require(set(external) == {4, 10, 30}, "CBM performance gate topology drift")
    checks = {
        "M4_predeclared_recipe_delta_ge_0p05": float(external[4]["mean_delta_r2"]) >= 0.05,
        "M4_breadth_ge_10_of_15": int(external[4]["positive_sessions"]) >= 10,
        "M10_predeclared_recipe_nonnegative": float(external[10]["mean_delta_r2"]) >= 0.0,
        "M30_predeclared_recipe_not_below_minus_0p03": float(external[30]["mean_delta_r2"]) >= -0.03,
    }
    return {"decision": "PASS" if all(checks.values()) else "STOP", "checks": checks,
            "external_predeclared_recipe": {f"M{budget}": external[budget] for budget in (4, 10, 30)}}


def execute_reviewed(root: Path) -> Mapping[str, object]:
    import numpy as np
    import torch
    from mc_maze import d_optimal_calibration_design as design
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
    from mc_maze.unit_side_features import _pool_trial_rate_matrix
    from src import low_cost_calibration_v1 as lowcost
    from src import calibration_budget_protocol_factorial_v1 as factorial
    from src import original_spint_short_budget_v1 as short_budget
    from src.posterior_marginalized_cell_d_v1 import plan

    root = Path(root).absolute()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
            "CBM score launch environment drift")
    failed_v1 = _validate_failed_v1(root)
    result_root = root / RESULT_ROOT_RELATIVE
    require(not result_root.exists() and not result_root.is_symlink(), "CBM score root is not fresh")
    phase1_body, phase1_sha = _read_pair(root / phase1.RESULT_ROOT_RELATIVE / "receipt.json")
    phase1_receipt = json.loads(phase1_body)
    require(phase1_receipt.get("status") == "PHASE1_COMPARATORS_COMPLETE", "Phase-1 authority incomplete")
    factorial_body, factorial_sha = _read_pair(
        root / factorial.RESULT_ROOT_RELATIVE / "receipt.json"
    )
    factorial_receipt = json.loads(factorial_body)
    require(factorial_receipt.get("status") == "PROTOCOL_FACTORIAL_COMPLETE",
            "protocol-factorial authority incomplete")
    short_body, short_sha = _read_pair(root / short_budget.RESULT_ROOT_RELATIVE / "receipt.json")
    short_receipt = json.loads(short_body)
    require(short_receipt.get("status") == "SHORT_BUDGET_BASELINES_COMPLETE"
            and short_receipt.get("phase1_receipt_sha256") == phase1_sha,
            "short-budget baseline authority incomplete")
    result_root.mkdir(parents=False, mode=0o700)
    attempt_sha = phase1._publish_pair(result_root, "attempt.json", {
        "schema": SCHEMA + "_attempt", "status": "ATTEMPT_RESERVED",
        "phase1_receipt_sha256": phase1_sha, "training_root": cbm.RESULT_ROOT_RELATIVE,
        "protocol_factorial_receipt_sha256": factorial_sha,
        "short_budget_receipt_sha256": short_sha,
        "failed_v1_predecessor": failed_v1,
        "formal_opened": False, "target_optimizer_steps": 0,
    })
    backend: Any | None = None
    try:
        profile = plan.validate_compatible_device_profile(plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
        backend, _identity, _authority = lowcost._build_reviewed_backend(root, profile)
        runtime = backend.base.runtime
        model, artifact = _load_cbm_model(root, runtime)
        state_before = runtime.state_digest(model)
        phase1_fixed_side = {
            (str(cell["surface"]), int(cell["budget"]), str(row["session"])):
                str(row["normalized_side_sha256"])
            for cell in phase1_receipt["cells"]
            if cell["system"] == "cell_d_ridge_t4_fixed_0p1"
            and cell["regime"] == "label_limited_m30_activity"
            for row in cell["sessions"]
        }
        require(len(phase1_fixed_side) == 21 * len(phase1.BUDGETS),
                "Phase-1 fixed-ridge side authority topology drift")
        mean = torch.tensor(plan.SEALED_OLS_T4_MEAN_FLOAT32, device="cuda:0")
        std = torch.tensor(plan.SEALED_OLS_T4_STD_FLOAT32, device="cuda:0")
        rows: dict[tuple[str, int, str, str, str], list[dict[str, object]]] = {}
        for surface in ("within", "external"):
            for session, prepared in backend.sessions(surface).items():
                private = prepared.opaque
                snapshot = private.held_asset.private_snapshot()
                try:
                    trials = list_datamodule_rewarded_trials(snapshot.path, bin_size_ms=20, window_size=50, trial_result_filter="R")
                    rates_ut, unit_count = _pool_trial_rate_matrix(snapshot.path, trials)
                    snapshot.reverify()
                finally:
                    snapshot.close()
                rates = np.ascontiguousarray(rates_ut.T, dtype=np.float64)
                theta = np.asarray([item["target_dir"] for item in trials], dtype=np.float64)
                directions = design.direction_indices_from_thetas(theta)
                times = np.asarray([item["start_time"] for item in trials], dtype=np.float64)
                require(unit_count == private.neural.shape[1], f"{session}: CBM score unit drift")
                for budget in phase1.BUDGETS:
                    support_names = ("chronological",) if budget == 30 else ("chronological", "doptimal_first30")
                    for support in support_names:
                        selected = (np.arange(budget, dtype=np.int64) if support == "chronological" else
                                    np.sort(design.greedy_forward_d_optimal_indices(theta[:30], budget)).astype(np.int64))
                        raw_ols = design.fit_carriers_from_selected_trials(rates, directions, selected)
                        canonical_theta = np.asarray(
                            [design.CANONICAL_DIRECTIONS_RAD[int(directions[index])] for index in selected],
                            dtype=np.float64,
                        )
                        raw_ridge, ridge_evidence = phase1.fit_ridge_t4(
                            rates[selected], canonical_theta,
                            normalized_lambda=phase1.RIDGE_T4_FIXED_LAMBDA,
                        )
                        estimators = (
                            ("ordinary_ols", raw_ols, None),
                            ("fixed_ridge_0p1", raw_ridge, ridge_evidence),
                        )
                        for estimator, raw, estimator_evidence in estimators:
                            side = ((torch.as_tensor(raw, dtype=torch.float32, device="cuda:0") - mean) / std).unsqueeze(0)
                            if support == "chronological" and estimator == "ordinary_ols" and budget in (4, 30):
                                require(torch.equal(side, private.side_by_budget[budget]), f"{session}/M{budget}: CBM OLS parity drift")
                            side_sha = phase1.array_sha256(side.cpu().numpy())
                            if support == "chronological" and estimator == "fixed_ridge_0p1":
                                require(side_sha == phase1_fixed_side[(surface, budget, session)],
                                        f"{session}/M{budget}: Phase-1 fixed-ridge side parity drift")
                            for regime, calibration_indices in (
                                ("label_limited_m30_activity", tuple(range(30))),
                                ("total_calibration_limited", tuple(int(index) for index in selected.tolist())),
                            ):
                                r2, prediction_sha = factorial._forward(
                                    runtime=runtime, model=model, prepared=prepared, side=side,
                                    calibration_indices=calibration_indices,
                                )
                                rows.setdefault((surface, budget, support, regime, estimator), []).append({
                                    "session": session, "r2": r2, "n_windows": prepared.n_windows,
                                    "prediction_sha256": prediction_sha,
                                    "normalized_side_sha256": side_sha,
                                    "b3s_calibration_indices": list(calibration_indices),
                                    "selected_indices": selected.tolist(),
                                    "selected_indices_sha256": phase1.array_sha256(selected),
                                    "estimator_evidence": estimator_evidence,
                                })
        require(runtime.state_digest(model) == state_before, "CBM score model state changed")
        cells = [{"surface": surface, "budget": budget, "system": "cbm_d", "regime": regime,
                  "support": support, "estimator": estimator,
                  "sessions": sessions, "summary": _summary(sessions)}
                 for (surface, budget, support, regime, estimator), sessions in rows.items()]
        require(len(cells) == 40, "CBM score cell topology drift")
        contrasts = _contrasts(cells, phase1_receipt["cells"])
        gate = _gate(contrasts)
        training_effects = _matched_training_effects(
            cells, factorial_receipt["cells"], phase1_receipt["cells"],
        )
        performance_gate = _performance_gate(training_effects)
        short_budget_contrasts = _short_budget_system_contrasts(cells, short_receipt["cells"])
        require(len(short_budget_contrasts) == 40,
                "CBM/short-budget descriptive comparison topology drift")
        receipt = {
            "schema": SCHEMA + "_receipt", "status": "CBM_MATCHED_SCORE_COMPLETE",
            "attempt_sha256": attempt_sha, "phase1_receipt_sha256": phase1_sha,
            "protocol_factorial_receipt_sha256": factorial_sha,
            "short_budget_receipt_sha256": short_sha,
            "failed_v1_predecessor": failed_v1,
            "model_artifact": artifact, "cells": cells, "paired_contrasts": contrasts,
            "decision_gate": gate, "phase1_historical_anchors": phase1_receipt["historical_anchors"],
            "matched_training_effects": training_effects,
            "short_budget_system_contrasts": short_budget_contrasts,
            "predeclared_performance_gate": performance_gate,
            "estimators": ["ordinary_ols", "fixed_ridge_0p1"],
            "supports": ["chronological", "doptimal_first30"],
            "comparison_roles": {
                "cell_d_same_recipe": "matched_training_effect",
                "cell_d_ordinary_ols": "mechanistic_attribution",
                "original_spint_b0_and_arm_a": "descriptive_system_comparison_not_treatment_attribution",
            },
            "boundaries": {"same_fixed_inputs": True, "formal_opened": False,
                           "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0},
        }
        receipt_sha = phase1._publish_pair(result_root, "receipt.json", receipt)
        terminal_sha = phase1._publish_pair(result_root, "terminal.json", {
            "schema": SCHEMA + "_terminal", "status": "TERMINAL", "attempt_sha256": attempt_sha,
            "receipt_sha256": receipt_sha, "decision": performance_gate["decision"],
            "ordinary_ols_attribution_decision": gate["decision"], "formal_opened": False,
        })
        os.chmod(result_root, 0o555)
        return {"receipt_sha256": receipt_sha, "terminal_sha256": terminal_sha, "receipt": receipt}
    except BaseException as error:
        phase1._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure", "status": "CBM_SCORE_FAILED", "attempt_sha256": attempt_sha,
            "error_class": type(error).__name__,
            "error_sha256": phase1.sha256(f"{type(error).__name__}: {error}".encode()),
            "formal_opened": False, "target_optimizer_steps": 0,
        })
        raise
    finally:
        if backend is not None:
            backend.close()


def dry_plan() -> dict[str, object]:
    return {"schema": SCHEMA + "_plan", "status": "DRY_NO_DATA_NO_GPU_NO_WRITE_NO_SCORE",
            "budgets": list(phase1.BUDGETS), "surfaces": ["within", "external"],
            "regimes": ["label_limited_m30_activity", "total_calibration_limited"],
            "estimators": ["ordinary_ols", "fixed_ridge_0p1"],
            "supports": ["chronological", "doptimal_first30"],
            "comparison_authorities": [
                phase1.RESULT_ROOT_RELATIVE,
                "tfpd_exploration/results/calibration_budget_protocol_factorial_v2",
                "tfpd_exploration/results/original_spint_short_budget_v2",
            ],
            "failed_v1_root": FAILED_V1_ROOT_RELATIVE,
            "cell_count": 40,
            "result_root": RESULT_ROOT_RELATIVE, "formal_opened": False, "target_optimizer_steps": 0}
