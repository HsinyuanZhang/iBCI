"""Actual M4/M10/M30 original-SPINT B0 and Arm-A total-calibration baselines."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src import calibration_budget_comparators_v1 as compare


SCHEMA = "original_spint_short_budget_v2"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/original_spint_short_budget_v2"
FAILED_V1_ROOT_RELATIVE = "tfpd_exploration/results/original_spint_short_budget_v1"
FAILED_V1_ATTEMPT_SHA256 = "2df49433981f6e7dbe74e343021f55108a59a32f41ed6c880960c7f9f33b920b"
FAILED_V1_FAILURE_SHA256 = "04ed11fd2b15937af81b3e22069518ecada783be0681e127dfac48d89d968d71"
BUDGETS = (4, 10, 30)


class ShortBudgetError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ShortBudgetError(message)


def _forward_b0(*, model: Any, prepared: Any, budget: int, runtime: Any) -> tuple[float, str]:
    import numpy as np
    import torch
    from scripts.eval_adaptation_dandi688 import decode_last_behavior

    private = prepared.opaque
    before = runtime.state_digest(model)
    predictions: list[Any] = []
    digest = hashlib.sha256()
    starts = tuple(int(item) for item in private.starts.tolist())
    first_probe = True
    with torch.no_grad():
        for offset in range(0, len(starts), compare.EVAL_BATCH_SIZE):
            chunk = starts[offset:offset + compare.EVAL_BATCH_SIZE]
            neural = torch.from_numpy(np.stack([private.neural[start:start + 50] for start in chunk])).to("cuda:0")
            calibration = torch.from_numpy(private.calibration[:budget]).to("cuda:0")
            calibration = calibration.unsqueeze(0).expand(len(chunk), -1, -1, -1)
            key_features = model.decoder_key_features(None)
            output, _identity = model.student(
                neural, calib_trials=calibration, side_features=None,
                decoder_key_features=key_features, electrode_ids=None,
            )
            value = decode_last_behavior(output).reshape(len(chunk), 2).detach().cpu().contiguous()
            require(torch.isfinite(value).all().item(), "B0 short-budget output nonfinite")
            if first_probe:
                repeated, _identity = model.student(
                    neural, calib_trials=calibration, side_features=None,
                    decoder_key_features=key_features, electrode_ids=None,
                )
                repeated_value = decode_last_behavior(repeated).reshape(len(chunk), 2).detach().cpu().contiguous()
                require(torch.equal(value, repeated_value), "B0 short-budget repeated first batch drift")
                first_probe = False
            digest.update(value.numpy().tobytes())
            predictions.append(value)
    prediction = torch.cat(predictions).contiguous()
    target = torch.from_numpy(private.last_targets).contiguous()
    score = compare.session_r2_float32(prediction.numpy(), target.numpy())
    require(before == runtime.state_digest(model), "B0 short-budget model state changed")
    return score, digest.hexdigest()


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return compare._summary(rows)


def _validate_failed_v1(root: Path) -> dict[str, object]:
    directory = root / FAILED_V1_ROOT_RELATIVE
    require(directory.is_dir() and not directory.is_symlink(), "B0-short failed-v1 root drift")
    require({leaf.name for leaf in directory.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
    }, "B0-short failed-v1 topology drift")
    payloads: dict[str, object] = {}
    for name, digest in {
        "attempt.json": FAILED_V1_ATTEMPT_SHA256,
        "failure.json": FAILED_V1_FAILURE_SHA256,
    }.items():
        body = (directory / name).read_bytes()
        require(compare.sha256(body) == digest, f"B0-short failed-v1 {name} drift")
        require((directory / f"{name}.sha256").read_bytes()
                == f"{digest}  {name}\n".encode("ascii"),
                f"B0-short failed-v1 {name} sidecar drift")
        payloads[name] = json.loads(body)
    attempt = payloads["attempt.json"]
    failure = payloads["failure.json"]
    require(isinstance(attempt, dict) and attempt.get("status") == "ATTEMPT_RESERVED",
            "B0-short failed-v1 attempt semantics drift")
    require(isinstance(failure, dict) and failure.get("status") == "SHORT_BUDGET_BASELINES_FAILED",
            "B0-short failed-v1 failure semantics drift")
    require(failure.get("attempt_sha256") == FAILED_V1_ATTEMPT_SHA256
            and failure.get("target_optimizer_steps") == 0,
            "B0-short failed-v1 lineage/update drift")
    return {
        "root": FAILED_V1_ROOT_RELATIVE, "attempt_sha256": FAILED_V1_ATTEMPT_SHA256,
        "failure_sha256": FAILED_V1_FAILURE_SHA256,
        "reason": "legacy_within_B0_receipt_did_not_match_current_unified_input_forward",
    }


def execute_reviewed(root: Path) -> Mapping[str, object]:
    import torch
    from mc_maze import subm_b0_external_score_bridge as bridge
    from src import low_cost_calibration_v1 as lowcost
    from src.posterior_marginalized_cell_d_v1 import plan

    root = Path(root).absolute()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1" and os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
            "short-budget baseline launch environment drift")
    failed_v1 = _validate_failed_v1(root)
    result_root = root / RESULT_ROOT_RELATIVE
    require(not result_root.exists() and not result_root.is_symlink(), "short-budget result root is not fresh")
    result_root.mkdir(parents=False, mode=0o700)
    attempt_sha = compare._publish_pair(result_root, "attempt.json", {
        "schema": SCHEMA + "_attempt", "status": "ATTEMPT_RESERVED", "budgets": list(BUDGETS),
        "b0_seed": 42, "b0_checkpoint_epoch": 12, "arm_a_seed": 42,
        "formal_opened": False, "target_optimizer_steps": 0,
        "failed_v1_predecessor": failed_v1,
    })
    backend: Any | None = None
    try:
        profile = plan.validate_compatible_device_profile(plan.COMPATIBLE_DEVICE_PROFILES["gpu1"])
        backend, _identity, _authority = lowcost._build_reviewed_backend(root, profile)
        runtime = backend.base.runtime
        arm_a, arm_artifact = compare._strict_load_arm_a(root, runtime)
        audit = bridge.audit_b0_source_bundle(verify_checkpoint_payload=True)
        checkpoint_path = bridge.b0_run_dir(42) / "epoch_ckpts/epoch_011.ckpt"
        checkpoint = bridge._read_verified_bytes(
            checkpoint_path, "B0 short-budget seed42 epoch12",
            expected_sha256=audit["source_runs"]["42"]["source_checkpoint_sha256_bundle"]["12"],
        )
        teacher = bridge._read_verified_bytes(
            bridge.TEACHER, "B0 short-budget teacher", expected_sha256=bridge.EXPECTED_TEACHER_SHA256,
        )
        b0, consumption = bridge._load_b0_model_from_verified_source_bytes(
            checkpoint, teacher, device=torch.device("cuda:0"),
        )
        b0.eval()
        phase1_body = (root / compare.RESULT_ROOT_RELATIVE / "receipt.json").read_bytes()
        phase1_receipt = json.loads(phase1_body)
        phase1_arm = {
            (cell["surface"], int(cell["budget"])): cell
            for cell in phase1_receipt["cells"]
            if cell["system"] == "arm_a_ols" and cell["regime"] == "label_limited_m30_activity"
        }
        require(len(phase1_arm) == 6, "short-budget Phase-1 Arm-A authority drift")
        external = json.loads((root / "sua_exploration/results/subm_b0_external_score_bridge_v1/external_subject_M_b0_s42.json").read_bytes())
        within = json.loads((root / "sua_exploration/results/a11_b0_convergence_full_access_v1/a11_b0_convergence_full_access_v1_full_cpu_forward_2843108a665b53b4.json").read_bytes())
        b0_m30_reference = {
            "external": external["per_epoch"]["12"]["per_session_r2"],
            "within": within["cpu_forward_result"]["per_seed"]["42"]["per_epoch"]["11"]["per_session_r2"],
        }
        rows: dict[tuple[str, int, str], list[dict[str, object]]] = {}
        m30_historical_comparison: list[dict[str, object]] = []
        for surface in ("within", "external"):
            for session, prepared in backend.sessions(surface).items():
                private = prepared.opaque
                for budget in BUDGETS:
                    b0_r2, b0_sha = _forward_b0(model=b0, prepared=prepared, budget=budget, runtime=runtime)
                    rows.setdefault((surface, budget, "original_spint_b0_epoch12"), []).append({
                        "session": session, "r2": b0_r2, "prediction_sha256": b0_sha,
                        "n_windows": prepared.n_windows, "b3s_calibration_trials": budget,
                        "t4_label_trials": 0,
                    })
                    if budget == 30:
                        historical = float(b0_m30_reference[surface][session])
                        m30_historical_comparison.append({
                            "surface": surface, "session": session,
                            "current_matched_r2": b0_r2, "historical_r2": historical,
                            "delta_current_minus_historical": b0_r2 - historical,
                            "historical_used_as_gate": False,
                        })
                    arm_cell = phase1_arm[(surface, budget)]
                    arm_row = next(row for row in arm_cell["sessions"] if row["session"] == session)
                    # Recompute Arm A with B3S=M while retaining the already
                    # frozen OLS-M side from the Phase-1 input materialization.
                    side = private.side_by_budget[budget]
                    arm_r2, arm_sha, _ = compare._forward_neural_model(
                        runtime=runtime, model=arm_a, prepared=prepared, side=side,
                        calibration_trials=budget,
                    )
                    rows.setdefault((surface, budget, "arm_a_total_calibration"), []).append({
                        "session": session, "r2": arm_r2, "prediction_sha256": arm_sha,
                        "n_windows": prepared.n_windows, "b3s_calibration_trials": budget,
                        "phase1_arm_label_limited_r2": float(arm_row["r2"]),
                    })
        cells = [{"surface": surface, "budget": budget, "system": system, "sessions": values,
                  "summary": _summary(values)} for (surface, budget, system), values in rows.items()]
        require(len(cells) == 12, "short-budget baseline cell topology drift")
        receipt = {
            "schema": SCHEMA + "_receipt", "status": "SHORT_BUDGET_BASELINES_COMPLETE",
            "attempt_sha256": attempt_sha, "cells": cells, "arm_a_artifact": arm_artifact,
            "b0": {"seed": 42, "checkpoint_epoch": 12, "checkpoint_sha256": checkpoint.sha256,
                   "teacher_sha256": teacher.sha256, "consumption": consumption,
                   "current_matched_m30_authority_created": True,
                   "legacy_m30_parity_required": False},
            "phase1_receipt_sha256": compare.sha256(phase1_body),
            "failed_v1_predecessor": failed_v1,
            "m30_historical_comparison": m30_historical_comparison,
            "boundaries": {"fixed_query_after_trial30": True, "formal_opened": False,
                           "target_optimizer_steps": 0, "target_backward_calls": 0,
                           "target_update_calls": 0,
                           "current_matched_m30_is_governing": True,
                           "historical_m30_is_diagnostic_only": True},
        }
        receipt_sha = compare._publish_pair(result_root, "receipt.json", receipt)
        terminal_sha = compare._publish_pair(result_root, "terminal.json", {
            "schema": SCHEMA + "_terminal", "status": "TERMINAL",
            "attempt_sha256": attempt_sha, "receipt_sha256": receipt_sha, "formal_opened": False,
        })
        os.chmod(result_root, 0o555)
        return {"receipt_sha256": receipt_sha, "terminal_sha256": terminal_sha, "receipt": receipt}
    except BaseException as error:
        compare._publish_pair(result_root, "failure.json", {
            "schema": SCHEMA + "_failure", "status": "SHORT_BUDGET_BASELINES_FAILED",
            "attempt_sha256": attempt_sha, "error_class": type(error).__name__,
            "error_sha256": compare.sha256(f"{type(error).__name__}: {error}".encode()),
            "formal_opened": False, "target_optimizer_steps": 0,
        })
        raise
    finally:
        if backend is not None:
            backend.close()


def dry_plan() -> dict[str, object]:
    return {"schema": SCHEMA + "_plan", "status": "DRY_NO_DATA_NO_GPU_NO_WRITE_NO_SCORE",
            "budgets": list(BUDGETS), "systems": ["original_spint_b0_epoch12", "arm_a_total_calibration"],
            "fixed_query_after_trial30": True, "formal_opened": False, "target_optimizer_steps": 0}
