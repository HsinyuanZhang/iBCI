#!/usr/bin/env python3
"""Prepare-only CPU audit for RT R4 carrier-budget response.

No NWB, outer evaluator, Trainer, CUDA context, or checkpoint is opened.  The
audit binds existing immutable q24 split manifests, the sealed M24 Full-MB4
aggregate, the earlier descriptor-reliability receipt, and the exact Hydra
configs/commands for a fixed three-fold pilot.  It never launches those
commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.rt_r4_budget_response_datamodule import (
    COMMON_QUERY_START,
    FIXED_ACTIVITY_BUDGET,
    PRIMARY_CARRIER_BUDGETS,
    R4_ARMS,
)


SCHEMA = "rt_r4_budget_response_common_q24_prepare_v1"
STATUS = "PASS_RT_R4_COMMON_Q24_SOURCE_ONLY_PREPARED_NOT_LAUNCHED"
PILOT_FOLDS = (0, 7, 14)
ALL_FOLDS = tuple(range(15))
EXPANSION_FOLDS = tuple(fold for fold in ALL_FOLDS if fold not in PILOT_FOLDS)
SEED = 42
EXPERIMENT = "rt_r4_budget_response_common_q24"
RELIABILITY_RECEIPT = (
    ROOT / "outputs/rt_k4_reliability_audit/"
    "rt_afc4_calibration_reliability_m6_m12_m18_m24_receipt.json"
)
M24_AGGREGATE = (
    WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/"
    "RT_MB4_MATCHED_FULL15_AGGREGATE_v1.json"
)
M24_MB4_IMPORT_ROOT = (
    WORKSPACE / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/"
    "imported_cells/afc4_mb4"
)
DEFAULT_OUTPUT = (
    WORKSPACE / "sua_exploration/results/rt_r4_budget_response_common_q24_v1/"
    "RT_R4_COMMON_Q24_SOURCE_ONLY_PREPARE_v1.json"
)


class R4PrepareError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise R4PrepareError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, *, immutable: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    _need(path.is_file() and not path.is_symlink(), f"missing/non-regular receipt: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if immutable:
        _need(mode == 0o444, f"receipt must be immutable mode0444: {path}")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R4PrepareError(f"invalid JSON receipt: {path}") from error
    _need(isinstance(body, dict), f"receipt is not a JSON object: {path}")
    return body, {"path": str(path), "sha256": _sha256(path), "size": path.stat().st_size,
                  "mode": f"{mode:04o}"}


def _compose(*, budget: int, arm: str, fold: int = 0) -> dict[str, Any]:
    command = [
        sys.executable, str(ROOT / "src/train.py"), f"experiment={EXPERIMENT}",
        f"data.side_feature_calibration_n_trials={budget}",
        f"data.side_feature_group={arm}", f"data.loso_fold={fold}",
        f"data.outer_loso_fold={fold}", f"seed={SEED}", "ckpt_path=null",
        "train=true", "test=false", "trainer.accelerator=cpu", "trainer.devices=1",
        "--cfg", "job",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    _need(result.returncode == 0,
          f"R4 Hydra composition failed for M{budget}/{arm}:\n{result.stdout}\n{result.stderr}")
    cfg = OmegaConf.create(result.stdout)
    values = OmegaConf.to_container(cfg, resolve=False)
    _need(isinstance(values, dict) and isinstance(values.get("data"), dict),
          "R4 Hydra composition did not return a data mapping")
    data = values["data"]
    data["calibration_n_trials"] = int(cfg.data.calibration_n_trials)
    data["side_feature_calibration_n_trials"] = int(
        cfg.data.side_feature_calibration_n_trials
    )
    data["query_start_trial"] = int(cfg.data.query_start_trial)
    data["outer_loso_fold"] = int(cfg.data.outer_loso_fold)
    return values


def _audit_composed_config(cfg: Mapping[str, Any], *, budget: int, arm: str) -> None:
    data, model, trainer = cfg.get("data"), cfg.get("model"), cfg.get("trainer")
    _need(isinstance(data, Mapping) and isinstance(model, Mapping)
          and isinstance(trainer, Mapping), "R4 config lacks data/model/trainer")
    _need(data.get("_target_") ==
          "src.data.rt_r4_budget_response_datamodule.RtR4BudgetResponseNestedLossoDataModule",
          "R4 config does not use the isolated budget-response DataModule")
    _need(data.get("calibration_n_trials") == FIXED_ACTIVITY_BUDGET
          and data.get("side_feature_calibration_n_trials") == budget
          and data.get("query_start_trial") == COMMON_QUERY_START,
          "R4 config did not separate fixed activity M24 from the carrier prefix")
    _need(data.get("rt_r4_common_query_start") is True
          and data.get("allow_conditional_m18") is False,
          "R4 primary config lacks the explicit common-q24/primary-budget guard")
    _need(data.get("side_feature_group") == arm and arm in R4_ARMS,
          "R4 config arm is not the matched Full/MB4 pair")
    _need(data.get("random_calibration") is False and data.get("smooth_calibration") is False,
          "R4 config is not chronological raw support")
    _need(cfg.get("seed") == SEED and cfg.get("train") is True and cfg.get("test") is False,
          "R4 config seed/train/test contract drift")
    _need(trainer.get("max_epochs") == 35 and model.get("freeze_decoder") is False
          and model.get("loss_mode") == "task_only",
          "R4 model/epoch contract differs from matched M24")
    callback = cfg.get("callbacks", {}).get("rt_nested_selection_receipt", {})
    _need(callback.get("monitor") == "val_heldin/r2_mean",
          "R4 checkpoint rule is not inner-validation val_heldin/r2_mean")


def _audit_reliability(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    body, row = _json(path, immutable=False)
    _need(body.get("schema") == "rt_afc4_calibration_reliability_audit_v1"
          and body.get("status") == "PASS__CPU_DIAGNOSTIC_ONLY__NO_GPU_NO_MODEL_NO_FORMAL_TEST",
          "R4 reliability receipt schema/status drift")
    aggregate = body.get("aggregate_by_budget")
    per_session = body.get("per_session")
    _need(isinstance(aggregate, Mapping) and isinstance(per_session, Mapping)
          and len(per_session) == 15, "R4 reliability receipt lacks 15 sessions")
    summary: dict[str, Any] = {}
    for budget in (*PRIMARY_CARRIER_BUDGETS, FIXED_ACTIVITY_BUDGET):
        entry = aggregate.get(str(budget), {})
        _need(entry.get("sessions") == 15 and entry.get("full_prefix_defined") == 15,
              f"R4 M{budget} descriptor is not defined on all 15 sessions")
        for name, session in per_session.items():
            fit = session.get("budgets", {}).get(str(budget), {}).get("full_prefix", {}).get("fit", {})
            _need(fit.get("status") == "defined",
                  f"R4 M{budget} full-prefix descriptor is undefined for {name}")
        direction = entry.get("chronological_w_direction_cosine_median", {})
        summary[str(budget)] = {
            "full_prefix_defined_sessions": 15,
            "chronological_split_half_defined_sessions": entry.get(
                "chronological_reliability_defined"
            ),
            "chronological_w_direction_cosine_across_session_median": direction.get("median"),
        }
    _need(summary["6"]["chronological_w_direction_cosine_across_session_median"]
          < summary["12"]["chronological_w_direction_cosine_across_session_median"]
          < summary["24"]["chronological_w_direction_cosine_across_session_median"],
          "sealed descriptor reliability does not order M6<M12<M24")
    row["source_is_mutable_but_sha_bound_fail_closed"] = row["mode"] != "0444"
    return summary, row


def _audit_q24_manifests(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets: list[str] = []
    for fold in ALL_FOLDS:
        path = root / f"fold_{fold:02d}/seed_42/fit/split_manifest.json"
        body, identity = _json(path, immutable=True)
        _need(body.get("outer_loso_fold") == fold
              and body.get("requested_side_feature_group") == "afc4_mb4",
              f"sealed M24 split identity drift at fold {fold}")
        _need(body.get("calibration", {}).get("budget_trials") == FIXED_ACTIVITY_BUDGET
              and body.get("query", {}).get("query_start_trial") == COMMON_QUERY_START,
              f"sealed M24 split is not activity-M24/q24 at fold {fold}")
        nested = body.get("nested_selection", {})
        _need(nested.get("inner_validation_only_for_checkpoint_selection") is True
              and nested.get("checkpoint_metric") == "val_heldin/r2_mean"
              and nested.get("checkpoint_metric_scope") == "inner_validation_session_only"
              and nested.get("outer_target_loaded_during_fit") is False,
              f"sealed M24 split lost inner-only selection at fold {fold}")
        query_rows = [*body.get("source_query_window_audit", {}).values(),
                      *body.get("inner_validation_query_window_audit", {}).values()]
        _need(len(query_rows) == 14
              and all(row.get("query_start_trial") == COMMON_QUERY_START
                      and row.get("full_window_disjoint") is True
                      and int(row.get("eligible_windows", 0)) > 0 for row in query_rows),
              f"sealed source q24 windows are not constructible at fold {fold}")
        target = str(body.get("target_session"))
        targets.append(target)
        rows.append({"fold": fold, "target_session": target,
                     "inner_validation_session": body.get("inner_validation_session"),
                     "minimum_source_or_inner_validation_q24_windows": min(
                         int(row["eligible_windows"]) for row in query_rows
                     ), "manifest": identity})
    _need(len(set(targets)) == 15, "sealed q24 manifests do not cover 15 unique outer sessions")
    return rows, {"folds": 15, "unique_target_sessions": 15,
                  "all_source_and_inner_validation_sessions_have_q24_windows": True,
                  "outer_target_payloads_opened_by_this_audit": 0}


def _audit_m24_aggregate(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    body, row = _json(path, immutable=True)
    _need(body.get("schema") == "rt_mb4_matched_full_minus_mb4_aggregate_v1"
          and body.get("status") == "PASS_RT_FULL_MINUS_MB4_ALL_15_PAIRED"
          and body.get("folds") == list(ALL_FOLDS) and body.get("seed") == SEED,
          "sealed M24 Full-MB4 aggregate identity drift")
    delta = body.get("full_minus_mb4", {})
    _need(delta.get("positive_folds") == 15,
          "sealed M24 Full-MB4 aggregate is not 15/15 positive")
    return {"mean": delta.get("mean"), "median": delta.get("median"),
            "positive_folds": delta.get("positive_folds"),
            "bootstrap_95": delta.get("paired_fold_bootstrap_95")}, row


def _train_command(*, budget: int, arm: str, fold: int, run_root: Path) -> list[str]:
    run_id = f"rt_r4_common_q24_m{budget}_{arm}_f{fold}_s{SEED}"
    return [
        sys.executable, str(ROOT / "src/train.py"), f"experiment={EXPERIMENT}",
        f"run_id={run_id}", f"data.side_feature_calibration_n_trials={budget}",
        f"data.side_feature_group={arm}", f"data.loso_fold={fold}",
        f"data.outer_loso_fold={fold}", f"seed={SEED}", "ckpt_path=null",
        "train=true", "test=false", f"hydra.run.dir={run_root / run_id / 'fit'}",
    ]


def build_plan(
    *, reliability_receipt: Path = RELIABILITY_RECEIPT,
    m24_aggregate: Path = M24_AGGREGATE,
    m24_manifest_root: Path = M24_MB4_IMPORT_ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Build the complete CPU-only plan without loading data or starting a Trainer."""

    output = Path(output).resolve()
    _need(not os.path.lexists(str(output)), f"R4 prepare receipt already exists: {output}")
    reliability, reliability_row = _audit_reliability(reliability_receipt)
    q24_rows, q24_summary = _audit_q24_manifests(Path(m24_manifest_root))
    m24_result, m24_row = _audit_m24_aggregate(m24_aggregate)
    composed: dict[str, Any] = {}
    for budget in PRIMARY_CARRIER_BUDGETS:
        for arm in R4_ARMS:
            cfg = _compose(budget=budget, arm=arm)
            _audit_composed_config(cfg, budget=budget, arm=arm)
            composed[f"m{budget}_{arm}"] = {
                "data_target": cfg["data"]["_target_"],
                "activity_calibration_trials": cfg["data"]["calibration_n_trials"],
                "carrier_calibration_trials": cfg["data"]["side_feature_calibration_n_trials"],
                "query_start_trial": cfg["data"]["query_start_trial"],
                "arm": arm,
                "max_epochs": cfg["trainer"]["max_epochs"],
                "checkpoint_monitor": cfg["callbacks"]["rt_nested_selection_receipt"]["monitor"],
            }
    run_root = output.parent / "gpu_runs"
    pilot = [
        {"budget": budget, "arm": arm, "fold": fold, "seed": SEED,
         "command": _train_command(budget=budget, arm=arm, fold=fold, run_root=run_root)}
        for budget in PRIMARY_CARRIER_BUDGETS
        for fold in PILOT_FOLDS
        for arm in R4_ARMS
    ]
    code_files = (
        "src/data/rt_r4_budget_response_datamodule.py",
        "src/data/rt_nested_loso_datamodule.py",
        "src/data/falcon_datamodule.py",
        "scripts/audit_rt_afc4_calibration_reliability.py",
        "scripts/prepare_rt_r4_budget_response.py",
        "configs/data/rt_r4_budget_response_common_q24.yaml",
        "configs/experiment/rt_r4_budget_response_common_q24.yaml",
    )
    code_closure = {relative: _sha256(ROOT / relative) for relative in code_files}
    return {
        "schema": SCHEMA,
        "status": "DRY_RUN_RT_R4_COMMON_Q24_SOURCE_ONLY_PREPARED_NOT_LAUNCHED",
        "objective": "within_RT_direction_content_delta_vs_carrier_prefix_reliability",
        "fixed_protocol": {
            "activity_neural_calibration_trials": FIXED_ACTIVITY_BUDGET,
            "activity_trial_index_range": [0, FIXED_ACTIVITY_BUDGET],
            "carrier_fit_budgets_primary": list(PRIMARY_CARRIER_BUDGETS),
            "carrier_trial_index_range_by_budget": {
                str(budget): [0, budget] for budget in PRIMARY_CARRIER_BUDGETS
            },
            "common_query_start_trial": COMMON_QUERY_START,
            "gap_trials_not_used_by_carrier_or_query": {
                str(budget): [budget, COMMON_QUERY_START] for budget in PRIMARY_CARRIER_BUDGETS
            },
            "folds": list(ALL_FOLDS), "seed": SEED,
            "inner_validation_checkpoint_rule": "val_heldin/r2_mean",
            "max_epochs": 35,
            "target_session_backpropagation": False,
        },
        "minimal_matched_contrast": {
            "full": "afc4_vel=[w_x,w_y,||W||,b]",
            "control": "afc4_mb4=[0,0,||W||,b] after the same source-only normalization",
            "delta": "Full_minus_MB4",
            "interpretation": "signed continuous-velocity direction content beyond modulation depth and baseline rate",
            "not_repeated": ["SPINT-scale R-S", "zero4", "row-shuffle", "XLSv2"],
        },
        "two_stage_execution": {
            "pilot_folds_fixed_before_results": list(PILOT_FOLDS),
            "pilot_cells": 12,
            "pilot_commands_not_executed": pilot,
            "per_budget_expansion_gate": {
                "positive_pilot_fold_deltas_required": "3/3",
                "mean_full_minus_mb4_at_least_r2": 0.03,
                "paired_by": ["budget", "fold", "seed", "query_start_trial"],
            },
            "expansion_folds_if_and_only_if_that_budget_passes": list(EXPANSION_FOLDS),
            "failed_budget_policy": "stop_that_budget; no added arm, seed, fold, or fusion path",
            "final_aggregate_policy": "retain pilot and every expansion fold with all signed deltas",
            "m18_policy": {
                "automatic_launch": False,
                "purpose": "curve-ambiguity diagnostic only",
                "predefined_ambiguity": (
                    "after complete M6/M12 aggregates are available, their ordered relationship to the "
                    "sealed M24 Full-MB4 delta is non-monotone despite both primary budgets passing "
                    "their expansion gates"
                ),
                "requires_separate_reviewed_prepare_receipt": True,
            },
        },
        "existing_evidence": {
            "descriptor_reliability": reliability,
            "descriptor_reliability_receipt": reliability_row,
            "sealed_m24_full_minus_mb4": m24_result,
            "sealed_m24_aggregate_receipt": m24_row,
            "sealed_source_q24_manifests": q24_rows,
            "q24_constructibility": q24_summary,
        },
        "composed_configs": composed,
        "code_closure_sha256": code_closure,
        "scope": {
            "nwb_files_opened": 0,
            "outer_target_payloads_opened": 0,
            "outer_scores_computed": 0,
            "trainer_constructed": False,
            "optimizer_constructed": False,
            "cuda_queried": False,
            "gpu_processes_started": 0,
            "pilot_commands_executed": 0,
            "gpu_authorized": False,
            "outer_evaluation_authorized": False,
        },
        "output": str(output),
    }


def write_immutable(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    path = Path(path).resolve()
    _need(not os.path.lexists(str(path)), f"refusing to overwrite R4 receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(body)
    payload["status"] = STATUS
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    return path, hashlib.sha256(data.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-immutable-receipt", action="store_true")
    args = parser.parse_args()
    plan = build_plan(output=args.output)
    if not args.write_immutable_receipt:
        print(json.dumps(plan, sort_keys=True))
        return
    path, digest = write_immutable(args.output, plan)
    print(json.dumps({"status": STATUS, "path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
