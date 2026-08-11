#!/usr/bin/env python3
"""CPU-only preflight for the future fresh RT sparse-endpoint three-arm matrix.

It never imports Torch, invokes a trainer, opens an NWB, or starts a process.
It statically checks the isolated production T4d adapter; it does not grant
GPU authority or run a decoder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "sua_exploration/docs/RT_SPARSE_ENDPOINT_LABEL_CPU_PROTOCOL_ADDENDUM_20260810.md"
PROTOCOL_SHA = "741e41249ab0d5fd771f5298f885afc1e468f09f51d29cb4beb78dd63da89581"
STAGE0B = ROOT / "sua_exploration/results/rt_simple_label_v1/stage0b/RT_SPARSE_ENDPOINT_STAGE0B_RECEIPT_v1.json"
STAGE0B_SHA = "b88c91ab5cfb30b4a9ef978622e00488193c4ad18b09498c84ba76e10b9943b1"
STAGE1 = ROOT / "sua_exploration/results/rt_simple_label_v1/stage1/RT_SPARSE_ENDPOINT_STAGE1_RECEIPT_v1.json"
STAGE1_SHA = "9b69eaaa2339610116a5db8efa23ba20ad61459373293d98e80ceec294c3d0e9"
STAGE1_REVIEW = ROOT / "sua_exploration/results/rt_simple_label_v1/RT_SPARSE_ENDPOINT_STAGE1_ROOT_REVIEW_v1.json"
STAGE1_REVIEW_SHA = "c94352899c9a55a253e3a730d1af6a247c73d3c77df00a1e597f1c4871fb411b"
RUNNER = ROOT / "streaming_calibration_exp/scripts/run_rt_clean_nested_loso.py"
DATA_MODULE = ROOT / "streaming_calibration_exp/src/data/rt_nested_loso_datamodule.py"
RT_LOADER = ROOT / "streaming_calibration_exp/src/data/rt_k4_loader.py"
T4D_LOADER = ROOT / "streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py"
EVALUATOR = ROOT / "streaming_calibration_exp/src/rt_clean_nested_loso_eval.py"
DATA_CONFIG = ROOT / "streaming_calibration_exp/configs/data/rt_nested_loso_m24.yaml"
T4D_CONFIG = ROOT / "streaming_calibration_exp/configs/experiment/rt_sparse_endpoint_t4d_clean_nested_loso_m24.yaml"


# The production CLI intentionally binds these immutable receipts by default.
# Unit tests and downstream packaging checks must inject their own bindings so
# importing/running the CPU preflight never silently reads ignored result
# artifacts from the author's worktree.
ReceiptInputs = Mapping[str, tuple[Path, str]]
SurfaceInputs = Mapping[str, Path]

ARMS = {
    "R-T4d": {"group": "rt_sparse_endpoint_t4d", "carrier": "[a,c,0,0]", "dense_allowed": False},
    "R-Full": {"group": "afc4_vel", "carrier": "[wx,wy,||W||,b]", "dense_allowed": True},
    "R-Zero4": {"group": "zero4", "carrier": "[0,0,0,0]", "dense_allowed": False},
}
FOLDS = tuple(range(15))
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bound_receipts(inputs: ReceiptInputs | None = None) -> dict[str, dict[str, str]]:
    if inputs is None:
        inputs = {
            "protocol": (PROTOCOL, PROTOCOL_SHA),
            "stage0b": (STAGE0B, STAGE0B_SHA),
            "stage1": (STAGE1, STAGE1_SHA),
            "stage1_review": (STAGE1_REVIEW, STAGE1_REVIEW_SHA),
        }
    required_names = {"protocol", "stage0b", "stage1", "stage1_review"}
    require(set(inputs) == required_names,
            f"receipt input keys must be exactly {sorted(required_names)}")
    result: dict[str, dict[str, str]] = {}
    for name, (path, expected) in inputs.items():
        actual = sha256(path); require(actual == expected, f"{name} SHA drift")
        result[name] = {"path": str(path), "sha256": actual}
    # Status validation must follow the same injected files whose digests were
    # checked above.  Falling back to the module-level production paths here
    # would make a synthetic test pass only on a developer machine that still
    # has the ignored results tree.
    stage0b = json.loads(inputs["stage0b"][0].read_text(encoding="utf-8"))
    stage1 = json.loads(inputs["stage1"][0].read_text(encoding="utf-8"))
    review = json.loads(inputs["stage1_review"][0].read_text(encoding="utf-8"))
    require(stage0b.get("status") == "PASS_STAGE0B_ENDPOINT_CONSTRUCTIBLE_NO_GPU", "Stage0B not passing")
    require(stage1.get("status") == "PASS_STAGE1_SPARSE_ENDPOINT_AC4_CONSTRUCTIBLE_NO_GPU", "Stage1 not passing")
    require(stage1.get("bound_inputs", {}).get("protocol", {}).get("sha256") == inputs["protocol"][1], "Stage1 protocol identity drift")
    require(review.get("status") == "PASS_REVIEW_STAGE2_GPU_PROPOSAL_ELIGIBLE_NOT_AUTHORIZED", "Stage1 review does not permit proposal preparation")
    return result


def source_provenance(paths: SurfaceInputs | None = None) -> dict[str, dict[str, str]]:
    if paths is None:
        paths = {
            "runner": RUNNER,
            "datamodule": DATA_MODULE,
            "rt_loader": RT_LOADER,
            "t4d_loader": T4D_LOADER,
            "outer_evaluator": EVALUATOR,
            "data_config": DATA_CONFIG,
            "t4d_config": T4D_CONFIG,
        }
    for path in paths.values(): require(path.is_file(), f"missing surface {path}")
    return {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}


def schedule() -> dict[str, Any]:
    cells = [{"fold": fold, "arm": arm, "seed": SEED, "fresh_fit": True, "one_shot_outer_eval": True} for fold in FOLDS for arm in ARMS]
    return {"outer_loso_folds": list(FOLDS), "fold_count": len(FOLDS), "arms": ARMS, "cells": cells, "fresh_fit_count": len(cells), "one_shot_outer_eval_count": len(cells), "historical_full_metric_allowed": False}


def inspect_minimal_reuse(paths: SurfaceInputs | None = None) -> dict[str, Any]:
    if paths is None:
        paths = {
            "runner": RUNNER,
            "datamodule": DATA_MODULE,
            "rt_loader": RT_LOADER,
            "t4d_loader": T4D_LOADER,
            "outer_evaluator": EVALUATOR,
            "data_config": DATA_CONFIG,
            "t4d_config": T4D_CONFIG,
        }
    runner, module, loader, t4d_loader, evaluator, config, t4d_config = (
        paths[name].read_text(encoding="utf-8")
        for name in ("runner", "datamodule", "rt_loader", "t4d_loader", "outer_evaluator", "data_config", "t4d_config")
    )
    common = all(token in config for token in ("window_size: 50", "calibration_n_trials: 24", "query_start_trial: 24", "session_window_budget: 4096")) and "nested_loso" in config
    full_zero = '"afc4_vel"' in runner and '"zero4"' in runner and '"afc4_vel"' in module and '"zero4"' in module
    t4d_group_exists = "rt_sparse_endpoint_t4d" in module and "rt_sparse_endpoint_t4d" in runner
    endpoint_before_dense = ("raw_feature, audit = _carrier_from_endpoint_payload" in t4d_loader and
                             t4d_loader.index("raw_feature, audit = _carrier_from_endpoint_payload") < t4d_loader.index("dense_raw = load_rt_session") and
                             "carrier_unchanged_after_dense_target" in t4d_loader and
                             "dense_target_never_enters_carrier" in t4d_loader and
                             "pos.data[:]" not in t4d_loader)
    # Dense velocity remains allowed only after the carrier freeze for decoder
    # target/evaluation arrays; the T4d estimator source has no cursor_vel.
    dense_loader_path = "cursor_vel" in loader and "load_rt_sparse_endpoint_t4d_session" in module and "build_outer_target_dataset" in evaluator
    no_target_bp_proof = all(token in evaluator for token in (
        "model_state_sha256_before", "model_state_sha256_after_target_carrier",
        "model_state_sha256_after", "model_state_three_point_unchanged",
        "if state_before_digest != state_after_target_carrier_digest",
        "if state_before_digest != state_after_digest",
    ))
    matched_config = all(token in t4d_config for token in (
        "side_feature_group: rt_sparse_endpoint_t4d", "max_epochs: 35",
    ))
    blockers = []
    if not t4d_group_exists: blockers.append("missing_rt_sparse_endpoint_t4d_arm_adapter")
    if not endpoint_before_dense: blockers.append("t4d_carrier_not_frozen_before_dense_decoder_target")
    if not no_target_bp_proof: blockers.append("outer_evaluator_target_no_bp_not_proven_by_static_preflight")
    if not matched_config: blockers.append("t4d_matched_config_missing_or_drifted")
    return {"common_clean_nested_contract_reusable": common, "full_and_zero4_groups_available": full_zero, "t4d_group_exists": t4d_group_exists, "shared_dense_loader_path": dense_loader_path, "t4d_carrier_before_dense_target": endpoint_before_dense, "outer_target_no_bp_static_proof": no_target_bp_proof, "t4d_matched_config_static_proof": matched_config, "blockers": blockers}


def build_preflight(*, receipt_inputs: ReceiptInputs | None = None,
                    surface_inputs: SurfaceInputs | None = None) -> dict[str, Any]:
    """Build the CPU-only report from explicit or production-bound inputs.

    ``receipt_inputs``/``surface_inputs`` are deliberately injectable.  This
    keeps contract tests hermetic while the command-line path retains the
    immutable production receipt pins above.
    """

    receipts = bound_receipts(receipt_inputs)
    surfaces = source_provenance(surface_inputs)
    plan = schedule()
    reuse = inspect_minimal_reuse(surface_inputs)
    status = "READY_FOR_SEPARATE_GPU_REVIEW" if not reuse["blockers"] else "STOP_STAGE2_IMPLEMENTATION_GAP_NO_GPU"
    return {"schema": "rt_sparse_endpoint_stage2_preflight_v1", "status": status, "bound_receipts": receipts, "source_provenance": surfaces, "contract": {"m24": 24, "query_start_trial": 24, "window_size": 50, "session_window_budget": 4096, "seed": SEED, "outer_loso_folds": 15, "primary_comparison": "R-T4d - R-Zero4", "secondary_comparison": "R-T4d - R-Full", "aggregate_required": ["mean", "median", "sign_count", "leave_largest_out_mean"], "historical_full_044195_forbidden": True, "target_backpropagation_forbidden": True}, "schedule": plan, "minimal_reuse_audit": reuse, "non_interference": {"gpu_started": False, "cuda_context_created": False, "nwb_opened": False, "decoder_constructed": False, "trainer_constructed": False, "processes_signalled": False, "paper_modified": False}}


def main() -> None:
    print(json.dumps(build_preflight(), indent=2, sort_keys=True))


if __name__ == "__main__": main()
