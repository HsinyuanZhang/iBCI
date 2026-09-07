#!/usr/bin/env python3
"""Write and validate the append-only C1 shared-encoder INT8 prelaunch.

This receipt is deliberately independent of the older ordinary-T4 INT8
selection receipt.  It binds the positive paired-view C1 result, one fixed
deployment checkpoint per seed, source-only dual-view scale fitting, and an
encoder-only W8A8/INT32 scope.  Formal sessions are names only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44)
VIEWS = ("sua", "pseudo_mua")
MANIFEST_REL = "sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json"
DATA_DIR_REL = "sua_exploration/data/dandi_000688/sub-C"
TEACHER_REL = (
    "sua_exploration/checkpoints/teacher_mc_maze/"
    "best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)
C1_FINAL_REL = (
    "sua_exploration/results/"
    "t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/finalization/aggregate.json"
)
C1_READINESS_REL = (
    "sua_exploration/results/"
    "t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/"
    "finalization/publication_readiness.json"
)
C1_CLOSURE_ROOT_REL = (
    "sua_exploration/results/"
    "t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/closure"
)
RUN_PREFIX = (
    "sua_exploration/checkpoints/"
    "t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist_shared_t4_s"
)
SOURCE_FILES = (
    "sua_exploration/scripts/write_t4_paired_view_c1_encoder_int8_prelaunch.py",
    "sua_exploration/scripts/eval_t4_paired_view_c1_encoder_int8.py",
    "sua_exploration/scripts/aggregate_t4_paired_view_c1_encoder_int8.py",
    "sua_exploration/scripts/run_t4_paired_view_c1_encoder_int8_ptq.sh",
    "software-to-hardware/b3_ckpt_loader.py",
    "software-to-hardware/b3_fake_quant.py",
    "software-to-hardware/b3_hw_golden.py",
    "software-to-hardware/b3_ptq.py",
    "software-to-hardware/b3_qat_encoder.py",
    "software-to-hardware/b3_quant_engine.py",
    "sua_exploration/scripts/eval_paired_view_c1_epoch_window.py",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _hashed_rel(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_manifest(manifest: dict[str, Any]) -> dict[str, list[str]]:
    splits = manifest.get("session_splits") or {}
    expected = {"train": 27, "val": 6, "test": 6}
    for split, count in expected.items():
        values = splits.get(split)
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f"strict manifest {split} count is not {count}")
        if len(values) != len(set(values)):
            raise ValueError(f"strict manifest {split} contains duplicates")
    if set(splits["train"]) & set(splits["val"]):
        raise ValueError("strict manifest train/val overlap")
    if (set(splits["train"]) | set(splits["val"])) & set(splits["test"]):
        raise ValueError("strict manifest formal names overlap opened splits")
    return {key: list(splits[key]) for key in expected}


def _validate_c1_evidence(aggregate: dict[str, Any], readiness: dict[str, Any]) -> None:
    if aggregate.get("c1_pass") is not True:
        raise ValueError("C1 frozen aggregate is not positive")
    if aggregate.get("formal_test_used") is not False:
        raise ValueError("C1 frozen aggregate does not seal formal test")
    gates = aggregate.get("gates") or {}
    if not gates or not all(value is True for value in gates.values()):
        raise ValueError("C1 frozen aggregate gates are incomplete")
    if readiness.get("status") not in {"publication_ready", "ready", "finalized"}:
        raise ValueError("C1 publication-readiness receipt is not ready")
    data_verification = readiness.get("data_verification") or {}
    if data_verification.get("formal_paths_resolved") is not False:
        raise ValueError("C1 readiness receipt resolved formal paths")


def _validate_run_metadata(metadata: dict[str, Any], seed: int) -> None:
    training = metadata.get("training") or {}
    checks = {
        "status": metadata.get("status") == "completed",
        "seed": metadata.get("seed") == seed,
        "variant": metadata.get("variant") == "B3S",
        "paired": metadata.get("training_kind") == "shared_paired_view",
        "signal_view": metadata.get("signal_view") == "paired_sua_pseudo_mua",
        "formal": metadata.get("held_out_test_evaluated") is False,
        "formal_sua": metadata.get("formal_sua_files_opened") is False,
        "forward_q": training.get("evaluation_forward_calibration_n") == 30,
        "pool": training.get("evaluation_pool_size") == 50,
        "start": training.get("evaluation_start_trial") == 50,
        "epochs": training.get("max_epochs") == 12,
    }
    views = metadata.get("view_configs") or {}
    for view in VIEWS:
        side = (views.get(view) or {}).get("side_features") or {}
        checks[f"{view}_view"] = (views.get(view) or {}).get("signal_view") == view
        checks[f"{view}_t4"] = (
            side.get("group") == "t4"
            and side.get("side_dim") == 4
            and side.get("pool_size") == 50
        )
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f"C1 seed {seed} metadata failed: {failed}")


def build_receipt() -> dict[str, Any]:
    manifest_row = _hashed_rel(MANIFEST_REL)
    manifest = _read_json(ROOT / MANIFEST_REL)
    splits = _validate_manifest(manifest)
    aggregate_row = _hashed_rel(C1_FINAL_REL)
    readiness_row = _hashed_rel(C1_READINESS_REL)
    aggregate = _read_json(ROOT / C1_FINAL_REL)
    readiness = _read_json(ROOT / C1_READINESS_REL)
    _validate_c1_evidence(aggregate, readiness)

    teacher_row = _hashed_rel(TEACHER_REL)
    seed_artifacts: dict[str, Any] = {}
    for seed in SEEDS:
        metadata_rel = f"{C1_CLOSURE_ROOT_REL}/shared_t4_s{seed}/run_metadata.json"
        checkpoint_rel = f"{RUN_PREFIX}{seed}/epoch_ckpts/epoch_011.ckpt"
        metadata_row = _hashed_rel(metadata_rel)
        checkpoint_row = _hashed_rel(checkpoint_rel)
        metadata = _read_json(ROOT / metadata_rel)
        _validate_run_metadata(metadata, seed)
        if metadata.get("train_val_manifest_sha256") != manifest_row["sha256"]:
            raise ValueError(f"seed {seed} manifest hash drift")
        if metadata.get("teacher_sha256") != teacher_row["sha256"]:
            raise ValueError(f"seed {seed} teacher hash drift")
        seed_artifacts[str(seed)] = {
            "seed": seed,
            "run_metadata": metadata_row,
            "checkpoint": checkpoint_row,
            "checkpoint_rule": "fixed_final_training_epoch_epoch_011_no_dev_argmax",
        }

    source_code = {relative: _hashed_rel(relative) for relative in SOURCE_FILES}
    return {
        "schema_version": 1,
        "status": "authorized_for_c1_shared_encoder_ptq",
        "created_at": datetime.now().astimezone().isoformat(),
        "scope": "C1_shared_B3S_T4_identity_encoder_W8A8_INT32_plus_FP32_decoder",
        "ordinary_t4_int8_selection_receipt_reused": False,
        "c2_authorized": False,
        "formal_test_files_opened": False,
        "c1_positive_trigger": {
            "aggregate": aggregate_row,
            "publication_readiness": readiness_row,
            "c1_pass": True,
        },
        "strict_manifest": {
            **manifest_row,
            "counts": [27, 6, 6],
            "source_sessions": splits["train"],
            "development_sessions": splits["val"],
            "sealed_formal_session_names": splits["test"],
            "formal_paths_resolved": False,
        },
        "data_dir": {"path": DATA_DIR_REL},
        "teacher": teacher_row,
        "seed_artifacts": seed_artifacts,
        "frozen_protocol": {
            "seeds": list(SEEDS),
            "views": list(VIEWS),
            "deployment_checkpoint": "epoch_011.ckpt",
            "checkpoint_selection_uses_development": False,
            "activity_calibration_n": 30,
            "t4_label_rate_pool_n": 50,
            "evaluation_start_trial": 50,
            "selection_mode": "chronological_first",
            "scale_fit_unique_source_sessions": 27,
            "scale_fit_records": 54,
            "scale_fit_view_weight": {"sua": 0.5, "pseudo_mua": 0.5},
            "scale_statistic_equal_mass_per_view": True,
            "scale_statistic_max_samples_per_view_per_tensor": 262144,
            "scale_statistic_sampling": "deterministic_uniform_stride_with_absolute_extremum_retained",
            "signed_t4_included_in_post0_input_scale_fit": True,
            "one_shared_scale_set_per_seed": True,
            "scale_selection_objective": "mean_identity_rmse_equal_weight_over_27x2_source_view_records",
            "scale_selection_uses_behavior_targets": False,
            "development_used_once_after_scale_selection": True,
            "paired_int8_fp32_same_checkpoint": True,
            "decoder_precision": "FP32",
            "decoder_quantized": False,
            "weight_bits": 8,
            "activation_bits": 8,
            "accumulator_bits": 32,
            "integer_requant": True,
        },
        "frozen_ptq_gates": {
            "each_view_overall_mean_delta_int8_minus_fp32_r2_min": -0.01,
            "each_seed_each_view_mean_delta_int8_minus_fp32_r2_min": -0.01,
            "max_edge_saturation": 0.005,
            "int32_overflow_count": 0,
            "integer_fake_quant_max_abs_E": 0.0,
            "formal_test_files_opened": False,
        },
        "conditional_qat": {
            "authorized_only_if_three_seed_ptq_aggregate_fails": True,
            "run_all_three_seeds_if_triggered": True,
            "mixed_ptq_qat_seed_aggregate_forbidden": True,
            "decoder_remains_fp32": True,
            "development_epoch_selection_forbidden": True,
            "status": "not_triggered_before_complete_three_seed_ptq_aggregate",
        },
        "source_code": source_code,
    }


def _resolve_row(row: dict[str, Any]) -> Path:
    path = ROOT / str(row.get("path", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != row.get("sha256"):
        raise ValueError(f"receipt-bound file hash drifted: {path}")
    return path


def validate_receipt(
    receipt_path: Path,
    *,
    seed: int | None = None,
    checkpoint: Path | None = None,
    run_metadata: Path | None = None,
) -> dict[str, Any]:
    receipt_path = receipt_path.expanduser().resolve()
    receipt = _read_json(receipt_path)
    expected = {
        "schema_version": 1,
        "status": "authorized_for_c1_shared_encoder_ptq",
        "scope": "C1_shared_B3S_T4_identity_encoder_W8A8_INT32_plus_FP32_decoder",
        "ordinary_t4_int8_selection_receipt_reused": False,
        "c2_authorized": False,
        "formal_test_files_opened": False,
    }
    failed = [key for key, value in expected.items() if receipt.get(key) != value]
    if failed:
        raise ValueError(f"C1 INT8 prelaunch failed: {failed}")
    protocol = receipt.get("frozen_protocol") or {}
    if protocol.get("seeds") != list(SEEDS) or protocol.get("views") != list(VIEWS):
        raise ValueError("C1 INT8 seed/view set drifted")
    if protocol.get("one_shared_scale_set_per_seed") is not True:
        raise ValueError("C1 INT8 must use one shared scale set per seed")
    for row in (receipt.get("c1_positive_trigger") or {}).values():
        if isinstance(row, dict) and "path" in row:
            _resolve_row(row)
    _resolve_row(receipt["strict_manifest"])
    _resolve_row(receipt["teacher"])
    for row in (receipt.get("source_code") or {}).values():
        _resolve_row(row)
    if seed is not None:
        if seed not in SEEDS:
            raise ValueError(f"unexpected seed: {seed}")
        bound = receipt["seed_artifacts"][str(seed)]
        metadata_path = (run_metadata or _resolve_row(bound["run_metadata"])).resolve()
        checkpoint_path = (checkpoint or _resolve_row(bound["checkpoint"])).resolve()
        if not metadata_path.is_file() or sha256_file(metadata_path) != bound["run_metadata"]["sha256"]:
            raise ValueError(f"seed {seed} run_metadata hash drift")
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != bound["checkpoint"]["sha256"]:
            raise ValueError(f"seed {seed} checkpoint hash drift")
        if checkpoint_path.name != "epoch_011.ckpt":
            raise ValueError("C1 INT8 deployment checkpoint is not epoch_011.ckpt")
        metadata = _read_json(metadata_path)
        _validate_run_metadata(metadata, seed)
        receipt["_validated_seed"] = {
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "run_metadata": str(metadata_path),
        }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable prelaunch: {out}")
    receipt = build_receipt()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_receipt(out)
    print(json.dumps({"out": str(out), "sha256": sha256_file(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
