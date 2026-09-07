#!/usr/bin/env python3
"""CPU-only, auditable gate for the RT AFC4 M24 development pilot.

This script deliberately does not invoke a Trainer, a remote host, EvalAI, or
any formal held-out route.  It builds the four matched RT arms on CPU, checks
their common chronological/event/sampling contract, executes one local
variable-N forward/backward smoke for ``afc4_vel``, and writes a receipt that a
reviewer can inspect before authorising even a one-epoch GPU runtime pilot.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.falcon_k4_features import deterministic_k4_row_permutation
from src.data.rt_datamodule import RtDataModule
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


ARMS = ("zero4", "afc4_vel", "afc4_rs", "afc4_ls")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _window_contract(dataset) -> dict[str, Any]:
    starts_by_session: dict[str, list[int]] = defaultdict(list)
    for session_name, start in dataset.window_indices:
        starts_by_session[session_name].append(int(start))
    rows: dict[str, Any] = {}
    for session_name, audit in dataset.query_window_audit.items():
        starts = starts_by_session.get(session_name, [])
        minimum = audit["minimum_window_start_padded_bin"]
        _assert(audit["query_start_trial"] == 24, f"{session_name}: query_start_trial is not M24")
        _assert(audit["support_trials"] == 24, f"{session_name}: support count is not M24")
        _assert(audit["full_window_disjoint"] is True, f"{session_name}: full-window disjoint flag is false")
        _assert(audit["eligible_windows"] > 0, f"{session_name}: no post-support query windows")
        _assert(bool(starts), f"{session_name}: no indexed post-support query windows")
        _assert(min(starts) >= minimum, f"{session_name}: query history reaches support bins")
        rows[session_name] = {
            "audit": audit,
            "observed_minimum_window_start_padded_bin": int(min(starts)),
            "observed_maximum_window_start_padded_bin": int(max(starts)),
            "full_window_disjoint_verified": True,
        }
    return rows


def _velocity_summary(dm: RtDataModule) -> dict[str, Any]:
    per_session: dict[str, Any] = {}
    all_values: list[np.ndarray] = []
    units: set[str] = set()
    conversions: set[float] = set()
    for session_name in dm.session_names:
        raw = dm.all_sessions[session_name]
        audit = raw["rt_velocity_audit"]
        unit = str(audit["loader_output_unit"])
        units.add(unit)
        conversions.add(float(audit["nwb_conversion"]))
        values = np.asarray(raw["covariates"], dtype=np.float64)[
            np.asarray(raw["eval_mask"], dtype=bool)
        ]
        _assert(values.size > 0 and np.isfinite(values).all(), f"{session_name}: invalid event velocity")
        all_values.append(values)
        per_session[session_name] = {
            "nwb_unit": unit,
            "nwb_conversion": float(audit["nwb_conversion"]),
            "loader_standardization": audit["loader_standardization"],
            "event_qualified_velocity_quantiles_cm_per_s": np.percentile(
                values, [0, 1, 25, 50, 75, 99, 100], axis=0
            ).tolist(),
            "velocity_binning": {
                key: audit[key]
                for key in (
                    "source_median_dt_s",
                    "output_bin_count",
                    "output_valid_bin_count",
                    "output_missing_bin_count",
                    "resampling",
                    "missing_bin_policy",
                )
            },
        }
    _assert(units == {"cm/s"}, f"unexpected RT velocity units: {sorted(units)}")
    _assert(conversions == {1.0}, f"unexpected RT velocity conversions: {sorted(conversions)}")
    all_event = np.concatenate(all_values, axis=0)
    return {
        "all_sessions_nwb_units": sorted(units),
        "all_sessions_nwb_conversions": sorted(conversions),
        "loader_standardization": "none",
        "global_event_qualified_velocity_quantiles_cm_per_s": np.percentile(
            all_event, [0, 1, 25, 50, 75, 99, 100], axis=0
        ).tolist(),
        "per_session": per_session,
    }


def _old_proxy_audit(repo_root: Path) -> dict[str, Any]:
    result_path = repo_root / "sua_exploration/results/k4_rt_loso_v1/results.json"
    script_path = repo_root / "sua_exploration/scripts/k4_rt_loso.py"
    if not result_path.is_file() or not script_path.is_file():
        return {
            "available": False,
            "status": "not_checked",
            "reason": "legacy result/script missing from this checkout",
        }
    old = json.loads(result_path.read_text())
    k4 = old.get("results", {}).get("K4", [])
    ks4 = old.get("results", {}).get("KS4", [])
    r2_deltas = [
        abs(float(left.get("r2_variance_weighted")) - float(right.get("r2_variance_weighted")))
        for left, right in zip(k4, ks4)
    ]
    paired = len(k4) == len(ks4) and all(
        left.get("fold") == right.get("fold") for left, right in zip(k4, ks4)
    ) and (max(r2_deltas, default=float("inf")) <= float(np.finfo(np.float32).eps))
    source = script_path.read_text()
    return {
        "available": True,
        "result_path": str(result_path),
        "legacy_k4_vs_ks4_r2_identical_up_to_one_float32_ulp": bool(paired),
        "legacy_k4_vs_ks4_r2_max_abs_delta": float(max(r2_deltas, default=float("nan"))),
        "legacy_k4_vs_ks4_r2_mean_abs_delta": float(np.mean(r2_deltas)) if r2_deltas else None,
        "legacy_code_contains_descriptor_row_mean": "mean(axis=0)" in source,
        "status": "INVALID_PROXY__NOT_EFFICACY_EVIDENCE",
        "causal_reason": (
            "The legacy proxy shuffled descriptor rows and then averaged over the unit/row axis; "
            "a row permutation leaves that mean invariant, so K4 and KS4 cannot test channel attachment."
        ),
        "new_pipeline_difference": (
            "B3S receives the full variable-N [N,4] descriptor, and afc4_rs permutes those rows only after "
            "source-only normalization; no unit-axis mean is taken."
        ),
    }


def _collect_arm(
    *,
    arm: str,
    args: argparse.Namespace,
) -> tuple[RtDataModule, dict[str, Any], dict[str, Any]]:
    dm = RtDataModule(
        data_dir=str(args.data_dir),
        batch_size=args.batch_size,
        window_size=args.window_size,
        calibration_n_trials=args.calibration_n_trials,
        query_start_trial=args.calibration_n_trials,
        max_trial_length=args.trial_length,
        loso_fold=args.loso_fold,
        side_feature_group=arm,
        side_feature_shuffle_seed=args.seed,
        session_window_budget=args.session_window_budget,
        session_balanced_sampling=True,
        sampler_reshuffle_each_epoch=True,
        num_workers=0,
        pin_memory=False,
        sampler_seed=args.seed,
    )
    dm.setup("fit")
    manifest = dm.get_split_manifest()
    _assert(manifest["formal_heldout_opened"] is False, f"{arm}: unexpectedly opened formal held-out")
    _assert(manifest["session_count"] == 15, f"{arm}: expected 15 RT sessions")
    _assert(manifest["target_session"] not in manifest["source_sessions"], f"{arm}: LOSO target leaks into source")
    _assert(len(manifest["source_sessions"]) == 14, f"{arm}: not 14 source sessions")
    per_session_batches = manifest["source_sampler"]["batches_per_source_session"]
    expected_batches = args.session_window_budget // args.batch_size
    _assert(
        set(per_session_batches.values()) == {expected_batches},
        f"{arm}: source sessions are not exactly budget-balanced: {per_session_batches}",
    )

    query_checks = {
        "source": _window_contract(dm.train_dataset),
        "target": _window_contract(dm.val_heldin_dataset),
    }
    support = manifest["m24_event_support_audit"]
    for session_name, summary in support.items():
        _assert(summary["budget_trials"] == 24, f"{arm}/{session_name}: M24 support budget changed")
        _assert(
            summary["accepted_reach_segments_within_budget"] > 0,
            f"{arm}/{session_name}: zero accepted M24 reach segments",
        )
    for session_name, event in manifest["rt_event_segment_audit"].items():
        _assert(event["complete_cue_trials"] > 0, f"{arm}/{session_name}: no complete-cue trials")
        _assert(event["accepted_reach_segments"] > 0, f"{arm}/{session_name}: no accepted reaches")
        _assert(event["event_qualified_bins"] > 0, f"{arm}/{session_name}: empty event mask")

    artifact: dict[str, Any] = {
        "manifest": manifest,
        "query_checks": query_checks,
        "session_schedule": [
            dm.train_dataset.window_indices[batch[0]][0]
            for batch in dm.train_batch_sampler.batched_indices
        ],
        "unit_counts": {name: int(dm.all_sessions[name]["neural"].shape[1]) for name in dm.session_names},
    }
    if arm != "zero4":
        raw_descriptors = {
            name: dm.train_dataset.k4_raw_features.get(
                name, dm.val_heldin_dataset.k4_raw_features.get(name)
            ).copy()
            for name in dm.session_names
        }
        normalised_descriptors = {}
        for name in dm.session_names:
            dataset = dm.train_dataset if name in dm.train_dataset.k4_raw_features else dm.val_heldin_dataset
            normalised_descriptors[name] = dataset._native_k4_side_features(name, 0, 24).copy()
        artifact["raw_descriptors"] = raw_descriptors
        artifact["normalised_descriptors"] = normalised_descriptors
        audits = {
            **manifest["source_k4_calibration_audit"],
            **manifest["target_k4_calibration_audit"],
        }
        for session_name, audit in audits.items():
            _assert(audit["design_rank"] == 3, f"{arm}/{session_name}: AFC4 design rank is not 3")
            _assert(np.isfinite(audit["design_condition"]), f"{arm}/{session_name}: non-finite design condition")
            _assert(audit["active_blocks"] >= 3, f"{arm}/{session_name}: fewer than three active blocks")
            _assert(
                audit["candidate_trial_bounded_blocks"] >= audit["segment_qualified_blocks"] >= audit["active_blocks"],
                f"{arm}/{session_name}: inconsistent block accounting",
            )
    return dm, manifest, artifact


def _check_matched_arms(artifacts: dict[str, dict[str, Any]], *, seed: int) -> dict[str, Any]:
    baseline = artifacts["afc4_vel"]
    common: dict[str, Any] = {
        "same_loso_source_target": True,
        "same_source_session_schedule": True,
        "same_query_window_audits": True,
        "same_session_window_budget": True,
    }
    for arm in ARMS:
        current = artifacts[arm]
        base_manifest = baseline["manifest"]
        manifest = current["manifest"]
        _assert(
            manifest["source_sessions"] == base_manifest["source_sessions"]
            and manifest["target_session"] == base_manifest["target_session"],
            f"{arm}: LOSO split differs from afc4_vel",
        )
        _assert(current["session_schedule"] == baseline["session_schedule"], f"{arm}: source sampler schedule differs")
        _assert(current["query_checks"] == baseline["query_checks"], f"{arm}: query windows differ")
        _assert(
            manifest["source_sampler"] == base_manifest["source_sampler"],
            f"{arm}: source window budget differs",
        )

    row_shuffle: dict[str, Any] = {}
    for session_name, aligned in baseline["normalised_descriptors"].items():
        shuffled = artifacts["afc4_rs"]["normalised_descriptors"][session_name]
        order = deterministic_k4_row_permutation(aligned.shape[0], session_name=session_name, seed=seed)
        _assert(not np.array_equal(order, np.arange(order.size)), f"{session_name}: row shuffle is identity")
        _assert(
            np.array_equal(shuffled, aligned[order]),
            f"{session_name}: afc4_rs is not a full descriptor-row permutation",
        )
        row_shuffle[session_name] = {
            "num_units": int(order.size),
            "permutation_is_nonidentity": True,
            "full_descriptor_rows_preserved": True,
        }

    label_shuffle: dict[str, Any] = {}
    ls_audits = {
        **artifacts["afc4_ls"]["manifest"]["source_k4_calibration_audit"],
        **artifacts["afc4_ls"]["manifest"]["target_k4_calibration_audit"],
    }
    for session_name, aligned in baseline["raw_descriptors"].items():
        null = artifacts["afc4_ls"]["raw_descriptors"][session_name]
        audit = ls_audits[session_name]
        _assert(not np.array_equal(aligned, null), f"{session_name}: label null did not change descriptor")
        _assert(
            audit["label_changed_blocks"] == audit["active_blocks"],
            f"{session_name}: label null left an active block paired with itself",
        )
        label_shuffle[session_name] = {
            "label_changed_blocks": int(audit["label_changed_blocks"]),
            "active_blocks": int(audit["active_blocks"]),
            "raw_descriptor_max_abs_delta_from_aligned": float(np.max(np.abs(aligned - null))),
            "policy": audit["label_shuffle_policy"],
            "permutation_sha256": audit["label_permutation_sha256"],
        }
    return {"common_contract": common, "row_shuffle": row_shuffle, "label_shuffle": label_shuffle}


def _forward_backward_smoke(dm: RtDataModule, *, teacher_ckpt_path: Path) -> dict[str, Any]:
    module = StreamingCalibrationLitModule(
        task="rt",
        variant="B3S",
        teacher_ckpt_path=str(teacher_ckpt_path),
        window_size=50,
        trial_length=100,
        side_dim=4,
        freeze_decoder=True,
        loss_mode="task_plus_y_plus_E",
        lambda_y=1.0,
        lambda_E=0.1,
        optimizer=partial(torch.optim.Adam, lr=1.0e-4, weight_decay=0.0),
        scheduler=None,
        compile=False,
    )
    module.setup("fit")
    module.train()
    batch = next(iter(dm.train_dataloader()))
    out = module.model_step(batch)
    loss = out["loss"]
    _assert(torch.isfinite(loss).item(), "CPU forward produced non-finite loss")
    loss.backward()
    assert module.student is not None
    decoder_frozen = all(not parameter.requires_grad for parameter in module.student.decoder.parameters())
    decoder_grad_free = all(parameter.grad is None for parameter in module.student.decoder.parameters())
    encoder_grad_count = sum(
        int(parameter.grad is not None and torch.isfinite(parameter.grad).all().item())
        for parameter in module.student.id_encoder.parameters()
        if parameter.requires_grad
    )
    _assert(decoder_frozen and decoder_grad_free, "frozen decoder received a trainable/gradient path")
    _assert(encoder_grad_count > 0, "no trainable calibration encoder gradient in CPU smoke")
    result = {
        "device": "cpu",
        "batch_shapes": {
            "neural": list(batch[0].shape),
            "behavior": list(batch[1].shape),
            "support": list(batch[2].shape),
            "side_features": list(batch[4].shape),
        },
        "batch_session": batch[3][0],
        "loss": float(loss.detach().cpu()),
        "decoder_parameter_count": int(sum(parameter.numel() for parameter in module.student.decoder.parameters())),
        "decoder_all_requires_grad_false": bool(decoder_frozen),
        "decoder_all_grads_none_after_backward": bool(decoder_grad_free),
        "trainable_encoder_parameters_with_finite_grads": int(encoder_grad_count),
        "target_session_backpropagation": False,
    }
    del out, loss, batch, module
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "sua_exploration/data/dandi_000688/sub-C",
    )
    parser.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=root / "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "streaming_calibration_exp/outputs/rt_k4_preflight/rt_k4_m24_fold0_seed42_cpu_receipt.json",
    )
    parser.add_argument("--loso-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--trial-length", type=int, default=100)
    parser.add_argument("--calibration-n-trials", type=int, default=24)
    parser.add_argument("--session-window-budget", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _assert(args.calibration_n_trials == 24, "preflight is frozen to M24")
    _assert(args.window_size == 50 and args.trial_length == 100, "preflight expects B3S M2 tensor geometry")
    _assert(args.session_window_budget % args.batch_size == 0, "equal budget must divide batch size")
    _assert(args.teacher_ckpt.is_file(), f"teacher checkpoint missing: {args.teacher_ckpt}")
    _assert(args.data_dir.is_dir(), f"RT data directory missing: {args.data_dir}")

    aligned_dm: RtDataModule | None = None
    artifacts: dict[str, dict[str, Any]] = {}
    manifests: dict[str, Any] = {}
    for arm in ARMS:
        dm, manifest, artifact = _collect_arm(arm=arm, args=args)
        artifacts[arm] = artifact
        manifests[arm] = manifest
        if arm == "afc4_vel":
            aligned_dm = dm
        else:
            # The comparison evidence is copied into ``artifacts`` above; do
            # not retain four 15-session raw tensors during a CPU preflight.
            del dm
            gc.collect()

    matched = _check_matched_arms(artifacts, seed=args.seed)
    variable_units = sorted(set(artifacts["afc4_vel"]["unit_counts"].values()))
    _assert(len(variable_units) > 1, "RT data unexpectedly has a fixed unit count")
    assert aligned_dm is not None
    smoke = _forward_backward_smoke(aligned_dm, teacher_ckpt_path=args.teacher_ckpt)
    velocity = _velocity_summary(aligned_dm)
    teacher_payload = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher_hparams = teacher_payload.get("hyper_parameters", {})
    _assert(teacher_hparams.get("task") == "m2", "expected FALCON-M2 teacher")
    _assert(teacher_hparams.get("behavior_scaling_factor") == 5.0, "unexpected frozen teacher scaling")

    receipt = {
        "schema": "rt_afc4_m24_cpu_preflight_v1",
        "status": "PASS__CPU_GATE_ONLY__GPU_NOT_AUTHORIZED_BY_THIS_RECEIPT",
        "cpu_only_execution": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_initialized": bool(torch.cuda.is_initialized()),
            "formal_heldout_opened": False,
        },
        "arguments": vars(args),
        "matched_arms": ["zero4", "afc4_vel", "afc4_rs", "afc4_ls"],
        "arm_manifests": manifests,
        "matched_arm_checks": matched,
        "variable_n_batch_contract": {
            "observed_session_unit_counts": artifacts["afc4_vel"]["unit_counts"],
            "distinct_unit_counts": variable_units,
            "per_unit_descriptor_shape": "[N,4]",
            "unit_axis_averaging": "forbidden_and_not_present",
        },
        "forward_backward_smoke": smoke,
        "cursor_velocity_scale_audit": {
            **velocity,
            "frozen_teacher": {
                "path": str(args.teacher_ckpt),
                "sha256": _sha256(args.teacher_ckpt),
                "task": teacher_hparams.get("task"),
                "predict_scaled_behavior": teacher_hparams.get("predict_scaled_behavior"),
                "behavior_scaling_factor": teacher_hparams.get("behavior_scaling_factor"),
                "disposition": (
                    "RT cursor_vel remains raw NWB cm/s.  The inherited M2 teacher divides decoder output "
                    "by its recorded factor 5.0 before loss/R2; this is an operational transfer convention, "
                    "not a hidden RT unit conversion or data standardisation."
                ),
            },
        },
        "legacy_proxy_disposition": _old_proxy_audit(Path(__file__).resolve().parents[2]),
        "pilot_gate": {
            "authorised_by_script": False,
            "maximum_after_root_review": {
                "fold": 0,
                "seed": 42,
                "max_epochs": 1,
                "arms": ["zero4", "afc4_vel", "afc4_rs", "afc4_ls"],
            },
            "prohibited_without_new_review": ["additional_folds", "additional_seeds", "epochs_gt_1", "formal_heldout"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=_json_default) + "\n")
    print(f"WROTE_CPU_PREFLIGHT_RECEIPT={args.output}")
    print(f"STATUS={receipt['status']}")


if __name__ == "__main__":
    main()
