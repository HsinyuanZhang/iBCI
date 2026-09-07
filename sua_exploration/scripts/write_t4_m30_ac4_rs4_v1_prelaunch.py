#!/usr/bin/env python3
"""CPU-only semantic audit and write-once prelaunch receipt for AC4-RS4 v1."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
sys.path.insert(0, str(SUA))

from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule  # noqa: E402
from mc_maze.unit_side_features import (  # noqa: E402
    ac4rs4_derived_seed,
    deterministic_nonidentity_row_permutation,
    load_unit_side_features,
    session_name_from_path,
    side_feature_stats_sha256,
)


OUT = SUA / "results/t4_m30_ac4_rs4_prelaunch_v1_20260804/receipt.json"
BASELINE_AGGREGATE = SUA / "results/sua_t4_m30_component_attribution_v10/aggregate_r11.json"
BASELINE_AGGREGATE_SHA256 = "965bfc9ed7c20d37e99cff838e0a17da67da45fc81fddfeb1d649b03ddbc2932"
NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
MANIFEST = SUA / "configs/subc_co_27_6_strict_train_val_manifest.json"
TEACHER = SUA / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
CACHE = SUA / "cache/t4_m30_ac4_rs4_v1"
SCREEN = "sua_t4_m30_ac4_rs4_v1"
SEEDS = (42, 43, 44)

SOURCES = (
    "streaming_calibration_exp/src/metrics/gate2_matrix.py",
    "streaming_calibration_exp/src/metrics/run_artifacts.py",
    "streaming_calibration_exp/src/models/components/neuron_dropout.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/falcon_module.py",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/t4_logit_residual_module.py",
    "streaming_calibration_exp/src/utils/clean_teacher_validation.py",
    "streaming_calibration_exp/src/utils/instantiators.py",
    "streaming_calibration_exp/src/utils/logging_utils.py",
    "streaming_calibration_exp/src/utils/pylogger.py",
    "streaming_calibration_exp/src/utils/rich_utils.py",
    "streaming_calibration_exp/src/utils/utils.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/sua_auxiliary_stage0.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/scripts/aggregate_t4_m30_ac4_rs4_v1.py",
    "sua_exploration/scripts/aggregate_t4_m30_experiment_a_v3.py",
    "sua_exploration/scripts/dandi688_gradient_free_protocol.py",
    "sua_exploration/scripts/eval_adaptation_dandi688.py",
    "sua_exploration/scripts/eval_epoch_window_generic_dandi688.py",
    "sua_exploration/scripts/eval_t4_m30_experiment_a.py",
    "sua_exploration/scripts/run_t4_m30_ac4_rs4_v1_one_cell.sh",
    "sua_exploration/scripts/schedule_t4_m30_ac4_rs4_v1_2gpu.sh",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
    "sua_exploration/scripts/train_variant_dandi688.py",
    "sua_exploration/scripts/verify_t4_m30_ac4_rs4_v1.py",
    "sua_exploration/scripts/write_t4_m30_ac4_rs4_v1_prelaunch.py",
    "sua_exploration/tests/test_t4_m30_ac4_rs4_v1.py",
    "sua_exploration/tests/test_t4_m30_experiment_a_descriptors.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    if sha256(BASELINE_AGGREGATE) != BASELINE_AGGREGATE_SHA256:
        raise ValueError("r11 aggregate drift")
    baseline = json.loads(BASELINE_AGGREGATE.read_text())
    if baseline.get("formal_test_used") is not False:
        raise ValueError("r11 formal-test scope drift")
    if baseline.get("conditional_row_shuffle_trigger_arms") != ["ac4"]:
        raise ValueError("r11 trigger is not exactly AC4")

    future_paths = []
    for seed in SEEDS:
        future_paths.extend(
            [
                SUA / f"checkpoints/{SCREEN}_ac4_rs4_dandi688_co_s{seed}",
                SUA / f"checkpoints/{SCREEN}_ac4_rs4_dandi688_co_s{seed}.lock",
                SUA / f"results/{SCREEN}/ac4_rs4_s{seed}.json",
                SUA / f"results/p3_{SCREEN}_ac4_rs4_dandi688_co_s{seed}_seed{seed}.json",
            ]
        )
    future_paths.extend(
        [
            SUA / f"results/{SCREEN}",
            SUA / f"results/{SCREEN}/aggregate_v1.json",
        ]
    )
    collisions = [str(path) for path in future_paths if path.exists()]
    if collisions:
        raise FileExistsError(f"future path collision: {collisions}")

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(SUA / "data/dandi_000688/sub-C"),
        task="CO",
        split_counts=(27, 6, 6),
        batch_size=32,
        window_size=50,
        calibration_n_trials=30,
        max_trial_length=100,
        bin_size_ms=20,
        num_workers=0,
        seed=42,
        max_units_exclusive=100,
        cache_dir=str(CACHE),
        signal_view="sua",
        side_feature_group="ac4rs4",
        side_feature_pool_size=30,
        side_permutation_seed=42,
        train_val_manifest_path=str(MANIFEST),
    )
    dm._initialize_splits()
    mean, std = dm._get_side_feature_stats()
    normalizer_sha = side_feature_stats_sha256(mean, std)
    if normalizer_sha != NORMALIZER_SHA256:
        raise ValueError(f"ordinary T4 normalizer drift: {normalizer_sha}")

    audit = {}
    audited_paths = list(dm.session_files["train"]) + list(dm.session_files["val"])
    if len(audited_paths) != 33:
        raise ValueError(f"expected 33 source/development sessions, got {len(audited_paths)}")
    for session_path in audited_paths:
        session_name = session_name_from_path(session_path)
        split = "train" if session_path in dm.session_files["train"] else "val"
        aligned, _ = load_unit_side_features(
            session_path,
            feature_group="ac4",
            pool_size=30,
            mean=mean,
            std=std,
            cache_dir=CACHE,
            bin_size_ms=20,
            window_size=50,
            trial_result_filter="R",
            signal_view="sua",
        )
        if aligned.shape[1] != 4 or not np.array_equal(aligned[:, 2:], np.zeros_like(aligned[:, 2:])):
            raise ValueError(f"AC4 mask drift: {session_name}")
        session_audit = {"split": split, "num_units": int(aligned.shape[0]), "seeds": {}}
        for seed in SEEDS:
            shuffled, _ = load_unit_side_features(
                session_path,
                feature_group="ac4rs4",
                pool_size=30,
                mean=mean,
                std=std,
                cache_dir=CACHE,
                permutation_seed=seed,
                bin_size_ms=20,
                window_size=50,
                trial_result_filter="R",
                signal_view="sua",
            )
            permutation = deterministic_nonidentity_row_permutation(
                aligned.shape[0], permutation_seed=seed, session_name=session_name
            )
            if np.array_equal(permutation, np.arange(aligned.shape[0])):
                raise ValueError(f"identity permutation: {session_name} seed {seed}")
            if not np.array_equal(shuffled, aligned[permutation]):
                raise ValueError(f"complete-row conservation failure: {session_name} seed {seed}")
            session_audit["seeds"][str(seed)] = {
                "derived_seed": ac4rs4_derived_seed(
                    permutation_seed=seed, session_name=session_name
                ),
                "permutation_sha256": hashlib.sha256(
                    permutation.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "is_nonidentity": True,
                "fixed_point_count": int(
                    np.sum(permutation == np.arange(aligned.shape[0]))
                ),
                "complete_row_equality": True,
            }
        audit[session_name] = session_audit

    # Cold/warm cache equality on one real source session, in an isolated temporary root.
    cold_path = audited_paths[0]
    cold_name = session_name_from_path(cold_path)
    with tempfile.TemporaryDirectory(prefix="ac4rs4_cold_warm_") as temporary:
        temporary_path = Path(temporary)
        cold, _ = load_unit_side_features(
            cold_path,
            feature_group="ac4rs4",
            pool_size=30,
            mean=mean,
            std=std,
            cache_dir=temporary_path,
            permutation_seed=42,
            bin_size_ms=20,
            window_size=50,
            trial_result_filter="R",
            signal_view="sua",
        )
        warm, _ = load_unit_side_features(
            cold_path,
            feature_group="ac4rs4",
            pool_size=30,
            mean=mean,
            std=std,
            cache_dir=temporary_path,
            permutation_seed=42,
            bin_size_ms=20,
            window_size=50,
            trial_result_filter="R",
            signal_view="sua",
        )
        if not np.array_equal(cold, warm):
            raise ValueError("cold/warm cache output drift")

    source_hashes = {}
    for relative in SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = sha256(path)

    references = {
        arm: {
            str(seed): {
                "result_sha256": baseline["artifact_evidence"][f"{arm}_s{seed}"]["artifact_sha256"],
                "metadata_sha256": baseline["artifact_evidence"][f"{arm}_s{seed}"]["run_metadata_sha256"],
            }
            for seed in SEEDS
        }
        for arm in ("ac4", "z4")
    }
    receipt = {
        "schema_version": 1,
        "status": "PASS",
        "created_at": datetime.now().astimezone().isoformat(),
        "authorization_id": "t4-m30-ac4-rs4-v1-user-continued-20260804",
        "gpu_launch_authorized": True,
        "authorized_matrix": {
            "arm": "AC4-RS4",
            "seeds": list(SEEDS),
            "cells": 3,
            "variant": "B3S",
            "side_dim": 4,
            "support_pool": 30,
            "epochs": 12,
            "epoch_window": list(range(5, 13)),
            "formal_test": False,
        },
        "formal_test_opened": False,
        "audited_session_scope": {"train": 27, "development_validation": 6, "formal_test": 0},
        "baseline_aggregate": {
            "path": str(BASELINE_AGGREGATE.relative_to(ROOT)),
            "sha256": BASELINE_AGGREGATE_SHA256,
            "trigger_arms": ["ac4"],
        },
        "qualified_references": references,
        "teacher_sha256": sha256(TEACHER),
        "strict_manifest_sha256": sha256(MANIFEST),
        "normalizer_sha256": normalizer_sha,
        "cache": {
            "root": str(CACHE),
            "raw_t4_only": True,
            "final_shuffled_matrix_cached": False,
            "cold_warm_equal": True,
            "cold_warm_session": cold_name,
        },
        "session_permutation_audit": audit,
        "source_sha256": source_hashes,
        "decision_rule": {
            "primary": "AC4 - AC4-RS4",
            "effective": "mean>=0.03; positive sessions>=5/6; positive seeds=3/3; paired 2SE lower>0; hierarchical lower>0",
            "ineffective": "mean + 2SE < 0.03",
            "otherwise": "indeterminate",
            "no_extra_seeds_or_shuffle_families": True,
        },
        "future_paths_absent_at_receipt": True,
        "independent_terra_review": {
            "scientific_trigger": "GO",
            "gpu_state_before_preflight": "NO-GO until this receipt and tests pass",
            "required_permutation_version": "ExperimentA-AC4-RS4-v1",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "out": str(OUT), "sessions": len(audit)}, sort_keys=True))


if __name__ == "__main__":
    main()
