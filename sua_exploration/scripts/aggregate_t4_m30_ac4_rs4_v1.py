#!/usr/bin/env python3
"""Fail-closed aggregate for the triggered AC4 versus AC4-RS4 control."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from aggregate_t4_m30_experiment_a_v3 import decide, summarize
from verify_t4_m30_ac4_rs4_v1 import RECEIPT, require_claim


SEEDS = (42, 43, 44)
EPOCHS = tuple(range(5, 13))
SESSIONS = (
    "sub-C_ses-CO-20151103",
    "sub-C_ses-CO-20151104",
    "sub-C_ses-CO-20151106",
    "sub-C_ses-CO-20151109",
    "sub-C_ses-CO-20151110",
    "sub-C_ses-CO-20151112",
)
BASELINE_AGGREGATE_SHA256 = (
    "965bfc9ed7c20d37e99cff838e0a17da67da45fc81fddfeb1d649b03ddbc2932"
)
NORMALIZER_SHA256 = "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_result(
    path: Path,
    *,
    logical_arm: str,
    metadata_group: str,
    seed: int,
    expected_result_sha: str | None,
    status_dir: Path | None,
    expected_train_source_sha: str,
) -> tuple[np.ndarray, dict]:
    need(path.is_file(), f"missing result {path}")
    if expected_result_sha is not None:
        need(sha256(path) == expected_result_sha, f"reference result drift {path}")
    result = json.loads(path.read_text())
    protocol = result.get("protocol", {})
    need(
        (
            result.get("variant"),
            result.get("seed"),
            result.get("signal_view"),
            result.get("epoch_list"),
            result.get("no_test_files_evaluated"),
            result.get("uses_backward_gradients"),
        )
        == ("B3S", seed, "sua", list(EPOCHS), True, False),
        f"result identity drift {path}",
    )
    need(
        (
            protocol.get("calibration_n"),
            protocol.get("pool_size"),
            protocol.get("total_epochs"),
            protocol.get("burn_in_epochs"),
        )
        == (30, 30, 12, 4),
        f"M30 scorer drift {path}",
    )
    metadata_path = Path(result["run_metadata_path"])
    need(
        metadata_path.is_file()
        and sha256(metadata_path) == result.get("run_metadata_sha256"),
        f"metadata binding {path}",
    )
    metadata = json.loads(metadata_path.read_text())
    side = metadata.get("side_features", {})
    training = metadata.get("training", {})
    need(
        (
            metadata.get("status"),
            metadata.get("variant"),
            metadata.get("seed"),
            metadata.get("task"),
            tuple(metadata.get("split_counts", [])),
            metadata.get("max_units_exclusive"),
            metadata.get("signal_view"),
            metadata.get("held_out_test_evaluated"),
            side.get("group"),
            side.get("side_dim"),
            side.get("pool_size"),
            side.get("normalization_sha256"),
        )
        == (
            "completed",
            "B3S",
            seed,
            "CO",
            (27, 6, 6),
            100,
            "sua",
            False,
            metadata_group,
            4,
            30,
            NORMALIZER_SHA256,
        ),
        f"metadata/descriptor drift {path}",
    )
    need(
        (
            training.get("calibration_n_trials"),
            training.get("max_epochs"),
            training.get("no_early_stopping"),
            training.get("checkpoint_every_epoch"),
            training.get("loss_mode"),
            training.get("learning_rate"),
            training.get("batch_size"),
            training.get("freeze_decoder"),
            training.get("identity_mode"),
            training.get("deterministic"),
        )
        == (30, 12, True, True, "task_only", 1e-4, 32, False, "calibrated", True),
        f"training drift {path}",
    )
    residual = metadata.get("t4_logit_residual")
    need(
        residual is None
        or (isinstance(residual, dict) and residual.get("enabled") is False),
        f"residual drift {path}",
    )

    evidence = {
        "artifact_sha256": sha256(path),
        "run_metadata_sha256": sha256(metadata_path),
    }
    if logical_arm == "ac4_rs4":
        contract = side.get("descriptor_contract", {})
        need(
            side.get("permutation_seed") == seed
            and contract.get("ordinary_t4_normalizer_reused") is True
            and contract.get("mask_applied_after_standardization") is True
            and contract.get("mask") == "[a,c,0,0]"
            and contract.get("complete_normalized_rows_permuted") is True
            and contract.get("nonidentity_row_permutation_required") is True
            and contract.get("row_permutation_seed") == seed
            and contract.get("row_permutation_version") == "ExperimentA-AC4-RS4-v1"
            and contract.get("row_permutation_inputs")
            == ["session_name", "training_seed"]
            and contract.get("neural_activity_permuted") is False
            and contract.get("target_labels_permuted") is False
            and contract.get("normalizer_refit") is False,
            f"AC4-RS4 contract drift {path}",
        )
        permutation_receipt = side.get("row_permutation_receipt", {})
        need(
            len(permutation_receipt) == 33
            and {item.get("split") for item in permutation_receipt.values()}
            == {"train", "val"}
            and all(
                item.get("training_seed") == seed
                and item.get("num_units", 0) >= 2
                and item.get("is_nonidentity") is True
                and isinstance(item.get("derived_seed"), int)
                and isinstance(item.get("permutation_sha256"), str)
                and len(item["permutation_sha256"]) == 64
                for item in permutation_receipt.values()
            ),
            f"AC4-RS4 per-session permutation receipt drift {path}",
        )
        need(status_dir is not None, "AC4-RS4 requires cell status directory")
        status_path = status_dir / f"ac4_rs4_s{seed}.json"
        need(status_path.is_file(), f"missing status {status_path}")
        status = json.loads(status_path.read_text())
        need(
            (
                status.get("arm"),
                status.get("seed"),
                status.get("status"),
                status.get("exit_code"),
                status.get("result_sha256"),
                status.get("metadata_sha256"),
            )
            == (
                "ac4_rs4",
                seed,
                "completed",
                0,
                sha256(path),
                sha256(metadata_path),
            ),
            f"status closure {status_path}",
        )
        cost_path = metadata_path.parent / "post_run_cost_receipt.json"
        need(cost_path.is_file(), f"missing cost receipt {cost_path}")
        cost = json.loads(cost_path.read_text())
        need(
            status.get("cost_sha256") == sha256(cost_path)
            and cost.get("run_metadata_sha256") == sha256(metadata_path)
            and cost.get("train_variant_source_sha256") == expected_train_source_sha
            and cost.get("accelerator") == "gpu",
            f"cost closure {cost_path}",
        )
        evidence["cost_sha256"] = sha256(cost_path)

    rows = []
    for epoch in EPOCHS:
        values = result["per_epoch"][str(epoch)]["per_session_r2"]
        need(sorted(values) == list(SESSIONS), f"session drift {path}")
        rows.append([values[session] for session in SESSIONS])
    array = np.asarray(rows, dtype=float)
    need(np.isfinite(array).all(), f"nonfinite result {path}")
    return array, evidence


def add_dispersion(stats: dict, left: np.ndarray, right: np.ndarray) -> None:
    left_seed = left.mean(axis=(1, 2))
    right_seed = right.mean(axis=(1, 2))
    corr = float(np.corrcoef(left_seed, right_seed)[0, 1])
    unpaired = float(
        math.sqrt(left_seed.var(ddof=1) / 3 + right_seed.var(ddof=1) / 3)
    )
    stats["paired_vs_unpaired"] = {
        "unpaired_seed_mean_se": unpaired,
        "seed_correlation": corr if np.isfinite(corr) else None,
        "paired_se_is_smaller": stats["seed_mean_se_paired"] < unpaired,
    }
    stats["per_seed_dispersion"] = {
        "left_seed_scores": left_seed.tolist(),
        "right_seed_scores": right_seed.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--status-dir", required=True, type=Path)
    parser.add_argument("--baseline-result-dir", required=True, type=Path)
    parser.add_argument("--baseline-aggregate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    need(not args.out.exists(), f"write-once output collision {args.out}")
    authorization = require_claim()
    need(
        sha256(args.baseline_aggregate) == BASELINE_AGGREGATE_SHA256,
        "baseline aggregate drift",
    )
    baseline = json.loads(args.baseline_aggregate.read_text())
    need(
        baseline.get("formal_test_used") is False
        and baseline.get("conditional_row_shuffle_trigger_arms") == ["ac4"],
        "AC4 trigger drift",
    )
    receipt = json.loads(RECEIPT.read_text())
    train_source_sha = receipt["source_sha256"][
        "sua_exploration/scripts/train_variant_dandi688.py"
    ]

    data: dict[str, np.ndarray] = {}
    evidence: dict[str, dict] = {}
    for logical_arm, metadata_group in (
        ("ac4", "ac4"),
        ("z4", "z4"),
        ("ac4_rs4", "ac4rs4"),
    ):
        arrays = []
        for seed in SEEDS:
            if logical_arm == "ac4_rs4":
                path = args.result_dir / f"ac4_rs4_s{seed}.json"
                expected = None
                status_dir = args.status_dir
            else:
                path = args.baseline_result_dir / f"{logical_arm}_s{seed}.json"
                expected = baseline["artifact_evidence"][f"{logical_arm}_s{seed}"][
                    "artifact_sha256"
                ]
                status_dir = None
            values, item_evidence = load_result(
                path,
                logical_arm=logical_arm,
                metadata_group=metadata_group,
                seed=seed,
                expected_result_sha=expected,
                status_dir=status_dir,
                expected_train_source_sha=train_source_sha,
            )
            arrays.append(values)
            evidence[f"{logical_arm}_s{seed}"] = item_evidence
        data[logical_arm] = np.stack(arrays)

    rng = np.random.default_rng(20260804)
    primary = summarize(data["ac4"] - data["ac4_rs4"], rng)
    add_dispersion(primary, data["ac4"], data["ac4_rs4"])
    primary["decision"] = decide(primary)
    secondary = summarize(data["ac4_rs4"] - data["z4"], rng)
    add_dispersion(secondary, data["ac4_rs4"], data["z4"])
    secondary["decision"] = decide(secondary)

    output = {
        "schema_version": 1,
        "status": "completed",
        "formal_test_used": False,
        "authorization": authorization,
        "baseline_aggregate_sha256": BASELINE_AGGREGATE_SHA256,
        "seeds": list(SEEDS),
        "epochs": list(EPOCHS),
        "sessions": list(SESSIONS),
        "artifact_evidence": evidence,
        "absolute_r2": {
            arm: {
                "mean_r2": float(values.mean()),
                "seed_mean_r2": values.mean(axis=(1, 2)).tolist(),
                "session_mean_r2": values.mean(axis=(0, 1)).tolist(),
            }
            for arm, values in data.items()
        },
        "primary_ac4_minus_ac4_rs4": primary,
        "secondary_ac4_rs4_minus_z4": secondary,
        "row_attachment_state": primary["decision"],
        "next_rule": (
            "Experiment B may enter source-only CPU estimator selection only if primary is effective; "
            "otherwise stop the AC4 attachment/estimator branch without extra seeds or shuffles."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
