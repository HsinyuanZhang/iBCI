#!/usr/bin/env python3
"""Fail-closed aggregate for the v3 fresh 3-seed by 4-arm C1 matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (42, 43, 44)
EPOCHS = tuple(range(5, 13))
CELLS = (
    "separate_sua_t4",
    "separate_pseudo_mua_t4",
    "shared_t4",
    "shared_ts4",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def hierarchical_ci(
    cells: np.ndarray, rng: np.random.Generator, draws: int = 50000
) -> dict[str, float]:
    need(cells.shape == (3, 6), f"hierarchical cells must be [3,6], got {cells.shape}")
    values = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        seeds = rng.integers(0, 3, size=3)
        sessions = rng.integers(0, 6, size=6)
        values[index] = cells[np.ix_(seeds, sessions)].mean()
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def summarize(delta: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Summarize paired seed x session values after epoch averaging."""
    need(delta.shape == (3, 6), f"paired delta must be [3,6], got {delta.shape}")
    seed_means = delta.mean(axis=1)
    session_means = delta.mean(axis=0)
    mean = float(delta.mean())
    se = float(seed_means.std(ddof=1) / math.sqrt(3))
    ci = hierarchical_ci(delta, rng)
    return {
        "all_18_seed_session_deltas": delta.tolist(),
        "mean_delta": mean,
        "seed_mean_deltas": seed_means.tolist(),
        "session_mean_deltas": session_means.tolist(),
        "positive_seed_means": int((seed_means > 0).sum()),
        "positive_session_means": int((session_means > 0).sum()),
        "seed_mean_se_paired": se,
        "paired_two_se_lower": mean - 2 * se,
        "paired_two_se_upper": mean + 2 * se,
        "hierarchical_bootstrap": ci,
    }


def _row(receipt: dict[str, Any], cell: str, seed: int) -> dict[str, Any]:
    rows = [
        row
        for row in receipt["fresh_matrix"]
        if row.get("cell") == cell and row.get("seed") == seed
    ]
    need(len(rows) == 1, f"fresh matrix row missing/duplicated for {cell}/s{seed}")
    return rows[0]


def _load_cell_view(
    *,
    receipt: dict[str, Any],
    program_receipt_sha: str,
    result_root: Path,
    cell: str,
    seed: int,
    view: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    row = _row(receipt, cell, seed)
    status_path = result_root / "status" / f"{cell}_s{seed}.complete.json"
    status = load_json(status_path)
    need(
        (
            status.get("status"),
            status.get("cell"),
            status.get("seed"),
            status.get("program_receipt_sha256"),
            status.get("cell_receipt_sha256"),
        )
        == (
            "completed",
            cell,
            seed,
            program_receipt_sha,
            program_receipt_sha,
        ),
        f"status drift: {status_path}",
    )
    need(status.get("formal_sua_files_opened") is False, f"formal flag drift: {status_path}")
    artifact_rows = status.get("artifacts") or []
    expected_relatives = row["logical_artifacts"]
    if cell.startswith("shared_"):
        suffix = f"_{view}.json"
        matches = [relative for relative in expected_relatives if relative.endswith(suffix)]
    else:
        matches = list(expected_relatives)
    need(len(matches) == 1, f"artifact mapping drift for {cell}/{view}/s{seed}")
    artifact = result_root / matches[0]
    status_artifacts = {item["relative"]: item["sha256"] for item in artifact_rows}
    need(matches[0] in status_artifacts, f"artifact absent from status: {artifact}")
    need(sha256_file(artifact) == status_artifacts[matches[0]], f"artifact hash drift: {artifact}")
    result = load_json(artifact)
    need(
        result.get("seed") == seed
        and result.get("signal_view") == view
        and result.get("no_test_files_evaluated") is True
        and result.get("uses_backward_gradients") is False
        and result.get("epoch_list") == list(EPOCHS),
        f"artifact identity drift: {artifact}",
    )
    protocol = result.get("protocol") or {}
    need(
        protocol.get("calibration_n") == 30
        and protocol.get("pool_size") == 50
        and protocol.get("total_epochs") == 12
        and protocol.get("burn_in_epochs") == 4,
        f"artifact protocol drift: {artifact}",
    )
    sessions = result["session_splits"]["val"]
    need(len(sessions) == 6 and len(set(sessions)) == 6, f"validation sessions drift: {artifact}")
    values = np.asarray(
        [
            [result["per_epoch"][str(epoch)]["per_session_r2"][session] for session in sessions]
            for epoch in EPOCHS
        ],
        dtype=np.float64,
    )
    need(np.isfinite(values).all(), f"non-finite score: {artifact}")
    closure_dir = result_root / row["logical_closure_dir"]
    metadata_path = closure_dir / "run_metadata.json"
    metadata = load_json(metadata_path)
    cost_path = closure_dir / "post_run_cost_receipt.json"
    cost = load_json(cost_path)
    need(result.get("run_metadata_sha256") == sha256_file(metadata_path), "artifact/metadata hash drift")
    need(sha256_file(metadata_path) == status["metadata_sha256"], "metadata/status hash drift")
    need(sha256_file(cost_path) == status["cost_sha256"], "cost/status hash drift")
    closure_path = result_root / status["closure_relative"]
    need(sha256_file(closure_path) == status["closure_manifest_sha256"], "closure hash drift")
    if cell.startswith("shared_"):
        static_cost = metadata["cost_receipt_reference"]
    else:
        static_cost = {
            "encoder": metadata["encoder_cost_profile_reference"],
            "decoder": metadata["decoder_architecture"][
                "decoder_cost_comparison_receipt_reference_n64"
            ],
            "descriptor_persistent_bytes_per_channel": 16,
            "shared_training_view_forwards_per_optimizer_step": 1,
            "deployment_weight_copies": 1,
            "deployment_extra_state_vs_separate": 0,
        }
    return values, {
        "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "cost_path": str(cost_path),
        "cost_sha256": sha256_file(cost_path),
        "closure_manifest": str(closure_path),
        "closure_manifest_sha256": sha256_file(closure_path),
        "sessions": sessions,
        "variant_score": float(result["variant_score"]),
        "fit_wall_clock_seconds": float(cost["fit_wall_clock_seconds"]),
        "cuda_peak_memory_allocated_bytes": int(cost["cuda_peak_memory_allocated_bytes"]),
        "cuda_peak_memory_reserved_bytes": int(cost["cuda_peak_memory_reserved_bytes"]),
        "runtime_environment": cost.get("runtime_environment"),
        "student_parameter_count": int(status["student_parameter_count"]),
        "fp32_student_weight_bytes": int(status["fp32_student_weight_bytes"]),
        "static_cost_reference": static_cost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.receipt.expanduser().resolve()
    result_root = args.result_root.expanduser().resolve()
    output = args.out.expanduser().resolve()
    need(not output.exists(), f"write-once aggregate exists: {output}")
    receipt = load_json(receipt_path)
    receipt_sha = sha256_file(receipt_path)
    data: dict[str, np.ndarray] = {}
    evidence: dict[str, Any] = {}
    canonical_sessions: list[str] | None = None
    for seed in SEEDS:
        for cell in CELLS:
            views = ("sua", "pseudo_mua") if cell.startswith("shared_") else (
                ("sua",) if cell == "separate_sua_t4" else ("pseudo_mua",)
            )
            for view in views:
                values, item = _load_cell_view(
                    receipt=receipt,
                    program_receipt_sha=receipt_sha,
                    result_root=result_root,
                    cell=cell,
                    seed=seed,
                    view=view,
                )
                if canonical_sessions is None:
                    canonical_sessions = item["sessions"]
                need(item["sessions"] == canonical_sessions, "cross-cell validation session order drift")
                key = f"{cell}:{view}"
                data.setdefault(key, np.empty((3, 8, 6), dtype=np.float64))
                data[key][SEEDS.index(seed)] = values
                evidence[f"{cell}_s{seed}_{view}"] = item

    # Epochs are a fixed estimator window, so primary paired inference first
    # averages each seed/session over the eight predeclared checkpoints.
    scores = {key: values.mean(axis=1) for key, values in data.items()}
    rng = np.random.default_rng(20260804)
    sua_noninferiority = summarize(
        scores["shared_t4:sua"] - scores["separate_sua_t4:sua"], rng
    )
    pseudo_noninferiority = summarize(
        scores["shared_t4:pseudo_mua"] - scores["separate_pseudo_mua_t4:pseudo_mua"], rng
    )
    sua_content = summarize(scores["shared_t4:sua"] - scores["shared_ts4:sua"], rng)
    pseudo_content = summarize(
        scores["shared_t4:pseudo_mua"] - scores["shared_ts4:pseudo_mua"], rng
    )
    separate_gap = np.abs(
        scores["separate_sua_t4:sua"] - scores["separate_pseudo_mua_t4:pseudo_mua"]
    )
    shared_gap = np.abs(scores["shared_t4:sua"] - scores["shared_t4:pseudo_mua"])
    gap_increase = summarize(shared_gap - separate_gap, rng)

    sua_pass = (
        sua_noninferiority["paired_two_se_lower"] >= -0.03
        and sua_noninferiority["hierarchical_bootstrap"]["lower_95"] >= -0.03
    )
    pseudo_pass = (
        pseudo_noninferiority["paired_two_se_lower"] >= -0.03
        and pseudo_noninferiority["hierarchical_bootstrap"]["lower_95"] >= -0.03
    )
    def content_pass(row: dict[str, Any]) -> bool:
        return (
            row["paired_two_se_lower"] > 0
            and row["hierarchical_bootstrap"]["lower_95"] > 0
            and row["positive_seed_means"] == 3
            and row["positive_session_means"] >= 5
        )

    gap_pass = (
        gap_increase["paired_two_se_upper"] <= 0.03
        and gap_increase["hierarchical_bootstrap"]["upper_95"] <= 0.03
    )
    gates = {
        "sua_noninferiority": sua_pass,
        "pseudo_mua_noninferiority": pseudo_pass,
        "sua_correct_content_attachment": content_pass(sua_content),
        "pseudo_mua_correct_content_attachment": content_pass(pseudo_content),
        "cross_view_gap_not_increased": gap_pass,
    }

    parameter_counts: dict[str, int] = {}
    for cell in CELLS:
        view = "pseudo_mua" if cell == "separate_pseudo_mua_t4" else "sua"
        counts = {
            int(evidence[f"{cell}_s{seed}_{view}"]["student_parameter_count"])
            for seed in SEEDS
        }
        need(len(counts) == 1, f"student parameter count drift across seeds for {cell}")
        parameter_counts[cell] = counts.pop()

    output_payload = {
        "schema_version": 1,
        "status": "completed",
        "formal_test_used": False,
        "historical_c1_artifacts_used": False,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "all_12_cells_owned_by_program_receipt": True,
        "independent_control_receipt_used": False,
        "seeds": list(SEEDS),
        "epochs": list(EPOCHS),
        "sessions": canonical_sessions,
        "absolute_r2": {
            key: {
                "mean_r2": float(values.mean()),
                "seed_mean_r2": values.mean(axis=(1, 2)).tolist(),
                "session_mean_r2": values.mean(axis=(0, 1)).tolist(),
            }
            for key, values in data.items()
        },
        "primary_contrasts": {
            "shared_t4_minus_separate_t4_sua": sua_noninferiority,
            "shared_t4_minus_separate_t4_pseudo_mua": pseudo_noninferiority,
            "shared_t4_minus_shared_ts4_sua": sua_content,
            "shared_t4_minus_shared_ts4_pseudo_mua": pseudo_content,
            "shared_minus_separate_absolute_cross_view_gap": gap_increase,
        },
        "gates": gates,
        "c1_pass": all(gates.values()),
        "decision": "pass_enter_conditional_c2" if all(gates.values()) else "stop_c_no_rescue",
        "cost": {
            "student_parameter_count_by_arm": parameter_counts,
            "fp32_weight_bytes_by_arm": {key: value * 4 for key, value in parameter_counts.items()},
            "deployment_weight_copies_shared": 1,
            "deployment_extra_state_shared_vs_separate": 0,
            "descriptor_persistent_bytes_per_channel": 16,
            "sealed_prelaunch_parameter_mac_state_receipt": receipt[
                "cpu_gate_receipts"
            ]["exact_model_parameter_mac_state_audit"],
            "static_reference_by_arm": {
                cell: evidence[
                    f"{cell}_s42_{'pseudo_mua' if cell == 'separate_pseudo_mua_t4' else 'sua'}"
                ]["static_cost_reference"]
                for cell in CELLS
            },
            "per_run": {
                key: {
                    field: row[field]
                    for field in (
                        "fit_wall_clock_seconds",
                        "cuda_peak_memory_allocated_bytes",
                        "cuda_peak_memory_reserved_bytes",
                        "runtime_environment",
                    )
                }
                for key, row in evidence.items()
            },
        },
        "artifact_evidence": evidence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"out": str(output), "c1_pass": output_payload["c1_pass"]}, sort_keys=True))


if __name__ == "__main__":
    main()
