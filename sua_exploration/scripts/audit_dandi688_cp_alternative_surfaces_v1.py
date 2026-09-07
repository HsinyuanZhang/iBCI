#!/usr/bin/env python3
"""Source-only output-calibration and identity-ensemble diagnostics.

The script uses the six named validation sessions and never constructs the
formal-test dataset.  It does not update any neural-network parameter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SEEDS = (42, 43, 44)
ENSEMBLE_MEMBERS = 16
ENSEMBLE_SUPPORT = 10


def _r2(target, prediction) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = np.square(target - prediction).sum()
    total = np.square(target - target.mean(axis=0, keepdims=True)).sum()
    if not total > 0.0:
        raise RuntimeError("target variance is degenerate")
    value = 1.0 - float(residual / total)
    if not np.isfinite(value):
        raise RuntimeError("nonfinite R2")
    return value


def _starts(trials) -> np.ndarray:
    values = [
        np.arange(int(row["start"]), int(row["stop"]) - plan.WINDOW + 1, dtype=np.int64)
        for row in trials
    ]
    if not values or any(value.size == 0 for value in values):
        raise RuntimeError("empty trial window range")
    return np.ascontiguousarray(np.concatenate(values), dtype=np.int64)


def _decode(student, record, starts: np.ndarray, identity, device) -> tuple[np.ndarray, np.ndarray]:
    predictions, targets = [], []
    with torch.no_grad():
        for offset in range(0, starts.size, 256):
            chunk = starts[offset : offset + 256]
            neural_np, target_np = runner._batch_arrays(record, chunk)
            neural = torch.from_numpy(neural_np).to(device)
            prediction = (
                student.decode_with_identity(neural, identity.expand(neural.shape[0], -1, -1))[:, -1, :]
                / plan.BEHAVIOR_SCALE
            )
            predictions.append(prediction.detach().cpu().numpy().astype(np.float32))
            targets.append(target_np)
    return (
        np.ascontiguousarray(np.concatenate(predictions), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(targets), dtype=np.float32),
    )


def _fit_output_laws(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    p = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    denominator = float(np.square(p).sum())
    if not denominator > 0.0:
        raise RuntimeError("scale-only output fit is singular")
    scale = float((p * y).sum() / denominator)

    affine = []
    conditions = []
    for coordinate in range(y.shape[1]):
        design = np.column_stack((p[:, coordinate], np.ones(p.shape[0], dtype=np.float64)))
        gram = design.T @ design
        conditions.append(float(np.linalg.cond(gram)))
        coefficient, *_ = np.linalg.lstsq(design, y[:, coordinate], rcond=None)
        affine.append([float(coefficient[0]), float(coefficient[1])])
    return {"scale": scale, "affine": affine, "affine_gram_conditions": conditions}


def _apply_affine(prediction: np.ndarray, affine) -> np.ndarray:
    out = np.empty_like(prediction, dtype=np.float32)
    for coordinate, (scale, offset) in enumerate(affine):
        out[:, coordinate] = prediction[:, coordinate] * scale + offset
    return np.ascontiguousarray(out)


def _ensemble_identities(material, student, carrier, seed: int, session: str, device):
    calibration = torch.from_numpy(
        np.asarray(material.record.calib_trials, dtype=np.float32)
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded = student.id_encoder.pre_pool(calibration.permute(0, 1, 3, 2))
    payload = f"DANDI688_CP_DIAGNOSTIC_ENSEMBLE_V1:{seed}:{session}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(derived))
    identities, subsets = [], []
    with torch.no_grad():
        for _ in range(ENSEMBLE_MEMBERS):
            subset = np.sort(rng.choice(plan.ACTIVITY_SUPPORT, size=ENSEMBLE_SUPPORT, replace=False))
            mean_feature = encoded[:, subset].mean(dim=1)
            identity = student.id_encoder.post_pool(torch.cat((mean_feature, carrier), dim=-1))
            identities.append(identity)
            subsets.append([int(value) for value in subset])
    return identities, subsets


def _decode_ensemble(student, record, starts, identities, device):
    summed = None
    target_ref = None
    for identity in identities:
        prediction, target = _decode(student, record, starts, identity, device)
        if target_ref is None:
            target_ref = target
            summed = prediction.astype(np.float64)
        else:
            if not np.array_equal(target, target_ref):
                raise RuntimeError("ensemble target drift")
            summed += prediction
    return np.ascontiguousarray(summed / len(identities), dtype=np.float32), target_ref


def execute(repo_root: Path) -> dict[str, object]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("diagnostic requires isolated physical GPU0")
    repo_root = repo_root.resolve()
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device")

    dm = data.prepare_datamodule(repo_root)
    materials = data.materialize(repo_root, dm)
    if dm.test_dataset is not None:
        raise RuntimeError("formal-test dataset unexpectedly constructed")
    validation = sorted(session for session, row in materials.items() if row.split == "val")
    if len(validation) != plan.VALIDATION_SESSIONS:
        raise RuntimeError("validation roster drift")
    trial_rows = {
        session: list_datamodule_rewarded_trials(
            repo_root / plan.DATA_RELATIVE / f"{session}_behavior+ecephys.nwb",
            bin_size_ms=plan.BIN_MS,
            window_size=plan.WINDOW,
            trial_result_filter="R",
        )
        for session in validation
    }

    result = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        student = runner._prepare_student(repo_root, seed, device)
        cache = runner._session_cache(materials, student, device, seed)
        rows = []
        for session in validation:
            material = materials[session]
            trials = trial_rows[session]
            guard_starts = _starts(trials[30:50])
            query_starts = material.q50_starts
            state = cache[session]
            native_identity = state["native_identity"]
            guard_prediction, guard_target = _decode(
                student, material.record, guard_starts, native_identity, device
            )
            fit = _fit_output_laws(guard_prediction, guard_target)
            native_prediction, query_target = _decode(
                student, material.record, query_starts, native_identity, device
            )
            scale_prediction = np.ascontiguousarray(native_prediction * fit["scale"], dtype=np.float32)
            affine_prediction = _apply_affine(native_prediction, fit["affine"])
            ensemble_identities, subsets = _ensemble_identities(
                material, student, state["carrier"], seed, session, device
            )
            ensemble_prediction, ensemble_target = _decode_ensemble(
                student, material.record, query_starts, ensemble_identities, device
            )
            if not np.array_equal(query_target, ensemble_target):
                raise RuntimeError("query target drift")
            r2 = {
                "NATIVE": _r2(query_target, native_prediction),
                "OUTPUT_SCALE": _r2(query_target, scale_prediction),
                "OUTPUT_AFFINE2": _r2(query_target, affine_prediction),
                "IDENTITY_ENSEMBLE_K10X16": _r2(query_target, ensemble_prediction),
            }
            subset_array = np.ascontiguousarray(subsets, dtype=np.int64)
            exposure = np.bincount(subset_array.reshape(-1), minlength=plan.ACTIVITY_SUPPORT)
            rows.append(
                {
                    "session": session,
                    "guard_trial_indices": [int(row["trial_index"]) for row in trials[30:50]],
                    "guard_window_count": int(guard_starts.size),
                    "query_window_count": int(query_starts.size),
                    "output_fit": fit,
                    "ensemble_subsets_sha256": runner.core.array_sha256(subset_array),
                    "ensemble_min_trial_exposure": int(exposure.min()),
                    "ensemble_max_trial_exposure": int(exposure.max()),
                    "r2": r2,
                    "delta_vs_native": {
                        key: float(value - r2["NATIVE"])
                        for key, value in r2.items()
                        if key != "NATIVE"
                    },
                }
            )
        result[str(seed)] = {
            "rows": rows,
            "equal_session_mean_r2": {
                key: float(np.mean([row["r2"][key] for row in rows]))
                for key in rows[0]["r2"]
            },
            "equal_session_mean_delta_vs_native": {
                key: float(np.mean([row["delta_vs_native"][key] for row in rows]))
                for key in rows[0]["delta_vs_native"]
            },
        }
        del student, cache
        torch.cuda.empty_cache()

    return {
        "schema": "dandi688_cp_alternative_surfaces_diagnostic_v1",
        "status": "POST_RESULT_SOURCE_ONLY_DIAGNOSTIC",
        "output_fit_labeled_guard": "rewarded_trials_30_through_49",
        "query": "rewarded_trials_50_onward",
        "ensemble": {"members": ENSEMBLE_MEMBERS, "support_per_member": ENSEMBLE_SUPPORT},
        "formal_test_names_only": list(dm.session_splits["test"]),
        "formal_test_files_opened": False,
        "model_parameter_updates": 0,
        "results": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "INERT", "formal_test_files_opened": False}, sort_keys=True))
        return
    global np, torch, data, plan, runner, list_datamodule_rewarded_trials
    import numpy as np
    import torch
    from mc_maze.dandi688_cp_film_v1 import data, plan, runner
    from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials

    result = execute(args.repo_root)
    if args.summary_only:
        result = {
            "schema": result["schema"],
            "status": result["status"],
            "formal_test_files_opened": result["formal_test_files_opened"],
            "model_parameter_updates": result["model_parameter_updates"],
            "results": {
                seed: {
                    "equal_session_mean_r2": row["equal_session_mean_r2"],
                    "equal_session_mean_delta_vs_native": row["equal_session_mean_delta_vs_native"],
                    "per_session_delta_vs_native": {
                        item["session"]: item["delta_vs_native"] for item in row["rows"]
                    },
                }
                for seed, row in result["results"].items()
            },
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
