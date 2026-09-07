#!/usr/bin/env python3
"""Run the velocity Kalman comparator and write an immutable CPU receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "SPINT-main", REPO_ROOT / "streaming_calibration_exp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sua_exploration.comparators.core import kalman_comparator as core


DEFAULT_SUBJECT_M_DATA = REPO_ROOT / "sua_exploration/data/dandi_000688"
DEFAULT_RT_DATA = REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"
DEFAULT_H1_DATA = REPO_ROOT / "SPINT-main/data/000954"
DEFAULT_V9_RUN_ROOT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
DEFAULT_STAGE2_CELL_ROOT = (
    REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical/matrix_v1/cells"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/kalman_comparator"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_immutable(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(path, 0o444)
    return hashlib.sha256(payload).hexdigest()


def run_subject_m(
    *,
    view: str,
    budget: int,
    data_dir: Path,
    v9_run_root: Path,
    q_floor: float,
    w_floor: float,
) -> dict[str, Any]:
    blocks = core.load_all_subject_m_blocks(
        repo_root=REPO_ROOT,
        data_dir=data_dir,
        v9_run_root=v9_run_root,
        view=view,
        budget=budget,
    )
    results = [core.evaluate_session(block, q_floor=q_floor, w_floor=w_floor) for block in blocks]
    return {
        "dataset": "subject_m",
        "view": view,
        "budget_trials": int(budget),
        "summary": core.summarize_sessions(results),
        "per_session": [
            {
                "session_name": item.session_name,
                "success": item.success,
                "failure_reason": item.failure_reason,
                "r2": item.r2,
                "metric_name": item.metric_name,
                "calibration_rows": item.calibration_rows,
                "query_rows": item.query_rows,
                "query_identity": item.query_identity,
                "fit": {
                    "q_floor_applied": item.fit.q_floor_applied,
                    "w_floor_applied": item.fit.w_floor_applied,
                    "parameters": item.parameters.as_dict() if item.parameters is not None else None,
                },
            }
            for item in results
        ],
    }


def run_rt(
    *,
    data_dir: Path,
    stage2_cell_root: Path,
    q_floor: float,
    w_floor: float,
) -> dict[str, Any]:
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions, load_rt_session
    from sua_exploration.comparators.core import rt_classical_comparators as rt

    sessions = {
        rt.session_name_from_nwb_path(Path(path)): Path(path) for path in find_rt_sessions(data_dir)
    }
    require(len(sessions) == core.RT_EXPECTED_FOLDS, f"expected {core.RT_EXPECTED_FOLDS} RT sessions")
    results: list[core.SessionEvaluation] = []
    for fold in range(core.RT_EXPECTED_FOLDS):
        sealed = rt.load_sealed_stage2_cell(stage2_cell_root, fold)
        nwb_path = sessions[sealed["session_name"]]
        blocks = core.build_rt_fold_blocks(
            fold=fold,
            nwb_path=nwb_path,
            stage2_cell_root=stage2_cell_root,
            raw_loader=load_rt_session,
        )
        results.append(core.evaluate_session(blocks, q_floor=q_floor, w_floor=w_floor))
    return {
        "dataset": "rt",
        "summary": core.summarize_sessions(results),
        "per_session": [
            {
                "session_name": item.session_name,
                "success": item.success,
                "failure_reason": item.failure_reason,
                "r2": item.r2,
                "metric_name": item.metric_name,
                "calibration_rows": item.calibration_rows,
                "query_rows": item.query_rows,
                "query_identity": item.query_identity,
                "fit": {
                    "q_floor_applied": item.fit.q_floor_applied,
                    "w_floor_applied": item.fit.w_floor_applied,
                    "parameters": item.parameters.as_dict() if item.parameters is not None else None,
                },
            }
            for item in results
        ],
    }


def run_h1(*, data_dir: Path, q_floor: float, w_floor: float) -> dict[str, Any]:
    blocks = core.load_all_h1_blocks(data_dir=data_dir, repo_root=REPO_ROOT)
    results = [core.evaluate_session(block, q_floor=q_floor, w_floor=w_floor) for block in blocks]
    pooled_truth = np.concatenate([block.query_velocity_truth for block in blocks], axis=0)
    # Pooled metric is session-concatenated query truths vs predictions recomputed below.
    predictions = []
    for block, item in zip(blocks, results):
        require(item.success and item.parameters is not None, f"H1 session failed: {block.session_name}")
        filtered = core.kalman_filter_query(
            block.query_rates,
            item.parameters,
            initial_state=block.calibration_states[-1],
        )
        predictions.append(filtered[:, core.velocity_slice(block.kinematic_dim)])
    pooled_prediction = np.concatenate(predictions, axis=0)
    pooled_r2 = core.r2_pooled(pooled_truth, pooled_prediction)
    return {
        "dataset": "falcon_h1",
        "summary": core.summarize_sessions(results),
        "pooled_r2": float(pooled_r2),
        "per_session": [
            {
                "session_name": item.session_name,
                "success": item.success,
                "failure_reason": item.failure_reason,
                "r2": item.r2,
                "metric_name": item.metric_name,
                "calibration_rows": item.calibration_rows,
                "query_rows": item.query_rows,
                "query_identity": item.query_identity,
                "fit": {
                    "q_floor_applied": item.fit.q_floor_applied,
                    "w_floor_applied": item.fit.w_floor_applied,
                    "parameters": item.parameters.as_dict() if item.parameters is not None else None,
                },
            }
            for item in results
        ],
    }


def run_experiment(
    *,
    dataset: str,
    view: str | None,
    budget: int,
    data_dir: Path,
    v9_run_root: Path,
    stage2_cell_root: Path,
    h1_data_dir: Path,
    output_dir: Path,
    q_floor: float,
    w_floor: float,
) -> dict[str, Any]:
    if dataset == "subject_m":
        require(view is not None, "subject-M requires --view")
        results = run_subject_m(
            view=view,
            budget=budget,
            data_dir=data_dir,
            v9_run_root=v9_run_root,
            q_floor=q_floor,
            w_floor=w_floor,
        )
        receipt_name = f"kalman_comparator_subject_m_{view}_m{budget}_receipt.json"
    elif dataset == "rt":
        results = run_rt(data_dir=data_dir, stage2_cell_root=stage2_cell_root, q_floor=q_floor, w_floor=w_floor)
        receipt_name = "kalman_comparator_rt_receipt.json"
    elif dataset == "falcon_h1":
        results = run_h1(data_dir=h1_data_dir, q_floor=q_floor, w_floor=w_floor)
        receipt_name = "kalman_comparator_falcon_h1_receipt.json"
    elif dataset == "falcon_m2":
        raise RuntimeError(
            "falcon_m2 experiment run is blocked: sealed M24 sources require held-out-calib NWBs "
            "which this agent session must not open"
        )
    else:
        raise RuntimeError(f"unknown dataset: {dataset}")

    receipt = {
        "schema": "kalman_comparator_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": time.strftime("%Y-%m-%d"),
        "dataset": dataset,
        "scope": {
            "cuda_used": False,
            "q_floor": float(q_floor),
            "w_floor": float(w_floor),
            "state_parameterisation": "[position, velocity, constant]",
        },
        "results": results,
        "input_bindings": {
            "data_dir": str(data_dir.resolve()),
            "v9_run_root": str(v9_run_root.resolve()) if dataset == "subject_m" else None,
            "stage2_cell_root": str(stage2_cell_root.resolve()) if dataset == "rt" else None,
            "h1_data_dir": str(h1_data_dir.resolve()) if dataset == "falcon_h1" else None,
            "implementation_sha256": {
                "kalman_comparator.py": core.sha256_file(Path(core.__file__)),
                "run_kalman_comparator.py": core.sha256_file(Path(__file__)),
            },
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    receipt_path = output_dir / receipt_name
    require(not receipt_path.exists(), f"refusing to overwrite {receipt_path}")
    receipt_sha = write_immutable(receipt_path, core.canonical_json_bytes(receipt))
    return {
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "dataset": dataset,
        "summary": results.get("summary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Velocity Kalman comparator runner")
    parser.add_argument(
        "--dataset",
        choices=("subject_m", "falcon_m2", "rt", "falcon_h1", "all_dry_run"),
        required=True,
    )
    parser.add_argument("--view", choices=("sua", "pseudo_mua"), default=None)
    parser.add_argument("--budget", type=int, default=core.SUBM_FIT_TRIALS)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_SUBJECT_M_DATA)
    parser.add_argument("--rt-data-dir", type=Path, default=DEFAULT_RT_DATA)
    parser.add_argument("--h1-data-dir", type=Path, default=DEFAULT_H1_DATA)
    parser.add_argument("--v9-run-root", type=Path, default=DEFAULT_V9_RUN_ROOT)
    parser.add_argument("--stage2-cell-root", type=Path, default=DEFAULT_STAGE2_CELL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--q-floor", type=float, default=core.DEFAULT_Q_FLOOR)
    parser.add_argument("--w-floor", type=float, default=core.DEFAULT_W_FLOOR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run or args.dataset == "all_dry_run":
        payload = core.dry_run_all(
            repo_root=REPO_ROOT,
            data_dir_subject_m=args.data_dir.resolve(),
            data_dir_rt=args.rt_data_dir.resolve(),
            data_dir_h1=args.h1_data_dir.resolve(),
            v9_run_root=args.v9_run_root.resolve(),
            stage2_cell_root=args.stage2_cell_root.resolve(),
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    summary = run_experiment(
        dataset=args.dataset,
        view=args.view,
        budget=args.budget,
        data_dir=args.data_dir.resolve(),
        v9_run_root=args.v9_run_root.resolve(),
        stage2_cell_root=args.stage2_cell_root.resolve(),
        h1_data_dir=args.h1_data_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        q_floor=float(args.q_floor),
        w_floor=float(args.w_floor),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
