#!/usr/bin/env python3
"""CPU-only runner for the FA-alignment comparator.  Real data: --audit-only only."""
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
REPO_ROOT = SCRIPT_DIR.parents[1]
for path in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "SPINT-main", REPO_ROOT / "streaming_calibration_exp"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sua_exploration.mc_maze import fa_alignment_comparator as core


DEFAULT_SUBJECT_M_DATA = REPO_ROOT / "sua_exploration/data/dandi_000688/sub-M"
DEFAULT_RT_DATA = REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"
DEFAULT_STAGE2_CELL_ROOT = (
    REPO_ROOT / "sua_exploration/results/rt_terminal_stage2_20260811_canonical/matrix_v1/cells"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/fa_alignment_comparator"


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


def receipt_path_for(dataset: str, view: str | None, output_dir: Path, *, scoring: bool = False) -> Path:
    if dataset == "subject_m":
        require(view is not None, "subject-M requires --view")
        suffix = "scoring" if scoring else "audit"
        return output_dir / f"fa_alignment_comparator_subject_m_{view}_{suffix}_receipt.json"
    if dataset == "rt":
        suffix = "scoring" if scoring else "audit"
        return output_dir / f"fa_alignment_comparator_rt_{suffix}_receipt.json"
    raise RuntimeError(f"unknown dataset: {dataset}")


def nwb_sha256_map(paths: dict[str, Path]) -> dict[str, str]:
    return {name: core.sha256_file(path) for name, path in sorted(paths.items())}


def run_audit(
    *,
    dataset: str,
    view: str | None,
    data_dir: Path,
    stage2_cell_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    start = time.monotonic()
    if dataset == "subject_m":
        require(view in core.VIEWS, "subject-M requires --view sua|pseudo_mua")
        sessions = core.discover_subject_m_sessions(data_dir)
        results = core.audit_subject_m(data_dir=data_dir, view=view)
        nwb_sha = nwb_sha256_map(sessions)
    elif dataset == "rt":
        from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions, load_rt_session
        from sua_exploration.mc_maze import rt_classical_comparators as rt

        sessions = {
            rt.session_name_from_nwb_path(Path(path)): core.assert_path_in_scope(Path(path))
            for path in find_rt_sessions(data_dir)
        }
        results = core.audit_rt(data_dir=data_dir, stage2_cell_root=stage2_cell_root, raw_loader=load_rt_session)
        nwb_sha = nwb_sha256_map(sessions)
    else:
        raise RuntimeError(f"unknown dataset: {dataset}")

    implementation_paths = {
        "fa_alignment_comparator.py": Path(core.__file__),
        "run_fa_alignment_comparator.py": Path(__file__),
        "source_pooled_ridge.py": REPO_ROOT / "sua_exploration/mc_maze/source_pooled_ridge.py",
        "rt_classical_comparators.py": REPO_ROOT / "sua_exploration/mc_maze/rt_classical_comparators.py",
        "subm_v9_f0_pv_ridge.py": REPO_ROOT / "sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py",
    }
    receipt = {
        "schema": "fa_alignment_comparator_v1",
        "status": "AUDIT_ONLY_CPU",
        "date": time.strftime("%Y-%m-%d"),
        "protocol_date": core.PROTOCOL_DATE,
        "dataset": dataset,
        "view": view if dataset == "subject_m" else "rt",
        "audit_only": True,
        "scoring_arm_executed": False,
        "scope": {
            "cuda_used": False,
            "target_labels_used_in_alignment": False,
            "alignment_fitted": False,
            "datasets_in_scope": ["subject_m", "rt"],
            "datasets_out_of_scope": ["h1", "m2"],
            "seed_material": core.SEED_MATERIAL,
            "seed": core.seed_from_material(),
        },
        "part_a": results,
        "input_bindings": {
            "data_dir": str(data_dir.resolve()),
            "stage2_cell_root": str(stage2_cell_root.resolve()) if dataset == "rt" else None,
            "nwb_sha256": nwb_sha,
            "implementation_sha256": {
                name: core.sha256_file(path) for name, path in sorted(implementation_paths.items())
            },
            "query_identity": results.get("query_identity"),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "elapsed_seconds": time.monotonic() - start,
        "receipt_policy": "exclusive new filename; never overwrite; chmod 0444",
    }
    receipt_path = receipt_path_for(dataset, view, output_dir)
    require(not receipt_path.exists(), f"refusing to overwrite {receipt_path}")
    receipt_sha = write_immutable(receipt_path, core.canonical_json_bytes(receipt))
    return {
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "dataset": dataset,
        "view": receipt["view"],
        "verdicts": results["verdicts"],
        "buildable_arms": results["buildable_arms"],
        "correspondence": {
            "correspondent_channel_count": results["correspondence"]["correspondent_channel_count"],
            "channel_id_sets_intersection_size": results["correspondence"]["channel_id_sets_intersection_size"],
            "channel_id_sets_union_size": results["correspondence"]["channel_id_sets_union_size"],
            "stable_index_mapping_exists": results["correspondence"]["stable_index_mapping_exists"],
            "all_sessions_identical_channel_ids": results["correspondence"]["all_sessions_identical_channel_ids"],
        },
        "shared_d": results["latent_dimensionality"],
    }


def run_scoring(
    *,
    dataset: str,
    data_dir: Path,
    stage2_cell_root: Path,
    output_dir: Path,
    arms: tuple[str, ...],
) -> dict[str, Any]:
    start = time.monotonic()
    require(dataset == "rt", "this runner scores RT only; subject-M remains out of scope")
    require("faa_a1_anchored" not in arms, "faa_a1_anchored is refused on RT")
    from streaming_calibration_exp.src.data.rt_k4_loader import find_rt_sessions, load_rt_session
    from sua_exploration.mc_maze import rt_classical_comparators as rt

    results = core.score_rt(
        data_dir=data_dir,
        stage2_cell_root=stage2_cell_root,
        raw_loader=load_rt_session,
        d=core.SELECTED_RT_D,
        arms=arms,
    )
    sessions = {
        rt.session_name_from_nwb_path(Path(path)): core.assert_path_in_scope(Path(path))
        for path in find_rt_sessions(data_dir)
    }
    implementation_paths = {
        "fa_alignment_comparator.py": Path(core.__file__),
        "run_fa_alignment_comparator.py": Path(__file__),
        "rt_classical_comparators.py": REPO_ROOT / "sua_exploration/mc_maze/rt_classical_comparators.py",
        "subm_v9_f0_pv_ridge.py": REPO_ROOT / "sua_exploration/mc_maze/subm_v9_f0_pv_ridge.py",
    }
    receipt = {
        "schema": "fa_alignment_comparator_scoring_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": time.strftime("%Y-%m-%d"),
        "protocol_date": core.PROTOCOL_DATE,
        "dataset": "rt",
        "view": "rt",
        "audit_only": False,
        "scoring_arm_executed": True,
        "scope": {
            "cuda_used": False,
            "target_labels_used_in_alignment": False,
            "alignment_fitted": True,
            "datasets_in_scope": ["rt"],
            "datasets_out_of_scope": ["subject_m", "h1", "m2"],
            "selected_d": core.SELECTED_RT_D,
            "shared_d_defensible": False,
            "shared_d_limitation": core.RT_SHARED_D_LIMITATION,
            "seed_material": core.SEED_MATERIAL,
            "seed": core.seed_from_material(),
        },
        "results": results,
        "input_bindings": {
            "data_dir": str(data_dir.resolve()),
            "stage2_cell_root": str(stage2_cell_root.resolve()),
            "nwb_sha256": nwb_sha256_map(sessions),
            "implementation_sha256": {
                name: core.sha256_file(path) for name, path in sorted(implementation_paths.items())
            },
            "query_identity": results["query_identity"],
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "elapsed_seconds": time.monotonic() - start,
        "receipt_policy": "exclusive new filename; never overwrite; chmod 0444",
    }
    receipt_path = receipt_path_for("rt", None, output_dir, scoring=True)
    require(not receipt_path.exists(), f"refusing to overwrite {receipt_path}")
    receipt_sha = write_immutable(receipt_path, core.canonical_json_bytes(receipt))
    return {
        "status": receipt["status"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "dataset": "rt",
        "query_identity": results["query_identity"],
        "target_labels_used_in_alignment": False,
        "arms": results["arms"],
        "contrasts": results["contrasts"],
        "shared_d_limitation": core.RT_SHARED_D_LIMITATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FA-alignment comparator (CPU)")
    parser.add_argument("--dataset", choices=("subject_m", "rt"), required=True)
    parser.add_argument("--view", choices=core.VIEWS, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(core.SCORING_ARMS_RT),
        choices=core.ARMS,
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--stage2-cell-root", type=Path, default=DEFAULT_STAGE2_CELL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.dataset == "subject_m":
        data_dir = (args.data_dir or DEFAULT_SUBJECT_M_DATA).resolve()
    else:
        data_dir = (args.data_dir or DEFAULT_RT_DATA).resolve()
    if args.audit_only:
        summary = run_audit(
            dataset=args.dataset,
            view=args.view,
            data_dir=data_dir,
            stage2_cell_root=args.stage2_cell_root.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    else:
        if args.dataset != "rt":
            raise SystemExit("scoring is implemented for RT only in this session")
        if "faa_a1_anchored" in args.arms:
            raise SystemExit("faa_a1_anchored is refused on RT: FAA_UNDEFINED_NO_CHANNEL_CORRESPONDENCE")
        summary = run_scoring(
            dataset=args.dataset,
            data_dir=data_dir,
            stage2_cell_root=args.stage2_cell_root.resolve(),
            output_dir=args.output_dir.resolve(),
            arms=tuple(args.arms),
        )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
