#!/usr/bin/env python3
"""Dry by default; reserve the k-curve-extension attempt or execute the grid.

CPU-only: launch with CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE=1,
OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TFPD_ROOT = REPO_ROOT / "tfpd_exploration"
for item in (REPO_ROOT, TFPD_ROOT, TFPD_ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.m2_kcurve_ext_v1 import plan  # noqa: E402

OWNED_RELATIVE = plan.OWNED_PATHS
PREDECESSOR_RELATIVE = plan.PREDECESSOR_RELATIVE
TESTS_RELATIVE = "tfpd_exploration/tests/test_m2_kcurve_ext_v1.py"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_attempt() -> dict[str, object]:
    root = plan.result_root(REPO_ROOT)
    target = root / "attempt.json"
    if target.exists():
        raise SystemExit(f"attempt receipt already exists: {target}")
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": "M2_KCURVE_EXT_V1",
        "authority": (
            "operator work order 2026-09-02: the M2 k-curve EXTENSION on top "
            "of the sealed k-curve (results/m2_kcurve_v1/) -- Q1: does the "
            "external k-decline of CDM survive UNBOUNDED accumulation "
            "(FIFO-cap-30 vs the m2_memory_law_scan_v1 UNCAPPED law imported "
            "verbatim)?  Q2: does the pool size B interact with k (D-opt "
            "prefix from the finite-angle candidates within the FIRST-B "
            "positions, B in {10,20,30}, B30 = the sealed column)?  Grid law "
            "x B x k with k in {4, min(5,usable_B), 8, min(10,usable_B), "
            "all-usable_B} capped at per-session usable within B; every cell "
            "scored on the SAME sealed post-30 window partition; STATIC rows "
            "for B30 (the sealed column, anchors) plus B10/B20 k=4; surfaces "
            "external_post30_local (primary) + within_post30 (secondary); "
            "CPU-only (CUDA_VISIBLE_DEVICES='', torch threads 4, 0 workers)"
        ),
        "inference_only": True,
        "pre_registration": {
            "foundation": {
                "sealed_kcurve_replay": plan.KCURVE_REPLAY_RELATIVE,
                "sealed_kcurve_replay_sha256": plan.KCURVE_REPLAY_SHA256,
                "sealed_memory_scan_replay": plan.MEMORY_SCAN_REPLAY_RELATIVE,
                "sealed_memory_scan_replay_sha256": plan.MEMORY_SCAN_REPLAY_SHA256,
                "sealed_reblock10_replay": plan.REBLOCK10_REPLAY_RELATIVE,
                "sealed_reblock10_replay_sha256": plan.REBLOCK10_REPLAY_SHA256,
                "reuse_law": "frozen modules imported verbatim; never edited",
            },
            "questions": {
                "q1": (
                    "does the external k-decline of CDM survive unbounded "
                    "accumulation? (UNCAPPED vs FIFO30 paired delta at every "
                    "(B,k); verdict CAPACITY_COMPETITION_CONFIRMED iff the "
                    "B30 external decline slope under UNCAPPED is not steeper "
                    "than the STATIC no-memory control's slope by more than "
                    "0.005, else CARRIER_OVERFIT_DOMINANT)"
                ),
                "q2": (
                    "does the pool size B interact with k? (paired B10/B20 vs "
                    "B30 deltas at matched k under UNCAPPED; pool narrowing "
                    "cost, descriptive)"
                ),
                "q3": "the best external cell overall (law, B, k) with its value",
            },
            "b_pool_law": dict(plan.B_POOL_LAW),
            "memory_laws": dict(plan.MEMORY_LAWS),
            "law_equivalence_disclosure": plan.LAW_EQUIVALENCE_DISCLOSURE,
            "b_axis": list(plan.B_AXIS),
            "k_grid_template": list(plan.K_GRID_TEMPLATE),
            "surface": dict(plan.SURFACE_LAW),
            "frozen_everything_else": dict(plan.FROZEN_EVERYTHING_ELSE),
            "anchors": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in plan.ANCHORS.items()
            },
            "readout_law": dict(plan.READOUT_LAW),
            "verdict_strings": list(plan.VERDICTS),
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "cpu_binding_plan": {
            **plan.ENVIRONMENT_LAW,
            "hard_timeout_seconds": plan.HARD_TIMEOUT_SECONDS,
            "one_launch_one_process": True,
        },
        "tests": {
            "path": TESTS_RELATIVE,
            "sha256": _sha256_file(REPO_ROOT / TESTS_RELATIVE),
            "required_green_before_execute": True,
            "scope": "no-data no-CUDA (first-B candidate law, per-B D-opt "
                     "prefix law, B-column grid resolution, B30 identity with "
                     "the sealed law, decline slope, Q1 verdict rule, matched-k "
                     "law, best-cell law, exact-CPU anchor matcher, plan "
                     "bindings)",
        },
        "owned_sha256s": {item: _sha256_file(REPO_ROOT / item) for item in OWNED_RELATIVE},
        "predecessor_sha256s": {
            item: _sha256_file(REPO_ROOT / item) for item in PREDECESSOR_RELATIVE
        },
        "target_gradients": 0,
        "parameter_updates": 0,
        "model_or_checkpoint_updated": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    target.write_bytes(body)
    target.chmod(0o444)
    sidecar = target.with_name(target.name + ".sha256")
    sidecar.write_text(f"{digest}  attempt.json\n", encoding="ascii")
    sidecar.chmod(0o444)
    return {"status": payload["status"], "attempt": str(target), "attempt_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dry", "attempt", "execute"), default="dry")
    parser.add_argument("--batch-size", type=int, default=plan.BATCH_SIZE)
    args = parser.parse_args()
    if args.stage == "dry":
        print(json.dumps({
            "schema": f"{plan.SCHEMA}_dry",
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE_NO_LAUNCH",
            "stages": ["attempt", "execute"],
            "laws": list(plan.LAWS),
            "b_axis": list(plan.B_AXIS),
            "families": list(plan.FAMILIES),
            "surfaces": list(plan.SURFACES),
            "k_grid_template": list(plan.K_GRID_TEMPLATE),
            "device": "cpu",
            "official_contract_claim": False,
        }, sort_keys=True))
        return
    if args.stage == "attempt":
        print(json.dumps(_write_attempt(), indent=2, sort_keys=True))
        return
    from src.m2_kcurve_ext_v1 import physical

    result = physical.execute(REPO_ROOT, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
