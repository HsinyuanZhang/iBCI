#!/usr/bin/env python3
"""Affine diagnostics on the frozen T0/C1 deployment predictions (§8, item 5).

Re-decodes both arms once through the EXACT deployment scoring law of
``src/cal_aug_v1/deployment.py`` (same materialization, same recipe, same
dropout-inactive proof, same last-bin/target/valid convention), caches the
per-(arm, surface, session, budget) last-bin streams under
``cache/cal_aug_affine_v1/``, verifies each row's prediction SHA against the
sealed deployment receipt, and runs the §8 affine diagnostics per arm-cell
plus the deployable cross-surface r0.  Zero training; read-only on the arms.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
# ROOT last: tfpd_exploration/src must win `import src`
sys.path.insert(0, str(ROOT))

from src.cal_aug_v1 import deployment, plan, receipts  # noqa: E402
from src.affine_diagnostics_v1 import affine  # noqa: E402

CACHE = ROOT / "cache/cal_aug_affine_v1"
OUT = ROOT / "results/cal_aug_v1/affine"
DEPLOYMENT_RECEIPT = ROOT / "results/cal_aug_v1/deployment/terminal.json"
ARMS = {
    "t0": ROOT / "results/cal_aug_v1/t0_operator_disabled/swa_final4.pt",
    "c1": ROOT / "results/cal_aug_v1/c1_prefix_cycle/swa_final4.pt",
}


def main() -> int:
    from src.calibration_gap_v1 import z1_oracle_cells as z1

    if not sys.flags.no_user_site:
        raise SystemExit("PYTHONNOUSERSITE=1 is mandatory")
    if OUT.exists():
        raise SystemExit(f"fresh root required: {OUT}")
    OUT.mkdir(parents=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write = receipts._write_json if hasattr(receipts, "_write_json") else None

    stack = receipts.load_sealed_runner_stack(REPO, ROOT)
    arm_common, pop_robust = stack["arm_common"], stack["pop_robust"]
    deployment_receipt = json.loads(DEPLOYMENT_RECEIPT.read_text())
    sealed_rows = deployment_receipt.get("result", deployment_receipt)["arms"]

    CACHE.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    streams_by_cell: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    wall0 = time.time()
    for arm, swa in ARMS.items():
        runtime = z1.HonestOracleRuntime(root=REPO)
        deployment._swap_runtime_model(runtime, pop_robust, arm_common, swa)
        import torch

        from src.calibration_gap_v1 import p4_stream_stats as p4

        for surface in deployment.DEPLOYMENT_SURFACES:
            roster = runtime.external_roster if surface == "external" else runtime.within_roster
            for session_name in roster:
                inputs = p4.materialize_session(runtime, surface, session_name)
                for budget in deployment.DEPLOYMENT_BUDGETS:
                    selected = inputs.selected_by_budget[budget]
                    prediction, _ = p4._decode_static(
                        runtime, inputs, inputs.calib[list(selected)],
                        inputs.side_by_budget[budget],
                    )
                    last = prediction[:, 49, :].detach().contiguous().numpy().astype(np.float64)
                    target = np.asarray(inputs.last_targets, dtype=np.float64)
                    valid = np.asarray(inputs.last_valid_mask, dtype=bool)
                    sha = hashlib.sha256(
                        prediction.detach().contiguous().numpy().tobytes()
                    ).hexdigest()
                    sealed = [
                        r for r in sealed_rows[arm]["rows"]
                        if r["surface"] == surface and r["session"] == session_name
                        and int(r["budget"]) == int(budget)
                    ][0]
                    if sha != sealed["prediction_sha256"]:
                        raise SystemExit(
                            f"prediction SHA drift vs sealed deployment receipt: "
                            f"{arm} {surface} {session_name} M{budget}"
                        )
                    key = f"{arm}:{surface}:{session_name}:m{int(budget)}"
                    rel = f"{key.replace(':', '_')}.npz"
                    np.savez(
                        CACHE / rel, pred=last, target=target, valid=valid,
                    )
                    manifest[key] = {
                        "relative": rel,
                        "sha256": hashlib.sha256((CACHE / rel).read_bytes()).hexdigest(),
                        "r2": float(sealed["r2"]),
                        "prediction_sha256": sha,
                    }
                    cell = f"{arm}_{surface}_m{int(budget)}"
                    streams_by_cell.setdefault(cell, {})[session_name] = (
                        last[valid], target[valid],
                    )
                print(f"[affine] {arm} {surface} {session_name} cached", flush=True)
        runtime.close()

    cells: dict[str, object] = {}
    cross: dict[str, object] = {}
    for cell, rows in sorted(streams_by_cell.items()):
        opportunity = affine.full_opportunity_rows(rows)
        disp = affine.dispersion_stats(
            [np.asarray(list(r["six_vector"].values())) for r in opportunity]
        )
        loo = affine.loo_r0_rows(rows)
        cells[cell] = {
            "n_sessions": len(rows),
            "full_affine_opportunity_equal_session_mean_gain": float(np.mean(
                [r["gain"] for r in opportunity]
            )),
            "dispersion": disp,
            "loo_r0_within_cell": loo,
        }
        print(
            f"{cell}: full={float(np.mean([r['gain'] for r in opportunity])):+.4f}"
            f" loo_r0={loo['equal_session_mean_r0_gain']:+.4f}"
            f" ({loo['n_positive']}/{loo['n_sessions']})",
            flush=True,
        )
    for arm in ("t0", "c1"):
        for budget in (4, 10, 30):
            within = streams_by_cell.get(f"{arm}_within_m{budget}")
            external = streams_by_cell.get(f"{arm}_external_m{budget}")
            if within and external:
                row = affine.cross_surface_r0_rows(within, external)
                cross[f"{arm}_external_m{budget}"] = row
                print(
                    f"{arm}_external_m{budget}: CROSS_R0"
                    f" {row['equal_session_mean_r0_gain']:+.4f}"
                    f" ({row['n_positive']}/{row['n_sessions']})",
                    flush=True,
                )

    body = {
        "schema": "cal_aug_v1_affine_diagnostics",
        "status": "COMPLETE",
        "design_authority": (
            "EXECUTION_GUIDANCE section 8 (queue item 5) on the frozen T0/C1 "
            "deployment predictions"
        ),
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deployment_receipt_sha256": hashlib.sha256(
            DEPLOYMENT_RECEIPT.read_bytes()
        ).hexdigest(),
        "arm_swa_sha256": {
            arm: receipts.sha256_file(path) for arm, path in ARMS.items()
        },
        "rows_prediction_sha_verified_against_deployment_receipt": True,
        "cache_manifest": manifest,
        "cells": cells,
        "cross_surface_r0_deployable": cross,
        "forwards_are_read_only_scoring": True,
        "target_optimizer_backward_update": 0,
        "model_or_checkpoint_updated": False,
        "wall_seconds": time.time() - wall0,
    }
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    tmp = OUT / "terminal.json.tmp"
    tmp.write_text(text)
    os.replace(tmp, OUT / "terminal.json")
    os.chmod(OUT / "terminal.json", 0o444)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (OUT / "terminal.json.sha256").write_text(f"{digest}  terminal.json\n")
    os.chmod(OUT / "terminal.json.sha256", 0o444)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
