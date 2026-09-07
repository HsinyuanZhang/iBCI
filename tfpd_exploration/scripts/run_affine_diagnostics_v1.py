#!/usr/bin/env python3
"""Run the §8 affine-miscalibration diagnostics on frozen cached streams.

Reads the sealed stream cache of the learnable-output-filter line (kinds
``cdm`` = activity-only CDM A0 rollouts and ``static`` = the frozen static
deployment recipe) and reports, per (kind, surface, budget) cell:

* full own-affine opportunity per session (leakage-labelled oracle rows);
* the dispersion statistics of the session six-vectors (§8.1 list);
* the deployable ``r0`` diagnostic (leave-one-out source-mean correction).

Zero forwards, zero training; target optimizer/backward/update counts are 0.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane.receipt import (
    enforce_environment,
    sha256_file,
    write_receipt_transactionally,
)
from src.learnable_output_filter_v1 import ladder, streams
from src.affine_diagnostics_v1 import affine

CACHE = ROOT / "cache/learnable_output_filter_v1"
OUT = ROOT / "results/affine_diagnostics_v1_r1"


def cell_rows(streams_map) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (surface, session, budget), stream in streams_map.items():
        prediction, target = ladder.joined_valid_rows(
            [block.raw for block in stream.blocks], stream, dtype="float64",
        )
        rows[session] = (prediction, target)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", default="cdm,static")
    args = parser.parse_args()

    enforce_environment()
    if OUT.exists():
        raise SystemExit(f"fresh successor root required: {OUT}")
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.mkdir(parents=True)
    write_receipt_transactionally(OUT / "attempt.json", {
        "schema": "affine_diagnostics_v1_attempt",
        "started_utc": started,
        "kinds": args.kinds,
        "design_authority": (
            "tfpd_exploration/docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_"
            "AUDIT_20260829.md section 8"
        ),
        "status": "ATTEMPT_PUBLISHED_BEFORE_DATA_ACCESS",
    })

    manifest_sha = sha256_file(streams.manifest_path(CACHE))
    cells: dict[str, object] = {}
    cross_surface: dict[str, object] = {}
    for kind in [item.strip() for item in args.kinds.split(",") if item.strip()]:
        loaded = streams.load_all(CACHE, kind)
        by_cell: dict[tuple[str, int], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for surface in sorted({key[0] for key in loaded}):
            for budget in sorted({key[2] for key in loaded if key[0] == surface}):
                subset = {
                    key: value for key, value in loaded.items()
                    if key[0] == surface and key[2] == budget
                }
                rows = cell_rows(subset)
                if len(rows) < 3:
                    continue
                by_cell[(surface, budget)] = rows
                cell_key = f"{kind}_{surface}_m{budget}"
                opportunity = affine.full_opportunity_rows(rows)
                gains = [row["gain"] for row in opportunity]
                disp = affine.dispersion_stats(
                    [np.asarray(list(row["six_vector"].values())) for row in opportunity],
                )
                r0 = affine.loo_r0_rows(rows)
                cells[cell_key] = {
                    "n_sessions": len(rows),
                    "full_affine_opportunity": {
                        "equal_session_mean_gain": float(np.mean(gains)),
                        **affine.paired_session_stats(gains),
                        "sessions": opportunity,
                    },
                    "dispersion": disp,
                    "loo_r0_within_cell": r0,
                }
                print(
                    f"{cell_key}: n={len(rows)}"
                    f" full={float(np.mean(gains)):+.4f}"
                    f" loo_r0={r0['equal_session_mean_r0_gain']:+.4f}"
                    f" ({r0['n_positive']}/{r0['n_sessions']})"
                    f" trace={disp['covariance_trace']:.3e}"
                    f" pc1={disp['first_pc_explained_fraction']:.3f}",
                    flush=True,
                )
        for budget in sorted({key[1] for key in by_cell if key[0] == "within"}
                             & {key[1] for key in by_cell if key[0] == "external"}):
            cross = affine.cross_surface_r0_rows(
                by_cell[("within", budget)], by_cell[("external", budget)],
            )
            cross_surface[f"{kind}_external_m{budget}"] = cross
            print(
                f"{kind}_external_m{budget}: CROSS_R0"
                f" {cross['equal_session_mean_r0_gain']:+.4f}"
                f" ({cross['n_positive']}/{cross['n_sessions']})",
                flush=True,
            )

    body = {
        "schema": "affine_diagnostics_v1",
        "status": "COMPLETE",
        "design_authority": (
            "tfpd_exploration/docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_"
            "AUDIT_20260829.md section 8 (queue item 5)"
        ),
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_manifest_sha256": manifest_sha,
        "cells": cells,
        "cross_surface_r0_deployable": cross_surface,
        "leakage_rule": (
            "own-affine rows are oracle diagnostics: target_label_leakage=true, "
            "checkpoint_selection_eligible=false, deployment_eligible=false; "
            "the r0 rows fit nothing on the held session"
        ),
        "forwards_performed_by_this_run": 0,
        "target_optimizer_backward_update": 0,
        "model_or_checkpoint_updated": False,
    }
    write_receipt_transactionally(OUT / "terminal.json", body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
