#!/usr/bin/env python3
"""Print the inert plan for the M1 matched T0/C1 50-epoch prefix pair.

This public CLI intentionally does not import the route package, Torch, the
M1 parser, or an artifact-root helper.  Only an in-process root-reviewed
caller may execute the smoke, the two 50-epoch arms (t0 and c1 may share
GPU 1), the overfitting probe, and the Phase-3 2x2x2 tables.
"""
from __future__ import annotations

import argparse
import json


def _payload() -> dict[str, object]:
    return {
        "cell": "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1",
        "phase": "m1_t0c1_prefix_v1_50ep",
        "prospective_roots": {
            "smoke": "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/smoke",
            "t0": "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/t0",
            "c1": "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/c1",
            "probe": "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/probe",
            "phase3": "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/phase3_table",
        },
        "arms": {"t0": "operator disabled, same runner", "c1": "prefix cycle (10, 5, 2)"},
        "budget": {
            "seed": 42, "epochs": 50, "steps_per_epoch": 4951,
            "total_optimizer_steps": 247550, "batch": 32, "swa": False,
        },
        "lr_schedule_law": {
            "kind": "warmup_then_cosine", "n_epochs": 50, "steps_per_epoch": 4951,
            "total_steps": 247550, "warmup_epochs": 2, "warmup_steps": 9902,
            "lr_warmup_start": 1e-5, "lr_warmup_end": 1e-4, "lr_final": 1e-6,
            "phase_local_steps": True,
        },
        "checkpoint_epoch_indices_0based": [9, 19, 29, 39, 49],
        "checkpoint_epochs_1based": [10, 20, 30, 40, 50],
        "dropout_proof_bound_before_launch": True,
        "dropout_block_sha256_both_trees": "eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884",
        "hard_timeout_seconds_per_arm": 43200,
        "runtime_abort_estimate_seconds": 40000,
        "linear_projection_50ep_hours": {"t0": 6.29, "c1": 5.43},
        "post_memo_seconds_per_step": {
            "t0": 0.08577056604618638, "c1": 0.07313887172792367,
        },
        "full_run_seconds_per_step_not_post_memo": {"t0": 0.10018, "c1": 0.08771},
        "selection_surface": "val_heldout",
        "val_heldout_sessions": ["20121004", "20121017", "20121024"],
        "probe_grid": "2 arms x 5 checkpoints x 7 sessions = 70 scored cells",
        "predecessor_20ep_terminal_sha256": {
            "t0": "4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9",
            "c1": "cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99",
        },
        "predecessor_20ep_phase3_terminal_sha256": (
            "821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096"
        ),
        "launch_envelope": {
            "PYTHON": "/home/xinyuan/miniconda3/envs/spint/bin/python",
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONNOUSERSITE": "1",
        },
        "order": ["smoke", "t0", "c1", "probe", "phase3"],
        "gpu": "CUDA_VISIBLE_DEVICES=1",
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the static pair contract")
    parser.add_argument("--execute", action="store_true", help="always rejected by the public CLI")
    arguments = parser.parse_args(argv)
    if arguments.execute:
        parser.error("m1 t0c1 50ep pair requires an in-process root-reviewed capability")
    print(json.dumps(_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
