#!/usr/bin/env python
"""AC3-0 runner: attempt -> materialize (brief GPU inference) -> select -> screen -> finalize.

Stages
------
``attempt``     reserve the result root and pin the owned module bytes +
                pre-registrations BEFORE any data access (attempt-before-data).
``materialize`` the only GPU touch: inference-only four-group trajectories of
                the six SOURCE sessions on the P2' never-commit line, gated on
                both GPUs being idle, wall time inside the brief-inference gate.
``select``      CPU hyperparameter selection on grouped source folds.
``screen``      CPU evaluation of the amended AC3-0 matrix + gates + controls.
``finalize``    terminal receipt and a read-only result root.
``dry-plan``    print the frozen contract without touching data.

Example
-------
    PYTHONNOUSERSITE=1 python scripts/run_ac3_action_continuity_v0.py attempt
    SUBC_DATA_ROOT=... SUBM_DATA_ROOT=... CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
        python scripts/run_ac3_action_continuity_v0.py materialize --gpu-index 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.ac3_action_continuity_v1 import materialize as mat
from src.ac3_action_continuity_v1 import plan, screen


def _base() -> Path:
    return Path(__file__).resolve().parents[2]


def _attempt(base: Path, output_root: Path | None = None) -> dict[str, object]:
    from src.ac3_action_continuity_v1 import screen as screen_module

    output = output_root or (base / plan.RESULT_ROOT_RELATIVE)
    if output.exists():
        raise SystemExit(f"the AC3-0 result root already exists: {output}")
    output.mkdir(mode=0o755, parents=True, exist_ok=False)
    payload = {
        "schema": f"{plan.SCHEMA}_attempt_v1",
        "status": "ATTEMPT_RESERVED",
        "cell": plan.CELL,
        "scope": "non_governing_frozen_representation_screen",
        "addendum_authority": plan.ADDENDUM_RELATIVE,
        "amendments": "section 23 operator decision of 2026-08-29 (binding)",
        "owned_sha256s": plan.owned_sha256s(base),
        "pre_registration": plan.pre_registration_payload(),
        "rows": list(plan.ROWS),
        "learned_rows": list(plan.LEARNED_ROWS),
        "source_surface": plan.SOURCE_SURFACE,
        "source_session_count": plan.SOURCE_SESSION_COUNT,
        "budget": plan.BUDGET,
        "external_roster_opened": False,
        "predecessor_evidence": {
            "p2prime_result": {
                "path": plan.P2PRIME_RESULT_RELATIVE,
                "sha256": plan.P2PRIME_RESULT_SHA256,
            },
            "p2prime_terminal": {
                "path": plan.P2PRIME_TERMINAL_RELATIVE,
                "sha256": plan.P2PRIME_TERMINAL_SHA256,
                "verdict": plan.P2PRIME_VERDICT,
            },
            "sealed_cell_d_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
        },
        "process_gate": dict(plan.PROCESS_GATE),
        "inference_only_decoder": True,
        "target_optimizer_backward_update": 0,
    }
    import hashlib

    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    (output / "attempt.json").write_text(body + "\n", encoding="utf-8")
    (output / "attempt.json.sha256").write_text(f"{digest}  attempt.json\n", encoding="ascii")
    del screen_module
    return {"stage": "attempt", "attempt_sha256": digest, "result_root": str(output)}


def main() -> int:
    parser = argparse.ArgumentParser(description="AC3-0 frozen action-continuity representation screen")
    parser.add_argument("stage", choices=["attempt", "amend", "materialize", "select", "screen", "finalize", "dry-plan"])
    parser.add_argument("--gpu-index", type=int, default=1, choices=(0, 1))
    parser.add_argument("--budget", type=int, default=plan.BUDGET)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--row", type=str, default=None, help="restrict select to one learned row")
    parser.add_argument("--assemble", action="store_true", help="assemble selection.json from per-row partials")
    parser.add_argument("--amendment-index", type=int, default=1)
    args = parser.parse_args()
    base = _base()
    output_root = Path(args.output_root).absolute() if args.output_root else None
    if args.stage == "dry-plan":
        print(json.dumps({
            "schema": f"{plan.SCHEMA}_dry_plan",
            "cell": plan.CELL,
            "rows": list(plan.ROWS),
            "row_specs": {key: dict(value) for key, value in plan.ROW_SPECS.items()},
            "gates": plan.GATES,
            "utility_rows": plan.UTILITY_ROWS,
            "process_gate": plan.PROCESS_GATE,
            "result_root_relative": plan.RESULT_ROOT_RELATIVE,
            "required_data_root_environment": dict(plan.EXACT_DATA_ROOT_ENV),
            "target_optimizer_backward_update": 0,
        }, indent=2, sort_keys=True))
        return 0
    if args.stage == "attempt":
        result = _attempt(base, output_root)
    elif args.stage == "amend":
        result = screen.write_amendment(base, output_root=output_root, index=args.amendment_index)
    elif args.stage == "materialize":
        result = mat.materialize(base, gpu_index=args.gpu_index, budget=args.budget,
                                 output_root=output_root)
    elif args.stage == "select":
        import hashlib

        output = output_root or (base / plan.RESULT_ROOT_RELATIVE)
        if args.assemble:
            body = screen.assemble_selection(output)
            digest = (output / "selection.json.sha256").read_text(encoding="ascii").split()[0]
            result = {
                "stage": "select", "selection_sha256": digest, "assembled": True,
                "selected": {row: value["selected"] for row, value in body["rows"].items()},
            }
        else:
            trial_set = screen.load_trial_set(output)
            table = screen.build_table(trial_set)
            rows = [args.row] if args.row else None
            selection = screen.select_hyperparameters(trial_set, table, rows=rows)
            body = json.dumps(selection, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if args.row:
                partial = output / "selection_rows" / f"{args.row}.json"
                partial.parent.mkdir(parents=True, exist_ok=True)
                if partial.exists():
                    raise SystemExit(f"refusing to overwrite an existing partial: {partial}")
                partial.write_text(body + "\n", encoding="utf-8")
                (output / "selection_rows" / f"{args.row}.json.sha256").write_text(
                    f"{digest}  {args.row}.json\n", encoding="ascii")
            else:
                path = output / "selection.json"
                if path.exists():
                    raise SystemExit(f"refusing to overwrite an existing receipt: {path}")
                path.write_text(body + "\n", encoding="utf-8")
                (output / "selection.json.sha256").write_text(
                    f"{digest}  selection.json\n", encoding="ascii")
            result = {
                "stage": "select", "selection_sha256": digest,
                "row": args.row,
                "wall_seconds": selection["wall_seconds"],
                "selected": {row: value["selected"] for row, value in selection["rows"].items()},
            }
    elif args.stage == "screen":
        outcome = screen.run_screen(base, output_root=output_root)
        result = {
            "stage": "screen", "screen_sha256": outcome["screen_sha256"],
            "verdict": outcome["verdict"],
        }
    else:
        outcome = screen.finalize(base, output_root=output_root)
        result = dict(outcome)
    print(json.dumps({key: value for key, value in result.items() if key != "payload"},
                     indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
