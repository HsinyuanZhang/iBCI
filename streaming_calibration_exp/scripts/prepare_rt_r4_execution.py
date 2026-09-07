#!/usr/bin/env python3
"""Write the superseding, no-launch execution preflight for RT R4.

The original immutable source-only prepare receipt is preserved byte-for-byte.
This receipt binds that parent plus the additional target evaluator, runner,
gate aggregator, selection callback, and data/config closure required to make
the frozen 12-cell pilot executable.  Building it opens no NWB or outer target,
constructs no Trainer/optimizer, does not query CUDA, and starts no process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts import rt_r4_pilot as pilot
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import rt_r4_pilot as pilot


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
DEFAULT_OUTPUT = (
    REPO
    / "sua_exploration/results/rt_r4_budget_response_common_q24_v1/"
      "RT_R4_EXECUTION_PREFLIGHT_v1.json"
)
CODE_CLOSURE = (
    "configs/data/rt_r4_budget_response_common_q24.yaml",
    "configs/experiment/rt_r4_budget_response_common_q24.yaml",
    "scripts/prepare_rt_r4_execution.py",
    "scripts/rt_r4_pilot.py",
    "src/callbacks/rt_nested_selection_receipt.py",
    "src/data/falcon_datamodule.py",
    "src/data/rt_nested_loso_datamodule.py",
    "src/data/rt_r4_budget_response_datamodule.py",
    "src/rt_clean_nested_loso_eval.py",
    "src/rt_r4_budget_response_eval.py",
    "tests/test_rt_r4_execution.py",
)


class RtR4ExecutionPreflightError(RuntimeError):
    """The immutable parent or executable R4 closure is incomplete."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RtR4ExecutionPreflightError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"R4 execution-preflight input missing: {path}")
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mode": f"{path.stat().st_mode & 0o777:04o}",
        "sha256": _sha256(path),
    }


def build_receipt(*, output: Path) -> dict[str, Any]:
    parent = pilot.PREPARE_RECEIPT.resolve()
    _need(parent.is_file(), f"R4 original prepare receipt missing: {parent}")
    _need((parent.stat().st_mode & 0o777) == 0o444,
          "R4 original prepare receipt is not immutable mode 0444")
    _need(_sha256(parent) == pilot.PREPARE_RECEIPT_SHA256,
          "R4 original prepare receipt SHA drift")
    parent_body = json.loads(parent.read_text(encoding="utf-8"))
    _need(
        isinstance(parent_body, Mapping)
        and parent_body.get("status")
        == "PASS_RT_R4_COMMON_Q24_SOURCE_ONLY_PREPARED_NOT_LAUNCHED",
        "R4 original prepare receipt status drift",
    )
    closure = {relative: _sha256(PROJECT / relative) for relative in CODE_CLOSURE}
    plan = pilot.build_plan(work_root=output.parent / "gpu_runs", execution_preflight=output)
    _need(plan["pilot_cell_count"] == 12, "R4 execution plan is not the frozen 12-cell pilot")
    _need(plan["scope"] == {
        "nwb_files_opened": 0,
        "outer_target_payloads_opened": 0,
        "trainer_constructed": False,
        "optimizer_constructed": False,
        "cuda_queried": False,
        "gpu_processes_started": 0,
        "tmux_sessions_created": 0,
        "commands_executed": 0,
    }, "R4 execution plan dry-run scope drift")
    return {
        "schema": pilot.EXECUTION_PREFLIGHT_SCHEMA,
        "status": pilot.EXECUTION_PREFLIGHT_STATUS,
        "objective": "make_frozen_R4_M6_M12_pilot_executable_without_launching_it",
        "supersedes_without_mutating": _binding(parent),
        "original_prepare_receipt_retained_as_historical_source_only_contract": True,
        "execution_code_closure_sha256": closure,
        "target_side_contract": {
            "activity_calibration_trials": 24,
            "carrier_prefix_budgets": [6, 12],
            "common_query_start_trial": 24,
            "target_builder": "build M24/q24 FalconDataset, then replace raw AFC4 only from target trials[0:M)",
            "normalizer": "verbatim inner-train-only mean/std bound by selected fit split manifest",
            "matched_arms": {
                "full": "afc4_vel=[wx,wy,||W||,b]",
                "control": "afc4_mb4=[0,0,||W||,b] after common normalization",
            },
            "one_shot_outer_receipt": pilot.OUTER_SCHEMA,
            "target_backpropagation": False,
            "optimizer_present": False,
            "model_state_sha256_before_equals_after_required": True,
            "common_query_identity_sha256_required": True,
        },
        "static_dual_3090_pilot": plan,
        "budgetwise_gate": {
            "budgets_evaluated_independently": [6, 12],
            "pilot_folds": [0, 7, 14],
            "paired_delta": "Full_minus_MB4",
            "positive_folds_required": "3/3",
            "mean_delta_at_least_r2": 0.03,
            "passing_budget_expansion_folds": list(pilot.EXPANSION_FOLDS),
            "failing_budget_policy": "stop; no new arms, seeds, folds, or fusion paths",
            "aggregator_launches_nothing": True,
            "m18_not_authorized": True,
        },
        "scheduling_trigger": (
            "root may schedule after local H1 source training releases the requested GPUs; "
            "RT launch is not conditioned on observing an H1 accuracy result"
        ),
        "authorization": {
            "gpu_launch_authorized_by_this_receipt": False,
            "outer_evaluation_authorized_by_this_receipt": False,
            "requires_separate_root_launch_instruction": True,
        },
        "scope": {
            "nwb_files_opened": 0,
            "outer_target_payloads_opened": 0,
            "trainer_constructed": 0,
            "optimizer_constructed": 0,
            "cuda_queried": 0,
            "gpu_processes_started": 0,
            "tmux_sessions_created": 0,
            "pilot_commands_executed": 0,
        },
        "output": str(output.resolve()),
    }


def write_immutable(path: Path, body: Mapping[str, Any]) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    _need(not path.exists(), f"refusing to overwrite R4 execution preflight: {path}")
    payload = json.dumps(dict(body), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    path.chmod(0o444)
    return path, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    body = build_receipt(output=output)
    if args.dry_run:
        print(json.dumps(body, indent=2, sort_keys=True))
        return 0
    path, digest = write_immutable(output, body)
    print(json.dumps({"status": body["status"], "output": str(path), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
