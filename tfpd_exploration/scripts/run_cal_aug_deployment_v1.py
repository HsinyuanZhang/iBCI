#!/usr/bin/env python3
"""CAL-AUG V1 deployment driver: the section 6 frozen matched-scorer scoring.

Thin driver with attempt/terminal receipt discipline over
``src.cal_aug_v1.deployment``: scores T0 and C1 once each on the frozen
M4/M10/M30 within-6 + external-15 inputs through the sealed Z1/P4 deployment
recipe (CPU-only), evaluates the lower-continuation and primary gates, and —
when the mechanism receipt is supplied — the combined disposition including
``MECHANISM_POSITIVE__DEPLOYMENT_INCONCLUSIVE``.  Zero target
optimizer/backward/update.  Fresh root ``results/cal_aug_v1/deployment/``.

Environment (the sealed Z1 harness enforces it; the driver fails closed first
with a clear message): ``CUDA_VISIBLE_DEVICES`` must be the empty string and
``SUBC_DATA_ROOT``/``SUBM_DATA_ROOT`` must point at the frozen dandi_000688
sub-C/sub-M roots inside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "sua_exploration"))
sys.path.insert(0, str(REPO / "streaming_calibration_exp"))
# ROOT last: tfpd_exploration/src must win `import src` over the
# streaming tree; the streaming components are either file-path loaded
# by the sealed arm runner or merged via src.__path__ inside tfpd_lane.
sys.path.insert(0, str(ROOT))

from src.cal_aug_v1 import deployment, plan, receipts  # noqa: E402

REQUIRED_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "SUBC_DATA_ROOT": str(REPO / "sua_exploration/data/dandi_000688/sub-C"),
    "SUBM_DATA_ROOT": str(REPO / "sua_exploration/data/dandi_000688/sub-M"),
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CAL-AUG V1 deployment scoring")
    parser.add_argument(
        "--t0-swa", type=Path,
        default=ROOT / "results/cal_aug_v1/t0_operator_disabled/swa_final4.pt",
    )
    parser.add_argument(
        "--c1-swa", type=Path,
        default=ROOT / "results/cal_aug_v1/c1_prefix_cycle/swa_final4.pt",
    )
    parser.add_argument(
        "--mechanism-receipt", type=Path,
        default=ROOT / "results/cal_aug_v1/mechanism/terminal.json",
        help="the mechanism receipt feeding the combined disposition (optional)",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/cal_aug_v1")
    return parser.parse_args(argv)


def _enforcement() -> dict:
    problems = []
    observed = {}
    for variable, expected in REQUIRED_ENV.items():
        value = os.environ.get(variable, None)
        observed[variable] = value
        if value != expected:
            problems.append(f"{variable}={value!r} (expected {expected!r})")
    if problems:
        print("deployment environment drift: " + "; ".join(problems), file=sys.stderr)
        raise SystemExit(3)
    return observed


def _require_arm_terminal(arm_dir: Path) -> dict:
    terminal = arm_dir / "terminal.json"
    if not terminal.is_file():
        raise SystemExit(f"arm terminal receipt missing: {terminal}")
    body = json.loads(terminal.read_text())
    if body.get("status") != "CAL_AUG_CELL_TERMINAL":
        raise SystemExit(f"arm not terminal: {arm_dir} -> {body.get('status')}")
    return {"terminal_path": str(terminal), "status": body["status"]}


def _load_mechanism_gate(path: Path):
    if path is None or not Path(path).is_file():
        return None, {"path": str(path), "present": False}
    body = json.loads(Path(path).read_text())
    sidecar = Path(str(path) + ".sha256")
    if sidecar.is_file():
        if sidecar.read_text().split()[0] != receipts.sha256_file(path):
            raise SystemExit("mechanism receipt SHA drift against its sidecar")
    gate = body.get("registration_gate")
    if gate is None:
        raise SystemExit("mechanism receipt lacks registration_gate")
    return gate, {
        "path": str(path), "present": True,
        "status": body.get("status"), "passed": gate.get("passed"),
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    _enforcement()

    out_dir = Path(args.out_root) / plan.RESULT_SUBROOTS["deployment"]
    if out_dir.exists():
        print(f"fresh deployment output directory required: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    context = {"stage": "initialization"}
    try:
        return _run(args, out_dir, started, context)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        stack = getattr(main, "_receipt_stack", None)
        if stack is not None:
            receipts.publish_failure(
                out_dir, stack["receipt"],
                {
                    "schema": plan.SCHEMA + "_deployment",
                    "started_utc": started,
                    "stage": context["stage"],
                    "failure": {
                        "kind": type(exc).__name__, "detail": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def _run(args, out_dir: Path, started: str, context: dict) -> int:
    context["stage"] = "pinned_verification"
    pinned = receipts.verify_pinned_files(REPO)
    sealed_predecessors = receipts.verify_sealed_predecessors(REPO)
    stack = receipts.load_sealed_runner_stack(REPO, ROOT)
    receipt_mod = stack["receipt"]
    main._receipt_stack = stack  # type: ignore[attr-defined]

    context["stage"] = "predecessors"
    arm_bindings = {
        "t0": {
            "swa_path": str(args.t0_swa),
            "swa_sha256": receipts.sha256_file(args.t0_swa),
            "terminal": _require_arm_terminal(args.t0_swa.parent),
        },
        "c1": {
            "swa_path": str(args.c1_swa),
            "swa_sha256": receipts.sha256_file(args.c1_swa),
            "terminal": _require_arm_terminal(args.c1_swa.parent),
        },
    }
    mechanism_gate, mechanism_binding = _load_mechanism_gate(args.mechanism_receipt)

    closure_launch = receipt_mod.source_closure(ROOT, plan.BOUND_PATTERNS)
    context["stage"] = "attempt_publication"
    attempt_sha = receipts.publish_attempt(
        out_dir, receipt_mod,
        {
            "schema": plan.SCHEMA + "_deployment_attempt",
            "status": "ATTEMPT_PUBLISHED",
            "cell": plan.CELL,
            "kind": "frozen_deployment_recipe_scoring",
            "started_utc": started,
            "arms": arm_bindings,
            "mechanism_binding": mechanism_binding,
            "surfaces": list(deployment.DEPLOYMENT_SURFACES),
            "budgets": list(deployment.DEPLOYMENT_BUDGETS),
            "recipe": (
                "sealed Z1/P4 deployment law: D-opt-first-30 @M4, chronological "
                "@M10/M30, ridge-T4 lambda=0.1 per budget, last-bin house R2"
            ),
            "environment_enforced": REQUIRED_ENV,
            "pinned_sha256": pinned,
            "sealed_predecessors": sealed_predecessors,
            "work_order": {
                "relative": plan.WORK_ORDER_RELATIVE,
                "sha256": pinned[plan.WORK_ORDER_RELATIVE],
            },
            "gate_spec": plan.gate_spec_payload(),
            "source_closure": closure_launch,
            "target_optimizer_steps": 0,
        },
    )
    receipts.require_attempt(out_dir, "scoring")

    context["stage"] = "scoring"
    result = deployment.evaluate_deployment(
        repo_root=REPO, t0_swa=args.t0_swa, c1_swa=args.c1_swa,
        mechanism_gate=mechanism_gate,
    )

    closure_final = receipt_mod.source_closure(ROOT, plan.BOUND_PATTERNS)
    combined = result["combined_disposition"]
    status = "DEPLOYMENT_SCORING_COMPLETE"
    receipt_mod.write_receipt_transactionally(
        out_dir / "terminal.json",
        {
            "schema": plan.SCHEMA + "_deployment",
            "status": status,
            "cell": plan.CELL,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attempt_sha256": attempt_sha,
            "arms": arm_bindings,
            "mechanism_binding": mechanism_binding,
            "disposition": combined["status"],
            **result,
            "target_optimizer_backward_update": 0,
            "source_closure": {
                "launch": closure_launch, "final": closure_final,
                "launch_final_closure_equal":
                    closure_final["closure_sha256"] == closure_launch["closure_sha256"],
            },
            "pinned_sha256": pinned,
        },
    )
    print(json.dumps({
        "status": status,
        "disposition": combined["status"],
        "lower_gate_passed": result["lower_continuation_gate"]["passed"],
        "primary_gate_passed": result["primary_claim_gate"]["passed"],
        "external_m4": result["gate_inputs"]["external_m4"],
        "external_m10": result["gate_inputs"]["external_m10"],
        "external_m30_mean": result["gate_inputs"]["external_m30_mean"],
        "within_means_by_budget": result["gate_inputs"]["within_means_by_budget"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
