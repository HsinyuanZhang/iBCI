#!/usr/bin/env python3
"""CAL-AUG V1 mechanism driver: the section 5 crossed source readout.

Thin driver with attempt/terminal receipt discipline over
``src.cal_aug_v1.mechanism``: materializes the source-27 sessions through the
sealed training loader, strict-loads both
arms' sealed SWA checkpoints, scores every (arm, M in {30,10,4}) x 27 cell with
the SAME training T4, and evaluates the registration gate.  No target data, no
optimizer steps, no checkpoint selection.  Fresh root
``results/cal_aug_v1/mechanism/``.
"""

from __future__ import annotations

import argparse
import json
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

from src.cal_aug_v1 import mechanism, plan, receipts  # noqa: E402


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CAL-AUG V1 source mechanism readout")
    parser.add_argument(
        "--t0-swa", type=Path,
        default=ROOT / "results/cal_aug_v1/t0_operator_disabled/swa_final4.pt",
    )
    parser.add_argument(
        "--c1-swa", type=Path,
        default=ROOT / "results/cal_aug_v1/c1_prefix_cycle/swa_final4.pt",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/cal_aug_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=plan.SEED)
    parser.add_argument("--train-batch-size", type=int, default=plan.TRAIN_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args(argv)


def _require_arm_terminal(arm_dir: Path, expected_status="CAL_AUG_CELL_TERMINAL") -> dict:
    terminal = arm_dir / "terminal.json"
    if not terminal.is_file():
        raise SystemExit(f"arm terminal receipt missing: {terminal}")
    body = json.loads(terminal.read_text())
    if body.get("status") != expected_status:
        raise SystemExit(f"arm not terminal ({expected_status}): {arm_dir} -> {body.get('status')}")
    return {"terminal_path": str(terminal), "status": body["status"]}


def main(argv=None) -> int:
    args = _parse_args(argv)
    if not sys.flags.no_user_site:
        print("PYTHONNOUSERSITE=1 is mandatory", file=sys.stderr)
        return 3
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"CUDA unavailable ({args.device})", file=sys.stderr)
        return 3

    out_dir = Path(args.out_root) / plan.RESULT_SUBROOTS["mechanism"]
    if out_dir.exists():
        print(f"fresh mechanism output directory required: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    context = {"stage": "initialization"}
    try:
        return _run(args, out_dir, device, started, context)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001
        stack = getattr(main, "_receipt_stack", None)
        if stack is not None:
            receipts.publish_failure(
                out_dir, stack["receipt"],
                {
                    "schema": plan.SCHEMA + "_mechanism",
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


def _run(args, out_dir: Path, device, started: str, context: dict) -> int:
    context["stage"] = "pinned_verification"
    pinned = receipts.verify_pinned_files(REPO)
    sealed_predecessors = receipts.verify_sealed_predecessors(REPO)
    stack = receipts.load_sealed_runner_stack(REPO, ROOT)
    arm_runner = stack["arm_runner"]
    arm_common = stack["arm_common"]
    receipt_mod = stack["receipt"]
    pop_robust = stack["pop_robust"]
    main._receipt_stack = stack  # type: ignore[attr-defined]

    context["stage"] = "arm_predecessors"
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

    closure_launch = receipt_mod.source_closure(ROOT, plan.BOUND_PATTERNS)
    context["stage"] = "attempt_publication"
    attempt_sha = receipts.publish_attempt(
        out_dir, receipt_mod,
        {
            "schema": plan.SCHEMA + "_mechanism_attempt",
            "status": "ATTEMPT_PUBLISHED",
            "cell": plan.CELL,
            "kind": "source_mechanism_readout",
            "started_utc": started,
            "arms": arm_bindings,
            "pinned_sha256": pinned,
            "sealed_predecessors": sealed_predecessors,
            "work_order": {
                "relative": plan.WORK_ORDER_RELATIVE,
                "sha256": pinned[plan.WORK_ORDER_RELATIVE],
            },
            "gate_spec": plan.gate_spec_payload(),
            "source_closure": closure_launch,
            "environment": receipts.environment_payload(
                receipt_mod, REPO, device, {"mode": "scoring (no training)"}
            ),
            "target_optimizer_steps": 0,
        },
    )
    receipts.require_attempt(out_dir, "data_materialization")

    context["stage"] = "data_materialization"
    source = mechanism.materialize_source_sessions(arm_runner, args)

    models = {}
    swa_state_shas = {}
    for arm, swa in (("t0", args.t0_swa), ("c1", args.c1_swa)):
        context["stage"] = f"load_{arm}"
        model, digest = mechanism.load_arm_swa_model(pop_robust, arm_common, swa, device)
        models[arm] = model
        swa_state_shas[arm] = {"artifact_sha256": digest,
                               "state_sha256": arm_common.state_sha256(model)}
        if digest != arm_bindings[arm]["swa_sha256"]:
            raise SystemExit(f"{arm} SWA digest drift during load")

    context["stage"] = "readout"
    readout = mechanism.mechanism_readout(
        models_by_arm=models, source=source, device=device, pop_robust=pop_robust
    )
    gate = readout["registration_gate"]

    closure_final = receipt_mod.source_closure(ROOT, plan.BOUND_PATTERNS)
    status = (
        "MECHANISM_READOUT_COMPLETE__GATE_PASSED" if gate["passed"]
        else "MECHANISM_READOUT_COMPLETE__GATE_NOT_PASSED"
    )
    receipt_mod.write_receipt_transactionally(
        out_dir / "terminal.json",
        {
            "schema": plan.SCHEMA + "_mechanism",
            "status": status,
            "cell": plan.CELL,
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attempt_sha256": attempt_sha,
            "arms": {**arm_bindings, "state": swa_state_shas},
            **readout,
            "no_target_path": readout["no_target_path"],
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
        "gate_passed": gate["passed"],
        "recovery_m10": gate["recovery_margin_m10"]["value"],
        "recovery_m4": gate["recovery_margin_m4"]["value"],
        "positive_m10": gate["breadth"]["m10_positive_sessions"],
        "positive_m4": gate["breadth"]["m4_positive_sessions"],
        "c1_minus_t0_m30": gate["m30_safety_c1_minus_t0"]["value"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
