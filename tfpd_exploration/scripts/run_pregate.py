"""Deliverable A receipt runner: zero-GPU pre-gate for the time-varying
weighting premise (HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md §5.5).

CONTRACT (frozen in `src/tfpd_lane/pregate.py` and enforced here):

- estimators share value map and readout head; ONLY the weights differ
  (static = non-negative carrier ridge fitted on source, frozen;
   fisher = time-varying handoff-literal Fisher/slope weights evaluated at
   the model's own causal estimate, no true target in any weight;
   align = leaked upper bound with the true direction inside the weights,
   reported separately and NEVER a kill);
- construction (i) isotropic cosine population (sealed `src.tfpd/synth.py`
  generator): PASS requires R2_fisher - R2_static >= +0.05 AND
  R2_align - R2_static < +0.05 on query bins;
- construction (ii) state-matched population (narrow tuning, high
  baselines: only state-matched units carry signal): PASS requires
  R2_align - R2_static >= +0.05;
- overall PASS requires both constructions — each construction can pass and
  fail, so the gate is non-vacuous.  Failing construction (i) is a
  substantive verdict on the time-varying premise, not a runner error.

Receipt discipline (replicates `scripts/run_stage0_synth.py` §4):

1. CPU/env hard gate: refuses unless PYTHONNOUSERSITE is in effect,
   CUDA_VISIBLE_DEVICES is the empty string, and CUDA is unavailable (3).
2. Launch/final closure equality over this deliverable's NEW files only
   (`src/tfpd_lane/*.py`, this runner, `tests/test_pregate.py`); drift
   writes a FAIL_SOURCE_CLOSURE_DRIFT receipt and exits nonzero.
3. Transactional write: fsynced temp file hard-linked into place with
   O_EXCL; an existing receipt can never be overwritten; sidecar sha256 by
   the same procedure.
4. Failure semantics: execution error or gate FAIL still writes a receipt
   transactionally and exits nonzero; no partial pass exists.

Usage (from the package root):
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= python scripts/run_pregate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # package root, so `src.tfpd_lane` resolves

BOUND_PATTERNS = ("src/tfpd_lane/*.py", "scripts/run_pregate.py", "tests/test_pregate.py")

HANDOFF_DOC = "docs/HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/pregate_timevarying_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-sessions", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true", help="print the plan matrix and exit 0 (no receipt)")
    args = parser.parse_args()

    if args.dry_run:
        plan = {
            "mode": "dry-run (no receipt written)",
            "constructions": {
                "isotropic_cosine": {
                    "generator": "src.tfpd.synth.generate_session (sealed)",
                    "estimators": list(("static", "fisher", "align")),
                    "pass_rule": "fisher - static >= +0.05 AND align - static < +0.05",
                },
                "state_matched": {
                    "generator": "src.tfpd_lane.pregate.generate_state_matched_session",
                    "estimators": list(("static", "fisher", "align")),
                    "pass_rule": "align - static >= +0.05",
                },
            },
            "num_sessions": args.num_sessions,
            "seed": args.seed,
            "weight_discipline": "no true y_t in static or fisher weights; align is a leaked upper bound, never a kill",
        }
        print(json.dumps(plan, indent=1))
        return 0

    from src.tfpd_lane.receipt import (
        enforce_environment,
        sha256_file,
        source_closure,
        write_receipt_transactionally,
    )

    environment = enforce_environment()

    import torch

    from src.tfpd_lane.pregate import run_pregate

    if torch.cuda.is_available():
        print("environment hard gate failed: CUDA is available to torch", file=sys.stderr)
        return 3

    receipt_path = args.output_root / "receipt.json"
    if receipt_path.exists():
        print(f"receipt root already exists: {args.output_root}", file=sys.stderr)
        return 2

    handoff = ROOT / HANDOFF_DOC
    launch_closure = source_closure(ROOT, BOUND_PATTERNS)

    failure_payload_base = {
        "schema": "tfpd_pregate_timevarying_v1",
        "handoff": {
            "path": HANDOFF_DOC,
            "section": "5.5",
            "sha256": sha256_file(handoff),
        },
        "seed": args.seed,
        "num_sessions": args.num_sessions,
        "environment": environment,
        "launch_closure": launch_closure,
    }

    try:
        results = run_pregate(seed=args.seed, num_sessions=args.num_sessions)
    except Exception as error:  # noqa: BLE001 — any gate crash is a fail-closed event
        failure_payload_base.update(
            {"status": "FAIL_EXECUTION_ERROR", "error": f"{type(error).__name__}: {error}"}
        )
        write_receipt_transactionally(receipt_path, failure_payload_base)
        print(f"execution error; failure receipt: {receipt_path}", file=sys.stderr)
        return 1

    final_closure = source_closure(ROOT, BOUND_PATTERNS)
    if final_closure["closure_sha256"] != launch_closure["closure_sha256"]:
        failure_payload_base.update(
            {"status": "FAIL_SOURCE_CLOSURE_DRIFT", "final_closure": final_closure}
        )
        write_receipt_transactionally(receipt_path, failure_payload_base)
        print("source closure drifted between launch and final", file=sys.stderr)
        return 1

    all_passed = all(c["passed"] for c in results["constructions"].values())
    receipt = {
        **failure_payload_base,
        "final_closure": final_closure,
        "launch_final_closure_equal": True,
        "pregate": results,
        "status": "PASS_ALL_CONSTRUCTIONS" if all_passed else "FAIL",
        "authorizes": (
            "proceeding to Stage-1-surface scoring of the time-varying lane (handoff 5.5-3)"
            if all_passed
            else "nothing; construction (i) failure is a substantive negative verdict on "
            "the time-varying premise for isotropic cosine populations and per handoff "
            "5.5-5 this pre-gate is never a sole kill for the lane"
        ),
    }
    write_receipt_transactionally(receipt_path, receipt)
    summary = {
        "status": receipt["status"],
        "receipt": str(receipt_path),
        "constructions": {
            name: {"passed": c["passed"], "deltas": c["deltas"]} for name, c in results["constructions"].items()
        },
    }
    print(json.dumps(summary, indent=1))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
