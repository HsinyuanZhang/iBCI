#!/usr/bin/env python3
"""Run the one predeclared RT source-only fold smoke; no target/CEBRA/GPU mode exists."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_rt_source_authority_runner as runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run only the predeclared rt_outer_fold_00 source smoke.")
    parser.add_argument("--_worker-run-kind", choices=("smoke", "full"), help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output-root", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-fold-id", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args._worker_run_kind is not None:
            if args.smoke or args._worker_output_root is None or args._worker_fold_id is None:
                parser.error("internal RT source worker arguments are incomplete")
            payload = runner.run_rt_source_authority_worker(
                run_kind=args._worker_run_kind,
                output_root=Path(args._worker_output_root),
                outer_fold_id=args._worker_fold_id,
            )
        else:
            if not args.smoke:
                parser.error("only --smoke is available; 15-fold expansion needs a later root-only continuation")
            payload = runner.run_rt_source_authority_smoke()
    except (base.TrackBV2ContractError, runner.TrackBV2RTSourceAuthorityRunnerError) as exc:
        parser.error(str(exc))
    sys.stdout.buffer.write(base.canonical_json_bytes(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
