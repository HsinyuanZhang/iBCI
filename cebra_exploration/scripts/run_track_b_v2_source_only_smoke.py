#!/usr/bin/env python3
"""Run the bounded Track-B v2 source-only canonical-loader smoke.

This command deliberately has no target-session, query, checkpoint, CEBRA,
or GPU option.  It may open only canonical development *source* NWBs selected
by frozen IDs.  A two-session ``--loader-smoke`` validates real SUA/pMUA/RT
loader semantics without publishing an authority.  The complete source roster
can additionally build development-only, non-citable immutable source
authorities; that path remains no-model/no-score and is not an official receipt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


# Must run before importing route code, which in turn is the only code that may
# import a canonical source loader after argument validation.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONNOUSERSITE"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_source_adapter as source  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("subject_m", "rt"))
    parser.add_argument("--view", choices=("sua", "pseudo_mua"))
    parser.add_argument(
        "--source-session", action="append", required=True,
        help="Canonical source session ID; repeat once per source session. Never accepts a path.",
    )
    parser.add_argument(
        "--loader-smoke", action="store_true",
        help="Allow a strict27 subset for a source-only loader smoke; cannot write authorities.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New directory for development-only full-source authority pairs (never an official receipt).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dataset == "subject_m" and args.view is None:
        _parser().error("--view is required for subject_m")
    if args.dataset == "rt" and args.view is not None:
        _parser().error("--view is forbidden for rt")
    if args.loader_smoke and args.output_dir is not None:
        _parser().error("--loader-smoke cannot publish an authority bundle")

    request, rows = source.materialize_canonical_source_sessions(
        dataset=args.dataset,
        view=args.view,
        source_session_ids=tuple(args.source_session),
        source_only_smoke=args.loader_smoke,
    )
    # Canonical loaders import Torch, but this route must remain CPU-only and
    # must not incidentally import the comparator package while loading source
    # arrays.  Check observed runtime state rather than merely serializing a
    # declarative false flag into the summary.
    import torch

    source.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "source-only smoke did not clear CUDA visibility")
    source.require(not torch.cuda.is_initialized(), "source-only smoke initialized CUDA")
    source.require(not any(name == "cebra" or name.startswith("cebra.") for name in sys.modules),
                   "source-only smoke imported CEBRA")
    payload: dict[str, object] = {
        "schema": "track_b_v2_source_only_smoke_summary_v1",
        "status": request["status"],
        "dataset": request["dataset"],
        "view": request["view"],
        "source_session_ids": request["source_session_ids"],
        "source_only_smoke": request["source_only_smoke"],
        "source_sessions": [row.as_coverage_dict() for row in rows],
        "target_data_opened": False,
        "target_query_opened": False,
        "cebra_imported": False,
        "cebra_solver_called": False,
        "checkpoint_loaded": False,
        "gpu_used": False,
        "score_emitted": False,
        "official_execution_receipt_minted": False,
    }
    if args.output_dir is not None:
        bundle = source.build_source_only_authority_bundle(request=request, sessions=rows)
        payload["development_only_authority_receipts"] = source.write_development_source_only_authority_bundle(
            output_dir=args.output_dir, bundle=bundle,
        )
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
