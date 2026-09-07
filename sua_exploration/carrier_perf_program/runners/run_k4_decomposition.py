#!/usr/bin/env python3
"""K4 component decomposition CLI — CPU-only diagnostic for M2 held-in calibration."""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUA = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SUA))

from carrier_perf.gpu_guard import assert_no_gpu_flag, force_cpu, require_reviewed_cpu  # noqa: E402
from carrier_perf.k4_component_decomposition import (  # noqa: E402
    build_audit,
    build_synthetic_audit,
    write_audit,
)

DEFAULT_FALCON = SUA.parent / "SPINT-main" / "data" / "000953"
DEFAULT_OUTPUT = SUA / "results" / "m2_k4_component_decomposition_v1"


def main() -> None:
    force_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--falcon-data-dir", type=Path, default=DEFAULT_FALCON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--gpu", action="store_true", help="refused")
    args = parser.parse_args()
    assert_no_gpu_flag(args.gpu)
    if sum([args.dry_run, args.execute]) != 1:
        parser.error("choose exactly one of --dry-run / --execute")

    if args.dry_run:
        audit = build_synthetic_audit()
        print(
            json.dumps(
                {
                    "schema": audit["schema"],
                    "status": audit["status"],
                    "no_gpu": True,
                    "synthetic": True,
                    "n_sessions": audit["n_sessions"],
                    "recommendation": audit["conclusions"]["any_session_or"]["recommendation"],
                    "per_k4_dim_verdict": audit["conclusions"]["any_session_or"]["per_k4_dim_verdict"],
                    "headline_is_aggregation_sensitive": audit["conclusions"][
                        "headline_is_aggregation_sensitive"
                    ],
                    "defensible_recommendation": audit["conclusions"]["defensible_recommendation"],
                    "hybrid_margin_within_noise": audit["conclusions"]["hybrid_margin_within_noise"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    require_reviewed_cpu(args.execute)
    audit = build_audit(data_dir=args.falcon_data_dir, max_sessions=args.max_sessions)
    path = write_audit(audit, args.output_dir)
    print(
        json.dumps(
            {
                "wrote": str(path),
                "status": audit["status"],
                "n_sessions": audit.get("n_sessions"),
                "recommendation": audit["conclusions"]["any_session_or"]["recommendation"],
                "per_k4_dim_verdict": audit["conclusions"]["any_session_or"]["per_k4_dim_verdict"],
                "headline_is_aggregation_sensitive": audit["conclusions"][
                    "headline_is_aggregation_sensitive"
                ],
                "defensible_recommendation": audit["conclusions"]["defensible_recommendation"],
                "hybrid_margin_within_noise": audit["conclusions"]["hybrid_margin_within_noise"],
                "identifiability_margin_analysis": audit["conclusions"].get(
                    "identifiability_margin_analysis", {}
                ).get("hybrid_t4w_k4b_vs_t4_full"),
                "dimension4_redundancy_sessions": [
                    {
                        "session": s["session"],
                        "pearson": s["dimension4_redundancy"]["pearson"],
                        "spearman": s["dimension4_redundancy"]["spearman"],
                    }
                    for s in audit.get("sessions", [])
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
