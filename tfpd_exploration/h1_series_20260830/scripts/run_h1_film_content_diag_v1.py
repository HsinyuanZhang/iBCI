#!/usr/bin/env python3
"""Run the H1 FiLM content diagnostic (V5): full / empty / rowshuffle.

Preregistration: tfpd_exploration/h1_series_20260830/docs/
WORKORDER_H1_FILM_CONTENT_DIAGNOSTIC_V5_20260904.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
DIAG_SRC = REPO_ROOT / "tfpd_exploration/h1_series_20260830/src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SPINT-main"))
sys.path.insert(0, str(DIAG_SRC))

from h1_calibration_profile_film_content_diag_v1 import plan, runner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--modes", nargs="*", default=list(plan.MODES),
                        choices=plan.MODES)
    args = parser.parse_args()

    result_root = REPO_ROOT / plan.RESULT_ROOT_RELATIVE
    if not args.execute:
        print(json.dumps({
            "schema": plan.SCHEMA, "status": "INERT_USE_EXECUTE",
            "modes": plan.MODES, "result_root": str(result_root),
            "perm_seed": plan.PERM_SEED, "drift_tolerance": plan.DRIFT_TOLERANCE,
            "content_floor": plan.CONTENT_FLOOR,
        }, indent=1, sort_keys=True))
        return 0

    if result_root.exists():
        raise RuntimeError(f"diagnostic result root is not fresh: {result_root}")
    result_root.mkdir(parents=True)
    preflight_info = runner.preflight(args.gpu)

    modes = args.modes or list(plan.MODES)
    summaries = {}
    for mode in modes:
        summaries[mode] = runner.run_mode(mode, REPO_ROOT, device="cuda:0")
        print(f"MODE_DONE mode={mode} ep_gain_mean={summaries[mode]['ep_gain_mean']:.6f}", flush=True)

    if set(summaries) == set(plan.MODES):
        sealed = runner.sealed_v2_reference()
        decision = runner.decide(summaries, sealed)
        terminal_sha = runner.publish_terminal(result_root, summaries, sealed, decision, preflight_info)
        print(json.dumps({"branch": decision["branch"],
                          "empty_mean": decision["empty_mean"],
                          "full_mean": decision["full_mean"],
                          "rowshuffle_mean": decision["rowshuffle_mean"],
                          "drift_abs": decision["drift_abs"],
                          "lp_canary_ok": decision["lp_canary_ok"],
                          "terminal_sha256": terminal_sha}, indent=1, sort_keys=True))
    else:
        (result_root / "PARTIAL.json").write_text(json.dumps(
            {"completed_modes": sorted(summaries)}, indent=1), encoding="utf-8")
        print("PARTIAL run; decision requires all three modes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
