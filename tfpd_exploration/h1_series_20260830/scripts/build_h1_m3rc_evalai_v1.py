#!/usr/bin/env python3
"""Build (but never submit) the exactly-M3 H1-M3RC EvalAI payload."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--package-path", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_NO_WRITE_NO_DATA_NO_CUDA_NO_SUBMISSION",
            "official_calibration_trials": 3,
            "result_root": str(args.result_root),
            "package_path": str(args.package_path),
        }, sort_keys=True))
        return 0
    from h1_m3_readout_calibration_v1.package import execute

    print(json.dumps(execute(
        repo_root=args.repo_root,
        legacy_root=args.legacy_root,
        result_root=args.result_root,
        package_path=args.package_path,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
