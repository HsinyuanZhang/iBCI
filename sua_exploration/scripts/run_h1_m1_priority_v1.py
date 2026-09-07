#!/usr/bin/env python3
"""Execute the held-in-calib H1/M1 priority-1--3 result bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SPINT_ROOT = ROOT / "SPINT-main"
for entry in (str(ROOT), str(SPINT_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--audit-existing", action="store_true")
    parser.add_argument("--h1-device", default="cuda:0")
    args = parser.parse_args()
    if args.execute and args.audit_existing:
        raise SystemExit("choose exactly one of --execute or --audit-existing")
    if not args.execute and not args.audit_existing:
        print(json.dumps({
            "status": "DRY_NO_DATA_NO_MODEL_NO_GPU_NO_WRITE",
            "tasks": ["M1_M10_DIRECTRIDGE", "H1_PER_DOF", "M1_OUTPUT_RANK", "H1_M2_SUBSPACE_GATE"],
            "result_root": str(ROOT / "sua_exploration/results/h1_m1_priority_v1"),
        }, sort_keys=True))
        return
    from sua_exploration.h1_m1_priority_v1.execute import audit_existing_results, execute_all
    result = audit_existing_results() if args.audit_existing else execute_all(h1_device=args.h1_device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
