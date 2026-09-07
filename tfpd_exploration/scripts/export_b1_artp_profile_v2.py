#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "INERT_DRY_RUN", "route": "B1_ARTP_PROFILE_PACKAGE_V2", "evalai_push": False}, sort_keys=True))
        return
    from tfpd_exploration.src.b1_artp_v2.export import export_and_verify
    from tfpd_exploration.src.b1_artp_v2.plan import RESULT_ROOT

    result = export_and_verify(
        artifact=RESULT_ROOT / "artp_profile_payload_v2.pkl",
        receipt=RESULT_ROOT / "package_receipt.json",
        source_screen_path=RESULT_ROOT / "source_screen.json",
    )
    print(json.dumps({"status": result["status"], "artifact_sha256": result["artifact_sha256"], "all_exact": result["local_source_parity"]["all_exact"]}, sort_keys=True))


if __name__ == "__main__":
    main()
