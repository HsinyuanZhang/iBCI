#!/usr/bin/env python3
"""S2 EvalAI helper. Invoke with /usr/bin/python3."""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "artifacts/evalai_candidate.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-image-id", default="")
    parser.add_argument("--confirm-payload-sha256", default="")
    args = parser.parse_args()
    import runpy
    import sys

    sys.argv = [
        "submit_evalai.py",
        "--manifest",
        str(MANIFEST),
    ]
    if args.execute:
        sys.argv.extend(
            [
                "--execute",
                "--confirm-image-id",
                args.confirm_image_id,
                "--confirm-payload-sha256",
                args.confirm_payload_sha256,
            ]
        )
    runpy.run_path(
        str(
            HERE.parents[1]
            / "src/two_mainlines_long_v1/m2_runtime/submit_evalai.py"
        ),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
