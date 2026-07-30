#!/usr/bin/env python3
"""Download DANDI 000688 SUA subsets into sua_exploration/data/dandi_000688/."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DANDI_ID = "000688"
DANDI_VERSION = "0.250122.1735"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "dandi_000688"


def run_dandi_download(subject: str, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://api.dandiarchive.org/api/dandisets/{DANDI_ID}/"
        f"versions/{DANDI_VERSION}/assets/?path={subject}"
    )
    cmd = [
        "dandi",
        "download",
        url,
        "--existing",
        "skip",
        "-o",
        str(output_root),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=str,
        choices=["sub-J", "sub-C", "sub-M", "all-sua"],
        default="sub-C",
        help="Which subject folder to download (default: sub-C for first cross-session exp)",
    )
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    manifest = {
        "dandiset_id": DANDI_ID,
        "version": DANDI_VERSION,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "subjects": [],
    }

    subjects = ["sub-J", "sub-C", "sub-M"] if args.subject == "all-sua" else [args.subject]
    for subject in subjects:
        run_dandi_download(subject, output_root)
        dest = output_root / subject
        files = sorted(dest.glob("*.nwb"))
        manifest["subjects"].append(
            {
                "subject": subject,
                "n_files": len(files),
                "total_bytes": sum(f.stat().st_size for f in files),
                "files": [f.name for f in files],
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"dandi download failed: {exc}", file=sys.stderr)
        sys.exit(exc.returncode)
