#!/usr/bin/env python3
"""Write the immutable, held-in-only input manifest before clean-teacher fit."""
import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "sua_exploration/results/m2_ssc_t4_v1/clean_teacher_input_manifest.json"
SESSIONS = {
    "ses-2020-10-19-Run1", "ses-2020-10-19-Run2", "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2", "ses-2020-10-27-Run1", "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
}
FORBIDDEN = [
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session(path: Path) -> str:
    return path.name.split("_")[1].split(".")[0]


def files(directory_name: str, token: str, role: str) -> list[dict]:
    directory = (ROOT / "SPINT-main/data/000953" / directory_name).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    rows = []
    for path in sorted(directory.glob(f"*{token}*.nwb")):
        resolved = path.resolve()
        try:
            resolved.relative_to(directory)
        except ValueError as error:
            raise ValueError(f"symlink escapes clean-teacher directory: {path}") from error
        if "held-out" in resolved.name.lower():
            raise ValueError(resolved)
        rows.append({"role": role, "session": session(resolved), "path": str(resolved), "size_bytes": resolved.stat().st_size, "sha256": sha(resolved)})
    if len(rows) != 7 or {row["session"] for row in rows} != SESSIONS:
        raise ValueError(f"{role} session mismatch")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    rows = files("sub-MonkeyN-held-in-calib", "held-in-calib", "heldin_calib")
    rows += files("sub-MonkeyN-held-in-minival", "held-in-minival", "heldin_minival")
    payload = {
        "schema_version": 1,
        "protocol": "m2_clean_teacher_v1",
        "task": "m2",
        "calibration_n_trials": 24,
        "expected_heldin_sessions": sorted(SESSIONS),
        "forbidden_heldout_sessions": FORBIDDEN,
        "include_heldout_in_fit": False,
        "include_heldout_in_test": False,
        "files": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
