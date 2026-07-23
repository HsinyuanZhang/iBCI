#!/usr/bin/env python3
"""Register a completed run artifact directory into gate2_revised_matrix.csv."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.metrics.gate2_matrix import (
    ArtifactValidationError,
    build_matrix_row,
    find_protocol_control_r2,
    load_matrix,
    refresh_d512_deltas,
    upsert_matrix_row,
    validate_artifact_complete,
)


def default_matrix_path(artifact_dir: Path) -> Path:
    return artifact_dir.parent / "gate2_revised_matrix.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path, help="Completed run output directory")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Path to gate2_revised_matrix.csv (default: sibling of artifact_dir parent)",
    )
    parser.add_argument("--comparison-role", default="", help="Override comparison_role")
    parser.add_argument("--notes", default="", help="Optional notes stored in the matrix row")
    parser.add_argument(
        "--refresh-d512-deltas",
        action="store_true",
        help="Recompute delta_vs_D512_LOSO for matching fold/seed rows",
    )
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    try:
        validated = validate_artifact_complete(artifact_dir)
    except ArtifactValidationError as exc:
        raise SystemExit(f"Artifact validation failed: {exc}") from exc

    matrix_path = args.matrix or default_matrix_path(artifact_dir)
    existing_rows = load_matrix(matrix_path)
    d512_r2 = find_protocol_control_r2(
        existing_rows,
        fold_id=int(validated["fold_id"]),
        seed=int(validated["seed"]),
    )
    row = build_matrix_row(
        artifact_dir,
        comparison_role=args.comparison_role,
        d512_r2=d512_r2,
        notes=args.notes,
    )
    upsert_matrix_row(matrix_path, row)
    if args.refresh_d512_deltas or row.get("comparison_role") == "protocol_control":
        refresh_d512_deltas(matrix_path)
    print(json.dumps(row, indent=2))
    print(f"Updated matrix: {matrix_path}")


if __name__ == "__main__":
    main()
