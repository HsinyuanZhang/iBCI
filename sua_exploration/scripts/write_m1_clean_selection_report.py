#!/usr/bin/env python3
"""Write descriptive report for M1 clean-selection sealed report-window results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCREEN = "m1_clean_selection_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out or root / f"sua_exploration/results/{SCREEN}"
    out_dir = out_dir.resolve()
    agg_path = out_dir / "aggregate_report.json"
    protocol_path = out_dir / "protocol_receipt.json"
    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))

    lines = [
        "# M1 clean-selection sealed report-window evaluation (development evidence)",
        "",
        "This report is **descriptive only**. No pass/fail gate, deployment verdict, or `effective` flag is applied.",
        "",
        "## Limitations (read first)",
        "",
        "1. **Session-level n is 2**, not 4. Three cells cover `ses-20120926` (fold 1, two seeds) and `ses-20120927` (fold 2).",
        "2. **Report window length differs by session**: `ses-20120926` has 199 trials; `ses-20120927` has 166 trials in `[210, end)`.",
        "3. **M1 direction labels span a half plane** (8 directions across 157.5°); `[a, c]` is extrapolated outside the sampled arc.",
        "4. **Development evidence only.** No hidden EvalAI query/test NWB was opened.",
        "5. Checkpoint selection during retraining used only the selection window `[10, 210)`; the report window was sealed until training finished.",
        "",
        f"Protocol receipt SHA-256: `{sha256(protocol_path)}`",
        "",
        "## Three-way comparison on sealed report window `[210, end)`",
        "",
    ]

    source_labels = {
        "clean_best": "Clean-selection `best.ckpt`",
        "fixed_last": "Fixed-epoch `last.ckpt` (epoch 12)",
        "frozen_best": "Frozen `native_mua_t4_v1` `best.ckpt` (matched control)",
    }

    for group in ("f0", "t4", "ts4"):
        lines.append(f"### {group.upper()}")
        lines.append("")
        for source in ("clean_best", "fixed_last", "frozen_best"):
            arm = aggregate["arms"][group][source]
            lines.append(f"#### {source_labels[source]}")
            lines.append("")
            lines.append(f"- Equal-cell mean R²: **{fmt(arm['equal_cell_mean_r2'])}**")
            lines.append("")
            lines.append("| Cell | Left-out session | R² | Query trials | Scored windows | Checkpoint SHA-256 |")
            lines.append("| --- | --- | ---: | ---: | ---: | --- |")
            for cell in ("fold1_seed42", "fold1_seed43", "fold2_seed42"):
                lines.append(
                    f"| {cell} | {arm['per_cell_left_out_session'][cell]} | "
                    f"{fmt(arm['per_cell_r2'][cell])} | {arm['per_cell_query_trials'][cell]} | "
                    f"{arm['per_cell_eligible_windows'][cell]} | `{arm['per_cell_checkpoint_sha256'][cell]}` |"
                )
            lines.append("")

    lines.extend(["## Paired comparisons (descriptive)", ""])
    for group in ("f0", "t4", "ts4"):
        pass
    for comp in ("T4_minus_F0", "T4_minus_TS4"):
        lines.append(f"### {comp}")
        lines.append("")
        for source in ("clean_best", "fixed_last", "frozen_best"):
            delta = aggregate["paired_deltas_r2"][comp][source]
            lines.append(f"- **{source_labels[source]}** equal-cell mean ΔR²: {fmt(delta['equal_cell_mean_delta_r2'])}")
            for cell, val in delta["per_cell_delta_r2"].items():
                lines.append(f"  - {cell}: {fmt(val)}")
        lines.append("")

    lines.append("Three cells cover two distinct left-out sessions (fold1 x2, fold2 x1); this is descriptive only.")
    lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "artifacts": {
            "protocol_receipt.json": sha256(protocol_path),
            "aggregate_report.json": sha256(agg_path),
            "report.md": sha256(report_path),
        },
    }
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path)
    print(sha256(report_path))
    print(json.dumps(manifest["artifacts"], indent=2))


if __name__ == "__main__":
    main()
