#!/usr/bin/env python3
"""Write a descriptive development report for the M1 held-in replay correction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--aggregate", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    aggregate_path = args.aggregate or root / "sua_exploration/results/m1_heldin_disjoint_replay_v1/aggregate_heldin.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    lines = [
        "# M1 held-in-calib post-support replay (development evidence)",
        "",
        "This report is **descriptive only**. No pass/fail gate, deployment verdict, or `effective` flag is applied.",
        "",
        "## Limitations (read first)",
        "",
        "1. **Checkpoint selection is contaminated.** Each `best.ckpt` was selected by `val_heldin/r2_mean` on the old 2-trial support-internal minival endpoint. This replay fixes the evaluation endpoint, not the selection.",
        "2. **Only two distinct left-out sessions.** The three cells cover `ses-20120926` (fold 1, two seeds) and `ses-20120927` (fold 2). Session-level n is 2, not 4.",
        "3. **Native FALCON internal-LOSO development evidence only.** No hidden EvalAI query/test NWB was opened.",
        "4. **M1 direction labels span a half plane** (8 directions across 157.5°); `[a, c]` is extrapolated outside the sampled arc.",
        "",
        "## Per-arm / per-cell results",
        "",
    ]
    for group in ("f0", "t4", "ts4"):
        arm = aggregate["arms"][group]
        lines.append(f"### {group.upper()}")
        lines.append("")
        lines.append(f"- Equal-cell mean R²: **{arm['equal_cell_mean_r2']:.6f}**")
        lines.append("")
        lines.append("| Cell | Left-out session | R² | Query trials | Scored windows | Checkpoint SHA-256 |")
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for cell in ("fold1_seed42", "fold1_seed43", "fold2_seed42"):
            lines.append(
                f"| {cell} | {arm['per_cell_left_out_session'][cell]} | "
                f"{arm['per_cell_r2'][cell]:.6f} | {arm['per_cell_query_trials'][cell]} | "
                f"{arm['per_cell_eligible_windows'][cell]} | `{arm['per_cell_checkpoint_sha256'][cell]}` |"
            )
        lines.append("")
    deltas = aggregate["paired_deltas_r2"]
    lines.extend([
        "## Paired comparisons (descriptive)",
        "",
        f"- **T4 − F0** equal-cell mean ΔR²: {deltas['T4_minus_F0']['equal_cell_mean_delta_r2']:.6f}",
        f"  - fold1_seed42: {deltas['T4_minus_F0']['per_cell_delta_r2']['fold1_seed42']:.6f}",
        f"  - fold1_seed43: {deltas['T4_minus_F0']['per_cell_delta_r2']['fold1_seed43']:.6f}",
        f"  - fold2_seed42: {deltas['T4_minus_F0']['per_cell_delta_r2']['fold2_seed42']:.6f}",
        f"- **T4 − TS4** equal-cell mean ΔR²: {deltas['T4_minus_TS4']['equal_cell_mean_delta_r2']:.6f}",
        f"  - fold1_seed42: {deltas['T4_minus_TS4']['per_cell_delta_r2']['fold1_seed42']:.6f}",
        f"  - fold1_seed43: {deltas['T4_minus_TS4']['per_cell_delta_r2']['fold1_seed43']:.6f}",
        f"  - fold2_seed42: {deltas['T4_minus_TS4']['per_cell_delta_r2']['fold2_seed42']:.6f}",
        "",
        deltas["T4_minus_F0"]["paired_note"],
        "",
    ])
    out = args.out or aggregate_path.parent / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
