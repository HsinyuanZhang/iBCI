"""Aggregate the preregistered M1 external MUA attention replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate_attention_architecture_screen import (
    CONTROLS,
    VARIANTS,
    find_mua_artifact,
    mean,
    read_mua_score,
    summarize_deltas,
    validate_mua_artifact,
)

M1_CELLS = ((0, 42), (0, 43), (1, 42))


def passes_attention_screen(summary: dict) -> bool:
    return (
        summary["mean"] >= 0.005
        and summary["minimum"] >= -0.03
        and summary["positive_count"] >= 2
        and summary["n"] == 3
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument("--parent-screen-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-path", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    screen_dir = root / "sua_exploration" / "results" / args.screen_id
    parent_path = root / "sua_exploration" / "results" / args.parent_screen_id / "aggregate.json"
    if not parent_path.is_file():
        raise FileNotFoundError(f"Missing pseudo-MUA bridge aggregate: {parent_path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("gates", {}).get("advance_to_external_mua_replication") is not True:
        raise ValueError("Pseudo-MUA bridge did not pass advance_to_external_mua_replication")

    mua_root = root / "streaming_calibration_exp" / "outputs" / "streaming_calibration"
    scores: dict[str, dict[str, float]] = {variant: {} for variant in VARIANTS}
    artifacts: dict[str, dict[str, str]] = {variant: {} for variant in VARIANTS}
    contracts: dict[str, dict[str, dict]] = {variant: {} for variant in VARIANTS}
    for fold, seed in M1_CELLS:
        key = f"fold{fold}_seed{seed}"
        for variant in VARIANTS:
            path = find_mua_artifact(
                mua_root, args.screen_id, variant, fold, seed, task="m1"
            )
            contracts[variant][key] = validate_mua_artifact(
                path, variant=variant, fold=fold, seed=seed, task="m1"
            )
            scores[variant][key] = read_mua_score(path / "metrics_summary.csv")
            artifacts[variant][key] = str(path)

    paired_deltas = {
        f"B15_minus_{control}": {
            "per_cell": {
                cell: scores["B15"][cell] - scores[control][cell]
                for cell in sorted(scores["B15"])
            }
        }
        for control in CONTROLS
    }
    for payload in paired_deltas.values():
        payload["summary"] = summarize_deltas(list(payload["per_cell"].values()))
    gates = {
        "m1_architecture_usable": (
            mean(list(scores["B15"].values())) > 0.0
            and paired_deltas["B15_minus_B3"]["summary"]["mean"] >= 0.0
            and paired_deltas["B15_minus_B3"]["summary"]["minimum"] >= -0.03
        ),
        "m1_attention_screen": all(
            passes_attention_screen(paired_deltas[f"B15_minus_{control}"]["summary"])
            for control in ("B15P", "B15D")
        ),
    }
    gates["advance_to_confirmation_planning"] = all(gates.values())
    payload = {
        "schema_version": 1,
        "purpose": "external_m1_mua_attention_replication_development_only",
        "screen_id": args.screen_id,
        "parent_screen_id": args.parent_screen_id,
        "task": "m1",
        "no_formal_test_sessions_evaluated": True,
        "internal_loso_cells": [f"fold{fold}_seed{seed}" for fold, seed in M1_CELLS],
        "scores": scores,
        "artifacts": artifacts,
        "artifact_contracts": contracts,
        "paired_deltas": paired_deltas,
        "gates": gates,
    }
    out_path = args.out_path or screen_dir / "aggregate.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2, sort_keys=True))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
