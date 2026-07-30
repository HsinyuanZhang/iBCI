"""Aggregate the preregistered pseudo-MUA attention bridge pilot."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


VARIANTS = ("B3", "B15P", "B15D", "B15")
CONTROLS = ("B3", "B15P", "B15D")


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def summarize_deltas(deltas: list[float]) -> dict[str, float | int]:
    return {
        "mean": mean(deltas),
        "median": float(statistics.median(deltas)),
        "minimum": float(min(deltas)),
        "maximum": float(max(deltas)),
        "positive_count": sum(value > 0.0 for value in deltas),
        "n": len(deltas),
    }


def selected_scores(payload: dict) -> dict[str, float]:
    selected = payload["selected_protocol"]
    name = f"gradient_free_calibrated_{selected['selection_mode']}_n{selected['calibration_n']}"
    return {
        session: float(configs[name])
        for session, configs in payload["per_session_r2"].items()
    }


def passes_attention_screen(summary: dict) -> bool:
    return (
        summary["mean"] >= 0.005
        and summary["minimum"] >= -0.03
        and summary["positive_count"] >= 4
        and summary["n"] == 6
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
        raise FileNotFoundError(f"Missing parent attention screen aggregate: {parent_path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("gates", {}).get("advance_to_paired_pilot") is not True:
        raise ValueError("Parent screen did not pass advance_to_paired_pilot")

    pseudo: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for variant in VARIANTS:
        for seed in (42, 43):
            path = screen_dir / f"pseudo_{variant.lower()}_s{seed}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing pseudo-MUA result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("signal_view") != "pseudo_mua":
                raise ValueError(f"Result does not declare pseudo_mua: {path}")
            if payload.get("fixed_protocol") is not True:
                raise ValueError(f"Pseudo-MUA result was not fixed protocol: {path}")
            selected = payload["selected_protocol"]
            if selected["selection_mode"] != "first" or selected["calibration_n"] != 30:
                raise ValueError(f"Unexpected pseudo-MUA protocol in {path}: {selected}")
            pseudo[variant][seed] = selected_scores(payload)

    variant_means = {
        variant: mean([mean(list(seed_scores.values())) for seed_scores in pseudo[variant].values()])
        for variant in VARIANTS
    }
    paired_deltas: dict[str, dict] = {}
    for control in CONTROLS:
        per_session = {}
        for session in sorted(pseudo["B15"][42]):
            values = [pseudo["B15"][seed][session] - pseudo[control][seed][session] for seed in (42, 43)]
            per_session[session] = mean(values)
        paired_deltas[f"B15_minus_{control}"] = {
            "per_session_seed_mean": per_session,
            "summary": summarize_deltas(list(per_session.values())),
        }

    raw_deltas = parent["sua"]["paired_deltas"]
    bridge_shift = {
        f"B15_minus_{control}": (
            paired_deltas[f"B15_minus_{control}"]["summary"]["mean"]
            - raw_deltas[f"B15_minus_{control}"]["summary"]["mean"]
        )
        for control in CONTROLS
    }
    gates = {
        "pseudo_mua_architecture_usable": (
            variant_means["B15"] > 0.0
            and paired_deltas["B15_minus_B3"]["summary"]["mean"] >= 0.0
            and paired_deltas["B15_minus_B3"]["summary"]["minimum"] >= -0.03
        ),
        "pseudo_mua_attention_screen": all(
            passes_attention_screen(paired_deltas[f"B15_minus_{control}"]["summary"])
            for control in ("B15P", "B15D")
        ),
    }
    gates["advance_to_external_mua_replication"] = all(gates.values())
    payload = {
        "schema_version": 1,
        "purpose": "pseudo_mua_attention_bridge_development_only",
        "screen_id": args.screen_id,
        "parent_screen_id": args.parent_screen_id,
        "no_formal_test_sessions_evaluated": True,
        "pseudo_mua_definition": "sum sorted-unit binned spike counts within each NWB electrode id",
        "fixed_protocol": {"selection_mode": "first", "calibration_n": 30, "pool_size": 50},
        "variant_mean_r2": variant_means,
        "paired_deltas": paired_deltas,
        "paired_delta_mean_shift_from_sua": bridge_shift,
        "gates": gates,
    }
    out_path = args.out_path or screen_dir / "aggregate.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["gates"], indent=2, sort_keys=True))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
