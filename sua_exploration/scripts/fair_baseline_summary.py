"""Summarize validation-only fair no-calibration baselines for P3 variants."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean_r2(payload: dict) -> float:
    scores = payload.get("per_session_r2")
    if not isinstance(scores, dict) or not scores:
        raise ValueError(f"per_session_r2 missing or empty in {payload}")
    return float(statistics.fmean(scores.values()))


def summarize_variant(
    *,
    variant: str,
    protocol_path: Path,
    zero_identity_path: Path,
    learned_prior_path: Path,
    equal_tolerance: float = 1e-5,
) -> dict:
    protocol = load_json(protocol_path)
    zero_identity = load_json(zero_identity_path)
    learned_prior = load_json(learned_prior_path)

    selected = protocol["selected_protocol"]
    selected_key = (
        f"gradient_free_calibrated_{selected['selection_mode']}_n{selected['calibration_n']}"
    )
    calibrated_mean = float(protocol["mean_r2"][selected_key])
    zero_mean = mean_r2(zero_identity)
    learned_mean = mean_r2(learned_prior)
    abs_diff = abs(learned_mean - zero_mean)
    equal_controls = abs_diff <= equal_tolerance

    entry = {
        "variant": variant,
        "selected_calibration_mean_r2": calibrated_mean,
        "zero_identity_no_calibration_mean_r2": zero_mean,
        "learned_prior_no_calibration_mean_r2": learned_mean,
        "control_mode_learned_prior_equal_zero_identity": equal_controls,
        "control_mode_learned_prior_zero_identity_abs_diff": abs_diff,
        "delta_vs_zero": calibrated_mean - zero_mean,
        "selection_config": {
            "selection_mode": selected["selection_mode"],
            "calibration_n": selected["calibration_n"],
            "pool_size": selected.get("pool_size", protocol.get("pool_size")),
            "validation_mean_r2": calibrated_mean,
            "validation_paired_delta_vs_zero_identity_no_calibration": calibrated_mean - zero_mean,
        },
    }
    if equal_controls:
        entry["delta_vs_learned_prior"] = None
        entry["invalid_reason"] = (
            "learned_prior control is numerically identical to zero_identity; "
            "delta_vs_learned_prior is not a fair calibration margin"
        )
    else:
        entry["delta_vs_learned_prior"] = calibrated_mean - learned_mean
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        default="B3,B15,B16",
        help="Comma-separated variants to summarize.",
    )
    parser.add_argument(
        "--results_dir",
        default="sua_exploration/results",
        help="Directory containing protocol and no-calibration JSON artifacts.",
    )
    parser.add_argument("--out_path", default=None)
    parser.add_argument(
        "--supersedes",
        default=None,
        help="Optional prior summary JSON to annotate with superseded_by.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    entries = []
    for variant in variants:
        lower = variant.lower()
        protocol_path = results_dir / f"p3_gradient_free_protocol_selection_{lower}_s{args.seed}.json"
        zero_path = results_dir / f"p3_no_calibration_validation_{lower}_s{args.seed}.json"
        learned_path = results_dir / f"p3_no_calibration_validation_{lower}_learnedprior_s{args.seed}.json"
        entries.append(
            summarize_variant(
                variant=variant,
                protocol_path=protocol_path,
                zero_identity_path=zero_path,
                learned_prior_path=learned_path,
            )
        )

    out_path = (
        Path(args.out_path).expanduser().resolve()
        if args.out_path
        else results_dir / f"p3_fair_baseline_summary_s{args.seed}.json"
    )
    payload = {
        "schema_version": 2,
        "created_by": "fair_baseline_summary.py",
        "created_at": datetime.now().astimezone().isoformat(),
        "seed": args.seed,
        "interpretation": (
            "Learned-prior no-calibration used as fairness-aware control when valid; "
            "zero_identity retained as strict lower-bound control."
        ),
        "variants": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote fair baseline summary: {out_path}")

    if args.supersedes:
        prior_path = Path(args.supersedes).expanduser().resolve()
        prior = load_json(prior_path)
        prior["superseded_by"] = str(out_path)
        prior_path.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Annotated prior summary with superseded_by: {prior_path}")


if __name__ == "__main__":
    main()
