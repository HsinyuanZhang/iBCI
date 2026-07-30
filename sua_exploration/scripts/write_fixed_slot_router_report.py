"""Render an auditable Markdown handoff from fixed-slot pilot artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-pilot-id", required=True)
    parser.add_argument("--sharp-pilot-id", default="fixed_slot_router_sharp_v1")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    initial_dir = root / "results" / args.initial_pilot_id
    sharp_dir = root / "results" / args.sharp_pilot_id
    initial = read_json(initial_dir / "aggregate.json")
    hardware = read_json(initial_dir / "hardware_profile.json")
    decision_path = initial_dir / "low_temperature_followup_decision.json"
    decision = read_json(decision_path) if decision_path.is_file() else None
    sharp_path = sharp_dir / "aggregate.json"
    sharp = read_json(sharp_path) if sharp_path.is_file() else None

    if initial.get("no_formal_test_sessions_evaluated") is not True:
        raise ValueError("Initial aggregate does not certify formal-test isolation")
    initial_router = initial.get("router_diagnostics", {})
    if not initial_router.get("complete"):
        raise ValueError("Initial routing diagnostics are incomplete")
    initial_cached_decode = initial.get("cached_decode_verification", {})
    if not initial_cached_decode.get("complete"):
        raise ValueError("Initial cached-decode verification is incomplete")
    if sharp is not None and sharp.get("no_formal_test_sessions_evaluated") is not True:
        raise ValueError("Sharp aggregate does not certify formal-test isolation")

    results = initial["results"]
    n64_hardware = hardware["unit_count_profiles"]["N64"]["fixed_slot_profiles"]
    lines = [
        "# Fixed-Slot NeuronID Pilot Report",
        "",
        "**Scope:** validation-only DANDI 000688 sub-C CO development pilot; formal held-out test sessions were not accessed.",
        "",
        "## Protocol",
        "",
        f"- Encoder: B3 NeuronID; calibration selection: `{initial['protocol']['selection_mode']}` / "
        f"`n={initial['protocol']['calibration_n']}` / `pool={initial['protocol']['pool_size']}`.",
        "- Evaluation uses trials after the shared calibration pool; encoder, router and decoder remain frozen.",
        "- Scores are seed means across the six chronological validation sessions, not formal test results.",
        "",
        "## Initial Soft Router",
        "",
        "| Interface | Mean R² | Δ vs. B15P reference | Positive validation sessions | K=32 feasibility gate | N=64 read-in reduction |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for slot_name in ("K32", "K16"):
        result = results[slot_name]
        profile = n64_hardware[slot_name]
        lines.append(
            f"| `{slot_name}` | {fmt(result['mean_r2'])} | {fmt(result['delta_vs_b15p_reference'])} | "
            f"{result['positive_sessions']}/6 | {str(result['pilot_accuracy_feasible']).lower()} | "
            f"{profile['readin_mac_reduction_vs_variable']:.1f}× |"
        )

    cached_decode = initial_cached_decode["by_slot_count"]
    lines.extend([
        "",
        "## Cached Deployment Path",
        "",
        "Each verification compares normal fixed-slot forward with one calibration-derived "
        "batch-1 state broadcast across eight online spike windows per validation session. "
        "It reads no behavior labels or test files.",
        "",
        "| Interface | Verified validation session-seed pairs | Maximum absolute output difference |",
        "|---|---:|---:|",
    ])
    for slot_name in ("K32", "K16"):
        metrics = cached_decode[slot_name]
        lines.append(
            f"| `{slot_name}` | {metrics['verified_session_count']} | "
            f"{metrics['max_absolute_difference']:.2e} |"
        )

    router = initial_router["by_slot_count"]
    lines.extend([
        "",
        "## Routing Diagnostics",
        "",
        "The routing diagnostics use validation spikes and trial metadata only: no behavior data, decoder outputs, gradients, or test sessions.",
        "",
        "| Interface | Mean normalized assignment entropy | Mean max assignment probability | Slot-mass CV | Effective slots |",
        "|---|---:|---:|---:|---:|",
    ])
    for slot_name in ("K32", "K16"):
        metrics = router[slot_name]
        lines.append(
            f"| `{slot_name}` | {fmt(metrics['mean_assignment_normalized_entropy'])} | "
            f"{fmt(metrics['mean_assignment_max_probability'])} | "
            f"{fmt(metrics['slot_mass_coefficient_of_variation'])} | "
            f"{fmt(metrics['effective_slot_count_from_mass'])} |"
        )

    lines.extend(["", "## Low-Temperature Follow-up", ""])
    if decision is None:
        lines.append("- No follow-up decision artifact is present yet.")
    else:
        lines.append(
            "- Predeclared trigger: `mean K32 normalized assignment entropy >= 0.95`; "
            f"observed `{decision['mean_assignment_normalized_entropy']:.4f}`; "
            f"launch decision: `{str(decision['launch_low_temperature_followup']).lower()}`."
        )
        if sharp is None:
            lines.append("- The triggered low-temperature pilot has not completed yet.")
        else:
            sharp_router = sharp.get("router_diagnostics", {})
            sharp_metrics = sharp_router.get("mean_across_seeds", {})
            lines.append(
                f"- `K=32`, soft FiLM, temperature `0.1`: mean R² `{sharp['mean_r2']:.4f}`; "
                f"Δ vs. B15P `{sharp['delta_vs_b15p_reference']:.4f}`; "
                f"accuracy gate `{str(sharp['accuracy_feasible']).lower()}`."
            )
            if sharp_metrics:
                lines.append(
                    "- Low-temperature routing: normalized entropy "
                    f"`{sharp_metrics['mean_assignment_normalized_entropy']:.4f}`, "
                    "mean max assignment probability "
                    f"`{sharp_metrics['mean_assignment_max_probability']:.4f}`."
                )

    k32_passed = results["K32"]["pilot_accuracy_feasible"]
    lines.extend([
        "",
        "## Interpretation Boundary",
        "",
        f"- Initial K=32 fixed-interface feasibility gate: `{str(k32_passed).lower()}`.",
        "- This pilot establishes only whether a calibration-frozen fixed-token interface is promising in the current SUA development regime.",
        "- It does not establish formal held-out performance, SUA/MUA reuse, biological unit identity across sessions, or superiority to exact top-K/random/activity pruning controls.",
        "",
    ])
    output = initial_dir / "REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
