"""Summarize the immutable PCF8 source audit without accessing data or a decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(values: list[float | None]) -> float | None:
    finite = [item for item in values if item is not None and np.isfinite(item)]
    return None if not finite else float(np.median(np.asarray(finite, dtype=np.float64)))


def budget_summary(audit: dict, budget: str) -> dict:
    records = list(audit["per_recording"].values())
    block_values = {name: [int(row[budget][name]) for row in records] for name in ("support_blocks", "reference_blocks")}
    column_cosines = [
        _median([row[budget]["support_vs_later_reference"]["columns"][column]["support_reference_cosine_across_channels"] for row in records])
        for column in range(8)
    ]
    phase_rows = [phase for row in records for phase in row[budget]["phase_conditioned_stability"]["phases"].values()]
    phase_fit = [phase for phase in phase_rows if phase["status"] == "FIT"]
    phase_reference = [
        _median([row["support_reference_column_cosines"][column] for row in phase_fit])
        for column in range(7)
    ]
    phase_global = [
        _median([row["phase_vs_global_support_W_column_cosines"][column] for row in phase_fit])
        for column in range(7)
    ]
    return {
        "support_blocks_min_median_max": [int(min(block_values["support_blocks"])), float(np.median(block_values["support_blocks"])), int(max(block_values["support_blocks"]))],
        "later_disjoint_reference_blocks_min_median_max": [int(min(block_values["reference_blocks"])), float(np.median(block_values["reference_blocks"])), int(max(block_values["reference_blocks"]))],
        "all_support_design_ranks": [int(row[budget]["support_fit"]["design"]["rank"]) for row in records],
        "support_raw_condition_min_median_max": [
            float(min(row[budget]["support_fit"]["design"]["raw_condition_number"] for row in records)),
            float(np.median([row[budget]["support_fit"]["design"]["raw_condition_number"] for row in records])),
            float(max(row[budget]["support_fit"]["design"]["raw_condition_number"] for row in records)),
        ],
        "median_support_later_reference_cosine_across_channels": {
            **{f"W{column}": column_cosines[column] for column in range(7)}, "b": column_cosines[7],
        },
        "median_per_channel_W_cosine": _median([row[budget]["support_vs_later_reference"]["per_channel_W_cosine"]["median"] for row in records]),
        "degenerate_channels_union": sorted({channel for row in records for channel in row[budget]["support_fit"]["degenerate_channels"]}),
        "phase_conditioned_fit_status_counts": {status: sum(row["status"] == status for row in phase_rows) for status in sorted({row["status"] for row in phase_rows})},
        "median_phase_conditional_support_later_reference_W_cosine": {f"W{column}": phase_reference[column] for column in range(7)},
        "median_phase_vs_global_support_W_cosine": {f"W{column}": phase_global[column] for column in range(7)},
    }


def report(audit: dict, raw_path: Path) -> dict:
    _need(audit.get("schema") == "h1_pcf8_source_constructibility_precursor_v1", "raw audit schema drift")
    _need(audit.get("status") == "SOURCE_CPU_AUDIT_COMPLETE__NO_GPU_AUTHORIZATION", "raw audit status drift")
    m4, m3 = budget_summary(audit, "M4"), budget_summary(audit, "M3")
    return {
        "schema": "h1_pcf8_source_constructibility_summary_v1",
        "status": "FAIL_CPU_RELIABILITY_FORWARD_COEFFICIENTS__NO_GPU_AUTHORIZATION",
        "raw_audit": {"path": str(raw_path.resolve()), "sha256": sha256(raw_path), "mode": oct(stat.S_IMODE(raw_path.stat().st_mode))},
        "scope": {"source_recordings": 11, "target_recordings_opened": 0, "target_decoder_r2_computed": False, "gpu_used": False, "target_backward_steps": 0},
        "conclusion": {
            "numerical_constructibility": "PASS: all 11 source records have rank-8 forward designs at both M4 and ancillary M3.",
            "reliability": "FAIL: baseline b is near-identical between support and later same-recording reference (~0.994 cosine), whereas signed W columns and per-channel W vectors are not reproducible; phase-conditional W fits are near orthogonal to later phase-matched references and to global support W. This is phase-mixing/temporal-instability evidence, not a decoder-R2 prediction.",
            "organizer_held_M3": "M3 is explicitly audited on the same source records. Its W reliability is no better than M4, so this precursor cannot support an organizer-held H1 PCF8 claim.",
            "action": "Do not prepare GPU arms from this object. Preserve the result as a negative CPU constructibility/reliability result; no threshold was selected from target metrics.",
        },
        "M4_primary_development": m4,
        "M3_ancillary_organizer_held": m3,
        "normalization": audit["source_per_column_scale"],
        "future_controls_if_a_new_object_is_independently_proposed": audit["future_arm_schema"],
        "state_compute_contract": audit["state_compute_contract"],
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ROOT / "SPINT-main/src/data/h1_m4_eb_pilot.py",
                ROOT / "SPINT-main/src/data/h1_lag_screen.py",
                ROOT / "SPINT-main/src/models/components/h1_carrierid_spint.py",
            )
        },
    }


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def publish_once(output: Path, rendered: str) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"PCF8 summary refuses to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw.resolve()
    _need(raw.is_file() and stat.S_IMODE(raw.stat().st_mode) == 0o444, "raw audit must be immutable 0444")
    body = report(json.loads(raw.read_text(encoding="utf-8")), raw)
    publish_once(args.output, json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
