"""Produce a compact immutable v2 PCF8 ridge100 conclusion from its raw audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _median(items: list[float]) -> float:
    return float(np.median(np.asarray(items, dtype=np.float64)))


def budget_summary(raw: dict, budget: str) -> dict:
    rows = [row[budget] for row in raw["per_recording"].values()]
    phase_rows = [phase for row in rows for phase in row["phase_conditioned_stability"]["phases"].values()]
    phase_fit = [row for row in phase_rows if row["status"] == "FIT"]
    return {
        "regularized_condition_min_median_max": [
            min(row["support_fit"]["design"]["ridge100"]["regularized_system_condition_number"] for row in rows),
            _median([row["support_fit"]["design"]["ridge100"]["regularized_system_condition_number"] for row in rows]),
            max(row["support_fit"]["design"]["ridge100"]["regularized_system_condition_number"] for row in rows),
        ],
        "effective_df_trace_hat_min_median_max": [
            min(row["support_fit"]["design"]["ridge100"]["effective_degrees_of_freedom_trace_hat"] for row in rows),
            _median([row["support_fit"]["design"]["ridge100"]["effective_degrees_of_freedom_trace_hat"] for row in rows]),
            max(row["support_fit"]["design"]["ridge100"]["effective_degrees_of_freedom_trace_hat"] for row in rows),
        ],
        "per_channel_7D_W_cosine_raw_min_median_max": [
            min(row["support_later_raw"]["per_channel_7D_W_cosine"]["median"] for row in rows),
            _median([row["support_later_raw"]["per_channel_7D_W_cosine"]["median"] for row in rows]),
            max(row["support_later_raw"]["per_channel_7D_W_cosine"]["median"] for row in rows),
        ],
        "per_channel_7D_W_cosine_source_scale_only_min_median_max": [
            min(row["support_later_after_source_scale_only"]["per_channel_7D_W_cosine"]["median"] for row in rows),
            _median([row["support_later_after_source_scale_only"]["per_channel_7D_W_cosine"]["median"] for row in rows]),
            max(row["support_later_after_source_scale_only"]["per_channel_7D_W_cosine"]["median"] for row in rows),
        ],
        "W_norm_median_across_recordings": _median([row["W_norm"]["summary"]["median"] for row in rows]),
        "near_zero_W_channels_total": int(sum(row["W_norm"]["near_zero_count"] for row in rows)),
        "phase_fit_status_counts": {status: sum(row["status"] == status for row in phase_rows) for status in sorted({row["status"] for row in phase_rows})},
        "phase_W_support_later_cosine_median_by_column": {
            f"W{column}": _median([row["support_reference_W_column_cosines"][column] for row in phase_fit])
            for column in range(7)
        },
        "phase_W_vs_global_support_cosine_median_by_column": {
            f"W{column}": _median([row["phase_vs_global_support_W_column_cosines"][column] for row in phase_fit])
            for column in range(7)
        },
        "predeclared_gates": raw["gates"][budget],
    }


def report(raw: dict, raw_path: Path) -> dict:
    if raw.get("schema") != "h1_pcf8_ridge100_source_constructibility_v2":
        raise ValueError("v2 raw schema drift")
    m4, m3 = budget_summary(raw, "M4"), budget_summary(raw, "M3")
    return {
        "schema": "h1_pcf8_ridge100_source_summary_v2",
        "status": raw["status"],
        "raw_audit": {"path": str(raw_path.resolve()), "sha256": sha256(raw_path), "mode": oct(stat.S_IMODE(raw_path.stat().st_mode))},
        "scope": raw["scope"],
        "decision": {
            "rule": "All four predeclared source constructibility gates must pass at M4 and M3. A failure closes PCF8; these gates never predict decoder R2.",
            "verdict": "CLOSED: ridge100 makes the system numerically well conditioned but almost entirely intercept-dominated (effective df near one). The signed W carrier fails normalized reliability and label-pairing encoding gates at both M4 and organizer-held M3.",
            "no_further_action": "No lambda grid, post-hoc feature standardization, GPU run, target decoder evaluation, or reuse of v1 OLS as decision evidence is permitted for PCF8.",
        },
        "M4_primary": m4,
        "M3_organizer_held_ancillary": m3,
        "source_per_column_scale": raw["source_per_column_scale"],
        "fit_contract": raw["fit_contract"],
        "future_arm_schema": raw["future_arm_schema"],
        "state_compute_contract": raw["state_compute_contract"],
    }


def publish_once(output: Path, body: dict) -> None:
    output = output.resolve()
    if output.exists(): raise FileExistsError(f"summary refuses overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n"); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444); os.link(temporary, output)
    finally:
        if temporary.exists(): temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--raw", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); raw = args.raw.resolve()
    if not raw.is_file() or stat.S_IMODE(raw.stat().st_mode) != 0o444: raise ValueError("raw must be immutable 0444")
    publish_once(args.output, report(json.loads(raw.read_text()), raw))


if __name__ == "__main__": main()
