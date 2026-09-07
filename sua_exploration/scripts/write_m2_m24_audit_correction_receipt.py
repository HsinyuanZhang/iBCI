#!/usr/bin/env python3
"""Bind invalid M24 audit v1 to corrected v2 without overwriting either file."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "sua_exploration/results/general_carrier_proxy_v1"
V1 = RESULTS / "audit_m2_m24_heldin_v1.json"
V2 = RESULTS / "audit_m2_m24_heldin_v2.json"
OUT = RESULTS / "audit_m2_m24_correction_receipt_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite correction receipt: {OUT}")
    v1, v2 = (json.loads(path.read_text(encoding="utf-8")) for path in (V1, V2))
    if v1.get("protocol", {}).get("fit_trials") != "0:10" or v1.get("protocol", {}).get("prediction_trials") != "10:20":
        raise ValueError("v1 is not the expected stale-M20-metadata artifact")
    protocol = v2.get("protocol", {})
    if protocol.get("name") != "m24" or protocol.get("fit_trials") != "0:12" or protocol.get("prediction_trials") != "12:24" or protocol.get("support_trials") != 24:
        raise ValueError("v2 M24 protocol metadata is incomplete")
    if v1.get("formal_heldout_evaluated") is not False or v2.get("formal_heldout_evaluated") is not False:
        raise ValueError("qualification audit must remain held-in-only")
    fields = (
        "n_fit_blocks", "n_prediction_blocks", "kreg_mse", "rate_only_fit_baseline_mse",
        "kreg_over_baseline_ratio", "w_only_null_median_mse", "kreg_over_w_only_null_median_ratio",
        "beats_all_100_w_only_nulls", "w_only_null_count", "flattened_W_A_B_correlation",
    )
    old_by_name = {row["session"]: row for row in v1["sessions"]}
    new_by_name = {row["session"]: row for row in v2["sessions"]}
    if set(old_by_name) != set(new_by_name) or len(old_by_name) != 7:
        raise ValueError("v1/v2 session cohort drift")
    for name in old_by_name:
        old, new = old_by_name[name], new_by_name[name]
        if old.get("fit_trials") != [0, 12] or old.get("prediction_trials") != [12, 24]:
            raise ValueError(f"v1 numeric row is not M24: {name}")
        if new.get("fit_trials") != [0, 12] or new.get("prediction_trials") != [12, 24]:
            raise ValueError(f"v2 numeric row is not M24: {name}")
        if any(old.get(field) != new.get(field) for field in fields):
            raise ValueError(f"v1/v2 numeric value drift for {name}")
        if old["all20_active_blocks"] != new["all_support_active_blocks"] or old["all20_design_rank"] != new["all_support_design_rank"] or old["all20_design_condition"] != new["all_support_design_condition"]:
            raise ValueError(f"v1/v2 renamed support-audit mismatch for {name}")
    payload = {
        "schema_version": 1,
        "purpose": "invalidate_stale_top_level_M20_metadata_and_bind_corrected_M24_receipt",
        "invalid_artifact": str(V1.resolve()), "invalid_artifact_sha256": sha256(V1),
        "invalid_reason": "v1 session rows are M24 but top-level protocol/scope fields incorrectly say 0:10/10:20/M20",
        "corrected_artifact": str(V2.resolve()), "corrected_artifact_sha256": sha256(V2),
        "numeric_rows_verified_identical": True, "session_count": 7,
        "corrected_protocol": protocol,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
