#!/usr/bin/env python3
"""CPU dry-run of H-U on one H1 source date.  No GPU, no target NWB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.h1_carrierid_hu_features import (
    DECLARED_DEAD_CHANNELS,
    FEATURE_NAMES,
    descriptor_rank,
    hu_from_record,
    pearson_corr_matrix,
    per_unit_column_stats,
)
from src.data.h1_m4_eb_pilot import carrier_sha256, load_record
from src.h1_m4_eb_normalized_v2_contract import write_immutable_json


CACHE = ROOT / "pilot_artifacts/h1_m4_eb_fold0/preflight_cache"
HC_NPZ = CACHE / "fold0_all_source_m4_carriers.npz"
HC_MANIFEST = CACHE / "fold0_all_source_m4_carriers.manifest.json"
DATA = ROOT / "data/000954"
SESSION = "ses-19250108T110520"
TRIAL_VALUES = (1.0, 2.0, 3.0, 4.0)
FROZEN_HC_CACHE_SHA256 = "88261cc03532b605da1790e8669760d4d47e2f87d2db1428060445541638b0af"
FROZEN_HC_FIRST = "a3fe44ca2ba767577add07c94aa0e7ba37c3604579cb3bdb66c37c2f1d4b24fe"


def _session_nwb() -> Path:
    calib = DATA / "sub-HumanPitt-held-in-calib"
    return next(path for path in sorted(calib.glob("*.nwb")) if SESSION in path.name)


def run() -> dict[str, Any]:
    manifest = json.loads(HC_MANIFEST.read_text(encoding="utf-8"))
    if manifest["cache_sha256"] != FROZEN_HC_CACHE_SHA256:
        raise RuntimeError("sealed H-C source cache hash drifted")
    with np.load(HC_NPZ, allow_pickle=False) as values:
        hc = np.asarray(values["carriers"][0], dtype=np.float64)
    if carrier_sha256(hc) != FROZEN_HC_FIRST:
        raise RuntimeError("sealed first H-C entry hash drifted")
    record = load_record(_session_nwb())
    hu_raw, audit = hu_from_record(record, TRIAL_VALUES)
    stats = per_unit_column_stats(hu_raw, silent=audit.silent_channels)
    live_idx = [i for i in range(hu_raw.shape[0]) if i not in set(audit.silent_channels)]
    corr = pearson_corr_matrix(hu_raw[live_idx], hc[live_idx])
    spike_sum = np.asarray(record.neural, dtype=np.float64).sum(axis=0)
    ch66 = float(spike_sum[66]) if spike_sum.size > 66 else None
    degenerate = any(col["degenerate_constant"] for col in stats["columns"].values())
    body = {
        "schema": "h1_carrierid_hu_cpu_dryrun_one_date_v1",
        "status": "PASS_HU_CPU_DRYRUN_ONE_DATE" if not degenerate else "WARN_HU_DEGENERATE_DESCRIPTOR",
        "gpu_launched": False,
        "target_nwb_opened": False,
        "session": SESSION,
        "date": "19250108",
        "fold0_outer_date": "19250101",
        "trial_values": list(TRIAL_VALUES),
        "feature_names": list(FEATURE_NAMES),
        "declared_dead_channels": sorted(DECLARED_DEAD_CHANNELS),
        "channel_66_spike_sum_whole_recording": ch66,
        "audit": audit.as_dict(),
        "per_unit_stats": stats,
        "rank_live": descriptor_rank(hu_raw, silent=audit.silent_channels),
        "hu_raw_sha256": hashlib.sha256(np.ascontiguousarray(hu_raw).tobytes()).hexdigest(),
        "hc_first_entry_sha256": FROZEN_HC_FIRST,
        "hc_cache_sha256": FROZEN_HC_CACHE_SHA256,
        "pearson_corr_hu_rows_vs_hc_cols": corr.tolist(),
        "pearson_corr_abs_max": float(np.max(np.abs(corr))),
        "label_free": True,
        "used_principal_components": False,
        "degenerate_warning": bool(degenerate),
    }
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "pilot_artifacts/h1_carrierid_hu/H1_CARRIERID_HU_CPU_DRYRUN_19250108_v1.json",
    )
    args = parser.parse_args()
    body = run()
    path, digest = write_immutable_json(args.output, body)
    print(json.dumps({"path": str(path), "sha256": digest, "summary": {
        "status": body["status"],
        "rank_live": body["rank_live"],
        "n_silent": body["audit"]["n_silent_channels"],
        "channel_66_spike_sum_whole_recording": body["channel_66_spike_sum_whole_recording"],
        "pearson_corr_abs_max": body["pearson_corr_abs_max"],
        "per_unit_stats": body["per_unit_stats"],
        "degenerate_warning": body["degenerate_warning"],
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
