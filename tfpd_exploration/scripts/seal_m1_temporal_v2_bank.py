#!/usr/bin/env python3
"""Seal M1-TEMPORAL-v2 / rSyn3-refit-v1. Source 26/27/28 only. No target NWB."""

from __future__ import annotations

import json
import os
import sys

os.environ["PYTHONNOUSERSITE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from tfpd_exploration.src.m1_temporal_v2.bank import seal_source_bank
from tfpd_exploration.src.m1_temporal_v2 import plan


def main() -> int:
    receipt = seal_source_bank()
    print(json.dumps({
        "revision": plan.REVISION,
        "carrier_name": plan.CARRIER_NAME,
        "npz": str(plan.BANK_NPZ),
        "npz_sha256": receipt["digests"]["npz"],
        "d0": receipt["digests"]["d0"],
        "not_old_d0": receipt["digests"]["d0"] != plan.OLD_D0_DIGEST,
        "target_loaded": receipt["target_loaded"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
