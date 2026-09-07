"""Calibration-gap decomposition route (handoff 2026-08-24).

Zero/low-cost裁决批次 + P1..P5 路径的代码基础。全部性能数字从 SHA 验证的
sealed 收据加载（ledger.py），禁止硬编码。纯 numpy/json/hashlib——无 torch
依赖，CPU 即可。
"""

from .ledger import (
    RECEIPTS,
    GapLedger,
    LedgerError,
    load_ledger,
)
from .coverage import (
    CoverageRecords,
    coverage_covariates,
    coverage_regression,
    outlier_report,
    per_dof_attribution,
)

__all__ = [
    "RECEIPTS",
    "GapLedger",
    "LedgerError",
    "load_ledger",
    "CoverageRecords",
    "coverage_covariates",
    "coverage_regression",
    "outlier_report",
    "per_dof_attribution",
]
