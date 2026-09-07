"""Fold-0 per-session Zero4 / rSyn3 M10 carriers. Target query values are unread."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import rectify
from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3

from . import data as fold_data
from . import plan
from . import stage0 as fold_stage0


class CarrierBankError(RuntimeError):
    """Fail closed for Stage-1 carrier construction."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CarrierBankError(message)


def _encode_session(record, basis) -> np.ndarray:
    emg_b, rates_b, ids_b = fold_stage0._mask_budget(record, plan.SUPPORT_TRIALS)
    rsyn3.require_support_bins(ids_b, budget=plan.SUPPORT_TRIALS)
    scores = rsyn3.project_basis(emg_b, basis)
    weights, intercepts = rsyn3.fit_all_units(scores, rates_b)
    return rsyn3.carrier_from_encoding(weights, intercepts)


def build_fold0_carrier_bank(repo_root: Path) -> dict[str, Any]:
    """Source-frozen rSyn3 M10 carriers for fold 0. Target EMG/neural stay in [0,10)."""
    repo_root = Path(repo_root)
    plan.verify_bound_documents(repo_root)
    rectify.assert_frozen_law()
    loaded = fold_data.load_fold_scope(fold=0)
    isolation = dict(loaded["isolation"])
    _require(isolation["target_query_values_read"] is False, "target query leak")
    sources = loaded["sources"]
    target = loaded["target"]
    _require(target.session == plan.FOLD0_TARGET_SESSION, "fold-0 target drift")
    source_emg = np.concatenate(
        [rectify.relu_nonnegative_projection(record.emg) for record in sources.values()],
        axis=0,
    )
    nmf = rsyn3.fit_source_nmf(source_emg)
    source_raw = {name: _encode_session(record, nmf) for name, record in sources.items()}
    norm_mean, norm_scale = rsyn3.source_normalizer(list(source_raw.values()))
    target_raw = _encode_session(target, nmf)
    _require(
        rsyn3.array_digest(target_raw) == plan.FOLD0_M10_RAW_CARRIER_DIGEST,
        "fold-0 target M10 raw carrier digest drift vs sealed Stage-0",
    )
    raw = dict(source_raw)
    raw[target.session] = target_raw
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for name, carrier in raw.items():
        syn = rsyn3.normalize_carriers(carrier, norm_mean, norm_scale)
        zero = np.zeros_like(syn)
        normalized[name] = {"rSyn3": syn, "Zero4": zero}
        _require(np.array_equal(zero, np.zeros(syn.shape)), "Zero4 is not exact zeros")
    return {
        "fold": 0,
        "target_session": target.session,
        "source_sessions": list(plan.FOLD0_SOURCE_SESSIONS),
        "isolation": isolation,
        "normalizer_mean": norm_mean,
        "normalizer_scale": norm_scale,
        "raw": raw,
        "normalized": normalized,
        "ls4_enabled_in_stage1_pilot": False,
        "query_values_read": False,
    }


def carrier_for_arm(bank: dict[str, Any], session_name: str, arm: str) -> np.ndarray:
    _require(arm in plan.STAGE1_ARMS, f"unknown arm {arm}")
    key = "Zero4" if arm == "Z-Fix" else "rSyn3"
    return np.asarray(bank["normalized"][session_name][key], dtype=np.float32)
