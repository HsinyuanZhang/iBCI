"""Audited RT M24 adapter for the strong cross-reach XLSv2 label null.

The adapter consumes only raw chronological support neural/velocity bins and
event-qualified reach IDs.  It applies the immutable, session-namespaced
cross-reach permutation to support velocity labels and then calls the same
four-coordinate OLS fit used by aligned AFC4.  It has no model, checkpoint,
query-label, score, inverse, rotation, Procrustes, or learned alignment input.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np

from src.data.afc4_xls_v2 import (
    SEED,
    V2_AUDIT_SCHEMA,
    V2_AUDIT_STATUS,
    assert_audited_session_parity,
    deterministic_random_cross_reach_derangement,
)
from src.data.falcon_k4_features import (
    K4_ACTIVE_EPSILON,
    K4_BEHAVIOR_LEAD_BINS,
    K4_BLOCK_WIDTH_BINS,
    K4_RAW_BIN_MS,
    K4Audit,
    collect_k4_support_blocks,
    fit_k4_descriptor_from_blocks,
)


CALIBRATION_TRIALS = 24
AUDIT_SHA256 = "ad2468ca04c3ed2542c7c37b0f1d27c17d4e517943fe64a7fa34e6cf9636c899"
DEFAULT_AUDIT_PATH = (
    Path(__file__).resolve().parents[3]
    / "sua_exploration/results/rt_afc4_ls_null_strength_audit_v2"
    / "RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json"
)


class Afc4XlsV2AdapterError(ValueError):
    """The active XLSv2 support/adaptation contract failed closed."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise Afc4XlsV2AdapterError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_immutable_xls_v2_audit(path: str | Path = DEFAULT_AUDIT_PATH) -> tuple[dict[str, Any], str]:
    """Load only the frozen support audit, with exact mode/content binding."""

    source = Path(path).resolve()
    _need(source.is_file(), f"XLSv2 support audit is missing: {source}")
    _need(stat.S_IMODE(source.stat().st_mode) == 0o444, f"XLSv2 support audit must be mode 0444: {source}")
    digest = _sha256(source)
    _need(digest == AUDIT_SHA256, f"XLSv2 support audit SHA mismatch: {digest}")
    try:
        receipt = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Afc4XlsV2AdapterError(f"invalid XLSv2 support audit JSON: {source}") from error
    _need(isinstance(receipt, dict), "XLSv2 support audit must be a JSON object")
    _need(
        receipt.get("schema") == V2_AUDIT_SCHEMA and receipt.get("status") == V2_AUDIT_STATUS,
        "XLSv2 support audit schema/status drift",
    )
    return receipt, digest


def _session_row(receipt: Mapping[str, Any], session_name: str) -> Mapping[str, Any]:
    rows = receipt.get("fold_rows")
    _need(isinstance(rows, list), "XLSv2 support audit fold_rows absent")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("session_name") == session_name]
    _need(len(matches) == 1, f"XLSv2 support audit requires one row for {session_name!r}")
    return matches[0]


def afc4_xls_v2_from_support(
    neural: np.ndarray,
    velocity: np.ndarray,
    trial_change: np.ndarray,
    *,
    segment_ids: np.ndarray,
    session_name: str,
    audit_receipt: Mapping[str, Any],
    audit_receipt_sha256: str,
    calibration_n_trials: int = CALIBRATION_TRIALS,
    seed: int = SEED,
) -> tuple[np.ndarray, K4Audit]:
    """Fit XLSv2 from M24 support only and return a parity-bound audit.

    There is intentionally no argument for query labels/covariates, decoder
    output, model score, checkpoint, or an inverse/alignment map.
    """

    _need(int(calibration_n_trials) == CALIBRATION_TRIALS, "XLSv2 is frozen to chronological M24")
    _need(int(seed) == SEED, "XLSv2 is frozen to seed 42")
    _need(bool(session_name), "XLSv2 requires a nonempty session namespace")
    _need(audit_receipt_sha256 == AUDIT_SHA256, "XLSv2 active adapter received the wrong audit SHA")
    _need(
        audit_receipt.get("schema") == V2_AUDIT_SCHEMA and audit_receipt.get("status") == V2_AUDIT_STATUS,
        "XLSv2 active adapter received a non-passing support audit",
    )
    blocks = collect_k4_support_blocks(
        neural,
        velocity,
        trial_change,
        calibration_n_trials=CALIBRATION_TRIALS,
        segment_ids=segment_ids,
    )
    row = _session_row(audit_receipt, session_name)
    _need(row.get("support_trial_index_range") == [0, CALIBRATION_TRIALS], "XLSv2 audit M24 range drift")
    _need(int(row.get("active_blocks", -1)) == int(blocks.rates.shape[0]), "XLSv2 active-block count drift")
    _need(int(row.get("accepted_reaches", -1)) == int(np.unique(blocks.group_ids).size), "XLSv2 reach count drift")
    _need(int(row.get("num_channels", -1)) == int(blocks.rates.shape[1]), "XLSv2 channel count drift")

    permutation, selection = deterministic_random_cross_reach_derangement(
        blocks.group_ids,
        blocks.velocity,
        session_name=session_name,
        seed=SEED,
    )
    permutation_sha = assert_audited_session_parity(
        permutation,
        audit_receipt=audit_receipt,
        session_name=session_name,
    )
    features, design_rank, design_condition = fit_k4_descriptor_from_blocks(
        blocks.rates,
        blocks.velocity[permutation],
    )
    audit = K4Audit(
        calibration_trials=CALIBRATION_TRIALS,
        active_blocks=int(blocks.rates.shape[0]),
        raw_bin_ms=K4_RAW_BIN_MS,
        block_width_bins=K4_BLOCK_WIDTH_BINS,
        behavior_lead_bins=K4_BEHAVIOR_LEAD_BINS,
        active_rule=f"all_samples_neural_and_shifted_behavior_active__not_all_abs_velocity_lt_{K4_ACTIVE_EPSILON}",
        max_trial_length_used=False,
        design_rank=design_rank,
        design_condition=design_condition,
        extra={
            "candidate_trial_bounded_blocks": int(blocks.candidate_blocks),
            "segment_qualified_blocks": int(blocks.segment_qualified_blocks),
            "segment_constrained": True,
            "label_shuffle": True,
            "label_shuffle_policy": "xls_v2_session_namespaced_random_cross_reach_derangement",
            "label_shuffle_seed": SEED,
            "label_changed_blocks": int(np.count_nonzero(permutation != np.arange(permutation.size))),
            "label_permutation_sha256": permutation_sha,
            "xls_v2_support_audit_sha256": AUDIT_SHA256,
            "xls_v2_selection_attempt": int(selection["attempt"]),
            "query_labels_available_to_generator": False,
            "common_inverse_or_alignment_map": False,
        },
    )
    return features, audit
