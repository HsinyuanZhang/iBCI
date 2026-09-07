#!/usr/bin/env python3
"""Receipt contracts for H1 h=32 H-LS source-only early launch v2.

Early v2 authorizes no target operation.  Its only purpose is to let the
complete, fixed five-date source-training grid run while the original H-S/H-C
five-date aggregate is still pending.  A separate post-upstream binder must
later prove compatibility with that completed aggregate before an evaluator
can consume any early terminal.
"""
from __future__ import annotations

from pathlib import Path

from scripts.h1_carrierid_date_lodo_hls_fivedate_contract import (
    DATES, NULL_AUDIT_DEFAULT, NULL_AUDIT_SHA256, ROOT, WAITING_SCHEMA,
    WAITING_STATUS, read_immutable_json, sha256_file, validate_null_strength_audit,
    write_immutable_json,
)


EARLY_PREFLIGHT_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_source_preflight_v1"
EARLY_PREFLIGHT_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_EARLY_V2_SOURCE_ONLY_NOT_LAUNCHED"
EARLY_LAUNCH_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_fivedate_launch_receipt_v1"
EARLY_LAUNCH_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_EARLY_V2_FIVEDATE_PREPARED_NOT_LAUNCHED"
EARLY_TERMINAL_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_source_e49_terminal_v1"
EARLY_TERMINAL_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_EARLY_V2_SOURCE_E49_NO_TARGET"
POST_BINDER_SCHEMA = "h1_carrierid_date_lodo_hls_early_v2_post_upstream_binder_v1"
POST_BINDER_STATUS = "PASS_H1_CARRIERID_DATE_LODO_HLS_EARLY_V2_BOUND_TO_COMPLETE_UPSTREAM_NO_TARGET"

PAIR_SCHEMA = "h1_carrierid_date_lodo_phase2_pair_cpu_preflight_v1"
PAIR_STATUS = "PASS_H1_CARRIERID_DATE_LODO_PHASE2_PAIR_SOURCE_ONLY_NOT_LAUNCHED"

STATIC_PARTITIONS = {
    "local3090_gpu0": (DATES[0], DATES[2], DATES[4]),
    "local3090_gpu1": (DATES[1], DATES[3]),
}

DEFAULT_WAITING_PLAN = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_hls_fivedate_waiting_plan/"
    "H1_CARRIERID_DATE_LODO_HLS_FIVEDATE_WAITING_PLAN_v1.json"
)
DEFAULT_PHASE1_PREFLIGHT = (
    ROOT / "pilot_artifacts/h1_carrierid_date_lodo_phase1/"
    "H1_CARRIERID_DATE_LODO_SOURCE_PREFLIGHT_v1.json"
)


class HlsEarlyV2ContractError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise HlsEarlyV2ContractError(message)


def validate_waiting_plan(path: str | Path) -> tuple[Path, dict, str]:
    candidate, body, digest = read_immutable_json(path, schema=WAITING_SCHEMA, status=WAITING_STATUS)
    need(tuple(body.get("future_complete_grid", {}).get("outer_dates", ())) == DATES,
         "early v2 waiting plan does not bind the exact five-date grid")
    control = body.get("control_semantics", {})
    need(control.get("strong_null_audit", {}).get("sha256") == NULL_AUDIT_SHA256,
         "early v2 waiting plan does not bind the sealed null-strength audit")
    return candidate, body, digest


__all__ = [
    "DATES", "NULL_AUDIT_DEFAULT", "ROOT", "read_immutable_json", "sha256_file",
    "validate_null_strength_audit", "write_immutable_json", "EARLY_PREFLIGHT_SCHEMA",
    "EARLY_PREFLIGHT_STATUS", "EARLY_LAUNCH_SCHEMA", "EARLY_LAUNCH_STATUS",
    "EARLY_TERMINAL_SCHEMA", "EARLY_TERMINAL_STATUS", "POST_BINDER_SCHEMA",
    "POST_BINDER_STATUS", "PAIR_SCHEMA", "PAIR_STATUS", "STATIC_PARTITIONS",
    "DEFAULT_WAITING_PLAN", "DEFAULT_PHASE1_PREFLIGHT", "HlsEarlyV2ContractError",
    "need", "validate_waiting_plan",
]
