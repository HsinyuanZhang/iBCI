"""Typed immutable V2 profile composed with the V1 build chain."""
from __future__ import annotations

from dataclasses import dataclass

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.export_aofs_static_payload import (
    ReceiptCodec,
    V2_SIDE_EVIDENCE_RECEIPT_CODEC,
)

from . import plan


@dataclass(frozen=True)
class AofsRecoveryProfile:
    name: str
    result_root_relative: str
    artifact_root_relative: str
    receipt_codec: ReceiptCodec
    predecessor_kind: str


V2_RECOVERY_PROFILE = AofsRecoveryProfile(
    name="aofs_static_v2_side_evidence_receipt_recovery",
    result_root_relative=plan.RESULT_ROOT_RELATIVE,
    artifact_root_relative=plan.ARTIFACT_ROOT_RELATIVE,
    receipt_codec=V2_SIDE_EVIDENCE_RECEIPT_CODEC,
    predecessor_kind="exact_aofs_v1_pre_payload_failure",
)

