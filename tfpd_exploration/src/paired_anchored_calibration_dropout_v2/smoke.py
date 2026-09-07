"""Thin V2 lineage wrapper around the reviewed V1 paired executor."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from src.paired_anchored_calibration_dropout_v1 import smoke as v1
from . import plan
from .predecessor import validate_v1_predecessor

def profile() -> v1.ExecutionProfile:
    return v1.ExecutionProfile(cell=plan.CELL,schema=plan.SCHEMA,result_root_relative=plan.RESULT_ROOT_RELATIVE,bound_patterns=plan.BOUND_PATTERNS,review_evidence_paths=plan.REVIEW_EVIDENCE_PATHS,expected_sealed_sha256=plan.EXPECTED_SEALED_SHA256,work_order_relative=plan.WORK_ORDER_RELATIVE,predecessor_validator=validate_v1_predecessor)

def execute(*, root: Path, out_dir: Path, args: Any) -> int:
    return v1.execute(root=root,out_dir=out_dir,args=args,profile=profile())

def dry_payload() -> dict[str, object]:
    payload=v1.dry_payload(); payload.update({"cell":plan.CELL,"schema":plan.SCHEMA,"result_root":plan.RESULT_ROOT_RELATIVE,"predecessor_required":True}); return payload
