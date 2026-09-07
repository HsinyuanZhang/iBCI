"""Inert V2 descriptor; execution composition awaits root profile admission."""
from __future__ import annotations
from . import plan
from .predecessor import validate_v1_failure
from src.paired_anchored_calibration_dropout_full_v1 import smoke as full_v1
V2_EXECUTION_PROFILE=full_v1.ExecutionProfile(identity="full-v2",plan=plan.RUNTIME_PLAN,predecessor_validator=validate_v1_failure)
def dry_payload():
 return {"cell":plan.CELL,"schema":plan.SCHEMA,"arms":plan.ARMS,"predecessor_required":True,"no_torch_import":True,"full_launch_authorized":False}
def issue_capability(*,root,arm,runtime_factory=full_v1.PRODUCTION_RUNTIME_FACTORY):
 return full_v1._issue_root_capability_after_preflight(root=root,arm=arm,runtime_factory=runtime_factory,profile=V2_EXECUTION_PROFILE)
def execute(*,root,arm,args,capability,runtime_factory=full_v1.PRODUCTION_RUNTIME_FACTORY,profile=V2_EXECUTION_PROFILE):
 return full_v1.execute(root=root,arm=arm,args=args,capability=capability,runtime_factory=runtime_factory,profile=profile)
