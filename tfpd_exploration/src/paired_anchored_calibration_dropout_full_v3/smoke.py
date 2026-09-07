"""V3 profile composition; execution remains the one shared full lifecycle."""
from __future__ import annotations

from . import plan
from .predecessor import validate_v2_failure
from src.paired_anchored_calibration_dropout_full_v1 import smoke as full_v1


V3_EXECUTION_PROFILE = full_v1.ExecutionProfile(
    identity="full-v3",
    plan=plan.RUNTIME_PLAN,
    predecessor_validator=validate_v2_failure,
)


def dry_payload():
    return {
        "cell": plan.CELL,
        "schema": plan.SCHEMA,
        "arms": {name: dict(value) for name, value in plan.ARMS.items()},
        "predecessor_required": True,
        "zero_encoder_policy": plan.ZERO_ENCODER_POLICY,
        "no_torch_import": True,
        "full_launch_authorized": False,
    }


def issue_capability(*, root, arm, runtime_factory=full_v1.PRODUCTION_RUNTIME_FACTORY):
    return full_v1._issue_root_capability_after_preflight(
        root=root, arm=arm, runtime_factory=runtime_factory, profile=V3_EXECUTION_PROFILE
    )


def execute(*, root, arm, args, capability, runtime_factory=full_v1.PRODUCTION_RUNTIME_FACTORY,
            profile=V3_EXECUTION_PROFILE):
    return full_v1.execute(
        root=root, arm=arm, args=args, capability=capability,
        runtime_factory=runtime_factory, profile=profile,
    )
