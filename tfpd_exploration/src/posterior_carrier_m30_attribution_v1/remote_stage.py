"""Static future-stage description for the M30 attribution route.

The module is intentionally standard-library-only.  It names the immutable
V3 predecessor and the exact new route files that a reviewed future transport
must stage, but it neither reads payloads nor opens NWB/checkpoint data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import attribution as contract


class AttributionStageError(RuntimeError):
    """Raised when a future stage plan drifts before transport is allowed."""


@dataclass(frozen=True)
class AttributionStagePlan:
    """A declarative plan; byte discovery remains root-reviewed and deferred."""

    local_root: Path
    identity: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        checked = contract.validate_identity(self.identity)
        return {
            "schema": "posterior_carrier_m30_attribution_remote_stage_plan_v1",
            "classification": contract.CLASSIFICATION,
            "cell": contract.CELL,
            "phase": contract.PHASE,
            "identity": checked,
            "remote_stage_root": contract.REMOTE_STAGE_ROOT,
            "remote_result_root_relative": contract.RESULT_ROOT_RELATIVE,
            "v3_predecessor": contract.V3Evidence().payload(),
            "staging_policy": {
                "network_performed": False,
                "nwb_opened": False,
                "checkpoint_tensor_opened": False,
                "cuda_initialized": False,
                "result_root_created": False,
                "requires_future_root_review_capability": True,
                "reuse_v3_fixed_six_evaluation_assets_only": True,
                "new_stage_must_be_fresh": True,
            },
        }


def dry_stage_plan(*, root: Path, identity: contract.AttributionIdentity) -> dict[str, object]:
    """Return a pure plan without lstat/open/transport side effects."""
    del root
    return AttributionStagePlan(local_root=Path("."), identity=contract.validate_identity(identity)).payload()
