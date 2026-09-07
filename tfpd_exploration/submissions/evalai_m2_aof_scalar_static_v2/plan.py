"""Immutable no-data authorities for AOF-S V2 receipt-codec recovery."""
from __future__ import annotations

import hashlib
from pathlib import Path

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import plan as _v1_plan

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v2/local_build"
ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/artifacts/local_build_v2"
INCIDENT_RELATIVE = "tfpd_exploration/docs/RESULT_M2_AOF_SCALAR_STATIC_LOCAL_BUILD_V1_20260903.md"
INCIDENT_SHA256 = "417c0efc8c7f657bb5a103f7d0456ef88a3f67f49ccb9e81eba56531b109e67f"

V1_FAILURE_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v1/local_build"
V1_FAILURE_CLOSURE_SHA256 = "568f544f14a962dccc3c3bfe80814f4c3f39d579937814f92bfd4613298654ad"
V1_FAILURE_BODIES = {
    "attempt.json": "8b25aeef0a5c6b39b97e7a5685e37fc69bbdf66bbaeef3bc0a3f7abcc9811872",
    "predecessor_authority.json": "c05e67949d3acc4599d9f996422af9e584342d9440463614a0bbbddaafb81df7",
    "failure.json": "65a7b119f4899e51340826c92e8b35e2084bdd8d474ffe4021eff5829293a2d2",
}

# V2 binds the same reviewed V1 design authority plus its own additive source
# and never asserts a self-approved closure digest.  An operator must provide
# an independently reviewed map/digest to a future production invocation.
STATIC_CLOSURE_RELATIVES = (
    INCIDENT_RELATIVE,
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/__init__.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/plan.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/binding.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/profile.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/driver.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/Dockerfile",
) + tuple(relative for relative in _v1_plan.STATIC_CLOSURE_RELATIVES if relative != INCIDENT_RELATIVE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_static_authority(repo_root: Path = REPO_ROOT) -> None:
    incident = repo_root / INCIDENT_RELATIVE
    if not incident.is_file() or sha256_file(incident) != INCIDENT_SHA256:
        raise RuntimeError("AOF-S V2 incident authority drift")
    from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import plan as v1_plan
    v1_plan.validate_static_authority(repo_root)
