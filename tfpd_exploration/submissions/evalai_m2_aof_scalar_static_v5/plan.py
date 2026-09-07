"""Immutable no-network authorities for AOF-S V5 parent-layout recovery."""
from __future__ import annotations

import hashlib
from pathlib import Path

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v4 import plan as _v4_plan

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v5/local_build"
ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5/artifacts/local_build_v5"

V4_INCIDENT_RELATIVE = "tfpd_exploration/docs/RESULT_M2_AOF_SCALAR_STATIC_LOCAL_BUILD_V4_20260903.md"
V4_INCIDENT_SHA256 = "ede62c1d2fc67ddd79e383d76a2f37e9da286e6c7217bd1c1d970051f92139c0"
V4_FAILURE_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v4/local_build"
V4_FAILURE_CLOSURE_SHA256 = "736ef936059c59d873040f24af23eda5df79547768a41cbe1e82ff2be9920610"
V4_FAILURE_BODIES = {
    "attempt.json": "fa653f6ebb0a8395e2b30152dea3c486665726f16551cfb7f763449c7b178a4d",
    "predecessor_authority.json": "229fda31ff4fc22a67d57cec0e902b3fd4ede73977507d2f5ad81ed00ade08ba",
    "failure.json": "5e2260f80e2ddb5e33e5f5e8b22b6371efdc91b9229a02a3c5b0682b6d95dc59",
}

V2_ARTIFACT_ROOT_RELATIVE = _v4_plan.V2_ARTIFACT_ROOT_RELATIVE
V2_ARTIFACT_BODIES = _v4_plan.V2_ARTIFACT_BODIES
PAYLOAD_NAME = _v4_plan.PAYLOAD_NAME
RECEIPT_NAME = _v4_plan.RECEIPT_NAME
DOCKER_EXECUTABLE = _v4_plan.DOCKER_EXECUTABLE
LOCAL_DOCKER_BASE_TAG = _v4_plan.LOCAL_DOCKER_BASE_TAG
LOCAL_DOCKER_BASE_ID = _v4_plan.LOCAL_DOCKER_BASE_ID
IMAGE_TAG = "spint-m2:aof-scalar-static-v5-local-v1"

STATIC_CLOSURE_RELATIVES = (
    V4_INCIDENT_RELATIVE,
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5/__init__.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5/plan.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5/binding.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5/docker_argv.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5/profile.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5/driver.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v5/Dockerfile",
    "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v5.py",
) + tuple(_v4_plan.STATIC_CLOSURE_RELATIVES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_static_authority(repo_root: Path = REPO_ROOT) -> None:
    incident = repo_root / V4_INCIDENT_RELATIVE
    if not incident.is_file() or incident.is_symlink() or sha256_file(incident) != V4_INCIDENT_SHA256:
        raise RuntimeError("AOF-S V5 V4 parent-layout incident authority drift")
    _v4_plan.validate_static_authority(repo_root)
