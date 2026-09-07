"""Immutable, no-network authorities for AOF-S V3 Docker-client recovery."""
from __future__ import annotations

import hashlib
from pathlib import Path

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v2 import plan as _v2_plan

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v3/local_build"
ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3/artifacts/local_build_v3"

V2_FAILURE_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v2/local_build"
V2_FAILURE_CLOSURE_SHA256 = "0f3160c98da08a255427041d2d3aec200af2c95f84eae4ccf9da7427fe03dd65"
V2_FAILURE_BODIES = {
    "attempt.json": "e400fe4240cc37e1f7c31e1355dd63a1978d0762c4a2b09f700b9a434d54d768",
    "predecessor_authority.json": "587dc61e6f4de9516384911dffdbf7d394472c3104d9217a3168adace80245cf",
    "input_authority.json": "f1695edc4155b0270e5663eaf036e86113e16887638a989bc91d50d8e409fe41",
    "failure.json": "9ee7000ec553b7e047f563a6002feeb1a78666519cea1063da0b7b826cb5e22e",
}
V2_ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/artifacts/local_build_v2"
V2_ARTIFACT_BODIES = {
    "t4_m2_seed42_dopt4_act30_aofs_identity.pkl": "65b8001156a7cdb58efd4bbff9547d88443cfa983546558009771f3d4c18ab20",
    "t4_m2_seed42_dopt4_act30_aofs_identity.receipt.json": "b17e5dc09d99e84ee570baedae68c7a4dae84c0bf7923705a02f6c19704ca336",
}
PAYLOAD_NAME = "t4_m2_seed42_dopt4_act30_aofs_identity.pkl"
RECEIPT_NAME = "t4_m2_seed42_dopt4_act30_aofs_identity.receipt.json"

DOCKER_EXECUTABLE = "/usr/bin/docker"
LOCAL_DOCKER_BASE_TAG = "spint-m2:e8-epoch027-76f0fb2"
LOCAL_DOCKER_BASE_ID = "sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8"
IMAGE_TAG = "spint-m2:aof-scalar-static-v3-local-v1"

# V3 imports V2 only for the fixed held-graph codec, then uses the previously
# reviewed V1 lifecycle and local validation primitives.  This remains an
# explicit, no-glob set; an operator reviews the computed map externally.
STATIC_CLOSURE_RELATIVES = (
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3/__init__.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3/plan.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3/binding.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3/docker_argv.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3/profile.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3/driver.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v3/Dockerfile",
    "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v3.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/__init__.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/plan.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/binding.py",
) + tuple(_v2_plan.STATIC_CLOSURE_RELATIVES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_static_authority(repo_root: Path = REPO_ROOT) -> None:
    """V3 has no mutable document authority; V2 and V1 authorities are fixed."""
    _v2_plan.validate_static_authority(repo_root)

