"""Immutable, no-network authorities for the AOF-S V4 layout-only recovery."""
from __future__ import annotations

import hashlib
from pathlib import Path

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v2 import plan as _v2_plan

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v4/local_build"
ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/artifacts/local_build_v4"

V3_INCIDENT_RELATIVE = "tfpd_exploration/docs/RESULT_M2_AOF_SCALAR_STATIC_LOCAL_BUILD_V3_20260903.md"
V3_INCIDENT_SHA256 = "38fbd52ea9bc1f713d270b0c4e6b71ac20841f93f887f732be9c72365a5db832"
V3_FAILURE_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v3/local_build"
V3_FAILURE_CLOSURE_SHA256 = "377824b212ff31354a9e25034d1fe128834aa9aab82ab3d36b819441948e1239"
V3_FAILURE_BODIES = {
    "attempt.json": "554afff1d91855fdaec905350170635d9a838c1b2b92957dd3997eead5ae4a13",
    "predecessor_authority.json": "9bdbea7dc088695fe60465409d657c95cac7a6b465abbbdf97bce76da0d105d7",
    "input_authority.json": "b6f66cb7cb404240e019201fed41a1721128807861a47099ed5726e0611bf8ed",
    "failure.json": "40573a526803614fcda0652d202ba0b4a4362478fe7df01c666486473d20e8c2",
}

V2_ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v2/artifacts/local_build_v2"
V2_ARTIFACT_BODIES = {
    "t4_m2_seed42_dopt4_act30_aofs_identity.pkl": "65b8001156a7cdb58efd4bbff9547d88443cfa983546558009771f3d4c18ab20",
    "t4_m2_seed42_dopt4_act30_aofs_identity.receipt.json": "b17e5dc09d99e84ee570baedae68c7a4dae84c0bf7923705a02f6c19704ca336",
}
PAYLOAD_NAME = "t4_m2_seed42_dopt4_act30_aofs_identity.pkl"
RECEIPT_NAME = "t4_m2_seed42_dopt4_act30_aofs_identity.receipt.json"
V1_DECODER_SHA256 = "5aa6c6af4efb035cc02011a3604ffd5e8f53e66ad040c071c1e9ef0bf11b447e"
V1_LAWS_SHA256 = "dc39d2ff64f36cb1c4aea148477c369eb1e6559e56bbe5f4881531c266f950be"
V1_DECODE_SHA256 = "562924bea28d2d356656b81a0ae3b301b0ba21566d5f99466c706ea9f18dd459"

DOCKER_EXECUTABLE = "/usr/bin/docker"
LOCAL_DOCKER_BASE_TAG = "spint-m2:e8-epoch027-76f0fb2"
LOCAL_DOCKER_BASE_ID = "sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8"
IMAGE_TAG = "spint-m2:aof-scalar-static-v4-local-v1"

# V4 adds only route, proof, and Docker-layout leaves.  The predecessor's
# V2/V1 closure remains an explicit fixed dependency; no science is copied.
STATIC_CLOSURE_RELATIVES = (
    V3_INCIDENT_RELATIVE,
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/__init__.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/plan.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/binding.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/docker_argv.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/profile.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/driver.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/Dockerfile",
    "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v4.py",
) + tuple(_v2_plan.STATIC_CLOSURE_RELATIVES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_static_authority(repo_root: Path = REPO_ROOT) -> None:
    incident = repo_root / V3_INCIDENT_RELATIVE
    if not incident.is_file() or incident.is_symlink() or sha256_file(incident) != V3_INCIDENT_SHA256:
        raise RuntimeError("AOF-S V4 V3 layout incident authority drift")
    _v2_plan.validate_static_authority(repo_root)
