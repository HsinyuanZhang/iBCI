"""Immutable no-network authorities for AOF-S V6 dependency-layout recovery."""
from __future__ import annotations

import hashlib
from pathlib import Path

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v5 import plan as _v5_plan

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v6/local_build"
ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/artifacts/local_build_v6"
STAGED_CONTEXT_NAME = "docker_context"
V5_INCIDENT_RELATIVE = "tfpd_exploration/docs/RESULT_M2_AOF_SCALAR_STATIC_LOCAL_BUILD_V5_20260903.md"
V5_INCIDENT_SHA256 = "3b9d41e6bc508ce9c1ff6174aa973f6ec36c6fc698e7fa4d3223bce6ffccdb55"
V5_FAILURE_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v5/local_build"
V5_FAILURE_CLOSURE_SHA256 = "daac9e48b71f5c570ebb059ef9b1e3a15058ad5bee5759e4b6d4f4e5d2bc6a73"
V5_FAILURE_BODIES = {
    "attempt.json": "c684705696cd93790133821938a7e7b67939542f583cb377a2c4ab971e888bf5",
    "predecessor_authority.json": "3a448fb162ba175c99f5f53d15de42cdaa49925a543091f2c0d5d96789269d0b",
    "input_authority.json": "1a10ec8411edcdcd99d339f5ffd9c642beef28e39d1b7b6ffd7bb69834a76aa2",
    "failure.json": "33607a6ea9b44625127a32e2103a3692f76702d960d0ec432e4fb3bafd6a28d2",
}
V2_ARTIFACT_ROOT_RELATIVE = _v5_plan.V2_ARTIFACT_ROOT_RELATIVE
V2_ARTIFACT_BODIES = _v5_plan.V2_ARTIFACT_BODIES
PAYLOAD_NAME, RECEIPT_NAME = _v5_plan.PAYLOAD_NAME, _v5_plan.RECEIPT_NAME
DOCKER_EXECUTABLE, LOCAL_DOCKER_BASE_TAG, LOCAL_DOCKER_BASE_ID = (_v5_plan.DOCKER_EXECUTABLE, _v5_plan.LOCAL_DOCKER_BASE_TAG, _v5_plan.LOCAL_DOCKER_BASE_ID)
IMAGE_TAG = "spint-m2:aof-scalar-static-v6-local-v1"
THIRD_PARTY_BODIES = {
    "SPINT-main/third_party/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "SPINT-main/third_party/falcon_challenge/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "SPINT-main/third_party/falcon_challenge/filtering.py": "5b31f5188355458456dc460a7cef628558404c0eaa5fcc24fdaaf36cfadab2b2",
}
V1_RUNTIME_BODIES = {
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py": "5aa6c6af4efb035cc02011a3604ffd5e8f53e66ad040c071c1e9ef0bf11b447e",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/laws.py": "dc39d2ff64f36cb1c4aea148477c369eb1e6559e56bbe5f4881531c266f950be",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/decode.py": "562924bea28d2d356656b81a0ae3b301b0ba21566d5f99466c706ea9f18dd459",
}

STATIC_CLOSURE_RELATIVES = (
    V5_INCIDENT_RELATIVE,
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/__init__.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/plan.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/binding.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/staged_context.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/docker_argv.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/profile.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/driver.py",
    "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v6/Dockerfile",
    "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v6.py",
) + tuple(_v5_plan.STATIC_CLOSURE_RELATIVES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_static_authority(repo_root: Path = REPO_ROOT) -> None:
    incident = repo_root / V5_INCIDENT_RELATIVE
    if not incident.is_file() or incident.is_symlink() or sha256_file(incident) != V5_INCIDENT_SHA256:
        raise RuntimeError("AOF-S V6 V5 dependency-layout incident authority drift")
    for relative, digest in THIRD_PARTY_BODIES.items():
        leaf = repo_root / relative
        if not leaf.is_file() or leaf.is_symlink() or sha256_file(leaf) != digest:
            raise RuntimeError(f"AOF-S V6 frozen third_party authority drift: {relative}")
    for relative, digest in V1_RUNTIME_BODIES.items():
        leaf = repo_root / relative
        if not leaf.is_file() or leaf.is_symlink() or sha256_file(leaf) != digest:
            raise RuntimeError(f"AOF-S V6 frozen runtime authority drift: {relative}")
    _v5_plan.validate_static_authority(repo_root)
