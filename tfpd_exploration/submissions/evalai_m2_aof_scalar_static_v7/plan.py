"""Immutable authorities for AOF-S V7's thin container-path successor."""
from __future__ import annotations
import hashlib
from pathlib import Path
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v6 import plan as _v6

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v7/local_build"
ARTIFACT_ROOT_RELATIVE = "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/artifacts/local_build_v7"
STAGED_CONTEXT_NAME = "docker_context"
V6_INCIDENT_RELATIVE = "tfpd_exploration/docs/RESULT_M2_AOF_SCALAR_STATIC_LOCAL_BUILD_V6_20260903.md"
V6_INCIDENT_SHA256 = "0ce9a791b7d2ee4e0f28a9df1ac6fb9ffd12ec19ba6424f888b2586b036d6e33"
V6_FAILURE_ROOT_RELATIVE = "tfpd_exploration/results/m2_aof_scalar_static_package_v6/local_build"
V6_FAILURE_CLOSURE_SHA256 = "cfe32b783ef9acc3a4429b21afa9080b0a6f65378b53018aa7eff4c39e59f7c9"
V6_FAILURE_BODIES = {"attempt.json": "e8fd7fc1cadf82e3029fffa28ba616d33440c94724ccea771d35bd08a3749dac", "predecessor_authority.json": "aa3fc7297ba0eb6a72a989d025f5b1deae9a7ce654c177cbe2fcac3a13078fd1", "input_authority.json": "347e89e88ab10dcaa187d82e61f1cdbe795f81b733f6f4f377eef9cc46d5010b", "failure.json": "d035048742e2ec8485e0ed43ef4d827fa1581d4a0fd5b2d60a9ee7d831f0ec6e"}
V2_ARTIFACT_ROOT_RELATIVE, V2_ARTIFACT_BODIES, PAYLOAD_NAME, RECEIPT_NAME = _v6.V2_ARTIFACT_ROOT_RELATIVE, _v6.V2_ARTIFACT_BODIES, _v6.PAYLOAD_NAME, _v6.RECEIPT_NAME
DOCKER_EXECUTABLE, LOCAL_DOCKER_BASE_TAG, LOCAL_DOCKER_BASE_ID = _v6.DOCKER_EXECUTABLE, _v6.LOCAL_DOCKER_BASE_TAG, _v6.LOCAL_DOCKER_BASE_ID
IMAGE_TAG = "spint-m2:aof-scalar-static-v7-local-v1"
THIRD_PARTY_BODIES, V1_RUNTIME_BODIES = _v6.THIRD_PARTY_BODIES, _v6.V1_RUNTIME_BODIES
STATIC_CLOSURE_RELATIVES = (V6_INCIDENT_RELATIVE, "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/__init__.py", "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/plan.py", "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/binding.py", "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/staged_context.py", "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/docker_argv.py", "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/profile.py", "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/driver.py", "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/Dockerfile", "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v7.py") + tuple(_v6.STATIC_CLOSURE_RELATIVES)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()
def validate_static_authority(repo_root: Path = REPO_ROOT) -> None:
    incident = repo_root / V6_INCIDENT_RELATIVE
    if not incident.is_file() or incident.is_symlink() or sha256_file(incident) != V6_INCIDENT_SHA256: raise RuntimeError("AOF-S V7 V6 incident drift")
    for mapping in (THIRD_PARTY_BODIES, V1_RUNTIME_BODIES):
        for relative, digest in mapping.items():
            p = repo_root / relative
            if not p.is_file() or p.is_symlink() or sha256_file(p) != digest: raise RuntimeError(f"AOF-S V7 frozen source drift: {relative}")
    _v6.validate_static_authority(repo_root)
