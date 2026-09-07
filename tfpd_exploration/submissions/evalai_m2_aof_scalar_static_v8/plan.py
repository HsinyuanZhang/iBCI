"""Immutable authorities for the additive AOF-S V8 host-validation repair."""
from __future__ import annotations
import hashlib
from pathlib import Path
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v7 import plan as _v7
REPO_ROOT=Path(__file__).resolve().parents[3]
PACKAGE_RELATIVE="tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v8"
RESULT_ROOT_RELATIVE="tfpd_exploration/results/m2_aof_scalar_static_package_v8/local_build"
ARTIFACT_ROOT_RELATIVE="tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v8/artifacts/local_build_v8"
V7_INCIDENT_RELATIVE="tfpd_exploration/docs/RESULT_M2_AOF_SCALAR_STATIC_LOCAL_BUILD_V7_20260903.md"
V7_INCIDENT_SHA256="6cb3135e0d5c5f2e40b25418f160b8b29930ef5687f6babda673f3ba0d8fa258"
V7_FAILURE_ROOT_RELATIVE="tfpd_exploration/results/m2_aof_scalar_static_package_v7/local_build"
V7_FAILURE_CLOSURE_SHA256="25e80bc4bb5ffbe176c8c898b0f6e2c2f5a38e5fb740dfb3cbd66e9362c5642c"
V7_FAILURE_BODIES={"attempt.json":"4d3a0761de6e0982f36248f8fd4e90dca3ed59195032339748c5238c92a79c21","predecessor_authority.json":"e9ab1a29ea4311f2ed381748006c0dd86c601912753840fb1d8beedfd628ccab","input_authority.json":"fcdd9a49ab38131e4cec8c105d961ca2c2fbe5b9a76f4dd565d9cd7a216972ab","build.json":"ed1ac4ce7a0027d5b6086c67db220c907a305c50f78c127eb0e9c4ef830199ac","failure.json":"12a6f12041cd964a32620f56a1483ec9b4fe7288adae43b1193db39b7919e298"}
V2_ARTIFACT_ROOT_RELATIVE,V2_ARTIFACT_BODIES,PAYLOAD_NAME,RECEIPT_NAME=_v7.V2_ARTIFACT_ROOT_RELATIVE,_v7.V2_ARTIFACT_BODIES,_v7.PAYLOAD_NAME,_v7.RECEIPT_NAME
STATIC_CLOSURE_RELATIVES=(V7_INCIDENT_RELATIVE,"tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v8/__init__.py","tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v8/plan.py","tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v8/binding.py","tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v8/profile.py","tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v8/driver.py","tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v8.py")+tuple(_v7.STATIC_CLOSURE_RELATIVES)
def sha256_file(p:Path)->str:
 d=hashlib.sha256()
 with p.open("rb") as h:
  for b in iter(lambda:h.read(1<<20),b""):d.update(b)
 return d.hexdigest()
def validate_static_authority(root:Path=REPO_ROOT)->None:
 p=root/V7_INCIDENT_RELATIVE
 if not p.is_file() or p.is_symlink() or sha256_file(p)!=V7_INCIDENT_SHA256:raise RuntimeError("V8 V7 incident drift")
 _v7.validate_static_authority(root)
