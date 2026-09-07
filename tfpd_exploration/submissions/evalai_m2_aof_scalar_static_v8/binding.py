"""Held-FD V7 receipt graph and sealed container-minival evidence verifier."""
from __future__ import annotations
import hashlib,json,os,stat
from pathlib import Path
from typing import Any
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v4 import binding as _fd
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v7 import binding as _v7
from . import plan
class BindingError(RuntimeError):pass
def _need(v:bool,m:str)->None:
 if not v:raise BindingError(m)
def _hash_regular(path:Path, wanted:str)->None:
 _need(path.is_file() and not path.is_symlink(),f"V7 reused container leaf missing/symlink: {path.name}"); info=path.stat(follow_symlinks=False);_need(stat.S_ISREG(info.st_mode),f"V7 reused container leaf nonregular: {path.name}");_need(hashlib.sha256(path.read_bytes()).hexdigest()==wanted,f"V7 reused container leaf digest drift: {path.name}")
def validate_v7_failure_graph(root:Path=plan.REPO_ROOT)->dict[str,Any]:
 bodies,identity=_fd._graph(root/plan.V7_FAILURE_ROOT_RELATIVE,plan.V7_FAILURE_BODIES,"V7 host-validation failure")
 attempt,pred,input_,build,failure=(json.loads(bodies[n]) for n in ("attempt.json","predecessor_authority.json","input_authority.json","build.json","failure.json"))
 _need(attempt.get("schema")=="m2_aof_scalar_static_package_v7_attempt" and attempt.get("closure_sha256")==plan.V7_FAILURE_CLOSURE_SHA256,"V7 attempt drift")
 _need(set(pred)=={"v6_failure","v2_sealed_artifact"} and pred["v2_sealed_artifact"].get("bodies")==plan.V2_ARTIFACT_BODIES,"V7 predecessor drift")
 _need(input_.get("scientific_rebuild") is False and input_.get("v2_receipt_sha256")==plan.V2_ARTIFACT_BODIES[plan.RECEIPT_NAME],"V7 payload reuse drift")
 _need(failure=={"exception_class":"ModuleNotFoundError","exception_message":"No module named 'src.models'","network_submission":False,"published_prefix":["attempt.json","predecessor_authority.json","input_authority.json","build.json"],"schema":"m2_aof_scalar_static_package_v7_failure","status":"LOCAL_BUILD_FAILED"},"V7 host failure semantics drift")
 container=build.get("docker",{}).get("container_minival",{});_need(container.get("image_id")=="sha256:e1451cb0db6913708fd779b78392b92ed7702f0038ce2bcd72960e4035cfc3c3" and container.get("network_disabled") is True and container.get("pull") is False and container.get("container_removed") is True,"V7 sealed container evidence drift")
 _need(container.get("prediction_sha256")=="38f7c206a492fb5b85266c17985b950ca841c662dee5f1dcf6a0094d643ec8db" and container.get("target_sha256")=="546e5a4a0b2260cdd4c7124d4d668fe0f136f84a81f7aa472e7dbe39a3115530","V7 sealed container outputs drift")
 art=root/"tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/artifacts/local_build_v7/container_minival";_hash_regular(art/"prediction.pkl",container["prediction_sha256"]);_hash_regular(art/"ground_truth.pkl",container["target_sha256"])
 return {"root_relative":plan.V7_FAILURE_ROOT_RELATIVE,"root":identity,"bodies":dict(plan.V7_FAILURE_BODIES),"closure_sha256":plan.V7_FAILURE_CLOSURE_SHA256,"container_minival":container,"v7_build_sha256":plan.V7_FAILURE_BODIES["build.json"],"failure_stage":"post_v7_container_pre_host_validation"}
def validate_v2_sealed_artifact(root:Path=plan.REPO_ROOT)->dict[str,Any]:return _v7.validate_v2_sealed_artifact(root)
