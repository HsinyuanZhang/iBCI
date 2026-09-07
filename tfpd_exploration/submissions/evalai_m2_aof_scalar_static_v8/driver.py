"""V8: shared lifecycle, V7 evidence reuse, and repaired host validation only."""
from __future__ import annotations
import hashlib,json,os
from dataclasses import dataclass
from pathlib import Path
from typing import Any,Callable,Mapping
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import driver as _life
from . import binding,plan
from .profile import V8_PROFILE
class RecoveryError(RuntimeError):pass
_TOKEN=object();_CONSUMED:set[tuple[int,int,str]]=set();_PRE={"attempt.json","attempt.json.sha256","predecessor_authority.json","predecessor_authority.json.sha256"}
@dataclass(frozen=True)
class _Capability:
 token:object;result_root:Path;parent_device:int;parent_inode:int;closure:dict[str,str];closure_sha256:str;v7_failure:dict[str,Any];v2_artifact:dict[str,Any]
def _need(v:bool,m:str)->None:
 if not v:raise RecoveryError(m)
def closure_map(root:Path=plan.REPO_ROOT)->dict[str,str]:
 plan.validate_static_authority(root);out={}
 for r in plan.STATIC_CLOSURE_RELATIVES:
  p=root/r;_need(p.is_file() and not p.is_symlink(),f"V8 closure missing/symlink: {r}");out[r]=plan.sha256_file(p)
 return out
def closure_sha256(m:Mapping[str,str])->str:return hashlib.sha256(json.dumps(dict(m),sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _reviewed(root:Path,m:Mapping[str,str],d:str)->dict[str,str]:
 got=closure_map(root);_need(dict(m)==got and closure_sha256(got)==d,"V8 reviewed closure mismatch");return got
def _issue(*,repo_root:Path,result_root:Path,reviewed:Mapping[str,str],digest:str)->_Capability:
 got=_reviewed(repo_root,reviewed,digest);art=repo_root/plan.ARTIFACT_ROOT_RELATIVE;_need(not os.path.lexists(result_root) and not os.path.lexists(art) and not os.path.lexists(art.parent),"V8 fresh result/artifact parent/root required");par=result_root.parent;_need(par.is_dir() and not par.is_symlink(),"V8 result parent invalid");info=par.stat(follow_symlinks=False);return _Capability(_TOKEN,result_root,info.st_dev,info.st_ino,got,digest,binding.validate_v7_failure_graph(repo_root),binding.validate_v2_sealed_artifact(repo_root))
def _consume(c:_Capability,root:Path)->None:
 _need(isinstance(c,_Capability) and c.token is _TOKEN,"V8 capability required");key=(c.parent_device,c.parent_inode,str(c.result_root));_need(key not in _CONSUMED,"V8 capability consumed");info=c.result_root.parent.stat(follow_symlinks=False);_need((info.st_dev,info.st_ino)==(c.parent_device,c.parent_inode) and not os.path.lexists(c.result_root),"V8 root drift");_need(_reviewed(root,c.closure,c.closure_sha256)==c.closure and binding.validate_v7_failure_graph(root)==c.v7_failure and binding.validate_v2_sealed_artifact(root)==c.v2_artifact,"V8 predecessor/closure drift");_CONSUMED.add(key)
def _final(c:_Capability,root:Path)->None:_need(_reviewed(root,c.closure,c.closure_sha256)==c.closure and binding.validate_v7_failure_graph(root)==c.v7_failure and binding.validate_v2_sealed_artifact(root)==c.v2_artifact,"V8 final drift")
def _create_artifact(*,root:Path,result:Path)->Path:
 _need(result.is_dir() and not result.is_symlink() and set(os.listdir(result))==_PRE,"V8 artifact parent only after attempt/predecessor");package,art=root/plan.PACKAGE_RELATIVE,root/plan.ARTIFACT_ROOT_RELATIVE;par=art.parent;_need(package.is_dir() and not package.is_symlink() and par.parent==package and not os.path.lexists(par) and not os.path.lexists(art),"V8 artifact route/freshness drift");par.mkdir(mode=0o755);art.mkdir(mode=0o755);_need(set(os.listdir(par))=={art.name},"V8 artifact topology drift");return art
def _profile()->_life.LifecycleProfile:
 return _life.LifecycleProfile(attempt_schema="m2_aof_scalar_static_package_v8_attempt",attempt_status="LOCAL_BUILD_V8_RESERVED_NO_NETWORK",terminal_schema="m2_aof_scalar_static_package_v8_terminal",terminal_status="LOCAL_BUILD_V8_VALIDATED_NOT_SUBMITTED",failure_schema="m2_aof_scalar_static_package_v8_failure",artifact_root_relative=plan.ARTIFACT_ROOT_RELATIVE,payload_name=plan.PAYLOAD_NAME,payload_path=lambda r:r/plan.V2_ARTIFACT_ROOT_RELATIVE/plan.PAYLOAD_NAME,attempt_authority=lambda c:{"closure":c.closure,"closure_sha256":c.closure_sha256,"recovery_profile":V8_PROFILE.name,"v7_failure_predecessor":c.v7_failure},predecessor_authority=lambda c:{"v7_failure":c.v7_failure,"v2_sealed_artifact":{"root_relative":c.v2_artifact["root_relative"],"root":c.v2_artifact["root"],"bodies":c.v2_artifact["bodies"]}},input_authority=lambda b:{"reused_v2_payload":b["payload"],"v2_receipt_sha256":b["receipt_sha256"],"reused_v7_container_minival":b["v7_container_minival"],"scientific_rebuild":False},final_revalidate=lambda r,c:_final(c,r))
def _reuse_build(*,root:Path,c:_Capability)->Callable[[Path],Mapping[str,Any]]:
 def build(path:Path)->Mapping[str,Any]:
  _need(path==root/plan.V2_ARTIFACT_ROOT_RELATIVE/plan.PAYLOAD_NAME,"V8 exact V2 payload only");art=_create_artifact(root=root,result=c.result_root);body={"schema":"m2_aof_scalar_static_package_v8_reused_v7_container_authority","v2_payload":c.v2_artifact["bodies"][plan.PAYLOAD_NAME],"v2_receipt":c.v2_artifact["bodies"][plan.RECEIPT_NAME],"v7_build":c.v7_failure["v7_build_sha256"],"container_minival":c.v7_failure["container_minival"],"scientific_rebuild":False};p=art/"reused_v7_container_authority.json";p.write_text(json.dumps(body,sort_keys=True,indent=2)+"\n");os.chmod(p,0o444);return {"payload_sha256":c.v2_artifact["bodies"][plan.PAYLOAD_NAME],"receipt_sha256":c.v2_artifact["bodies"][plan.RECEIPT_NAME],"payload":_life._seal_artifact(p),"v7_container_minival":c.v7_failure["container_minival"],"v7_build_sha256":c.v7_failure["v7_build_sha256"]}
 return build
def execute_production_local(*,reviewed_closure:Mapping[str,str],reviewed_closure_sha256:str,repo_root:Path=plan.REPO_ROOT)->dict[str,Any]:
 c=_issue(repo_root=repo_root,result_root=repo_root/plan.RESULT_ROOT_RELATIVE,reviewed=reviewed_closure,digest=reviewed_closure_sha256);from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.validate_local import run_production_validation;art=repo_root/plan.ARTIFACT_ROOT_RELATIVE;return _life._execute_with_stages(c,repo_root=repo_root,build_payload=_reuse_build(root=repo_root,c=c),validate_payload=lambda p:run_production_validation(p,art),docker_preflight=lambda:{"container_minival":c.v7_failure["container_minival"],"client":"reused_v7_sealed_build_json","network":"none","pull":False},profile=_profile(),consume=_consume)
def _execute_synthetic_for_test(*,repo_root:Path,result_root:Path,build:Callable[[Path],Mapping[str,Any]],validate:Callable[[Path],Mapping[str,Any]],docker:Callable[[],Mapping[str,Any]])->dict[str,Any]:
 m=closure_map(repo_root);c=_issue(repo_root=repo_root,result_root=result_root,reviewed=m,digest=closure_sha256(m));return _life._execute_with_stages(c,repo_root=repo_root,build_payload=build,validate_payload=validate,docker_preflight=docker,profile=_profile(),consume=_consume)
