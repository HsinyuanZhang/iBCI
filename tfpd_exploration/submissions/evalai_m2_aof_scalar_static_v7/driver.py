"""AOF-S V7 shared lifecycle successor; only image PYTHONPATH differs."""
from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import driver as _life
from . import binding, docker_argv, plan, staged_context
from .profile import V7_PROFILE
class RecoveryError(RuntimeError): pass
_TOKEN = object(); _CONSUMED: set[tuple[int,int,str]] = set(); _PRE_BUILD_PREFIX = {"attempt.json","attempt.json.sha256","predecessor_authority.json","predecessor_authority.json.sha256"}
@dataclass(frozen=True)
class _Capability:
    token: object; result_root: Path; parent_device: int; parent_inode: int; closure: dict[str,str]; closure_sha256: str; v6_failure: dict[str,Any]; v2_artifact: dict[str,Any]
def _need(v: bool, m: str) -> None:
    if not v: raise RecoveryError(m)
def closure_map(repo_root: Path = plan.REPO_ROOT) -> dict[str,str]:
    plan.validate_static_authority(repo_root); out = {}
    for r in plan.STATIC_CLOSURE_RELATIVES:
        p = repo_root / r; _need(p.is_file() and not p.is_symlink(), f"V7 closure missing/symlink: {r}"); out[r] = plan.sha256_file(p)
    return out
def closure_sha256(mapping: Mapping[str,str]) -> str: return hashlib.sha256(json.dumps(dict(mapping),sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _reviewed(root: Path, m: Mapping[str,str], d: str) -> dict[str,str]:
    got = closure_map(root); _need(dict(m)==got and closure_sha256(got)==d, "V7 reviewed closure mismatch"); return got
def _issue(*, repo_root: Path, result_root: Path, reviewed: Mapping[str,str], digest: str) -> _Capability:
    got = _reviewed(repo_root,reviewed,digest); art=repo_root/plan.ARTIFACT_ROOT_RELATIVE; _need(not os.path.lexists(result_root) and not os.path.lexists(art) and not os.path.lexists(art.parent), "V7 fresh result/artifact parent/root required"); parent=result_root.parent; _need(parent.is_dir() and not parent.is_symlink(), "V7 result parent invalid"); info=parent.stat(follow_symlinks=False); return _Capability(_TOKEN,result_root,info.st_dev,info.st_ino,got,digest,binding.validate_v6_failure_graph(repo_root),binding.validate_v2_sealed_artifact(repo_root))
def _consume(cap: _Capability, root: Path) -> None:
    _need(isinstance(cap,_Capability) and cap.token is _TOKEN, "V7 capability required"); key=(cap.parent_device,cap.parent_inode,str(cap.result_root)); _need(key not in _CONSUMED, "V7 capability consumed"); info=cap.result_root.parent.stat(follow_symlinks=False); _need((info.st_dev,info.st_ino)==(cap.parent_device,cap.parent_inode) and not os.path.lexists(cap.result_root), "V7 root drift"); _need(_reviewed(root,cap.closure,cap.closure_sha256)==cap.closure, "V7 closure drift"); _need(binding.validate_v6_failure_graph(root)==cap.v6_failure and binding.validate_v2_sealed_artifact(root)==cap.v2_artifact,"V7 predecessor drift"); _CONSUMED.add(key)
def _final(cap: _Capability, root: Path) -> None: _need(_reviewed(root,cap.closure,cap.closure_sha256)==cap.closure and binding.validate_v6_failure_graph(root)==cap.v6_failure and binding.validate_v2_sealed_artifact(root)==cap.v2_artifact,"V7 final predecessor/closure drift")
def _create_route_artifact_root(*, repo_root: Path, result_root: Path) -> Path:
    _need(result_root.is_dir() and not result_root.is_symlink() and set(os.listdir(result_root))==_PRE_BUILD_PREFIX,"V7 artifact parent may be created only after attempt/predecessor prefix"); package,art=repo_root/plan.PACKAGE_RELATIVE,repo_root/plan.ARTIFACT_ROOT_RELATIVE; parent=art.parent; _need(package.is_dir() and not package.is_symlink() and parent.parent==package and not os.path.lexists(parent) and not os.path.lexists(art),"V7 artifact route/freshness drift"); parent.mkdir(mode=0o755); art.mkdir(mode=0o755); _need(set(os.listdir(parent))=={art.name},"V7 artifact topology drift"); return art
def _profile() -> _life.LifecycleProfile:
    return _life.LifecycleProfile(attempt_schema="m2_aof_scalar_static_package_v7_attempt",attempt_status="LOCAL_BUILD_V7_RESERVED_NO_NETWORK",terminal_schema="m2_aof_scalar_static_package_v7_terminal",terminal_status="LOCAL_BUILD_V7_VALIDATED_NOT_SUBMITTED",failure_schema="m2_aof_scalar_static_package_v7_failure",artifact_root_relative=plan.ARTIFACT_ROOT_RELATIVE,payload_name=plan.PAYLOAD_NAME,payload_path=lambda r:r/plan.V2_ARTIFACT_ROOT_RELATIVE/plan.PAYLOAD_NAME,attempt_authority=lambda c:{"closure":c.closure,"closure_sha256":c.closure_sha256,"recovery_profile":V7_PROFILE.name,"v6_failure_predecessor":c.v6_failure},predecessor_authority=lambda c:{"v6_failure":c.v6_failure,"v2_sealed_artifact":{"root_relative":c.v2_artifact["root_relative"],"root":c.v2_artifact["root"],"bodies":c.v2_artifact["bodies"]}},input_authority=lambda b:{"reused_v2_payload":b["payload"],"v2_receipt_sha256":b["receipt_sha256"],"session_records":b["session_records"],"scientific_rebuild":False},final_revalidate=lambda r,c:_final(c,r))
def _reuse_build(*, repo_root: Path, capability: _Capability) -> Callable[[Path],Mapping[str,Any]]:
    def build(path: Path) -> Mapping[str,Any]:
        _need(path==repo_root/plan.V2_ARTIFACT_ROOT_RELATIVE/plan.PAYLOAD_NAME,"V7 exact V2 payload only"); art=_create_route_artifact_root(repo_root=repo_root,result_root=capability.result_root); context=staged_context.stage_docker_context(repo_root=repo_root,artifact_root=art); receipt=capability.v2_artifact["receipt"]; body={"schema":"m2_aof_scalar_static_package_v7_reused_v2_payload_authority","v2_artifact_root":capability.v2_artifact["root_relative"],"payload":capability.v2_artifact["bodies"][plan.PAYLOAD_NAME],"receipt":capability.v2_artifact["bodies"][plan.RECEIPT_NAME],"scientific_rebuild":False}; authority=art/"reused_v2_payload_authority.json"; authority.write_text(json.dumps(body,sort_keys=True,indent=2)+"\n"); os.chmod(authority,0o444); return {"payload_sha256":capability.v2_artifact["bodies"][plan.PAYLOAD_NAME],"receipt_sha256":capability.v2_artifact["bodies"][plan.RECEIPT_NAME],"payload":_life._seal_artifact(authority),"session_records":receipt["session_records"],"docker_context":context}
    return build
def execute_production_local(*,reviewed_closure:Mapping[str,str],reviewed_closure_sha256:str,repo_root:Path=plan.REPO_ROOT)->dict[str,Any]:
    cap=_issue(repo_root=repo_root,result_root=repo_root/plan.RESULT_ROOT_RELATIVE,reviewed=reviewed_closure,digest=reviewed_closure_sha256); from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.validate_local import run_production_validation; art=repo_root/plan.ARTIFACT_ROOT_RELATIVE; return _life._execute_with_stages(cap,repo_root=repo_root,build_payload=_reuse_build(repo_root=repo_root,capability=cap),validate_payload=lambda p:run_production_validation(p,art),docker_preflight=lambda:{"container_minival":docker_argv.build_and_validate_container(repo_root=repo_root,artifact_root=art),"client":"route_owned_subprocess_argv","network":"none","pull":False},profile=_profile(),consume=_consume)
def _execute_synthetic_for_test(*,repo_root:Path,result_root:Path,build:Callable[[Path],Mapping[str,Any]],validate:Callable[[Path],Mapping[str,Any]],docker:Callable[[],Mapping[str,Any]])->dict[str,Any]:
    m=closure_map(repo_root); cap=_issue(repo_root=repo_root,result_root=result_root,reviewed=m,digest=closure_sha256(m)); return _life._execute_with_stages(cap,repo_root=repo_root,build_payload=build,validate_payload=validate,docker_preflight=docker,profile=_profile(),consume=_consume)
