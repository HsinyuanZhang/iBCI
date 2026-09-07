"""No-follow V1 incident binding and immutable narrow V2 receipt lifecycle."""
from __future__ import annotations
import hashlib, json, os, stat, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from . import plan
class LifecycleError(RuntimeError): pass
def _need(ok: bool, message: str) -> None:
    if not ok: raise LifecycleError(message)
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def closure(repo: Path) -> dict[str,str]:
    values={relative:sha(repo/relative) for relative in plan.STATIC_CLOSURE_RELATIVES}
    for rel,value in ((plan.DESIGN_RELATIVE,plan.DESIGN_SHA256),(plan.WORKORDER_RELATIVE,plan.WORKORDER_SHA256),(plan.INCIDENT_RELATIVE,plan.INCIDENT_SHA256)):
        _need(values[rel]==value,f"AJPF V2 static authority drift: {rel}")
    return values
def combined_closure(repo: Path) -> dict[str,str]:
    """V2 wrapper closure plus the exact current V1 executor closure."""
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle as v1
    result={"v2/"+key:value for key,value in closure(repo).items()}
    result.update({"v1/"+key:value for key,value in v1.closure_map(repo).items()})
    return result
def _held(root: Path,name: str,expected: str) -> dict[str,Any]:
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        leaf=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            info=os.fstat(leaf); _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o444 and info.st_nlink==1,"V2 held V1 leaf mode/link drift")
            raw=b"".join(iter(lambda:os.read(leaf,1<<20),b""))
        finally: os.close(leaf)
        side=os.open(name+".sha256",os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            info=os.fstat(side); _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o444 and info.st_nlink==1,"V2 held V1 sidecar mode/link drift")
            side_raw=b"".join(iter(lambda:os.read(side,1<<20),b""))
        finally: os.close(side)
    finally: os.close(fd)
    got=hashlib.sha256(raw).hexdigest(); _need(got==expected and side_raw==f"{got}  {name}\n".encode(),"V2 V1 incident digest drift")
    return json.loads(raw.decode())
def hold_v1_failure(repo: Path) -> dict[str,Any]:
    root=repo/plan.V1_FAILURE_ROOT_RELATIVE
    attempt=_held(root,"attempt.json",plan.V1_ATTEMPT_SHA256); launch=_held(root,"launch.json",plan.V1_LAUNCH_SHA256); failure=_held(root,"failure.json",plan.V1_FAILURE_SHA256)
    message=str(failure.get("exception_message"))
    _need(failure.get("exception_class")=="AttributeError" and "physical" in message and "_state_digest" in message and failure.get("published_prefix")==["attempt.json","launch.json"],"V2 V1 pre-step incident semantics drift")
    return {"v1_attempt_sha256":plan.V1_ATTEMPT_SHA256,"v1_launch_sha256":plan.V1_LAUNCH_SHA256,"v1_failure_sha256":plan.V1_FAILURE_SHA256,"pre_source_pre_step":True,"v1_closure_sha256":attempt.get("closure_sha256"),"v1_launch_uuid":launch.get("canonical_uuid")}
def pair(root: Path,name: str,body: Mapping[str,Any]) -> str:
    raw=(json.dumps(dict(body),sort_keys=True,indent=2)+"\n").encode(); digest=hashlib.sha256(raw).hexdigest(); target=root/name; side=root/(name+".sha256")
    _need(not target.exists() and not side.exists(),"V2 immutable leaf exists")
    fd,tmp=tempfile.mkstemp(dir=root,prefix=".v2.")
    with os.fdopen(fd,"wb") as out: out.write(raw);out.flush();os.fsync(out.fileno())
    os.chmod(tmp,0o444);os.replace(tmp,target); side.write_text(f"{digest}  {name}\n",encoding="ascii");os.chmod(side,0o444); return digest

_TOKEN=object();_USED:set[tuple[int,int,str]]=set()
@dataclass(frozen=True)
class Capability:
    token:object; repo_root:Path; root:Path; parent_dev:int; parent_ino:int; closure:dict[str,str]; digest:str
def closure_digest(value: Mapping[str,str]) -> str:
    return hashlib.sha256(json.dumps(dict(value),sort_keys=True,separators=(",",":")).encode()).hexdigest()
def issue_after_independent_review(*, reviewed_map: Mapping[str,str], digest: str, repo_root: Path, outer_root: Path) -> Capability:
    current=combined_closure(repo_root)
    _need(dict(reviewed_map)==current and digest==closure_digest(current),"AJPF V2 independent reviewed closure drift")
    _need(not os.path.lexists(outer_root) and outer_root.parent.is_dir() and not outer_root.parent.is_symlink(),"AJPF V2 fresh outer root required")
    info=outer_root.parent.stat(follow_symlinks=False)
    return Capability(_TOKEN,repo_root,outer_root,int(info.st_dev),int(info.st_ino),current,digest)
def consume(cap: Capability) -> tuple[Path,tuple[int,int]]:
    _need(isinstance(cap,Capability) and cap.token is _TOKEN,"AJPF V2 opaque capability required")
    key=(cap.parent_dev,cap.parent_ino,str(cap.root));_need(key not in _USED,"AJPF V2 capability consumed")
    parent=cap.root.parent.stat(follow_symlinks=False);_need((parent.st_dev,parent.st_ino)==(cap.parent_dev,cap.parent_ino) and not os.path.lexists(cap.root),"AJPF V2 parent/freshness drift")
    _need(combined_closure(cap.repo_root)==cap.closure,"AJPF V2 closure drift before attempt");_USED.add(key)
    cap.root.mkdir(mode=0o755); info=cap.root.stat(follow_symlinks=False);return cap.root,(int(info.st_dev),int(info.st_ino))
def validate_outer(root: Path, *, terminal: bool, published: tuple[str,...] | None = None) -> None:
    names={"attempt.json","v1_incident.json","strict_digest_regression.json"} if terminal else set(published or ())
    names.add("terminal.json" if terminal else "failure.json")
    expected=names|{name+".sha256" for name in names}|{"training","score"}
    _need({item.name for item in root.iterdir()}==expected,"AJPF V2 outer topology drift")
    for name in names:
        body=root/name;side=root/(name+".sha256")
        for path in (body,side):
            info=path.stat(follow_symlinks=False);_need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o444 and info.st_nlink==1,"AJPF V2 outer leaf mode/link drift")
        _need(side.read_bytes()==f"{sha(body)}  {name}\n".encode(),"AJPF V2 outer sidecar drift")
    for child in ("training","score"):
        info=(root/child).stat(follow_symlinks=False);_need(stat.S_ISDIR(info.st_mode) and not (root/child).is_symlink(),"AJPF V2 nested directory drift")
