"""V4's small immutable outer lifecycle; all predecessors are held inputs."""
from __future__ import annotations
import hashlib, json, os, stat, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from . import plan

class Error(RuntimeError): pass
def need(ok: bool, message: str) -> None:
    if not ok: raise Error(message)
def sha_bytes(raw: bytes) -> str: return hashlib.sha256(raw).hexdigest()
def digest(mapping: Mapping[str,str]) -> str:
    return sha_bytes(json.dumps(dict(mapping),sort_keys=True,separators=(',',':')).encode())

def _read_fd(fd: int) -> bytes: return b''.join(iter(lambda: os.read(fd,1<<20),b''))
def _dir(root: Path) -> tuple[int,tuple[int,int]]:
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW); info=os.fstat(fd)
    need(stat.S_ISDIR(info.st_mode),'V4 held directory type'); return fd,(int(info.st_dev),int(info.st_ino))
def held_json(root: Path,name: str) -> tuple[dict[str,Any],str]:
    """No-follow body and sidecar read with immutable regular-file evidence."""
    fd,identity=_dir(root)
    try:
        leaf=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            m=os.fstat(leaf);need(stat.S_ISREG(m.st_mode) and stat.S_IMODE(m.st_mode)==0o444 and m.st_nlink==1,'V4 held body mode/link')
            raw=_read_fd(leaf)
        finally: os.close(leaf)
        side=os.open(name+'.sha256',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            m=os.fstat(side);need(stat.S_ISREG(m.st_mode) and stat.S_IMODE(m.st_mode)==0o444 and m.st_nlink==1,'V4 held sidecar mode/link')
            side_raw=_read_fd(side)
        finally: os.close(side)
    finally: os.close(fd)
    d=sha_bytes(raw);need(side_raw==f'{d}  {name}\n'.encode(),'V4 held sidecar digest')
    value=json.loads(raw.decode('utf-8'));need(isinstance(value,dict),'V4 held JSON object')
    now=root.stat(follow_symlinks=False);need((now.st_dev,now.st_ino)==identity and not root.is_symlink(),'V4 held root swap')
    return value,d
def held_file(path: Path) -> tuple[bytes,str]:
    """Read one frozen authority through a held parent descriptor, no follow."""
    parent_fd,identity=_dir(path.parent)
    try:
        leaf=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=parent_fd)
        try:
            m=os.fstat(leaf);need(stat.S_ISREG(m.st_mode) and stat.S_IMODE(m.st_mode)==0o444 and m.st_nlink==1,'V4 held authority mode/link')
            raw=_read_fd(leaf)
        finally: os.close(leaf)
    finally: os.close(parent_fd)
    now=path.parent.stat(follow_symlinks=False);need((now.st_dev,now.st_ino)==identity and not path.parent.is_symlink(),'V4 held authority parent swap')
    return raw,sha_bytes(raw)
def pair(root: Path,name: str,body: Mapping[str,Any]) -> str:
    target,side=root/name,root/(name+'.sha256');need(not os.path.lexists(target) and not os.path.lexists(side),'V4 immutable leaf exists')
    raw=(json.dumps(dict(body),sort_keys=True,indent=2)+'\n').encode(); d=sha_bytes(raw)
    fd,tmp=tempfile.mkstemp(dir=root,prefix='.'+name)
    try:
        os.write(fd,raw);os.fsync(fd);os.close(fd);os.chmod(tmp,0o444);os.replace(tmp,target)
    finally:
        try: os.close(fd)
        except OSError: pass
        if os.path.exists(tmp):os.unlink(tmp)
    with side.open('x',encoding='ascii') as h:h.write(f'{d}  {name}\n')
    os.chmod(side,0o444);return d

def own_closure(repo: Path) -> dict[str,str]:
    leaves=(plan.DESIGN_RELATIVE,plan.WORKORDER_RELATIVE,plan.V3_INCIDENT_RELATIVE,
            'tfpd_exploration/src/m2_anchored_joint_postfusion_v4/__init__.py',
            'tfpd_exploration/src/m2_anchored_joint_postfusion_v4/plan.py',
            'tfpd_exploration/src/m2_anchored_joint_postfusion_v4/lifecycle.py',
            'tfpd_exploration/src/m2_anchored_joint_postfusion_v4/production.py',
            'tfpd_exploration/scripts/run_m2_anchored_joint_postfusion_v4.py')
    answer={}
    for rel in leaves:
        path=repo/rel;need(path.is_file() and not path.is_symlink(),f'V4 closure leaf: {rel}')
        answer[rel]=sha_bytes(path.read_bytes())
    need(answer[plan.DESIGN_RELATIVE]==plan.DESIGN_SHA256 and answer[plan.WORKORDER_RELATIVE]==plan.WORKORDER_SHA256,
         'V4 frozen design/workorder drift')
    return answer
def combined(repo: Path) -> dict[str,str]:
    # V3 carries the current reviewed V2/V1 executor map; V4 adds only its route.
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v3 import lifecycle as v3
    return {**{'v4/'+k:v for k,v in own_closure(repo).items()},**{'v3/'+k:v for k,v in v3.combined(repo).items()}}

def _exact_names(root: Path,names: set[str]) -> tuple[int,int]:
    fd,identity=_dir(root)
    try: got=set(os.listdir(fd))
    finally: os.close(fd)
    expected=names|{x+'.sha256' for x in names};need(got==expected,'V4 exact topology')
    return identity
def admit_predecessors(repo: Path) -> dict[str,Any]:
    """Bind V3's frozen failure, V2's held successful producer, and POOLED law."""
    incident=repo/plan.V3_INCIDENT_RELATIVE
    _incident,incident_sha=held_file(incident)
    need(incident_sha==plan.V3_INCIDENT_SHA256,'V4 V3 incident drift')
    v3root=repo/plan.V3_ROOT_RELATIVE; vid=_exact_names(v3root,{'attempt.json','v2_authority.json','failure.json'})
    va,ad=held_json(v3root,'attempt.json');vv,vd=held_json(v3root,'v2_authority.json');vf,fd=held_json(v3root,'failure.json')
    need({'attempt':ad,'v2_authority':vd,'failure':fd}==plan.V3_HELD_SHA256,'V4 V3 literal authority drift')
    need(va.get('schema')=='m2_anchored_joint_postfusion_v3_attempt' and vf.get('terminal_xor_failure') is False and tuple(vf.get('published_prefix',()))==('attempt.json','v2_authority.json'),'V4 V3 failure semantics')
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v3 import lifecycle as v3
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import binding
    v2=v3.admit_v2(repo)
    pooled=binding.validate_pooled_comparator_score(repo/'tfpd_exploration/results/m2_precision_cdm_v2_screen_v1')
    need(pooled.get('body_sha256')==plan.POOLED_SCORE_SHA256,'V4 POOLED literal drift')
    historical_device=pooled.get('score',{}).get('device')
    need(historical_device=={'batch_size':1024,'cublas_workspace_config':':4096:8','cuda_visible_devices':'1',
                             'logical_device':'cuda:0','name':'NVIDIA GeForce RTX 3090',
                             'tf32_cudnn':False,'tf32_matmul':False},
         'V4 historical POOLED GPU device law drift')
    now=v3root.stat(follow_symlinks=False);need((now.st_dev,now.st_ino)==vid and not v3root.is_symlink(),'V4 V3 root swap')
    return {'v3_root_identity':list(vid),'v3_attempt_sha256':ad,'v3_v2_authority_sha256':vd,'v3_failure_sha256':fd,
            'v3_incident_sha256':plan.V3_INCIDENT_SHA256,'v2_training':{k:v for k,v in v2.items() if not k.startswith('_')},
            'pooled_root_identity':pooled['root_identity'],'pooled_score_sha256':pooled['body_sha256'],
            'historical_pooled_device':historical_device,
            '_checkpoint_bytes':v2['_checkpoint_bytes'],'_pooled':pooled}

_TOKEN=object();_USED:set[tuple[int,int,str]]=set()
@dataclass(frozen=True)
class Capability:
    token:object;repo:Path;root:Path;closure:dict[str,str];parent_identity:tuple[int,int]
def issue_after_independent_review(*,reviewed_map:Mapping[str,str],reviewed_digest:str,repo_root:Path,outer_root:Path)->Capability:
    canonical=repo_root/plan.ROOT_RELATIVE; current=combined(repo_root)
    need(outer_root==canonical and dict(reviewed_map)==current and reviewed_digest==digest(current),'V4 reviewed closure/canonical root drift')
    need(not os.path.lexists(outer_root) and outer_root.parent.is_dir() and not outer_root.parent.is_symlink(),'V4 fresh root')
    p=outer_root.parent.stat(follow_symlinks=False);return Capability(_TOKEN,repo_root,outer_root,current,(int(p.st_dev),int(p.st_ino)))
def _issue_for_test(*,repo_root:Path,outer_root:Path)->Capability:
    current=combined(repo_root);need(not os.path.lexists(outer_root) and outer_root.parent.is_dir(),'V4 test fresh root');p=outer_root.parent.stat(follow_symlinks=False);return Capability(_TOKEN,repo_root,outer_root,current,(int(p.st_dev),int(p.st_ino)))
def consume(cap:Capability)->tuple[Path,tuple[int,int]]:
    need(isinstance(cap,Capability) and cap.token is _TOKEN,'V4 opaque capability required');key=(cap.parent_identity[0],cap.parent_identity[1],str(cap.root));p=cap.root.parent.stat(follow_symlinks=False)
    need(key not in _USED and (p.st_dev,p.st_ino)==cap.parent_identity and not os.path.lexists(cap.root) and combined(cap.repo)==cap.closure,'V4 consumed capability drift')
    _USED.add(key);cap.root.mkdir();r=cap.root.stat(follow_symlinks=False);need(stat.S_ISDIR(r.st_mode) and not cap.root.is_symlink(),'V4 created root type');return cap.root,(int(r.st_dev),int(r.st_ino))
def revalidate(cap:Capability,root_identity:tuple[int,int],admission:Mapping[str,Any])->None:
    p=cap.root.parent.stat(follow_symlinks=False);r=cap.root.stat(follow_symlinks=False)
    need((p.st_dev,p.st_ino)==cap.parent_identity and (r.st_dev,r.st_ino)==root_identity and not cap.root.is_symlink() and combined(cap.repo)==cap.closure,'V4 root/closure drift')
    now=admit_predecessors(cap.repo);need({k:v for k,v in now.items() if not k.startswith('_')}=={k:v for k,v in admission.items() if not k.startswith('_')},'V4 predecessor drift')

OUTER_PREFIX=('attempt.json','predecessor_authority.json','cross_device_bridge.json','input_authority.json','score.json','gates.json')
def validate_outer(root:Path,*,terminal:bool,published:tuple[str,...]|None=None)->None:
    prefix=OUTER_PREFIX if terminal else tuple(published or ())
    need(prefix==OUTER_PREFIX[:len(prefix)],'V4 outer prefix')
    names=set(prefix)|{('terminal.json' if terminal else 'failure.json')};_exact_names(root,names)
    observed={name:held_json(root,name)[1] for name in names}
    final_name='terminal.json' if terminal else 'failure.json'
    final,_=held_json(root,final_name)
    need(final.get('terminal_xor_failure') is terminal,'V4 terminal/failure XOR')
    if terminal:
        link_fields={'attempt.json':'attempt_sha256','predecessor_authority.json':'predecessor_authority_sha256',
                     'cross_device_bridge.json':'cross_device_bridge_sha256','input_authority.json':'input_authority_sha256',
                     'score.json':'score_sha256','gates.json':'gates_sha256'}
        need(all(final.get(field)==observed[name] for name,field in link_fields.items()),
             'V4 terminal body-link drift')
    else:
        need(tuple(final.get('published_prefix',()))==prefix,'V4 failure published-prefix drift')
