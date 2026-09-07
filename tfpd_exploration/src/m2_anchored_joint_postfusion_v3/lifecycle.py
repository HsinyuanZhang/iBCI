"""V3 read-only V2 admission and immutable outer receipt helpers."""
from __future__ import annotations
import hashlib,json,os,stat,tempfile
from pathlib import Path
from typing import Any,Mapping
from dataclasses import dataclass
from . import plan
class Error(RuntimeError): pass
def need(ok:bool,msg:str)->None:
    if not ok: raise Error(msg)
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def held_json(root:Path,name:str)->tuple[dict[str,Any],str]:
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        leaf=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            m=os.fstat(leaf);need(stat.S_ISREG(m.st_mode) and stat.S_IMODE(m.st_mode)==0o444 and m.st_nlink==1,'V3 held leaf mode/link');raw=b''.join(iter(lambda:os.read(leaf,1<<20),b''))
        finally:os.close(leaf)
        side=os.open(name+'.sha256',os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            m=os.fstat(side);need(stat.S_ISREG(m.st_mode) and stat.S_IMODE(m.st_mode)==0o444 and m.st_nlink==1,'V3 held sidecar mode/link');s=b''.join(iter(lambda:os.read(side,1<<20),b''))
        finally:os.close(side)
    finally:os.close(fd)
    d=hashlib.sha256(raw).hexdigest();need(s==f'{d}  {name}\n'.encode(),'V3 held sidecar');return json.loads(raw),d
def pair(root:Path,name:str,body:Mapping[str,Any])->str:
    raw=(json.dumps(dict(body),sort_keys=True,indent=2)+'\n').encode();d=hashlib.sha256(raw).hexdigest();p=root/name;q=root/(name+'.sha256');need(not p.exists() and not q.exists(),'V3 immutable leaf')
    fd,tmp=tempfile.mkstemp(dir=root);os.write(fd,raw);os.close(fd);os.chmod(tmp,0o444);os.replace(tmp,p);q.write_text(f'{d}  {name}\n');os.chmod(q,0o444);return d
def admit_v2(repo:Path)->dict[str,Any]:
    root=repo/plan.V2_ROOT_RELATIVE
    before=root.stat(follow_symlinks=False);need(stat.S_ISDIR(before.st_mode) and not root.is_symlink(),'V3 V2 root type')
    # Reuse the reviewed exact V1 nested-success validator, including held
    # checkpoint descriptors/sidecars and all epoch/manifest links.
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle as v1
    training_witness=v1.validate_held_training_success_graph(root=root/'training',repo_root=repo)
    outer_names={'attempt.json','v1_incident.json','strict_digest_regression.json','failure.json','training','score'}
    need({item.name for item in root.iterdir()}==outer_names|{name+'.sha256' for name in outer_names if name not in {'training','score'}},'V3 exact V2 outer topology')
    outer_attempt,oa=held_json(root,'attempt.json');outer_incident,oi=held_json(root,'v1_incident.json');outer_regression,orr=held_json(root,'strict_digest_regression.json');outer,od=held_json(root,'failure.json');train,td=held_json(root/'training','terminal.json');manifest,md=held_json(root/'training','manifest.json');score,sd=held_json(root/'score','failure.json')
    fixed=plan.V2_HELD_SHA256
    observed={'outer_attempt':oa,'outer_incident':oi,'outer_regression':orr,'outer_failure':od,'training_attempt':sha(root/'training'/'attempt.json'),'training_launch':sha(root/'training'/'launch.json'),'training_manifest':md,'training_terminal':td,'score_attempt':sha(root/'score'/'attempt.json'),'score_launch':sha(root/'score'/'launch.json'),'score_producer':sha(root/'score'/'producer_authority.json'),'score_failure':sd}
    observed.update({arm:training_witness['descriptor_witness']['checkpoint_sha256'][arm] for arm in ('J-MEAN','J-NATIVE','J-R1')})
    need(observed==fixed,'V3 frozen V2 authority literal drift')
    prefix=tuple(score.get('published_prefix',()))
    v1.validate_score_failure_topology(root/'score',published=prefix)
    need(sha(repo/plan.V2_INCIDENT_RELATIVE)==plan.V2_INCIDENT_SHA256,'V3 V2 incident doc drift')
    need(outer_attempt.get('schema')=='m2_anchored_joint_postfusion_v2_attempt' and outer_incident.get('pre_source_pre_step') is True and outer_regression.get('strict_load') is True and outer.get('terminal_xor_failure') is False and train.get('terminal_xor_failure') is True and 'historical sentinel' in str(score.get('exception_message','')),'V3 V2 incident semantics')
    need(tuple(outer.get('published_prefix',()))==('attempt.json','v1_incident.json','strict_digest_regression.json') and prefix==('attempt.json','launch.json','producer_authority.json'),'V3 V2 failure prefix drift')
    need(td==training_witness['terminal_sha256'] and md==training_witness['manifest_sha256'],'V3 V2 training held witness drift')
    after=root.stat(follow_symlinks=False);need((before.st_dev,before.st_ino)==(after.st_dev,after.st_ino),'V3 V2 root swap')
    return {'v2_root_identity':[before.st_dev,before.st_ino],'v2_outer_failure_sha256':od,'v2_training_terminal_sha256':td,'v2_training_manifest_sha256':md,'v2_score_failure_sha256':sd,'v2_checkpoint_sha256':training_witness['descriptor_witness']['checkpoint_sha256'],'_checkpoint_bytes':training_witness['checkpoint_bytes']}
_TOKEN=object();_USED=set()
@dataclass(frozen=True)
class Capability: token:object;repo:Path;root:Path;closure:dict[str,str];parent_dev:int;parent_ino:int
def combined(repo:Path)->dict[str,str]:
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v2 import lifecycle as v2
    own={x:sha(repo/x) for x in (plan.DESIGN_RELATIVE,plan.WORKORDER_RELATIVE,plan.V2_INCIDENT_RELATIVE,'tfpd_exploration/src/m2_anchored_joint_postfusion_v3/__init__.py','tfpd_exploration/src/m2_anchored_joint_postfusion_v3/plan.py','tfpd_exploration/src/m2_anchored_joint_postfusion_v3/lifecycle.py','tfpd_exploration/src/m2_anchored_joint_postfusion_v3/production.py')}
    return {**{'v3/'+k:v for k,v in own.items()},**{'v2/'+k:v for k,v in v2.combined_closure(repo).items()}}
def digest(v:Mapping[str,str])->str:return hashlib.sha256(json.dumps(dict(v),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def issue(*,reviewed:Mapping[str,str],reviewed_digest:str,repo:Path,root:Path)->Capability:
    canonical=repo/plan.ROOT_RELATIVE;current=combined(repo);need(root==canonical and dict(reviewed)==current and reviewed_digest==digest(current) and not os.path.lexists(root) and root.parent.is_dir() and not root.parent.is_symlink(),'V3 reviewed/fresh capability drift');m=root.parent.stat(follow_symlinks=False);return Capability(_TOKEN,repo,root,current,int(m.st_dev),int(m.st_ino))
def _issue_for_test(*,repo:Path,root:Path)->Capability:
    current=combined(repo);need(not os.path.lexists(root) and root.parent.is_dir(),'V3 test fresh root');m=root.parent.stat(follow_symlinks=False);return Capability(_TOKEN,repo,root,current,int(m.st_dev),int(m.st_ino))
def consume(cap:Capability)->tuple[Path,tuple[int,int]]:
    key=(cap.parent_dev,cap.parent_ino,str(cap.root));m=cap.root.parent.stat(follow_symlinks=False);need(isinstance(cap,Capability) and cap.token is _TOKEN and key not in _USED and (m.st_dev,m.st_ino)==(cap.parent_dev,cap.parent_ino) and not os.path.lexists(cap.root) and combined(cap.repo)==cap.closure,'V3 capability drift');_USED.add(key);cap.root.mkdir();r=cap.root.stat(follow_symlinks=False);return cap.root,(int(r.st_dev),int(r.st_ino))
def revalidate_cap(cap:Capability,root_identity:tuple[int,int])->None:
    p=cap.root.parent.stat(follow_symlinks=False);r=cap.root.stat(follow_symlinks=False);need((p.st_dev,p.st_ino)==(cap.parent_dev,cap.parent_ino) and (r.st_dev,r.st_ino)==root_identity and combined(cap.repo)==cap.closure,'V3 terminal/failure capability drift')
def validate_outer(root:Path,*,terminal:bool,published:tuple[str,...]|None=None)->None:
    names={'attempt.json','v2_authority.json','sentinel_authority.json'} if terminal else set(published or ())
    names.add('terminal.json' if terminal else 'failure.json'); expected=names|{name+'.sha256' for name in names}|({'score'} if terminal or (root/'score').exists() else set())
    need({x.name for x in root.iterdir()}==expected,'V3 outer exact topology')
    for name in names:
        body=root/name;side=root/(name+'.sha256')
        for path in (body,side):
            m=path.stat(follow_symlinks=False);need(stat.S_ISREG(m.st_mode) and stat.S_IMODE(m.st_mode)==0o444 and m.st_nlink==1,'V3 outer leaf mode/link')
        need(side.read_bytes()==f'{sha(body)}  {name}\n'.encode(),'V3 outer sidecar')
    if (root/'score').exists() or os.path.lexists(root/'score'):
        m=(root/'score').stat(follow_symlinks=False);need(stat.S_ISDIR(m.st_mode) and not (root/'score').is_symlink(),'V3 inner score directory drift')
        from tfpd_exploration.src.m2_anchored_joint_postfusion_v1 import lifecycle as v1
        if terminal:
            v1.validate_score_topology(root/'score',terminal=True)
            outer,_=held_json(root,'terminal.json');need(outer.get('inner_score_terminal_sha256')==sha(root/'score'/'terminal.json'),'V3 inner terminal link drift')
        elif (root/'score'/'failure.json').exists():
            failure,_=held_json(root/'score','failure.json');v1.validate_score_failure_topology(root/'score',published=tuple(failure.get('published_prefix',())))
        else:
            raise Error('V3 partial inner score graph')
