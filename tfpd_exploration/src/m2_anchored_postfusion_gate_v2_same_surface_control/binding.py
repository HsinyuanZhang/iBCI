"""Held-FD V1-failure predecessor validator for V2."""
from __future__ import annotations
import hashlib, json, os, stat
from pathlib import Path
from typing import Any, Mapping
from . import plan
class BindingError(RuntimeError): pass
def _need(ok, msg):
    if not ok: raise BindingError(msg)
def _flag(name):
    value=getattr(os,name,None); _need(isinstance(value,int) and value != 0, f"platform lacks {name}"); return value
def _read(fd, name):
    leaf=os.open(name, os.O_RDONLY|_flag("O_NOFOLLOW"), dir_fd=fd)
    try:
        info=os.fstat(leaf); _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode)==0o444 and info.st_nlink==1, f"V1 leaf mode/link drift: {name}")
        chunks=[]
        while True:
            block=os.read(leaf,1<<20)
            if not block: return b''.join(chunks)
            chunks.append(block)
    finally: os.close(leaf)
def validate_v1_failure(repo_root: Path) -> dict[str, Any]:
    root=Path(repo_root).absolute()/plan.V1_ROOT_RELATIVE; before=os.lstat(root)
    _need(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),"V1 predecessor root drift")
    names=set(plan.V1_BODIES); expected=names|{n+".sha256" for n in names}
    fd=os.open(root,os.O_RDONLY|_flag("O_DIRECTORY")|_flag("O_NOFOLLOW"))
    try:
        _need(set(os.listdir(fd))==expected,"V1 predecessor topology drift")
        payloads={}; digests={}
        for name, literal in plan.V1_BODIES.items():
            body=_read(fd,name); digest=hashlib.sha256(body).hexdigest(); _need(digest==literal,f"V1 body SHA drift: {name}")
            _need(_read(fd,name+'.sha256')==f"{digest}  {name}\n".encode(),f"V1 sidecar drift: {name}")
            payloads[name]=json.loads(body); digests[name]=digest
        after=os.fstat(fd); _need((after.st_dev,after.st_ino)==(before.st_dev,before.st_ino),"V1 predecessor root swap")
    finally: os.close(fd)
    a,l,s,alpha,inp,f=(payloads[n] for n in plan.V1_BODIES)
    _need(a.get("schema") == "m2_anchored_postfusion_gate_v1_attempt_v1"
          and a.get("status") == "ATTEMPT_RESERVED", "V1 attempt semantic drift")
    _need(l.get("schema") == "m2_anchored_postfusion_gate_v1_launch_v1", "V1 launch semantic drift")
    _need(s.get("schema") == "m2_anchored_postfusion_gate_v1_source_authority_v1", "V1 source authority schema drift")
    _need(alpha.get("schema") == "m2_anchored_postfusion_gate_v1_alpha_selection_v1", "V1 alpha schema drift")
    _need(inp.get("schema") == "m2_anchored_postfusion_gate_v1_input_authority_v1", "V1 input schema drift")
    _need(f.get("schema") == "m2_anchored_postfusion_gate_v1_failure_v1"
          and f.get("status") == "FAILED" and f.get("terminal_xor_failure") is True,
          "V1 failure schema/status drift")
    _need(a.get('closure_sha256')==plan.V1_CLOSURE_SHA256 and l.get('attempt_closure_sha256')==plan.V1_CLOSURE_SHA256,"V1 closure linkage drift")
    sel=s.get('source_selection',{})
    _need(sel.get('source_safety_gate_passed') is True and sel.get('selected_epoch')==plan.SELECTED_EPOCH
          and float(sel.get('delta_vs_zero'))==0.002202600400827759
          and sel.get('per_session_delta_vs_zero') == {
              'ses-2020-10-27-Run2': 0.0026953303948594742,
              'ses-2020-10-28-Run1': 0.0017098704067960435,
          }, "V1 source selection drift")
    _need(alpha.get('selection') == sel and float(alpha.get('refit_alpha'))==plan.REFIT_ALPHA
          and alpha.get('selected_epoch_fixed30_descriptive',{}).get('epoch')==plan.SELECTED_EPOCH
          and alpha.get('activity_authority') == 'pooled_g00m_linear'
          and alpha.get('teacher_forward_calls') == 0, "V1 alpha receipt drift")
    _need(s.get('selected_t4_checkpoint_sha256')==plan.SELECTED_CHECKPOINT_SHA256 and s.get('prepared_evidence',{}).get('selected_t4',{}).get('student_state_after_load_sha256')==plan.SELECTED_STUDENT_STATE_SHA256,"V1 strict checkpoint/state drift")
    _need(inp.get('source_authority_sha256') == digests['source_authority.json']
          and inp.get('activity_authority') == 'pooled_g00m_linear'
          and isinstance(inp.get('records'), Mapping) and len(inp['records']) == 13,
          "V1 target input authority drift")
    _need(f.get('attempt_sha256') == digests['attempt.json']
          and f.get('exception_class')=='tfpd_exploration.src.m2_anchored_postfusion_gate_v1.scoring.ScoringError'
          and f.get('diagnostic_message')=='APFG zero POOLED prediction mismatch'
          and f.get('error_sha256')=='59297c1eebb9e534872f780b192d5e45bf3f24a2fb7f84cee38533b894bc559b',"V1 failure semantics drift")
    _need(f.get('published_prefix',{}).get('input_authority.json')==digests['input_authority.json'] and not (root/'score.json').exists() and not (root/'terminal.json').exists(),"V1 target failure placement drift")
    progress=f.get('progress',{})
    _need(progress.get('stage')=='launch'
          and progress.get('published_prefix')==['attempt.json','launch.json','source_authority.json','alpha_selection.json']
          and progress.get('target_attempted') is True and progress.get('target_opened') is True
          and progress.get('target_complete') is True and progress.get('source_complete') is True,
          "V1 historical progress split-topology drift")
    return {"root_relative":plan.V1_ROOT_RELATIVE,"root_identity":[before.st_dev,before.st_ino],"body_sha256":digests,"payloads":payloads}
