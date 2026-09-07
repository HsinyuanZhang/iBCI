"""Held-FD validation of accepted PACD V2 terminal lineage."""
from __future__ import annotations
import hashlib,json,os,stat
from pathlib import Path
from . import plan
from src.paired_anchored_calibration_dropout_v2 import plan as v2_plan
class PredecessorError(RuntimeError): pass
LEAVES=("attempt.json","attempt.json.sha256","terminal.json","terminal.json.sha256")
def _read(fd):
 out=[]
 while (b:=os.read(fd,65536)): out.append(b)
 return b''.join(out)
def _file(d,name,expected):
 st=os.stat(name,dir_fd=d,follow_symlinks=False)
 if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode)!=0o444: raise PredecessorError(name)
 f=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=d)
 try: b=_read(f)
 finally: os.close(f)
 if hashlib.sha256(b).hexdigest()!=expected: raise PredecessorError("sha "+name)
 return b,json.loads(b)
def validate_v2_terminal(root:Path,relative:str=plan.V2_RELATIVE)->dict:
 try: d=os.open(Path(root)/relative,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 except OSError as e: raise PredecessorError(str(e)) from e
 try:
  if set(os.listdir(d))!=set(LEAVES): raise PredecessorError("topology")
  ab,a=_file(d,"attempt.json",plan.V2_ATTEMPT_SHA); tb,t=_file(d,"terminal.json",plan.V2_TERMINAL_SHA)
  for n,b in (("attempt.json",ab),("terminal.json",tb)):
   s=n+".sha256"; st=os.stat(s,dir_fd=d,follow_symlinks=False)
   if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode)!=0o444: raise PredecessorError("sidecar "+s)
   f=os.open(s,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=d)
   try: value=_read(f).decode()
   finally: os.close(f)
   if value!=f"{hashlib.sha256(b).hexdigest()}  {n}\n": raise PredecessorError("sidecar "+s)
  if t.get("attempt_sha256")!=plan.V2_ATTEMPT_SHA or t.get("status")!="PACD_SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE" or t.get("target_access") is not False or t.get("source_only") is not True: raise PredecessorError("terminal semantics")
  predecessor=t.get("predecessor",{})
  if predecessor != {"relative": v2_plan.PREDECESSOR_RELATIVE, "attempt_sha256": v2_plan.V1_ATTEMPT_SHA256, "failure_sha256": v2_plan.V1_FAILURE_SHA256, "topology": ["attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"]}: raise PredecessorError("exact V1 predecessor")
  c=t.get("source_closure",{}); 
  if c.get("launch",{}).get("closure_sha256")!=plan.V2_CLOSURE_SHA or c.get("final",{}).get("closure_sha256")!=plan.V2_CLOSURE_SHA: raise PredecessorError("closure")
  facts=t.get("no_target_facts",{})
  if facts.get("within_dev_sessions_opened") is not False or facts.get("external_sub_m_opened") is not False or facts.get("formal_or_organizer_held_data_opened") is not False or facts.get("scorer_called") is not False or facts.get("source_roster_n")!=27 or facts.get("val_paths_resolved")!=[] or facts.get("test_paths_resolved")!=[] or facts.get("single_source_datamodule_materialization") is not True: raise PredecessorError("no-target facts")
  arms=t.get("arms",[])
  if [x.get("arm",{}).get("name") for x in arms] != ["p0","p1","p2"] or any(len(x.get("steps",[]))!=8 for x in arms): raise PredecessorError("arms")
  for item in arms:
   for step in item["steps"]:
    if not step.get("dropout_pair_equal") or not step.get("rng_short_transition_equal") or not step.get("rng_pair_transition_equal"): raise PredecessorError("paired evidence")
    pf=step.get("parameter_finiteness",{})
    if pf.get("materialized")!=29 or pf.get("skipped_uninitialized_lazy")!=2: raise PredecessorError("parameter counts")
    if item["arm"]["name"]=="p0" and (not step.get("prediction_pair_equal") or not step.get("identity_pair_equal")): raise PredecessorError("P0 evidence")
  return {"relative":relative,"attempt_sha256":plan.V2_ATTEMPT_SHA,"terminal_sha256":plan.V2_TERMINAL_SHA,"topology":list(LEAVES)}
 finally: os.close(d)
