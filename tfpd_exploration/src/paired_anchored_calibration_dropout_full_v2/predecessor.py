"""Held-FD exact validation of the honest V1 P0 pre-update failure."""
from __future__ import annotations
import hashlib,json,os,stat
from pathlib import Path
from . import plan

class PredecessorError(RuntimeError): pass
LEAVES=tuple(sorted([name for body in plan.V1_FAILURE_SHAS for name in (body,body+".sha256")]))
def _read(fd):
 out=[]
 while (chunk:=os.read(fd,65536)): out.append(chunk)
 return b"".join(out)
def validate_v1_failure(root:Path,relative:str=plan.V1_FAILURE_RELATIVE):
 try: d=os.open(root/relative,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 except OSError as e: raise PredecessorError(str(e)) from e
 try:
  if set(os.listdir(d))!=set(LEAVES): raise PredecessorError("topology")
  bodies={}
  for name,expected in plan.V1_FAILURE_SHAS.items():
   st=os.stat(name,dir_fd=d,follow_symlinks=False)
   if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode)!=0o444: raise PredecessorError("body mode")
   fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=d)
   try: body=_read(fd)
   finally: os.close(fd)
   if hashlib.sha256(body).hexdigest()!=expected: raise PredecessorError("body sha")
   side=name+".sha256"; st=os.stat(side,dir_fd=d,follow_symlinks=False)
   if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode)!=0o444: raise PredecessorError("side mode")
   fd=os.open(side,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=d)
   try: sidebody=_read(fd).decode()
   finally: os.close(fd)
   if sidebody!=f"{expected}  {name}\n": raise PredecessorError("sidecar")
   bodies[name]=json.loads(body)
  a,l,s,f=(bodies[x] for x in ("attempt.json","launch.json","source_authority.json","failure.json"))
  if a.get("arm")!="p0" or a.get("source_closure",{}).get("closure_sha256")!="c6e4a4811c58ddf4530e200dd6ac704828b4a0f079b5bf1e303127257a380f8e": raise PredecessorError("attempt semantics")
  if f.get("status")!="CELL_FAILED" or f.get("progress",{}).get("epochs_published")!=0 or f.get("progress",{}).get("checkpoints_published")!=0 or f.get("progress",{}).get("swa_published") is not False or f.get("target_access") is not False: raise PredecessorError("failure semantics")
  if "UninitializedParameter" not in f.get("failure",{}).get("detail","") and "uninitialized parameter" not in f.get("failure",{}).get("detail","").lower(): raise PredecessorError("failure cause")
  return {"relative":relative,"failure_sha256":plan.V1_FAILURE_SHAS["failure.json"],"topology":list(LEAVES)}
 finally: os.close(d)
