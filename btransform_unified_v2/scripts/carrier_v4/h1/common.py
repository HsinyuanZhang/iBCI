"""H1 v4 source-only adapters for the shared latent-state estimator."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
from typing import Any
import numpy as np
HERE=Path(__file__).resolve().parent; V4=HERE.parent; ROOT=V4.parents[1]; WS=ROOT.parent; SPINT=WS/"SPINT-main"; DATA=SPINT/"data"/"000954"; HO_DIR=DATA/"sub-HumanPitt-held-out-calib"; SOURCE_PAYLOAD=SPINT/"local_data"/"h1_epfilm_evalai_v1"/"decoder.pt"; PROFILE_DIR=ROOT/"scripts"/"carrier_profile_v2"; RIFT_DIR=ROOT/"scripts"/"rift_v1"
OFFICIAL_HO_M3={"submission_id":582073,"selected_epoch":22,"val_ho_m3_grouped/r2_mean":0.3748232633647425,"worst_session_r2":0.20587240655236508}
def sha_file(p:Path)->str:
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_array(a:np.ndarray)->str:
 a=np.ascontiguousarray(a);return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def jsonable(x:Any)->Any:
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Path):return str(x)
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [jsonable(v) for v in x]
 return x
def atomic_json(p:Path,x:Any):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(jsonable(x),indent=2,sort_keys=True)+'\n');q.replace(p)
def setup_imports():
 for x in (str(WS),str(WS/'btransform_unified_v1'/'src'),str(WS/'btransform_unified_v1'/'scripts'),str(ROOT/'src'),str(RIFT_DIR),str(PROFILE_DIR),str(HERE),str(V4),str(SPINT)):
  if x in sys.path:sys.path.remove(x)
  sys.path.insert(0,x)
def session_from_path(p:Path)->str:return p.stem[p.stem.index('_ses-')+1:]
def index_heldout_calib():
 rows={session_from_path(p.resolve()):p.resolve() for p in sorted(HO_DIR.glob('*.nwb'))};
 if len(rows)!=14:raise RuntimeError('expected exact public H1 HO14')
 return rows
def load_public_heldout_m3(path:Path,pilot:Any):
 from falcon_challenge.config import FalconTask
 from falcon_challenge.dataloaders import load_nwb
 from pynwb import NWBHDF5IO
 neural,vel,change,mask=load_nwb(path,FalconTask.h1)
 with NWBHDF5IO(str(path),'r',load_namespaces=True) as h:trial=np.asarray(h.read().acquisition['TrialNum'].data[:],float)
 vals=[]
 for x in trial[np.asarray(mask,bool)&np.isfinite(trial)]:
  if not vals or x!=vals[-1]:vals.append(float(x))
 if len(vals)!=3:raise RuntimeError('HO public calibration must exact M3')
 blocks=tuple(pilot._trial_blocks(v,np.asarray(neural,float),np.asarray(vel,float),np.asarray(mask,bool),trial) for v in vals)
 return pilot.H1PilotRecord(pilot.session_from_path(path),pilot.session_date(pilot.session_from_path(path)),Path(path),sha_file(path),np.asarray(neural,np.float32),np.asarray(vel,np.float32),np.asarray(change,bool),np.asarray(mask,bool),trial,tuple(vals),blocks)
