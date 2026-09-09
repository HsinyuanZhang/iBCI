"""Immutable authority for carrier-profile last-one chronological ablations."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/carrier_last1_v2'; ARMS=('B_ACTIVITY_ONLY','D_JOINT'); PROFILES=('old','encoding','state','muscle'); SEED=42
STAGES={
 'inner':{'m1':{'sources':['ses-20120924','ses-20120926'],'targets':['ses-20120927']},'h1':{'source_through':'1925-01-15','targets':['1925-01-19']}},
 'outer':{'m1':{'sources':['ses-20120924','ses-20120926','ses-20120927'],'targets':['ses-20120928']},'h1':{'source_through':'1925-01-19','targets':['1925-01-20']}}}
def protocol():
 return {'schema':'carrier_last1_v2','seed':SEED,'stages':STAGES,'arms':{'B_ACTIVITY_ONLY':'activity only; ignores carrier pack','D_JOINT':'activity plus the named frozen carrier pack'},'profiles':{'old':'literal legacy profile','encoding':'new encoding profile','state':'new state/prototype profile'},'pack_contract':'NPZ keys carrier/<session> with finite [N,4] rows plus metadata JSON; source-only fitting and target M10/three-trial support only','selection':'source validation only; no target optimizer steps or query labels','fixed_primary_epochs':{'m1':24,'h1':32},'run_layout':'{stage}/{dataset}/{profile_id}/{arm}/s42','seals':'B, old-D, and each new-D pack/receipt must bind stage, profile, roster, and SHA before target score'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');t.replace(p)
if __name__=='__main__':
 p=OUT/'protocol.json'
 if p.exists() and json.loads(p.read_text()) != protocol():raise RuntimeError('refuse protocol drift')
 atomic_json(p,protocol());print(json.dumps({'path':str(p),'sha256':sha(p)}))
