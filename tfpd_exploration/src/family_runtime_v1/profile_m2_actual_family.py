"""Same-process, source-train-only M2 family causal-runtime timing attribution."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders
from tfpd_exploration.src.m2_dual_track_v1 import data,plan
from .m2_family_causal import M2FamilyCausalRuntime,M2RuntimeBank
ROOT=Path(__file__).resolve().parents[3]; FINAL=ROOT/'tfpd_exploration/results/m2/family_v1/finalize_pair_v1'; OUT=ROOT/'tfpd_exploration/results/m2/family_runtime_v1/m2_actual_family_profile_t1_v1.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 if OUT.exists(): raise FileExistsError(OUT)
 if os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','-1'): raise RuntimeError('CPU only')
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 r=json.loads((FINAL/'receipt.json').read_text()); out={'schema':'m2_actual_family_profile_t1_v1','status':'SOURCE_TRAIN_RAW_ONLY_SAME_PROCESS_NOT_SPINT_COMPARISON','threads':1,'receipt_sha256':sha(FINAL/'receipt.json'),'arms':{}}
 s=plan.HELDIN_SESSIONS[0]; d=data._session_dir('source_train',s); raw=np.asarray(np.load(d/'X_store.npy',mmap_mode='r'),dtype=np.float32); bank0=data.load_session_bank('source_train',s,device='cpu'); bank=M2RuntimeBank(bank0.E0,bank0.T,bank0.unit_mask)
 for arm in ('FLAT','ROUTE'):
  key=next(k for k in r['exports'] if k.startswith(arm+'_selected_')); p=FINAL/f'{key}_ema_state.pt'; m=make_paired_decoders(42)[0 if arm=='FLAT' else 1];m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);m.eval(); rt=M2FamilyCausalRuntime(m,bank,batch_size=7)
  times={k:[] for k in ('audit','repair','last','predict')}; oa,or_,ol=rt._audit,rt.spatial.repair,rt._last
  def wrap(name,fn):
   def f(*a,**kw):
    t=time.perf_counter_ns();v=fn(*a,**kw);times[name].append(time.perf_counter_ns()-t);return v
   return f
  rt._audit=wrap('audit',oa);rt.spatial.repair=wrap('repair',or_);rt._last=wrap('last',ol)
  for x in raw[49:49+256]:
   t=time.perf_counter_ns(); rt.predict(np.ascontiguousarray(np.broadcast_to(x,(7,96)).copy()));times['predict'].append(time.perf_counter_ns()-t)
  out['arms'][arm]={'export_sha256':sha(p),'raw_sha256':sha(d/'X_store.npy'),'bank_files':{n:sha(d/n) for n in ('e0_u.pt','T.npy','mapping.json','provenance.json')},'bins':256,'mean_ms':{k:float(np.mean(v)/1e6) for k,v in times.items()},'p95_ms':{k:float(np.percentile(v,95)/1e6) for k,v in times.items()}}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');return out
if __name__=='__main__': print(json.dumps(run(),sort_keys=True))
