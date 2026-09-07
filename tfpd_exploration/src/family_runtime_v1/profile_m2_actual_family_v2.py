"""Bound frozen selected M2 T1 timing diagnostic; no fitting/scoring."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders
from tfpd_exploration.src.m2_dual_track_v1 import data,plan
from .m2_family_causal import M2FamilyCausalRuntime,M2RuntimeBank
ROOT=Path(__file__).resolve().parents[3];F=ROOT/'tfpd_exploration/results/m2/family_v1/finalize_pair_v1';O=ROOT/'tfpd_exploration/results/m2/family_runtime_v1/m2_actual_family_profile_t1_v2.json'; FINAL='3be4a096f98bd9f77726094501271e2836e37c6f61c06ce136ce72505488cf47'
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def authority(r):
 files=[Path(__file__),Path(__file__).with_name('m2_family_causal.py'),Path(__file__).with_name('m2_family_spatial.py'),Path(__file__).with_name('linear_conv.py'),Path(__file__).with_name('grouped_value.py'),ROOT/'tfpd_exploration/src/m2_family_v1/decoder.py',ROOT/'tfpd_exploration/src/m2_family_v1/routing.py']
 out={'code':{str(p.relative_to(ROOT)):h(p) for p in files},'banks':{}}
 for s in plan.HELDIN_SESSIONS:
  d=data._session_dir('source_train',s);out['banks'][s]={n:h(d/n) for n in ('X_store.npy','e0_u.pt','T.npy','mapping.json','provenance.json')}
 return out
def run():
 if O.exists():raise FileExistsError(O)
 if os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','-1'):raise RuntimeError('CPU only')
 torch.set_num_threads(1);torch.set_num_interop_threads(1); r=json.loads((F/'receipt.json').read_text());
 if h(F/'receipt.json')!=FINAL:raise RuntimeError('finalizer SHA drift')
 pre=authority(r);result={'schema':'m2_actual_family_profile_t1_v2','status':'TIMING_DIAGNOSTIC_NOT_PROOF_OR_QUALITY','finalizer_sha256':FINAL,'pre':pre,'arms':{}}
 for arm in ('FLAT','ROUTE'):
  key=[k for k in r['exports'] if k.startswith(arm+'_selected_')]
  if len(key)!=1:raise RuntimeError('selected identity')
  key=key[0]
  p=F/f'{key}_ema_state.pt'
  if h(p)!=r['exports'][key]['export_sha256']:raise RuntimeError('export hash drift')
  m=make_paired_decoders(42)[0 if arm=='FLAT' else 1];m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);m.eval(); t={k:[] for k in ('predict','audit','repair','temporal_first3','temporal_remainder')}; errors=[]; done=0
  for si,s in enumerate(plan.HELDIN_SESSIONS):
   d=data._session_dir('source_train',s); raw=np.asarray(np.load(d/'X_store.npy',mmap_mode='r'),dtype=np.float32);b0=data.load_session_bank('source_train',s,device='cpu');b=M2RuntimeBank(b0.E0,b0.T,b0.unit_mask);rt=M2FamilyCausalRuntime(m,b,batch_size=7); oa,or_,ol=rt._audit,rt.spatial.repair,rt._last
   def wrap(k,fn):
    def q(*a,**w):q0=time.perf_counter_ns();v=fn(*a,**w);t[k].append(time.perf_counter_ns()-q0);return v
    return q
   rt._audit=wrap('audit',oa);rt.spatial.repair=wrap('repair',or_)
   for j,x in enumerate(raw[49:49+192]):
    # `_last` is split analytically: hooks time blocks0..2; remainder is total-last minus these.
    block=[]; hooks=[z.register_forward_hook(lambda *a,bb=block,**w:bb.append(time.perf_counter_ns())) for z in m.temporal.blocks[:3]]
    q0=time.perf_counter_ns();rt.predict(np.ascontiguousarray(np.broadcast_to(x,(7,96)).copy())); total=time.perf_counter_ns()-q0
    for z in hooks:z.remove()
    if j>=64 and done<128:
     t['predict'].append(total); first=sum(block[k+1]-block[k] for k in range(0,len(block)-1,2)) if len(block)>=2 else 0;t['temporal_first3'].append(first);t['temporal_remainder'].append(max(0,t['predict'][-1]-t['audit'][-1]-t['repair'][-1]-first));
     if j in (64,113,114,191):errors.append(float(np.abs(rt.raw[0].numpy()-raw[j:j+50]).max()));done+=1
   if done>=128:break
  result['arms'][arm]={'selected_key':key,'export_sha256':h(p),'warmup_bins':64,'timed_bins':len(t['predict']),'mean_ms':{k:float(np.mean(v)/1e6) for k,v in t.items() if v},'native_raw_window_subset_max_abs':max(errors,default=None)}
 if authority(r)!=pre:raise RuntimeError('post authority drift')
 result['post']=authority(r);O.parent.mkdir(parents=True,exist_ok=True);O.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');return result
if __name__=='__main__':print(json.dumps(run(),sort_keys=True))
