#!/usr/bin/env python3
"""Read-only M1 HO-calib M10 carrier-support stability audit."""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; V1=ROOT/'btransform_unified_v1'; WS=ROOT
for p in (V1/'src',V1/'scripts',WS): sys.path.insert(0,str(p))
import m1_projadd_depth2_series as legacy
from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
from tfpd_exploration.src.m1_optimized_v2 import bank as source_bank
from tfpd_exploration.src.m1_optimized_v2 import plan as source_plan
OUT=ROOT/'btransform_unified_v2/results/diagnostics_v1/m1_carrier_support_stability_v2.json'
HO=('20121004','20121017','20121024')
def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def fh(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def basis():
 z=np.load(source_plan.BANK_NPZ,allow_pickle=False)
 return syn3.SourceBasis('nnmf',np.asarray(z['scale']),np.asarray(z['d0']),np.asarray(z['activations']),tuple(int(x) for x in z['nmf_order']),str(z['reconstruction_digest'][0]),{'source':'rSyn3-refit-v1.sealed'}, {})
def fit(rec, ids, b, norm):
 t=time.perf_counter(); raw=rsyn3_bank._encode_record(rec,b,selected_trial_ids=ids); solve=time.perf_counter()-t
 c=np.ascontiguousarray(syn3.normalize_carriers(raw,norm['normalizer_mean'],norm['normalizer_scale']),dtype=np.float32)
 mask=np.isin(rec.rate_trial_ids,ids); scores=syn3.project_basis(rec.emg[np.isin(rec.emg_trial_ids,ids)],b)
 cov=syn3.coverage_report(scores,rec.rates[mask],trial_ids=rec.rate_trial_ids[mask],budget=10)
 return c,solve,cov
def row(session, provider):
 p=legacy._heldout_calib_path(session); rec=rsyn3_bank.load_public_calib_support(p); b=basis(); norm=source_bank.load()
 full=np.arange(10,dtype=np.int64); base,solve,cov=fit(rec,full,b,norm); prod,meta=legacy.encode_heldout_calib_carrier(session)
 if not np.allclose(base,prod,rtol=0,atol=0): raise RuntimeError(f'{session}: production carrier mismatch')
 opened=legacy.open_heldout_calib_session(session); samples=[]
 for _ in range(3):
  t=time.perf_counter(); e0=provider(opened['calib10']); samples.append(time.perf_counter()-t)
 e0s=float(np.median(samples))
 runs=[]
 for frac in (.5,.75):
  k=int(np.ceil(10*frac))
  for seed in range(101,111):
   ids=np.sort(np.random.default_rng(seed).choice(full,k,replace=False)); c,dt,cv=fit(rec,ids,b,norm); den=np.linalg.norm(base); runs.append({'fraction_nominal':frac,'selected_count':k,'seed':seed,'trial_ids':ids.tolist(),'normalized_frobenius':float(np.linalg.norm(c-base)/den),'cosine':float(c.ravel().dot(base.ravel())/(np.linalg.norm(c)*den)),'fit_seconds':dt,'coverage':cv})
 return {'nwb':str(p),'nwb_sha256':fh(p),'support_trial_ids_full':full.tolist(),'query_manifest':{'reader':'legacy.open_heldout_calib_session','query_windows':int(len(opened['dataset'].window_indices)),'same_visible_ho_calib_development_surface':True,'query_values_read_for_carrier':False,'claim':'carrier reads M10 support view; query windows can overlap this visible development recording and are not an independent post-support evaluation'},'production_meta':meta,'full_carrier_sha256':ah(base),'production_carrier_sha256':ah(prod),'byte_allclose':bool(np.array_equal(base,prod)),'full':{'normalized_frobenius':0.,'cosine':1.,'fit_seconds':solve,'coverage':cov},'frozen_b3_e0':{'sha256':ah(e0),'shape':list(e0.shape),'forward_seconds_median_3':e0s,'forward_seconds_3':samples},'resamples':runs}
def main():
 os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ['OMP_NUM_THREADS']='2'; os.environ['MKL_NUM_THREADS']='2'
 t=time.perf_counter(); cold=time.perf_counter(); provider=legacy.mp.default_identity_provider(); cold_load=time.perf_counter()-cold; rows={s:row(s,provider) for s in HO}; OUT.parent.mkdir(parents=True,exist_ok=True)
 receipt={'schema':'m1_carrier_support_stability_v1','status':'COMPLETED','cpu_only':True,'support':'held-out-calib chronological M10 whole trials','estimator':'frozen all-source NNMF basis + source normalizer + NNLS activations + production ridge unit encodings','fractions_nominal':[.5,.75,1.],'actual_selected_counts':{'0.5':5,'0.75':8,'1.0':10},'seeds':list(range(101,111)),'source_hashes':{str(p):fh(p) for p in (Path(__file__),V1/'scripts/m1_projadd_depth2_series.py',WS/'tfpd_exploration/src/m1_b3_allsource_v1/rsyn3_bank.py',WS/'tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py',source_plan.BANK_NPZ,Path(source_bank.__file__),Path(__import__('tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config',fromlist=['S_FIX_PATH']).S_FIX_PATH))},'b3_checkpoint_cold_load_seconds':cold_load,'elapsed_seconds':time.perf_counter()-t,'sessions':rows}; OUT.write_text(json.dumps(receipt,indent=2)+'\n'); print(OUT)
if __name__=='__main__':main()
