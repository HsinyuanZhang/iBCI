"""Independent, refuse-overwrite audit of the sealed H1 fixed M3 MAT7 readout.

This checks persisted support/fit/application artefacts.  It intentionally
does not attempt to reproduce the neural model predictions which formed P.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.decoder_validation_v2.audit_predictions import audit_h1
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import _index_split, load_session_arrays

ROOT=Path('/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000')
OUT=ROOT/'h1/v4_full_continue24_deterministic_v1/canonical_m3_mat7_readout_v1'
RECEIPT=OUT/'receipt.json'; TARGET=OUT/'independent_support_refit_audit_v1.json'
AUTH=Path('tfpd_exploration/h1_series_20260830/results/h1_m3_readout_calibration_evalai_package_v1/calibration_authority.json')
EXPECTED_RECEIPT='7818f6f274073fede373c36039bee615d94ca041f0d3fa387c1ff2838248a06a'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ahash(a):
 a=np.ascontiguousarray(a); h=hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode()); h.update(a.tobytes()); return h.hexdigest()
def fit(p,y):
 p=np.asarray(p,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
 if p.ndim!=2 or p.shape!=y.shape or p.shape[1]!=7 or len(p)<8 or not(np.isfinite(p).all() and np.isfinite(y).all()): raise ValueError('invalid MAT7 support')
 pm,ym=p.mean(0),y.mean(0); ps=np.maximum(p.std(0),1e-6); ys=np.maximum(y.std(0),1e-6)
 z=(p-pm)/ps; zy=(y-ym)/ys; d=np.c_[z,np.ones(len(z),dtype=np.float64)]
 sol=np.linalg.lstsq(d,zy,rcond=None)[0]
 return dict(family='MAT7',ridge=0.0,p_mean=pm,p_scale=ps,y_mean=ym,y_scale=ys,weight=sol[:7],intercept=sol[7])
def apply(m,p): return ((np.asarray(p,np.float64)-m['p_mean'])/m['p_scale']@m['weight']+m['intercept'])*m['y_scale']+m['y_mean']

def main():
 if TARGET.exists(): raise FileExistsError(TARGET)
 if sha(RECEIPT)!=EXPECTED_RECEIPT: raise RuntimeError('fixed M3 receipt SHA mismatch')
 receipt=json.loads(RECEIPT.read_text()); canonical=json.loads(AUTH.read_text())
 official={x['session']:x for x in canonical['sessions'] if x['scope']=='held-in-calib'}
 if len(official)!=13 or set(official)!=set(receipt['sessions']): raise RuntimeError('official held-in roster drift')
 paths=_index_split('held-in-calib'); persisted=torch.load(OUT/'per_session_maps.pt',map_location='cpu',weights_only=False)['maps']
 sessions={}; total=0
 for session,c in sorted(official.items()):
  support=OUT/f'{session}_canonical_m3_support.npz'
  if not support.is_file() or sha(support)!=receipt['sessions'][session]['support_npz_sha256']: raise RuntimeError(f'{session}: support archive SHA drift')
  with np.load(support,allow_pickle=False) as z:
   if set(z.files)!={'endpoint','trial_id','prediction_native_velocity','target_native_velocity'}: raise RuntimeError(f'{session}: support schema drift')
   end,trial,p,y=(np.asarray(z[k]) for k in ('endpoint','trial_id','prediction_native_velocity','target_native_velocity'))
  rec=load_session_arrays(paths[session],session,skip_first3=True)
  if sha(paths[session])!=c['nwb_sha256']: raise RuntimeError(f'{session}: actual raw NWB SHA drift')
  valid=np.flatnonzero(np.asarray(rec.eval_mask,dtype=bool)); first=tuple(np.unique(np.asarray(rec.trial_num)[valid])[:3].astype(float))
  wanted=np.flatnonzero(np.asarray(rec.eval_mask,dtype=bool)&np.isin(rec.trial_num,np.asarray(first)))
  if first!=tuple(c['calibration_trials']) or not(np.array_equal(end,wanted) and np.array_equal(trial,np.asarray(rec.trial_num)[wanted]) and np.array_equal(y,np.asarray(rec.velocity)[wanted])): raise RuntimeError(f'{session}: official endpoint/trial/target coverage drift')
  if len(end)!=c['calibration_bins']: raise RuntimeError(f'{session}: support count drift')
  m=fit(p,y); saved=persisted[session]; diffs={k:float(np.max(np.abs(np.asarray(m[k])-np.asarray(saved[k])))) for k in ('p_mean','p_scale','y_mean','y_scale','weight','intercept')}
  if max(diffs.values())>1e-10: raise RuntimeError(f'{session}: independent OLS map mismatch {diffs}')
  sessions[session]={'n':int(len(end)),'first_three_valid_trialnums':list(first),'raw_nwb_sha256':sha(paths[session]),'support_sha256':sha(support),'map_max_abs_diffs':diffs}; total+=len(end)
 if total!=30879: raise RuntimeError(f'support total {total} != 30879')
 exports={}
 for label in ('selected_selection','selected_complete','epoch24_selection','epoch24_complete'):
  corrected=OUT/f'full_{label}_m3mat7_native.npz'; row=receipt['applied_exports'][label]
  if sha(corrected)!=row['sha256']: raise RuntimeError(f'{label}: corrected archive SHA drift')
  src=Path(row['source_npz'])
  with np.load(corrected,allow_pickle=False) as c, np.load(src,allow_pickle=False) as o:
   if c.files!=o.files or any(not np.array_equal(c[k],o[k]) for k in o.files if k!='prediction_native_velocity'): raise RuntimeError(f'{label}: non-prediction fields changed')
   manual=np.empty_like(o['prediction_native_velocity'])
   for s,m in persisted.items(): manual[o['session_id']==s]=apply(m,o['prediction_native_velocity'][o['session_id']==s]).astype(manual.dtype)
   err=float(np.max(np.abs(manual-c['prediction_native_velocity'])))
   if err>1e-10: raise RuntimeError(f'{label}: manual map mismatch {err}')
  surface='selection' if label.endswith('selection') else 'complete'
  exports[label]={'corrected_sha256':sha(corrected),'source_sha256':sha(src),'manual_map_max_abs_error':err,'score':audit_h1(corrected,surface=surface)}
 result={'schema':'h1_m3_independent_support_refit_audit_v1','status':'PASS','fixed_receipt_sha256':EXPECTED_RECEIPT,'scope':'support provenance, float64 MAT7 OLS fit math, and corrected-export map application; not neural-model P prediction fidelity (separate runtime replay required)','contract':{'family':'MAT7','ridge':0.0,'scale_floor':1e-6,'fit_dtype':'float64','official_held_in_sessions':13,'support_total_bins':total},'sessions':sessions,'exports':exports}
 TARGET.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps({'receipt':str(TARGET),'sha256':sha(TARGET),'status':'PASS'},sort_keys=True))
if __name__=='__main__': main()
