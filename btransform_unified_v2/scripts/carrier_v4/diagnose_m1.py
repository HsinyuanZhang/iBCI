#!/usr/bin/env python3
"""CPU-only source-four M1 carrier alignment and stability diagnosis.

The program opens only the four held-in calibration NWBs.  It never resolves a
held-out-calib/query/test path and does not fit an NMF/SVD basis: both bases,
normalizers, and the candidate pack are supplied frozen inputs.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any
import numpy as np

HERE=Path(__file__).resolve().parent; V2=HERE.parents[1]; WS=V2.parent
for p in (V2,V2/"scripts",WS,WS/"btransform_unified_v1"/"src"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from carrier_profile_v2.m1_muscle_profile import MuscleProfileFit,_raw16,DIM,M10,BIN_SECONDS
from m1_carrier_refinement_v1.aligned_carrier import load_aligned_support
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as old_data
from tfpd_exploration.src.m1_optimized_v2 import bank as old_bank, plan as old_plan
from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank
from btransform_unified_v1 import m1_projadd as old_projadd

SOURCES=("ses-20120924","ses-20120926","ses-20120927","ses-20120928")
OLD_FIT_SOURCES=("ses-20120926","ses-20120927","ses-20120928")
DATA=WS/"SPINT-main/data/000941/sub-MonkeyL-held-in-calib"
def need(x:bool,msg:str):
 if not x: raise RuntimeError(msg)
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def arr_sha(a:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def pear(a,b):
 a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel();a=a-a.mean();b=b-b.mean();d=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/d) if d>0 else None
def rms(a):return float(np.sqrt(np.mean(np.square(np.asarray(a,float)))))
def atomic_json(p:Path,x:Any):
 q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n");q.replace(p)
def session_path(s):return DATA/f"sub-MonkeyL-held-in-calib_{s}_behavior+ecephys.nwb"
def movement_slices(ids):
 ids=np.asarray(ids);out=[]
 for t in np.unique(ids):out.append(np.flatnonzero(ids==t))
 return out
def shifted_metrics(old,new,ids,global_indices,timestamps):
 """Same-time and +/- one *within-trial* row correlations; no cross-trial pair."""
 old=np.asarray(old,float);new=np.asarray(new,float);same=pear(old,new);res={"same_time_pearson":same,"same_time_rms_difference":rms(old-new)}
 for shift in (-1,1):
  x=[];y=[];candidate_rows=0;dropped_gap_rows=0
  for ix in movement_slices(ids):
   if len(ix)>1:
    left,right=(ix[1:],ix[:-1]) if shift<0 else (ix[:-1],ix[1:])
    candidate_rows+=len(left)
    good=(np.abs(np.asarray(global_indices)[left]-np.asarray(global_indices)[right])==1)&(np.abs(np.abs(np.asarray(timestamps)[np.asarray(global_indices)[left]]-np.asarray(timestamps)[np.asarray(global_indices)[right]])-BIN_SECONDS)<=1e-3)
    dropped_gap_rows+=int((~good).sum())
    if good.any():x.append(old[left[good]]);y.append(new[right[good]])
  need(x,f"no legal within-trial shift pairs for {shift}")
  x=np.concatenate(x);y=np.concatenate(y);res[f"shift_{shift:+d}_pearson"]=pear(x,y);res[f"shift_{shift:+d}_rms_difference"]=rms(x-y);res[f"shift_{shift:+d}_legal_rows"]=int(len(x));res[f"shift_{shift:+d}_candidate_rows"]=candidate_rows;res[f"shift_{shift:+d}_dropped_gap_rows"]=dropped_gap_rows
 return res
def old_record(path:Path,n:int):return old_data.load_support_bins(path,emg_trial_stop=n,neural_trial_stop=n)
def old_basis():
 z=old_bank.load();blob=np.load(old_plan.BANK_NPZ,allow_pickle=False)
 return z,syn3.SourceBasis(kind="nnmf",scale=np.asarray(blob["scale"]),dictionary=np.asarray(blob["d0"]),activations=np.asarray(blob["activations"]),order=tuple(int(x) for x in blob["nmf_order"]),reconstruction_digest=str(blob["reconstruction_digest"][0]),library={"source":"sealed"},extra={})
def slice_new(z,lo,hi):
 e=(z.emg_trial_ids>=lo)&(z.emg_trial_ids<hi);r=(z.rate_trial_ids>=lo)&(z.rate_trial_ids<hi)
 need(np.array_equal(z.emg_trial_ids[e],z.rate_trial_ids[r]),"new EMG/rate trial IDs drift")
 return np.asarray(z.emg[e],float),np.asarray(z.rates[r],float)
def new_carrier(fit,z,lo,hi):
 e,r=slice_new(z,lo,hi);return ((_raw16(fit,e,r)@fit.svd_components.T-fit.normalizer_mean)/fit.normalizer_scale).astype(np.float32)
def old_carrier(record,basis,norm,ids):
 raw=rsyn3_bank._encode_record(record,basis,selected_trial_ids=np.asarray(ids,dtype=np.int64));return np.asarray(syn3.normalize_carriers(raw,norm["normalizer_mean"],norm["normalizer_scale"]),np.float32),raw
def per_axis(a,b):return [pear(a[:,k],b[:,k]) for k in range(a.shape[1])]
def axis_vs_vector(a,v):return [pear(a[:,k],v) for k in range(a.shape[1])]
def aligned_rsyn(z,basis,norm,lo,hi):
 e,r=slice_new(z,lo,hi);scores=syn3.project_basis(np.maximum(e,0.0),basis);w,b=syn3.fit_all_units(scores,r);raw=syn3.carrier_from_encoding(w,b);return np.asarray(syn3.normalize_carriers(raw,norm["normalizer_mean"],norm["normalizer_scale"]),np.float32),raw
def effective_rank(a):
 s=np.linalg.svd(np.asarray(a,float)-np.asarray(a,float).mean(0),compute_uv=False);p=s*s/max(float((s*s).sum()),1e-30);return float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-300)))))
def fit_loaded(path:Path):
 with np.load(path,allow_pickle=False) as z:
  meta=json.loads(str(z["metadata"].item()));need(meta.get("status")=="PASSED", "fit status must be PASSED")
  return MuscleProfileFit(z["muscle_rms"].copy(),z["svd_components"].copy(),z["normalizer_mean"].copy(),z["normalizer_scale"].copy(),tuple(meta["source_sessions"]),meta)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fit-npz",type=Path,required=True);ap.add_argument("--carrier-pack",type=Path,required=True);ap.add_argument("--output-dir",type=Path,required=True);a=ap.parse_args()
 out=a.output_dir.resolve();need(not out.exists(),f"refuse overwrite {out}");out.mkdir(parents=True)
 fit=fit_loaded(a.fit_npz.resolve());need(tuple(fit.source_sessions)==SOURCES,"candidate fit must be source4")
 with np.load(a.carrier_pack,allow_pickle=False) as pack:
  need(set(pack.files)==set(SOURCES+("20121004","20121017","20121024")),"carrier pack roster drift")
  candidate={s:np.asarray(pack[s],np.float32) for s in SOURCES}
 need(all(x.shape==(64,DIM) for x in candidate.values()),"candidate source geometry")
 norm,basis=old_basis();need(tuple(norm["receipt"]["source_sessions"])==OLD_FIT_SOURCES,"old frozen basis source roster")
 summary={"schema":"carrier_v4_m1_source_alignment_diagnosis_v1","status":"COMPLETED","scope":{"sessions":list(SOURCES),"reads":"held-in calibration NWB only; M20 maximum","no_ho_io":True,"no_query_labels":True,"basis_policy":"old source NMF and new SVD fit loaded frozen; neither basis refit"},"inputs":{"fit_npz":str(a.fit_npz.resolve()),"fit_sha256":sha(a.fit_npz.resolve()),"carrier_pack":str(a.carrier_pack.resolve()),"carrier_pack_sha256":sha(a.carrier_pack.resolve()),"old_bank_npz":str(old_plan.BANK_NPZ),"old_bank_sha256":sha(old_plan.BANK_NPZ),"old_bank_receipt":str(old_plan.BANK_RECEIPT),"old_bank_receipt_sha256":sha(old_plan.BANK_RECEIPT)},"code_sha256":{str(Path(__file__).resolve()):sha(Path(__file__).resolve()),str(Path(__file__).parents[1]/"m1_muscle_r100_v1/carrier.py"):sha(Path(__file__).parents[1]/"m1_muscle_r100_v1/carrier.py"),str(Path(__file__).parents[1]/"m1_carrier_refinement_v1/aligned_carrier.py"):sha(Path(__file__).parents[1]/"m1_carrier_refinement_v1/aligned_carrier.py")},"sessions":{},"limitations":["Correlation is descriptive and does not establish decoder R2 causality.","Old and aligned count conventions are compared; no conclusion about which clock is semantically correct follows from this report.","No HO file, HO query value, test surface, training, or basis refit is used."]}
 arrays={}
 all_new=[];all_old=[];all_raw4=[]
 for s in SOURCES:
  path=session_path(s);need(path.is_file(),f"missing source NWB {path}")
  new20=load_aligned_support(path,emg_trial_stop=20,neural_trial_stop=20);old20=old_record(path,20)
  # Both readers select the same EMG movement rows by contract.  Check directly
  # before comparing rate clock placement.
  need(np.array_equal(new20.emg_trial_ids,old20.emg_trial_ids) and np.array_equal(new20.emg,old20.emg),f"{s}: old/new movement-row EMG mismatch")
  need(np.array_equal(new20.rate_trial_ids,old20.rate_trial_ids),f"{s}: old/new rate trial IDs mismatch")
  count_old=np.asarray(old20.rates)*BIN_SECONDS;count_new=np.asarray(new20.counts)
  new10=load_aligned_support(path,emg_trial_stop=10,neural_trial_stop=10);old10=old_record(path,10)
  need(np.array_equal(new10.emg,old10.emg) and np.array_equal(new10.emg_trial_ids,old10.emg_trial_ids),f"{s}: M10 EMG rows mismatch")
  old10syn,old10raw=old_carrier(old10,basis,norm,range(10));old05,_=old_carrier(old10,basis,norm,range(5));old510,_=old_carrier(old10,basis,norm,range(5,10));old1020,_=old_carrier(old20,basis,norm,range(10,20));aligned10syn,aligned10raw=aligned_rsyn(new10,basis,norm,0,10);aligned05,_=aligned_rsyn(new10,basis,norm,0,5);aligned510,_=aligned_rsyn(new10,basis,norm,5,10);aligned1020,_=aligned_rsyn(new20,basis,norm,10,20)
  new10syn=new_carrier(fit,new10,0,10);new05=new_carrier(fit,new10,0,5);new510=new_carrier(fit,new10,5,10);new1020=new_carrier(fit,new20,10,20)
  # Source 26/27/28 must reproduce the stored frozen old rSyn normalized bytes.
  legacy_agree=None;sealed_deploy_agree=None
  if s in OLD_FIT_SOURCES:
   legacy_agree=float(np.max(np.abs(old10syn-norm["normalized"][s])));need(legacy_agree<=1e-6,f"{s}: reconstructed old rSyn differs from frozen bank by {legacy_agree}")
  elif s=="ses-20120924":
   # The formal run obtains 24 through the sealed-basis deployment API, not
   # from a source-bank NPZ.  Bind this reconstruction to that actual path.
   deployed,_meta=old_projadd.encode_outer_carrier();sealed_deploy_agree=float(np.max(np.abs(old10syn-deployed)));need(sealed_deploy_agree<=1e-6,f"{s}: reconstruction differs from sealed-basis deployment by {sealed_deploy_agree}")
  mean_rate=np.asarray(new10.rates,float).mean(0)
  raw4=(_raw16(fit,*slice_new(new10,0,10))@fit.svd_components.T)
  pack_delta=float(np.max(np.abs(candidate[s]-new10syn)));need(pack_delta<=1e-6,f"{s}: pack/recomputed muscle carrier drift {pack_delta}")
  row={"nwb_sha256":sha(path),"movement_rows":{"emg_exact":True,"trial_ids_exact":True,"m10_emg_exact":True,"m10_trial_ids_exact":True},"counts":{"old_rms":rms(count_old),"aligned_rms":rms(count_new),"comparison":shifted_metrics(count_old,count_new,new20.rate_trial_ids,new20.rate_global_time_index,new20.timestamps)},"old_frozen_rsyn":{"m10_reconstruct_max_abs_vs_frozen":legacy_agree,"m10_reconstruct_max_abs_vs_sealedbasis_deploy":sealed_deploy_agree,"axis_l2":np.linalg.norm(old10syn,axis=0).tolist(),"axis_mean":old10syn.mean(0).tolist(),"raw_axis_vs_normalized_axis_pearson":per_axis(old10raw,old10syn),"axis_vs_mean_rate_pearson":axis_vs_vector(old10syn,mean_rate),"intercept_raw_vs_mean_rate_pearson":pear(old10raw[:,3],mean_rate),"aligned_reader_same_basis":{"normalized_axis_pearson_vs_old":per_axis(aligned10syn,old10syn),"raw_axis_pearson_vs_old":per_axis(aligned10raw,old10raw),"stability":{"first5_vs_last5":per_axis(aligned05,aligned510),"m10_vs_next10":per_axis(aligned10syn,aligned1020)}},"stability":{"first5_vs_last5":per_axis(old05,old510),"m10_vs_next10":per_axis(old10syn,old1020)}},"new_muscle":{"pack_vs_recomputed_m10_max_abs":pack_delta,"raw4_vs_pack_axis_pearson":per_axis(raw4,candidate[s]),"axis_vs_mean_rate_pearson":axis_vs_vector(new10syn,mean_rate),"stability":{"first5_vs_last5":per_axis(new05,new510),"m10_vs_next10":per_axis(new10syn,new1020)}}}
  summary["sessions"][s]=row;arrays[f"old_rsyn_m10/{s}"]=old10syn;arrays[f"aligned_rsyn_m10/{s}"]=aligned10syn;arrays[f"new_muscle_m10/{s}"]=new10syn;arrays[f"new_muscle_pack/{s}"]=candidate[s];arrays[f"mean_rate/{s}"]=mean_rate.astype(np.float64);all_new.append(candidate[s]);all_old.append(old10syn);all_raw4.append(raw4)
 stack=np.concatenate(all_new);rawstack=np.concatenate(all_raw4);std=rawstack.std(0);energy=std*std;global_scale=float(np.sqrt(np.mean(np.square(np.maximum(std,1e-6)))));normalized=(rawstack-rawstack.mean(0))/global_scale
 oldstack=np.concatenate(all_old)
 summary["candidate_svd4"]={"raw4_source4_axis_std":std.tolist(),"raw4_source4_energy_fraction":(energy/max(float(energy.sum()),1e-30)).tolist(),"effective_rank_raw4":effective_rank(rawstack),"recomputed_global_rms":global_scale,"fit_normalizer_scale":np.asarray(fit.normalizer_scale).tolist(),"effective_rank_after_global_rms":effective_rank(normalized),"normalized_pack_axis_l2":np.linalg.norm(stack,axis=0).tolist()}
 summary["old_rsyn_source4"]={"axis_l2":np.linalg.norm(oldstack,axis=0).tolist(),"axis_std":oldstack.std(0).tolist(),"note":"24 is sealed-basis deployment; 26/27/28 are frozen-bank sources."}
 np.savez_compressed(out/"diagnostic_arrays.npz",**arrays);summary["outputs"]={"arrays":"diagnostic_arrays.npz","arrays_sha256":sha(out/"diagnostic_arrays.npz")};atomic_json(out/"diagnosis.json",summary)
if __name__=="__main__":main()
