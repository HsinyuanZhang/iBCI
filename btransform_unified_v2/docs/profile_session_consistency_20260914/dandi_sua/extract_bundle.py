#!/usr/bin/env python3
"""Extract the DANDI 2015 SUA MOVE-T4 distribution-profile bundle.

This is the only repo-dependent step.  It reads source/development prepared
SUA archives and the frozen source normalizer, never query labels, final data,
or model weights.  `plot.py` is self-contained and only reads this directory.
"""
from __future__ import annotations
import csv, hashlib, json, shutil, sys
from datetime import datetime
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]  # btransform_unified_v2
WS=ROOT.parent
PKG=ROOT/'dandi688_bench_v2'
sys.path.insert(0,str(ROOT))
from dandi688_bench_v2.carrier import estimate_move_t4, apply_carrier_normalizer

PREP=PKG/'results/prepared_2015_m33_v2'
RECEIPT=PREP/'prepared_receipt.json'
STATS=PKG/'results/formal_campaign_concat_2015_m33_allseeds_20260912/pretrain_sua_s42/source_stats.json'
OUT=HERE
SEED=20260914
MAX_SOURCE_ROWS=2048

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def jsdump(p:Path,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def kernel(a,b,ell):
 d=a[:,None,:]-b[None,:,:]
 return np.exp(-np.sum(d*d,axis=-1)/(2*ell*ell))
def sim(a,b,ell):
 kxy=kernel(a,b,ell).mean();kxx=kernel(a,a,ell).mean();kyy=kernel(b,b,ell).mean()
 s=float(kxy/np.sqrt(kxx*kyy));m=float(kxx+kyy-2*kxy)
 return s,max(0.0,m),m

def main():
 receipt=json.loads(RECEIPT.read_text()); stats=json.loads(STATS.read_text())
 if stats['representation']!='sua' or stats['status']!='FORMAL':raise RuntimeError('wrong stats')
 wanted=[]
 for sid,row in receipt['sessions'].items():
  if row.get('split') in ('train','dev'): wanted.append((sid,row['split']))
 wanted=sorted(wanted)
 if len(wanted)!=24 or sum(s=='train' for _,s in wanted)!=18 or sum(s=='dev' for _,s in wanted)!=6:raise RuntimeError('expected 18+6')
 arrays={}; rows=[]; norm={}; full={}; odd={}; even={}; raw={}; rawodd={}; raweven={}
 provenance={}
 for sid,split in wanted:
  p=PREP/(sid+'.sua.npz')
  with np.load(p,allow_pickle=False) as z:
   counts=np.asarray(z['carrier_counts'],np.float32); angles=np.asarray(z['carrier_angles'],np.float64)
   ids=np.asarray(z['channel_indices'],np.int64); observed=np.asarray(z['observed_mask'],bool)
  if counts.shape!=(33,len(ids)) or angles.shape!=(33,) or not np.all(observed[ids]):raise RuntimeError(sid+' archive geometry')
  r=estimate_move_t4(counts,angles)
  ro=estimate_move_t4(counts[::2],angles[::2]); re=estimate_move_t4(counts[1::2],angles[1::2])
  x=apply_carrier_normalizer(r,stats['carrier']); xo=apply_carrier_normalizer(ro,stats['carrier']); xe=apply_carrier_normalizer(re,stats['carrier'])
  for name,v in [('raw',r),('raw_odd',ro),('raw_even',re),('norm',x),('norm_odd',xo),('norm_even',xe)]:
   if v.shape!=(len(ids),4) or not np.isfinite(v).all():raise RuntimeError(sid+' '+name)
  raw[sid],rawodd[sid],raweven[sid]=r,ro,re
  norm[sid]=x
  full[sid],odd[sid],even[sid]=x.astype(np.float64),xo.astype(np.float64),xe.astype(np.float64)
  date=sid[-8:]
  datetime.strptime(date,'%Y%m%d')
  arrays.update({f'raw/{sid}':r.astype(np.float64),f'raw_odd/{sid}':ro.astype(np.float64),f'raw_even/{sid}':re.astype(np.float64),f'norm/{sid}':x.astype(np.float64),f'full/{sid}':x.astype(np.float64),f'norm_odd/{sid}':xo.astype(np.float64),f'norm_even/{sid}':xe.astype(np.float64),f'row_ids/{sid}':ids,f'observed_mask/{sid}':observed})
  rows += [{'session':sid,'date':date,'split':split,'local_row':int(i),'channel_index':int(c)} for i,c in enumerate(ids)]
  provenance[sid]={'split':split,'date':date,'prepared_archive':p.name,'prepared_archive_sha256':sha(p),'nunits':len(ids),'channel_indices_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'observed_mask_sha256':hashlib.sha256(observed.tobytes()).hexdigest()}
 # fixed source-only bandwidth: all source rows; deterministic capped sample if needed
 source_rows=np.concatenate([full[s] for s,sp in wanted if sp=='train'])
 rng=np.random.default_rng(SEED); sampled=source_rows if len(source_rows)<=MAX_SOURCE_ROWS else source_rows[rng.choice(len(source_rows),MAX_SOURCE_ROWS,replace=False)]
 dd=np.linalg.norm(sampled[:,None,:]-sampled[None,:,:],axis=-1); vals=dd[np.triu_indices(len(sampled),1)]; vals=vals[vals>0]
 if not len(vals):raise RuntimeError('no nonzero source distances')
 ell=float(np.median(vals))
 # pair results
 pairrows=[]; clip_count=0
 for i,(a,spa) in enumerate(wanted):
  for b,spb in wanted[i+1:]:
   s,m,unclip=sim(full[a],full[b],ell); clip_count += int(unclip<0)
   da=datetime.strptime(a[-8:],'%Y%m%d');db=datetime.strptime(b[-8:],'%Y%m%d')
   pairrows.append({'session_a':a,'session_b':b,'date_a':a[-8:],'date_b':b[-8:],'split_a':spa,'split_b':spb,'days':abs((db-da).days),'nunits_a':len(full[a]),'nunits_b':len(full[b]),'kernel_cosine_similarity':s,'mmd2':m,'mmd2_unclipped':unclip})
 splitrows=[]
 for s,sp in wanted:
  q,m,u=sim(odd[s],even[s],ell); clip_count+=int(u<0)
  splitrows.append({'session':s,'date':s[-8:],'split':sp,'nunits':len(full[s]),'odd_trials':17,'even_trials':16,'kernel_cosine_similarity':q,'mmd2':m,'mmd2_unclipped':u})
 np.savez_compressed(OUT/'profiles.npz',**arrays)
 with (OUT/'rows.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (OUT/'pairs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(pairrows[0]));w.writeheader();w.writerows(pairrows)
 with (OUT/'split_half.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(splitrows[0]));w.writeheader();w.writerows(splitrows)
 summary={'schema':'dandi_sua_move_t4_distribution_consistency_v1','status':'COMPLETED','sessions':24,'source_sessions':18,'development_sessions':6,'pairs':len(pairrows),'rows':len(rows),'bandwidth':{'ell':ell,'rule':'median of nonzero Euclidean distances among all normalized source profile rows; source only','source_rows_total':len(source_rows),'source_rows_sampled':len(sampled),'fixed_seed':SEED},'kernel':'S=mean k(X,Y)/sqrt(mean k(X,X)*mean k(Y,Y)); k=exp(-||x-y||^2/(2 ell^2))','mmd2':'mean k(X,X)+mean k(Y,Y)-2 mean k(X,Y); negative floating-point values clipped to zero','mmd2_negative_clip_count':clip_count,'pair_similarity_mean':float(np.mean([x['kernel_cosine_similarity'] for x in pairrows])),'pair_similarity_median':float(np.median([x['kernel_cosine_similarity'] for x in pairrows])),'split_half_similarity_mean':float(np.mean([x['kernel_cosine_similarity'] for x in splitrows])),'split_half_similarity_median':float(np.median([x['kernel_cosine_similarity'] for x in splitrows])),'provenance':{'prepared_receipt_sha256':sha(RECEIPT),'source_stats_sha256_file':sha(STATS),'source_stats_declared_sha256':stats['sha256'],'carrier_normalizer':{'mean':stats['carrier']['mean'],'std':stats['carrier']['std']},'prepared_root':'btransform_unified_v2/dandi688_bench_v2/results/prepared_2015_m33_v2','source_stats':'formal_campaign_concat_2015_m33_allseeds_20260912/pretrain_sua_s42/source_stats.json','sessions':provenance},'limitations':['Profiles are empirical distributions over each session\'s local SUA rows. They do not assert cross-day unit identity.','Kernel distribution similarity is bandwidth-dependent, is not rowwise Pearson correlation, and is not comparable numerically with rowwise r from other datasets. Pair values are dependent descriptive summaries; no p-value is reported.','The extraction reads carrier_counts, carrier_angles, local channel_indices, and observed_mask from source/development prepared archives. It reads no query labels, final data, or model weights.','Odd/even split-half is a measurement-noise reference, not a drift correction.']}
 jsdump(OUT/'summary.json',summary)
 jsdump(OUT/'provenance.json',summary['provenance'])
 print(json.dumps({k:summary[k] for k in ('sessions','pairs','rows','bandwidth','pair_similarity_mean','pair_similarity_median','split_half_similarity_mean','split_half_similarity_median')},indent=2))
if __name__=='__main__':main()
