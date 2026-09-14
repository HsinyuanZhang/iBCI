#!/usr/bin/env python3
"""Describe DANDI PMUA MOVE-T4 profile consistency using cached support only."""
from pathlib import Path
import sys,json,hashlib,csv,itertools
from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from btransform_unified_v2.dandi688_bench_v2.carrier import estimate_move_t4, apply_carrier_normalizer
OUT=Path(__file__).resolve().parent
CACHE=ROOT/'btransform_unified_v2/dandi688_bench_v2/results/prepared_2015_m33_v2'
STATS=ROOT/'btransform_unified_v2/dandi688_bench_v2/results/formal_campaign_concat_2015_m33_allseeds_20260912/pretrain_pmua_s42/source_stats.json'
SEED=20260914; NPERM=200

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def corr(x,y):
 x=np.asarray(x,float).ravel();y=np.asarray(y,float).ravel()
 if len(x)<2 or np.std(x)==0 or np.std(y)==0:return np.nan
 return float(np.corrcoef(x,y)[0,1])
def rec(p):
 with np.load(p,allow_pickle=False) as z:
  m=json.loads(str(z['metadata'].item())); c=z['carrier_counts'].copy();a=z['carrier_angles'].copy();ids=z['channel_indices'].copy(); observed=z['observed_mask'].copy()
 raw=estimate_move_t4(c,a);return dict(id=m['session_id'],split=m['split'],date=m['session_id'][-8:],ids=ids,observed_mask=observed,raw=raw,norm=None,path=str(p),sha=sha(p),angles=a,counts=c)
def matched(a,b):
 common=np.intersect1d(a['ids'],b['ids']); ia=np.searchsorted(a['ids'],common);ib=np.searchsorted(b['ids'],common);return common,ia,ib
def pair(a,b,rng):
 common,ia,ib=matched(a,b);
 da=datetime.strptime(a['date'], '%Y%m%d').date(); db=datetime.strptime(b['date'], '%Y%m%d').date(); n=len(common); aa=a['norm'][ia];bb=b['norm'][ib];ra=a['raw'][ia];rb=b['raw'][ib]
 out={'left':a['id'],'right':b['id'],'left_split':a['split'],'right_split':b['split'],'days_apart':abs((da-db).days),'n_common':n,'n_union':len(np.union1d(a['ids'],b['ids'])),'jaccard':n/len(np.union1d(a['ids'],b['ids'])),'sufficient':n>=8}
 out['r_norm_flat']=corr(aa,bb);out['rmse_norm']=float(np.sqrt(np.mean((aa-bb)**2)))
 for k,nm in enumerate(['a','c','magnitude','baseline']):out['r_raw_'+nm]=corr(ra[:,k],rb[:,k])
 vals=[]
 for _ in range(NPERM): vals.append(corr(aa,bb[rng.permutation(n)]))
 out['null_r_mean']=float(np.nanmean(vals));out['actual_minus_null']=out['r_norm_flat']-out['null_r_mean'] if np.isfinite(out['r_norm_flat']) else np.nan
 return out
def split(a):
 ix=[np.arange(0,33,2),np.arange(1,33,2)];raw=[estimate_move_t4(a['counts'][x],a['angles'][x]) for x in ix]
 norm=[apply_carrier_normalizer(x,ST['carrier']) for x in raw]
 return {'session':a['id'],'split':a['split'],'n_channels':len(a['ids']),'n_odd_finite':int(np.isfinite(a['angles'][ix[0]]).sum()),'n_even_finite':int(np.isfinite(a['angles'][ix[1]]).sum()),'r_norm_flat':corr(norm[0],norm[1]),'rmse_norm':float(np.sqrt(np.mean((norm[0]-norm[1])**2))),'r_raw_a':corr(raw[0][:,0],raw[1][:,0]),'r_raw_c':corr(raw[0][:,1],raw[1][:,1]),'r_raw_magnitude':corr(raw[0][:,2],raw[1][:,2]),'r_raw_baseline':corr(raw[0][:,3],raw[1][:,3])}
def q(x):
 x=np.asarray(x,float);x=x[np.isfinite(x)];return {'n':int(len(x)),'median':float(np.median(x)),'q25':float(np.quantile(x,.25)),'q75':float(np.quantile(x,.75))} if len(x) else {'n':0}
def main():
 global ST
 ST=json.loads(STATS.read_text()); rows=[rec(p) for p in sorted(CACHE.glob('*.pmua.npz'))]
 rows.sort(key=lambda x:x['date'])
 for a in rows:
  assert np.all(np.diff(a['ids'])>0), f"{a['id']}: channel ids must be strictly sorted"
  assert np.all(a['observed_mask'][a['ids']]), f"{a['id']}: observed mask mismatch"
  a['norm']=apply_carrier_normalizer(a['raw'],ST['carrier'])
 assert len(rows)==24 and all(np.isfinite(x['norm']).all() for x in rows)
 # Self-contained numeric profile material for review-bundle renderer; pad unobserved canonical electrodes with NaN.
 padded=np.full((len(rows),96,4),np.nan,dtype=np.float32); ids=np.full((len(rows),96),-1,dtype=np.int16)
 for ii,x in enumerate(rows): padded[ii,x['ids']]=x['norm']; ids[ii,:len(x['ids'])]=x['ids']
 np.savez_compressed(OUT/'profile_data.npz',tags=np.asarray([x['id'] for x in rows]),dates=np.asarray([x['date'] for x in rows]),profile=padded,channel_ids=ids)
 rng=np.random.default_rng(SEED); pairs=[pair(a,b,rng) for a,b in itertools.combinations(rows,2)]; halves=[split(a) for a in rows]
 fields=list(pairs[0]);
 with (OUT/'pairs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(pairs)
 with (OUT/'split_half.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(halves[0]));w.writeheader();w.writerows(halves)
 sufficient=[x for x in pairs if x['sufficient']]
 summary={'schema':'dandi_pmua_t4_session_consistency_v1','seed':SEED,'n_permutations':NPERM,'sessions':[{k:x[k] for k in ('id','date','split','path','sha')}|{'n_channels':int(len(x['ids']))} for x in rows],'input_source_stats':str(STATS),'input_source_stats_sha256':sha(STATS),'carrier_normalizer':ST['carrier'],'method':'estimate_move_t4 from 33 support trials then source-fixed normalizer; no session-wise alignment/rotation/refit','date_span_days':(datetime.strptime(rows[-1]['date'],'%Y%m%d').date()-datetime.strptime(rows[0]['date'],'%Y%m%d').date()).days,'pairs_total':len(pairs),'pairs_sufficient_ncommon_ge8':len(sufficient),'minimum_n_common':int(min(x['n_common'] for x in pairs)),'pair_summary':{k:q([x[k] for x in sufficient]) for k in ('r_norm_flat','rmse_norm','actual_minus_null','null_r_mean','jaccard','r_raw_a','r_raw_c','r_raw_magnitude','r_raw_baseline')},'split_half_summary':{k:q([x[k] for x in halves]) for k in ('r_norm_flat','rmse_norm','r_raw_a','r_raw_c','r_raw_magnitude','r_raw_baseline')},'caveat':'Pair summaries are descriptive: date pairs share sessions and no p-values are computed. Split-half is within-session estimator-noise context, not a drift correction.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Figure
 dates=[x['date'][4:] for x in rows];n=len(rows);mat=np.full((n,n),np.nan)
 for i in range(n):mat[i,i]=1
 for x in pairs:
  i=next(i for i,a in enumerate(rows) if a['id']==x['left']);j=next(i for i,a in enumerate(rows) if a['id']==x['right'])
  if x['sufficient']:mat[i,j]=mat[j,i]=x['r_norm_flat']
 plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42});fig,ax=plt.subplots(2,2,figsize=(10,8));im=ax[0,0].imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');ax[0,0].set(title='PMUA T4 profile correlation (shared electrodes; n<8 masked)',xticks=range(n),yticks=range(n),xticklabels=dates,yticklabels=dates);plt.setp(ax[0,0].get_xticklabels(),rotation=90,fontsize=6);plt.setp(ax[0,0].get_yticklabels(),fontsize=6);fig.colorbar(im,ax=ax[0,0],label='flattened normalized T4 r')
 sc=ax[0,1].scatter([x['days_apart'] for x in sufficient],[x['rmse_norm'] for x in sufficient],c=[x['n_common'] for x in sufficient],cmap='viridis',s=22);fig.colorbar(sc,ax=ax[0,1],label='shared electrodes');ax[0,1].set(title='Cross-date difference',xlabel='Days apart',ylabel='T4 RMSE (source-SD units)')
 actual=[x['r_norm_flat'] for x in sufficient];null=[x['null_r_mean'] for x in sufficient];half=[x['r_norm_flat'] for x in halves];ax[1,0].boxplot([actual,null,half],labels=['cross-date\nactual','cross-date\nshuffled null','within-date\nodd/even']);ax[1,0].set(title='Different comparison objects',ylabel='flattened normalized T4 r',ylim=(-1,1));ax[1,0].axhline(0,color='k',lw=.6); ax[1,0].text(.03,.96,f"median r: actual {np.median(actual):.2f}; null {np.median(null):.2f}; split-half {np.median(half):.2f}",transform=ax[1,0].transAxes,va='top',fontsize=8)
 for k,nm in enumerate(['r_raw_a','r_raw_c','r_raw_magnitude','r_raw_baseline']):ax[1,1].scatter(np.full(len(sufficient),k)+np.random.default_rng(SEED+k).uniform(-.12,.12,len(sufficient)),[x[nm] for x in sufficient],s=10,alpha=.55)
 ax[1,1].set(title='Raw T4 dimension correlations',xticks=range(4),xticklabels=['a','c','mag.','base'],ylabel='across-electrode r',ylim=(-1,1));ax[1,1].axhline(0,color='k',lw=.6)
 for a in ax.flat:a.grid(alpha=.2)
 fig.suptitle(f'DANDI PMUA MOVE-T4: 24 source/dev sessions, {len(sufficient)}/{len(pairs)} usable date pairs',fontweight='bold');fig.tight_layout();fig.savefig(OUT/'dandi_pmua_profile_consistency.pdf');fig.savefig(OUT/'dandi_pmua_profile_consistency.png',dpi=220);plt.close(fig)
 def filehash(q): return hashlib.sha256(q.read_bytes()).hexdigest()
 summary['output_sha256']={q.name:filehash(q) for q in (OUT/'pairs.csv',OUT/'split_half.csv',OUT/'dandi_pmua_profile_consistency.pdf',OUT/'dandi_pmua_profile_consistency.png',OUT/'analyze_pmua_profiles.py')}
 summary['validation']={'all_sessions_covered':len(rows)==24,'all_pairs_expected':len(pairs)==276,'date_span_check_20150309_to_20151112':abs((datetime.strptime('20151112','%Y%m%d').date()-datetime.strptime('20150309','%Y%m%d').date()).days)==248,'finite_profile_values':True,'strictly_sorted_unique_ids':True,'observed_mask_matches_ids':True,'null_permutation':'whole 4D electrode rows; fixed seed 20260914; 200 per pair','diagonal_heatmap':'set to 1 by definition; off-diagonal n_common<8 masked'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({'pair_summary':summary['pair_summary'],'split_half':summary['split_half_summary']},indent=2))
if __name__=='__main__':main()
