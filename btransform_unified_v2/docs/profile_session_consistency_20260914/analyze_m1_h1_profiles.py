#!/usr/bin/env python3
"""Read-only fixed-column profile consistency for official M1 and H1 Full banks."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path('/home/xinyuan/Work_host/SPINT'); SEED=20260914; N=200
FEATURES=['axis_1','axis_2','axis_3','axis_4']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def corr(a,b):
 a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
 return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>0 and b.std()>0 else float('nan')
def q(x):
 x=np.asarray(x,float);return {'n':int(len(x)),'median':float(np.median(x)),'q25':float(np.quantile(x,.25)),'q75':float(np.quantile(x,.75)),'mean':float(np.mean(x)),'min':float(np.min(x)),'max':float(np.max(x))}
def wr(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load(dataset):
 if dataset=='m1':
  r=ROOT/'btransform_unified_v2/results/m1_muscle_r100_v1/carrier_official4'; z=np.load(r/'carrier_pack.npz'); ts={k:np.asarray(z[k],np.float32) for k in z.files}; meta=json.loads((r/'carrier_pack.json').read_text());
  dates={k:datetime.strptime(k.removeprefix('ses-'),'%Y%m%d').date() for k in ts}; source={'ses-20120924','ses-20120926','ses-20120927','ses-20120928'}
  prov={'input':'carrier_pack.npz','input_sha256':sha(r/'carrier_pack.npz'),'fit':'fit.npz','fit_sha256':sha(r/'fit.npz'),'metadata':'carrier_pack.json','metadata_sha256':sha(r/'carrier_pack.json'),'definition':'M1 official Full 582413 carrier profile: frozen source-only muscle-response16 SVD4, then fixed source normalizer; fixed input columns 0..63.'}
 elif dataset=='h1':
  r=ROOT/'btransform_unified_v2/results/h1_signed_state_r300_v1/banks_official13_20260909';z=np.load(r/'banks_27.npz');ts={k[2:]:np.asarray(z[k],np.float32) for k in z.files if k.startswith('T/')}; rec=json.loads((r/'receipt.json').read_text());
  dates={k:datetime.strptime(rec['tags'][k]['session'][4:12],'%Y%m%d').date() for k in ts}; source={k for k in ts if int(k.split('_')[0][1:])<6}
  prov={'input':'banks_27.npz','input_sha256':sha(r/'banks_27.npz'),'fit':'signed_state14_plan.npz','fit_sha256':sha(r/'signed_state14_plan.npz'),'metadata':'receipt.json','metadata_sha256':sha(r/'receipt.json'),'definition':'H1 official Full 582241 signed-state14 profile: source-plan projection/normalization; fixed columns 0..175.'}
 else:raise ValueError(dataset)
 return ts,dates,source,prov
def main():
 ap=argparse.ArgumentParser();ap.add_argument('dataset',choices=['m1','h1']);a=ap.parse_args();ts,dates,source,prov=load(a.dataset);out=ROOT/'btransform_unified_v2/docs/profile_session_consistency_20260914'/a.dataset;out.mkdir(exist_ok=True)
 keys=sorted(ts,key=lambda k:(dates[k],k)); nrow=next(iter(ts.values())).shape[0]; assert all(x.shape==(nrow,4) and np.isfinite(x).all() for x in ts.values())
 sess=[{'order':i,'tag':k,'date':str(dates[k]),'source_or_held_out':'source_held_in' if k in source else 'held_out','n_fixed_columns':nrow,'profile_shape':f'{nrow}x4'} for i,k in enumerate(keys)]
 prof=[]
 for s in sess:
  for i in range(nrow):
   for j,name in enumerate(FEATURES):prof.append({'tag':s['tag'],'date':s['date'],'source_or_held_out':s['source_or_held_out'],'fixed_column':i,'feature_index':j,'feature':name,'T_source_normalized':float(ts[s['tag']][i,j])})
 rng=np.random.default_rng(SEED); prs=[]; nul=[];R=np.eye(len(keys));E=np.zeros_like(R)
 for i,l in enumerate(keys):
  for j,r in enumerate(keys[i+1:],i+1):
   x,y=ts[l],ts[r];act=corr(x,y);rm=float(np.sqrt(np.mean((x-y)**2))); ns=[]
   for rep in range(N):
    val=corr(x,y[rng.permutation(nrow)]);ns.append(val);nul.append({'left_tag':l,'right_tag':r,'replicate':rep,'r_fixed_column_row_shuffle':val})
   gap=abs((dates[l]-dates[r]).days);row={'left_tag':l,'right_tag':r,'left_date':str(dates[l]),'right_date':str(dates[r]),'day_gap':gap,'pair_group':'same_date_different_recording' if gap==0 else 'different_date','n_fixed_columns':nrow,'flattened_r_actual':act,'rmse_fixed_source_normalized_units':rm,'null_r_mean':float(np.mean(ns)),'null_r_sd':float(np.std(ns,ddof=1))}
   row.update({f'r_{FEATURES[d]}':corr(x[:,d],y[:,d]) for d in range(4)});prs.append(row);R[i,j]=R[j,i]=act;E[i,j]=E[j,i]=rm
 wr(out/'sessions.csv',sess);wr(out/'profile_rows.csv',prof);wr(out/'pairs.csv',prs);wr(out/'shuffle_null.csv',nul)
 actual=np.array([r['flattened_r_actual'] for r in prs]);rmse=np.array([r['rmse_fixed_source_normalized_units'] for r in prs]);null=np.array([r['r_fixed_column_row_shuffle'] for r in nul]); groups={}
 for g in sorted(set(r['pair_group'] for r in prs)):
  z=[r for r in prs if r['pair_group']==g];groups[g]={'flattened_r_actual':q([r['flattened_r_actual'] for r in z]),'rmse_fixed_source_normalized_units':q([r['rmse_fixed_source_normalized_units'] for r in z])}
 summary={'dataset':a.dataset,'n_sessions':len(keys),'n_pairs':len(prs),'fixed_column_contract':prov['definition'],'all_pairs':{'flattened_r_actual':q(actual),'rmse_fixed_source_normalized_units':q(rmse)},'shuffle_null':q(null),'pair_groups':groups,'feature_correlations':{f:q([r[f'r_{f}'] for r in prs]) for f in FEATURES},'caveat':'Fixed-column engineering correspondence only; this does not establish cross-session biological single-unit identity. No query labels or outcome scores were read.'}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'input_sources.json').write_text(json.dumps(prov,indent=2)+'\n')
 labels=[str(dates[k])[5:]+'\n'+k.split('_')[-1] for k in keys];pdf=out/f'{a.dataset}_profile_session_consistency.pdf'
 with PdfPages(pdf) as pages:
  fig,ax=plt.subplots(figsize=(11,9),constrained_layout=True);im=ax.imshow(R,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(keys)),labels,rotation=60,ha='right',fontsize=7);ax.set_yticks(range(len(keys)),labels,fontsize=7);fig.colorbar(im,ax=ax,label='Flattened Pearson r');ax.set_title(f'{a.dataset.upper()} official Full: fixed-column four-dimensional T profiles');pages.savefig(fig,dpi=200);fig.savefig(out/f'{a.dataset}_profile_session_consistency.png',dpi=240);plt.close(fig)
  fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True);axs[0].scatter([r['day_gap'] for r in prs],actual,s=25);axs[0].axhline(0,color='k',lw=.6);axs[0].set(xlabel='Calendar day gap',ylabel='Flattened r');axs[1].hist(null,bins=40,alpha=.6,label='row shuffle null');axs[1].hist(actual,bins=min(20,len(actual)),alpha=.7,label='actual');axs[1].legend();axs[1].set(xlabel='Flattened r',ylabel='count');pages.savefig(fig,dpi=200);plt.close(fig)
 print(json.dumps({'dataset':a.dataset,'r_median':summary['all_pairs']['flattened_r_actual']['median'],'rmse_median':summary['all_pairs']['rmse_fixed_source_normalized_units']['median'],'n_pairs':len(prs)}))
if __name__=='__main__':main()
