#!/usr/bin/env python3
"""Export M2-only figures from locked verified development evidence."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/diagnostics_v1/m2_verified_evidence_plots_v1'
B=ROOT/'results/rift_v1/m2_r50_joint_b_s42_formal_v1/score_receipt.json'
D=ROOT/'results/rift_v1/m2_r50_joint_d_s42_formal_v1/score_receipt.json'
REPORT=ROOT/'results/diagnostics_v1/m2_carrier_support_stability_full_v1/report.json'
SUMMARY=ROOT/'results/diagnostics_v1/m2_carrier_support_stability_full_v1/validated_summary.json'
SESSIONS=('ses-2020-10-30-Run1','ses-2020-10-30-Run2','ses-2020-11-18-Run1','ses-2020-11-19-Run1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def score(p,arm):
 x=load(p); require(x.get('schema')=='m2_rift_joint_ext4_epoch_scan_v1' and x.get('status')=='COMPLETED',f'bad score receipt {p}')
 require(x.get('arm')==arm and x.get('seed')==42 and x.get('sampler_seed')==42 and x.get('official_test_used') is False,f'wrong score binding {p}')
 e=x.get('ema_by_epoch',{}); require(set(e)=={str(i) for i in range(1,25)},f'missing 24 epochs {p}')
 for r in e.values(): require(r.get('partial') is False and set(r.get('per_session',{}))==set(SESSIONS),f'bad ext4 query {p}')
 return x
def main():
 b,d=score(B,'B_ACTIVITY_ONLY'),score(D,'D_JOINT'); require(b['cache_hashes']['ext4']==d['cache_hashes']['ext4'],'B/D ext4 query/cache drift')
 r,s=load(REPORT),load(SUMMARY); require(r.get('schema')=='m2_carrier_support_stability_v2' and r.get('status')=='COMPLETED','bad stability report')
 require(s.get('schema')=='m2_support_stability_summary_v1' and s.get('status')=='VERIFIED' and s.get('source_report_sha256')==sha(REPORT),'bad validated stability summary')
 require(set(k for k in r['sessions'] if k.startswith('ext4/'))=={f'ext4/{z}' for z in SESSIONS},'stability ext4 sessions drift')
 budgets=[8,16,25,33]; labels=['24.2%','48.5%','75.8%','100%']; curves={}
 for name,x in [('B',b),('D',d)]: curves[name]=[x['ema_by_epoch'][str(i)]['equal_session_mean'] for i in range(1,25)]
 selected={'B':b['selection'],'D':d['selection']}; require(abs(selected['B']['equal_session_mean']-.26682546439211335)<1e-12 and abs(selected['D']['equal_session_mean']-.3935032532903647)<1e-12,'selected evidence values drift')
 paired={name:{'B':b['ema_by_epoch'][str(selected['B']['epoch'])]['per_session'][name]['r2'],'D':d['ema_by_epoch'][str(selected['D']['epoch'])]['per_session'][name]['r2']} for name in SESSIONS}
 per={}; agg={}
 for budget in budgets:
  rows=[]; per[str(budget)]={}
  for name in SESSIONS:
   runs=r['sessions'][f'ext4/{name}']['runs']; use=[z for z in runs if z['budget_trials']==budget and z['valid_production_carrier']]
   fail=sum(1 for z in runs if z['budget_trials']==budget and not z['valid_production_carrier']); require(len(use)+fail==10,'expected 10 resamples/session')
   per[str(budget)][name]={'total':10,'failures':fail,'valid':len(use),'mean_relative_frobenius':float(np.mean([z['relative_frobenius_vs_m33'] for z in use])),'mean_cosine':float(np.mean([z['cosine_vs_m33'] for z in use]))}
   rows+=use
  agg[str(budget)]={'total':40,'failures':40-len(rows),'valid':len(rows),'failure_fraction':(40-len(rows))/40,'mean_relative_frobenius_constructible_only':float(np.mean([z['relative_frobenius_vs_m33'] for z in rows])),'mean_cosine_constructible_only':float(np.mean([z['cosine_vs_m33'] for z in rows]))}
  expected=s['budgets']['ext4'][str(budget)]; require(agg[str(budget)]['failures']==expected['production_failures'],'summary failure mismatch')
 OUT.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
 fig,(ax,px)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.2,1]})
 epochs=np.arange(1,25)
 for n,c in [('B','#4C78A8'),('D','#E45756')]: ax.plot(epochs,curves[n],lw=2,label=f'{n} (selected e{selected[n]["epoch"]})',color=c); ax.scatter(selected[n]['epoch'],selected[n]['equal_session_mean'],color=c,s=45,zorder=3)
 ax.set(xlabel='Training epoch',ylabel='EMA equal-session $R^2$',title='Matched RIFT development evidence (seed 42)');ax.legend(frameon=False);ax.grid(alpha=.25)
 for i,name in enumerate(SESSIONS):
  px.plot([0,1],[paired[name]['B'],paired[name]['D']],color='.55',marker='o')
  px.annotate(name.replace('ses-2020-',''),xy=(0,paired[name]['B']),xytext=(-14,(-12,12,-12,12)[i]),textcoords='offset points',ha='right',va='center',fontsize=8)
 px.set(xlim=(-.42,1.18),xticks=[0,1],xticklabels=['B: activity only\ne7','D: joint\ne17'],ylabel='Selected epoch session $R^2$',title='Same four ext4 sessions');px.grid(axis='y',alpha=.25)
 fig.text(.5,.01,'Development surface: visible ext4 only. Arms independently select the earliest maximum EMA equal-session mean; no seed variance or p-values shown.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.07,1,1))
 for ext in ('png','pdf'):fig.savefig(OUT/f'm2_rift_b42_d42_ext4.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4.8)); x=np.arange(4)
 axes[0].bar(x,[agg[str(q)]['failure_fraction'] for q in budgets],color='#D65F5F');axes[0].set(xticks=x,xticklabels=[f'M{q}\n{z}' for q,z in zip(budgets,labels)],ylabel='Failure fraction (all 40)',title='Constructibility');axes[0].text(.5,.92,'M8: 14/40 failures\n10-30-R1 4/10; 10-30-R2 4/10\n11-18-R1 3/10; 11-19-R1 3/10',transform=axes[0].transAxes,ha='center',va='top',fontsize=7);axes[0].set_ylim(0,.45)
 for ai,key,title,ylabel in [(1,'mean_relative_frobenius_constructible_only','Relative Frobenius vs M33','Mean relative Frobenius'),(2,'mean_cosine_constructible_only','Cosine vs M33','Mean cosine')]:
  ax=axes[ai]
  for j,name in enumerate(SESSIONS): ax.plot(x,[per[str(q)][name]['mean_relative_frobenius' if ai==1 else 'mean_cosine'] for q in budgets],marker='o',lw=1,label=name.replace('ses-2020-',''))
  ax.plot(x,[agg[str(q)][key] for q in budgets],color='black',lw=2.8,marker='s',label='aggregate valid subsets');ax.set(xticks=x,xticklabels=[f'M{q}\n{z}' for q,z in zip(budgets,labels)],title=title,ylabel=ylabel);ax.grid(alpha=.25)
 axes[2].legend(fontsize=7,frameon=False,loc='lower right');fig.text(.5,.01,'ext4 only: 4 sessions × 10 resamples = 40/budget. Means use constructible subsets only; failures are not zeros. M33 is the reference.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.07,1,1))
 for ext in ('png','pdf'):fig.savefig(OUT/f'm2_support_stability_ext4.{ext}',dpi=240,bbox_inches='tight')
 plt.close(fig)
 data={'schema':'m2_verified_evidence_figure_data_v1','sources':{str(p):sha(p) for p in (B,D,REPORT,SUMMARY)},'rift':{'seed':42,'surface':'ext4 visible development','curves':curves,'selection':selected,'paired_selected_session_r2':paired},'support_stability':{'surface':'ext4','budgets':budgets,'support_fraction_labels':labels,'aggregate':agg,'per_session':per},'notes':['No timing arrays used; calibration timing correction is deliberately excluded.','No unmeasured-seed variance, confidence interval, or p-value is plotted.']}
 (OUT/'figure_data.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');(OUT/'README.md').write_text('# M2 verified evidence figures\n\nTwo exportable figures use only the four stated source files. Source SHA-256 values and every plotted array are in `figure_data.json`. The support figure uses ext4 only, 40 resamples per budget; valid-subset means do not convert construction failures to zero. No timing data are used.\n')
if __name__=='__main__': main()
