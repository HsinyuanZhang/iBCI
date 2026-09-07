"""65-row same-surface controls, contrasts, and sole external promotion gate."""
from __future__ import annotations
import re
import math
import numpy as np
from . import plan
class LawError(ValueError): pass
def _need(ok,msg):
    if not ok: raise LawError(msg)
def _boot(values):
    x=np.asarray(values,dtype=np.float64); rng=np.random.default_rng(plan.BOOTSTRAP_SEED)
    draw=x[rng.integers(0,len(x),size=(plan.BOOTSTRAP_RESAMPLES,len(x)))].mean(1)
    return {"seed":42,"resamples":10000,"ci95":[float(np.quantile(draw,.025)),float(np.quantile(draw,.975))]}
def validate_rows(rows):
    _need(len(rows)==65,"V2 row cardinality drift")
    keys=[]; canonical=[]
    for surface in plan.SURFACES:
        sessions=sorted({r['session'] for r in rows if r['surface']==surface}); _need(len(sessions)==plan.ROSTER_SIZES[surface],"V2 roster drift")
        for session in sessions:
            canonical.extend((surface,session,law,system) for system,law in (
                ('NATIVE-POOLED','FIXED30'),('APFG-ZERO','FIXED30'),('APFG-ZERO','UNCAPPED'),
                ('APFG-LEARNED','FIXED30'),('APFG-LEARNED','UNCAPPED')))
            group=[r for r in rows if r['surface']==surface and r['session']==session]
            _need(len(group)==5,"V2 five-row session topology drift")
            _need([(row['system'],row['memory_law']) for row in group] == [
                ('NATIVE-POOLED','FIXED30'),('APFG-ZERO','FIXED30'),('APFG-ZERO','UNCAPPED'),
                ('APFG-LEARNED','FIXED30'),('APFG-LEARNED','UNCAPPED')], 'V2 system/law topology drift')
            native=[r for r in group if r['system']=='NATIVE-POOLED']; _need(len(native)==1 and native[0]['memory_law']=='FIXED30',"V2 native law drift")
            for field in ('input_authority_key','target_sha256','query_starts_sha256','window_count'):_need(len({str(r[field]) for r in group})==1,"V2 same-input drift")
            z=[r for r in group if r['system']=='APFG-ZERO' and r['memory_law']=='FIXED30'][0]
            for field in ('prediction_sha256','target_sha256','query_starts_sha256','window_count','r2'):_need(z[field]==native[0][field],f"V2 native-zero sentinel drift: {field}")
            _need(native[0]['model_state_before_sha256']==native[0]['model_state_after_sha256'], 'V2 native state mutation drift')
            _need(z['model_state_before_sha256']==z['model_state_after_sha256'], 'V2 zero state mutation drift')
            _need(all(row.get('native_substate_sha256')==plan.SELECTED_STUDENT_STATE_SHA256 for row in group),
                  'V2 selected native-substate drift')
            _need(z.get('alpha_exact_positive_zero') is True, 'V2 zero alpha sentinel drift')
            _need(all(row.get('learned_refit_alpha') == float(plan.REFIT_ALPHA)
                      for row in group if row['system']=='APFG-LEARNED'),
                  'V2 learned refit alpha drift')
            keys.extend((r['surface'],r['session'],r['memory_law'],r['system']) for r in group)
            for r in group:
                _need(all(isinstance(r[k],str) and re.fullmatch(r'[0-9a-f]{64}',r[k]) for k in ('prediction_sha256','target_sha256','query_starts_sha256','model_state_before_sha256','model_state_after_sha256','native_substate_sha256')),"V2 digest drift")
                _need(math.isfinite(float(r['r2'])) and int(r.get('parameter_updates',-1))==0 and int(r.get('target_updates',-1))==0,
                      'V2 target update/R2 drift')
                _need(r['model_state_before_sha256']==r['model_state_after_sha256'], 'V2 per-row model mutation drift')
    _need(len(set(keys))==65,"V2 duplicate rows")
    _need(keys==canonical,"V2 canonical row order drift")
def recompute(rows):
    validate_rows(rows); out={}
    specs=(("memory_only","APFG-ZERO","UNCAPPED","APFG-ZERO","FIXED30"),("bounded_gate","APFG-LEARNED","FIXED30","APFG-ZERO","FIXED30"),("uncapped_gate","APFG-LEARNED","UNCAPPED","APFG-ZERO","UNCAPPED"),("total_deployed","APFG-LEARNED","UNCAPPED","NATIVE-POOLED","FIXED30"))
    for surface in plan.SURFACES:
      for name,asys,alaw,bsys,blaw in specs:
        a={r['session']:float(r['r2']) for r in rows if r['surface']==surface and r['system']==asys and r['memory_law']==alaw}; b={r['session']:float(r['r2']) for r in rows if r['surface']==surface and r['system']==bsys and r['memory_law']==blaw}; _need(set(a)==set(b),"V2 paired contrast roster drift")
        d={s:a[s]-b[s] for s in sorted(a)}; x=list(d.values()); out[f'{surface}|{name}']={"per_session_delta":d,"mean_delta":float(np.mean(x)),"median_delta":float(np.median(x)),"positive_sessions":int(np.sum(np.asarray(x)>0)),"bootstrap":_boot(x)}
    e=out['external_post30_local|total_deployed']; return {"contrasts":out,"promotion_gate":{"mean_threshold":.010,"positive_threshold":4,"mean_delta":e['mean_delta'],"positive_sessions":e['positive_sessions'],"passed":e['mean_delta']>=.010 and e['positive_sessions']>=4}}
