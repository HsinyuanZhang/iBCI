"""Selected-EMA complete source replay through V3, bound to independent audit."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .runtime import BankBatch, HeterogeneousCurrentQueryStream

ROOT=Path("/home/xinyuan/Work_host/SPINT/tfpd_exploration/results/decoder_validation_v2/20260905_190000/m1")
POST=ROOT/"formal_formal12_chron80_v2_current_query_postscore.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def r2(p,y):
    p,y=np.asarray(p,np.float64),np.asarray(y,np.float64); return float(1-np.square(p-y).sum()/np.square(y-y.mean(0)).sum())

def run(output:Path,device:str):
    from tfpd_exploration.src.decoder_validation_v2.audit_predictions import audit_m1
    from tfpd_exploration.src.m1_optimized_v2 import plan
    from tfpd_exploration.src.m1_optimized_v2.source_dev import _model
    if output.exists() or output.with_suffix('.npz').exists(): raise FileExistsError(output)
    post=json.loads(POST.read_text()); row=post['selected']
    state,reference=Path(row['plain_ema_model_state']),Path(row['prediction_export'])
    if sha(state)!=row['plain_ema_model_state_sha256'] or sha(reference)!=row['prediction_export_sha256']: raise RuntimeError('sealed selected authority drift')
    reference_audit=audit_m1(reference,checkpoint=Path(row['checkpoint']))
    with np.load(reference,allow_pickle=False) as z: export={k:z[k].copy() for k in z.files}
    cache_path=plan.RESULT_ROOT/'m1_optimized_v2_source_runtime_cache.npz'; receipt=json.loads(cache_path.with_suffix('.receipt.json').read_text())
    if sha(cache_path)!=receipt['npz_sha256']: raise RuntimeError('cache digest drift')
    prov_path=plan.RESULT_ROOT/'rSyn3-refit-v1.source-only.provenance-supplement.npz'; prov_receipt=json.loads(prov_path.with_suffix('.receipt.json').read_text())
    if sha(prov_path)!=prov_receipt['supplement_npz_sha256']: raise RuntimeError('unit provenance digest drift')
    model=_model('current_query'); model.load_state_dict(torch.load(state,map_location='cpu',weights_only=False),strict=True); model.to(device).eval()
    for x in model.parameters(): x.requires_grad_(False)
    offline=export['prediction'].copy(); prediction=np.empty_like(offline); checks={}
    with np.load(cache_path,allow_pickle=False) as cache, np.load(prov_path,allow_pickle=False) as provenance:
      for name in plan.SOURCE_SESSIONS:
        ids=np.flatnonzero(export['session']==name); ids=ids[np.argsort(export['window_start'][ids],kind='stable')]; starts=export['window_start'][ids].astype(np.int64)
        raw=np.asarray(cache[f'raw_neural/{name}'],dtype=np.float32)
        if not np.all(raw[:99]==0): raise RuntimeError('expected 99-bin source prefix')
        e0,t,mask=(np.asarray(cache[f'bank_{key}/{name}']) for key in ('e0','t','unit_mask'))
        if array_sha(e0)!=receipt['arrays'][name]['e0_sha256'] or tuple(e0.shape)!=(64,100): raise RuntimeError('bank E0 authority drift')
        roster=np.asarray(provenance[f'nwb_unit_ids_in_rate_column_order/{name}'])
        if roster.shape!=(64,) or array_sha(roster)!=prov_receipt['rows'][name]['nwb_unit_ids_sha256']: raise RuntimeError('physical unit roster authority drift')
        bank=BankBatch(torch.from_numpy(e0).to(device).unsqueeze(0),torch.from_numpy(t).to(device).unsqueeze(0),torch.from_numpy(mask).to(device).unsqueeze(0), (name,), (tuple(roster.tolist()),))
        first=int(starts[0]); stream=HeterogeneousCurrentQueryStream(model,bank); stream.refresh_state(history=torch.from_numpy(raw[None,first:first+100]).to(device)); cursor=first+99; maximum=0.; consumed=0
        for ix,start in zip(ids,starts,strict=True):
          endpoint=int(start)+99
          if cursor==endpoint:
            # The initial padded W=100 state has no just-observed public bin;
            # take an explicit detached/copy native value exactly once.
            value=stream.current_prediction().detach().cpu().numpy()[0].astype(np.float32,copy=True)
          else:
            while cursor<endpoint:
              cursor+=1; value=stream.predict(np.ascontiguousarray(raw[cursor:cursor+1])); consumed+=1
            value=value[0]
          ref=offline[ix]; err=np.abs(value-ref); maximum=max(maximum,float(err.max()))
          if not np.all(err<=1e-5+1e-5*np.abs(ref)): raise AssertionError(f'parity {name}:{endpoint} {err.max()}')
          prediction[ix]=value
        delta=r2(prediction[ids],export['target'][ids])-r2(offline[ids],export['target'][ids])
        if abs(delta)>1e-5: raise AssertionError(f'R2 {name} {delta}')
        checks[name]={'n':len(ids),'max_abs_error':maximum,'r2_delta':delta,'raw_bins_after_initial_window':consumed,'physical_unit_ids_sha256':array_sha(roster)}
    export['prediction']=prediction; np.savez_compressed(output.with_suffix('.npz'),**export)
    streamed_audit=audit_m1(output.with_suffix('.npz'),checkpoint=Path(row['checkpoint']))
    if len(prediction)!=31252: raise AssertionError('complete 31252 source endpoint requirement')
    report={'schema':'m1_runtime_v3_complete_selected_stream_v1','status':'PASS','device':device,'n':len(prediction),'selected_ema_state_sha256':row['plain_ema_model_state_sha256'],'reference_export_sha256':row['prediction_export_sha256'],'runtime_cache_sha256':receipt['npz_sha256'],'unit_provenance_sha256':prov_receipt['supplement_npz_sha256'],'runtime_code_sha256':sha(Path(__file__).with_name('runtime.py')),'checks':checks,'max_abs_error':float(np.abs(prediction-offline).max()),'independent_reference_audit':reference_audit,'independent_stream_audit':streamed_audit,'stream_export':str(output.with_suffix('.npz')),'stream_export_sha256':sha(output.with_suffix('.npz'))}
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1');a=p.parse_args();print(json.dumps(run(a.output,a.device),indent=2))
