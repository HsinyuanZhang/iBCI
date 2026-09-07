"""Isolated, fixed-contract M3 MAT7 readout for sealed V4-FULL24 only.

No data is accessed on import.  The entry point fits exactly one map per
held-in session from canonical legal M3 support bins; it never uses C2 map
coefficients and never selects a model, epoch, family, or ridge after fitting.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v2.score import _windows,CHUNK
from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.data import _index_split,load_session_arrays
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from tfpd_exploration.h1_series_20260830.src.h1_m3_readout_calibration_v1.core import fit,apply

ATTEMPT=ROOT/'v4_full_continue24_deterministic_v1';EXPORT=ATTEMPT/'independent_score_export';OUT=ATTEMPT/'canonical_m3_mat7_readout_v1';CANON=Path('tfpd_exploration/h1_series_20260830/results/h1_m3_readout_calibration_evalai_package_v1/calibration_authority.json');RIDGE=0.;FAMILY='MAT7';SCALE_FLOOR=1e-6
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ahash(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.tobytes());return h.hexdigest()
def r2(p,y):
 p,y=p.astype(np.float64),y.astype(np.float64);return float(1-np.square(p-y).sum()/np.square(y-y.mean(0,keepdims=True)).sum())
def code_hashes():
 here=Path(__file__).resolve();src=here.parents[1];files={'fitter':here,'v4_model':src/'h1_optimized_v4/model.py','v4_continue':here.with_name('v4_full_continue24.py'),'m3_core':Path('tfpd_exploration/h1_series_20260830/src/h1_m3_readout_calibration_v1/core.py'),'m3_plan':Path('tfpd_exploration/h1_series_20260830/src/h1_m3_readout_calibration_v1/plan.py'),'source_data':src/'h1_temporal_decoder_quick_product_v1/data.py','cache':src/'h1_optimized_v2/cache.py'}
 if any(not p.is_file() for p in files.values()):raise FileNotFoundError('operator source missing')
 return {k:sha(v) for k,v in files.items()}
def support_rows(cache):
 canonical=json.loads(CANON.read_text());official={x['session']:x for x in canonical['sessions'] if x['scope']=='held-in-calib'}
 if len(official)!=13:raise RuntimeError('canonical held-in roster drift')
 paths=_index_split('held-in-calib');out={}
 for session,row in cache['train'].items():
  rec=load_session_arrays(paths[session],session,skip_first3=True);c=official.get(session);actual_raw_sha=file_sha(paths[session])
  if c is None or row['sha256']!=c['nwb_sha256'] or actual_raw_sha!=c['nwb_sha256']:raise RuntimeError(f'{session}: canonical source hash drift')
  values=tuple(float(x) for x in row['first3']);expected=tuple(float(x) for x in c['calibration_trials'])
  if values!=tuple(float(x) for x in rec.first3) or values!=expected:raise RuntimeError(f'{session}: M3 identity drift')
  ends=np.flatnonzero(rec.eval_mask & np.isin(rec.trial_num,np.asarray(values)))
  if len(ends)!=int(c['calibration_bins']) or len(ends)<8:raise RuntimeError(f'{session}: legal M3 support cardinality drift')
  out[session]={'rec':rec,'ends':ends,'canonical':c,'first3':values,'raw_nwb_sha256':actual_raw_sha}
 return out
def predict_support(model,cache,support,device):
 result={};model.eval()
 with torch.inference_mode():
  for session,item in support.items():
   row=cache['train'][session];bank=H1Bank(*[row['bank'][k].to(device) for k in ('E0','T','unit_mask')]);ends=item['ends'];parts=[]
   # This is the exact finite causal W700 operator at every legal M3 bin;
   # _windows supplies the model's defined zero startup history before bin 699.
   for off in range(0,len(ends),CHUNK):
    x=torch.as_tensor(_windows(item['rec'].neural,ends[off:off+CHUNK]),device=device);parts.append((model.forward_last(x,bank)/20).cpu().numpy().astype(np.float32,copy=False))
   result[session]=np.concatenate(parts)
 return result
def map_payload(mapping):return {'family':mapping.family,'ridge':mapping.ridge,'p_mean':mapping.p_mean,'p_scale':mapping.p_scale,'y_mean':mapping.y_mean,'y_scale':mapping.y_scale,'weight':mapping.weight,'intercept':mapping.intercept}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 need=[ATTEMPT/'input_authority.json',ATTEMPT/'protocol_freeze.json',ATTEMPT/'selection_freeze.json',ATTEMPT/'final.json',EXPORT/'manifest.json',CANON]
 if any(not p.is_file() for p in need):raise FileNotFoundError('sealed inputs/exports/canonical authority required')
 freeze=json.loads((ATTEMPT/'selection_freeze.json').read_text());selected=Path(freeze['selected']['checkpoint']);endpoint=ATTEMPT/'full_epoch_024.pt'
 if selected!=endpoint or freeze['selected']['epoch']!=24:raise RuntimeError('this fixed fitter is authorized only for frozen V4 FULL24')
 manifest=json.loads((EXPORT/'manifest.json').read_text());record=manifest['arms']['full']['epoch24']
 if Path(record['checkpoint'])!=endpoint:raise RuntimeError('export/checkpoint binding drift')
 # Commit the fixed law before cache/NWB access, forward prediction, fitting,
 # transformed archive construction, or descriptive score calculation.
 protocol={'schema':'v4full24_canonical_m3_mat7_protocol_freeze_v1','frozen_before_fit':True,'model':'sealed V4 FULL continuation selected epoch24 only','family':FAMILY,'ridge':RIDGE,'scale_floor':SCALE_FLOOR,'fit_dtype':'float64','support':'canonical held-in-calib exact first-three legal TrialNum/eval-mask rows','prediction':'frozen EMA native /20 causal W700 with zero startup padding','prohibitions':['no C2 coefficients','no hyperparameter tuning','no model/epoch selection after fit','no support-prefix sampling invention']}
 OUT.mkdir();(OUT/'protocol_freeze.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
 # Verify every manifest binding against its committed file before consuming it.
 for name,digest in manifest['committed_input_bindings'].items():
  path=(ROOT/name) if name=='source_cache_authority.json' else ATTEMPT/name
  if sha(path)!=digest:raise RuntimeError(f'committed binding drift: {name}')
 if sha(endpoint)!=record['checkpoint_sha256'] or sha(Path(record['ema_metadata']))!=record['ema_metadata_sha256']:raise RuntimeError('checkpoint/EMA metadata export binding drift')
 if sha(Path(record['plain_ema_model_state']))!=record['plain_ema_model_state_sha256']:raise RuntimeError('plain EMA export binding drift')
 cache=build_or_load();authority=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,authority);support=support_rows(cache)
 device=torch.device('cuda:0');model=make_matched_pair()[0].to(device);payload=torch.load(endpoint,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'],strict=True)
 if int(model.frontend_contract_version.item())!=4:raise RuntimeError('frontend contract drift')
 ema=DecoderEMA(model,decay=float(payload['ema']['decay']));ema.load_checkpoint_state(payload['ema']);plain=torch.load(record['plain_ema_model_state'],map_location='cpu',weights_only=True)
 fitted={}
 def run(view):
  actual=view.state_dict()
  if actual.keys()!=plain.keys() or any(not torch.equal(actual[k].detach().cpu(),plain[k]) for k in actual):raise RuntimeError('EMA-applied model differs from frozen plain EMA export')
  predictions=predict_support(view,cache,support,device)
  for session,item in support.items():
   target=np.ascontiguousarray(item['rec'].velocity[item['ends']],dtype=np.float32);trial=np.ascontiguousarray(item['rec'].trial_num[item['ends']],dtype=np.float64);path=OUT/f'{session}_canonical_m3_support.npz';np.savez_compressed(path,endpoint=item['ends'].astype(np.int64),trial_id=trial,prediction_native_velocity=predictions[session],target_native_velocity=target)
   saved=np.load(path);mapping=fit(saved['prediction_native_velocity'],saved['target_native_velocity'],family=FAMILY,ridge=RIDGE);fitted[session]={'mapping':mapping,'prediction':saved['prediction_native_velocity'],'target':saved['target_native_velocity'],'trial_id':saved['trial_id'],'support_npz':path}
 ema.score_with_ema(model,run)
 rows={}
 for session,item in fitted.items():
  m=item['mapping'];rows[session]={'first3':list(support[session]['first3']),'raw_nwb_sha256':support[session]['raw_nwb_sha256'],'support_n_bins':int(len(item['target'])),'support_npz':str(item['support_npz']),'support_npz_sha256':sha(item['support_npz']),'support_endpoint_sha256':ahash(support[session]['ends'].astype(np.int64)),'support_trial_id_sha256':ahash(item['trial_id']),'support_prediction_native_sha256':ahash(item['prediction']),'support_target_native_sha256':ahash(item['target']),'map':map_payload(m),'map_array_sha256':{name:ahash(getattr(m,name)) for name in ('p_mean','p_scale','y_mean','y_scale','weight','intercept')}}
 maps=OUT/'per_session_maps.pt';torch.save({'schema':'v4full24_canonical_m3_mat7_maps_v1','family':FAMILY,'ridge':RIDGE,'scale_floor':SCALE_FLOOR,'maps':{s:map_payload(x['mapping']) for s,x in fitted.items()}},maps)
 receipt={'schema':'v4full24_canonical_m3_mat7_readout_v1','status':'FITTED_FIXED_CONTRACT','protocol_freeze_sha256':sha(OUT/'protocol_freeze.json'),'disclosure':'model-specific frozen V4-FULL24 readout fitted once from canonical legal M3 labels; no C2 coefficients and no hyperparameter/model/epoch selection','family':FAMILY,'ridge':RIDGE,'scale_floor':SCALE_FLOOR,'fit_dtype':'float64','checkpoint':str(endpoint),'checkpoint_sha256':sha(endpoint),'checkpoint_ema_metadata_sha256':sha(Path(record['ema_metadata'])),'frozen_plain_ema_state_sha256':sha(Path(record['plain_ema_model_state'])),'frozen_selection_sha256':sha(ATTEMPT/'selection_freeze.json'),'canonical_authority_sha256':sha(CANON),'source_cache_authority':authority,'source_cache_authority_sha256':sha(ROOT/'source_cache_authority.json'),'code_sha256':code_hashes(),'maps':str(maps),'maps_sha256':sha(maps),'sessions':rows,'support_total_bins':int(sum(x['support_n_bins'] for x in rows.values())),'applied_exports':{}}
 # Transform pre-existing exports only. Labels are preserved byte-for-byte;
 # scores are descriptive fixed-map replay metrics, never a selection signal.
 for label in ('selected','epoch24'):
  for surface in ('selection','complete'):
   source=Path(manifest['arms']['full'][label]['surfaces'][surface]['npz']);expected=manifest['arms']['full'][label]['surfaces'][surface]['sha256']
   if sha(source)!=expected:raise RuntimeError(f'export archive binding drift: {label}/{surface}')
   data=np.load(source);p=np.asarray(data['prediction_native_velocity']);outp=np.empty_like(p);sessions=np.asarray(data['session_id'])
   for session,item in fitted.items():outp[sessions==session]=apply(item['mapping'],p[sessions==session])
   if set(np.unique(sessions))!=set(fitted) or not np.isfinite(outp).all():raise RuntimeError(f'coverage/nonfinite drift: {label}/{surface}')
   transformed={key:data[key] for key in data.files};transformed['prediction_native_velocity']=outp;archive=OUT/f'full_{label}_{surface}_m3mat7_native.npz';np.savez_compressed(archive,**transformed);receipt['applied_exports'][f'{label}_{surface}']={'source_npz':str(source),'source_sha256':sha(source),'npz':str(archive),'sha256':sha(archive),'n_bins':int(len(outp)),'session_count':int(len(np.unique(sessions))),'preserved_source_fields':sorted(transformed),'r2_concat_float64_descriptive':r2(outp,data['target_native_velocity'])}
 target=OUT/'receipt.json';target.write_text(json.dumps(receipt,indent=2,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x)+'\n');print(json.dumps({'receipt':str(target),'sha256':sha(target)},sort_keys=True))
if __name__=='__main__':main()
