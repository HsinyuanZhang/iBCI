"""Prepared (not self-launching) fixed canonical MAT7 M3 fitter for V7 arms."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v2.score import _windows,CHUNK
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from tfpd_exploration.h1_series_20260830.src.h1_m3_readout_calibration_v1.core import fit,apply
from tfpd_exploration.src.h1_optimized_v5.v4full24_canonical_m3_readout import sha,ahash,r2,support_rows,CANON
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from .model import make_matched_pair
from .ema import V7EMA

ATTEMPT=ROOT/'v7_dropout30_paired12_v1';EXPORT=ATTEMPT/'independent_score_export';FAMILY='MAT7';RIDGE=0.;SCALE_FLOOR=1e-6
def hashes():
 here=Path(__file__).resolve();src=here.parents[1];files={'v7_fitter':here,'v7_model':here.with_name('model.py'),'v7_ema':here.with_name('ema.py'),'v7_train':here.with_name('paired_train.py'),'v7_export':here.with_name('export_predictions.py'),'v6_model':src/'h1_optimized_v6/model.py','v4_model':src/'h1_optimized_v4/model.py','m3_core':Path('tfpd_exploration/h1_series_20260830/src/h1_m3_readout_calibration_v1/core.py'),'cache':src/'h1_optimized_v2/cache.py'};return {k:sha(v) for k,v in files.items()}
def predict(model,cache,support,d):
 out={};model.eval()
 with torch.inference_mode():
  for n,item in support.items():
   bank=H1Bank(*[cache['train'][n]['bank'][q].to(d) for q in ('E0','T','unit_mask')]);parts=[]
   for off in range(0,len(item['ends']),CHUNK):
    x=torch.as_tensor(_windows(item['rec'].neural,item['ends'][off:off+CHUNK]),device=d);parts.append((model.forward_last(x,bank)/20).cpu().numpy().astype(np.float32,copy=False))
   out[n]=np.concatenate(parts)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=('full_v4','t_v6'),required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();arm=a.arm;out=ATTEMPT/f'v7_canonical_m3_mat7_{arm}_v1'
 if out.exists():raise FileExistsError(out)
 need=[ATTEMPT/'input_authority.json',ATTEMPT/'selection_protocol_freeze.json',ATTEMPT/'selection_freeze.json',ATTEMPT/'final.json',EXPORT/'manifest.json',CANON]
 if any(not x.is_file() for x in need):raise FileNotFoundError('sealed V7 inputs missing')
 manifest=json.loads((EXPORT/'manifest.json').read_text())
 if manifest.get('schema')!='h1_v7_dropout30_independent_native_score_export_v1' or set(manifest.get('arms',{}))!={'full_v4','t_v6'}:raise RuntimeError('V7 export manifest schema/arm roster drift')
 rec=manifest['arms'][arm]['epoch12'];endpoint=ATTEMPT/f'{arm}_epoch_012.pt';freeze=json.loads((ATTEMPT/'selection_freeze.json').read_text())
 expected_operator='v4_causal_full_window_no_query_temporal_contract' if arm=='full_v4' else 'v6_recency_query_temporal_contract4'
 if rec.get('operator_contract')!=expected_operator or rec.get('frontend_contract_version')!=4:raise RuntimeError('V7 export arm metadata drift')
 if arm=='full_v4' and ('temporal_contract_version' in rec or 'immutable_temporal_buffer_names' in rec):raise RuntimeError('FULL export carries query-only temporal tags')
 if arm=='t_v6' and (rec.get('temporal_contract_version')!=4 or not rec.get('immutable_temporal_buffer_names')):raise RuntimeError('T export temporal tags absent')
 if Path(freeze['selected'][arm]['checkpoint'])!=endpoint or freeze['selected'][arm]['epoch']!=12 or Path(rec['checkpoint'])!=endpoint:raise RuntimeError('fixed V7 M3 requires selected e12 only')
 protocol={'schema':'v7_canonical_m3_mat7_protocol_freeze_v1','frozen_before_fit':True,'arm':arm,'model':'sealed V7 selected/e12 only','family':FAMILY,'ridge':RIDGE,'scale_floor':SCALE_FLOOR,'support':'exact canonical held-in raw legal first3 TrialNum/eval-mask bins','prediction':'V7 EMA native /20 causal W700 zero startup','prohibitions':['no tuning','no epoch/model selection','no new training','no cross-arm coefficients']};out.mkdir();(out/'protocol_freeze.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
 for name,digest in manifest['committed_input_bindings'].items():
  path=(ROOT/name) if name=='source_cache_authority.json' else ATTEMPT/name
  if sha(path)!=digest:raise RuntimeError('committed input drift '+name)
 if sha(endpoint)!=rec['checkpoint_sha256'] or sha(Path(rec['ema_metadata']))!=rec['ema_metadata_sha256'] or sha(Path(rec['plain_ema_model_state']))!=rec['plain_ema_model_state_sha256']:raise RuntimeError('export binding drift')
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);support=support_rows(cache)
 if len(support)!=13 or set(support)!=set(cache['train']) or sum(len(x['ends']) for x in support.values())!=30879:raise RuntimeError('canonical support must be exactly 13 sessions / 30879 bins before fit')
 d=torch.device(a.device);full,t=make_matched_pair();model={'full_v4':full,'t_v6':t}[arm].to(d);state=torch.load(endpoint,map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
 if int(model.frontend_contract_version)!=4:raise RuntimeError('frontend contract drift')
 if arm=='t_v6':
  if state.get('operator_contract')!='v6_recency_query_temporal_contract4' or int(model.temporal.temporal_contract_version)!=4:raise RuntimeError('query operator contract drift')
  ema=V7EMA(model,.9995)
 else:
  if state.get('operator_contract')!='v4_causal_full_window_no_query_temporal_contract' or 'temporal_contract_version' in state:raise RuntimeError('full operator contract drift')
  ema=DecoderEMA(model,.9995)
 ema.load_checkpoint_state(state['ema']);plain=torch.load(rec['plain_ema_model_state'],map_location='cpu',weights_only=True);fitted={}
 def run(view):
  actual=view.state_dict()
  if actual.keys()!=plain.keys() or any(not torch.equal(actual[k].detach().cpu(),plain[k]) for k in actual):raise RuntimeError('EMA/plain mismatch')
  ps=predict(view,cache,support,d)
  for n,item in support.items():
   y=np.ascontiguousarray(item['rec'].velocity[item['ends']],np.float32);trial=np.ascontiguousarray(item['rec'].trial_num[item['ends']],np.float64);path=out/f'{n}_canonical_m3_support.npz';np.savez_compressed(path,endpoint=item['ends'].astype(np.int64),trial_id=trial,prediction_native_velocity=ps[n],target_native_velocity=y);saved=np.load(path);mapping=fit(saved['prediction_native_velocity'],saved['target_native_velocity'],family=FAMILY,ridge=RIDGE)
   if any(getattr(mapping,key).dtype!=np.dtype('float64') for key in ('p_mean','p_scale','y_mean','y_scale','weight','intercept')):raise RuntimeError('canonical map must be literal float64')
   fitted[n]={'map':mapping,'p':saved['prediction_native_velocity'],'y':saved['target_native_velocity'],'trial':saved['trial_id'],'path':path}
 ema.score_with_ema(model,run);maps=out/'per_session_maps.pt';torch.save({'schema':'v7_canonical_m3_mat7_maps_v1','arm':arm,'family':FAMILY,'ridge':RIDGE,'maps':{n:{'family':x['map'].family,'ridge':x['map'].ridge,'p_mean':x['map'].p_mean,'p_scale':x['map'].p_scale,'y_mean':x['map'].y_mean,'y_scale':x['map'].y_scale,'weight':x['map'].weight,'intercept':x['map'].intercept} for n,x in fitted.items()}},maps)
 rows={n:{'first3':list(support[n]['first3']),'raw_nwb_sha256':support[n]['raw_nwb_sha256'],'n_bins':len(x['y']),'support_npz':str(x['path']),'support_npz_sha256':sha(x['path']),'endpoint_sha256':ahash(support[n]['ends'].astype(np.int64)),'trial_sha256':ahash(x['trial']),'prediction_sha256':ahash(x['p']),'target_sha256':ahash(x['y'])} for n,x in fitted.items()}
 receipt={'schema':'v7_canonical_m3_mat7_readout_v1','status':'FITTED_FIXED_CONTRACT','arm':arm,'operator_contract':state['operator_contract'],'protocol_sha256':sha(out/'protocol_freeze.json'),'checkpoint_sha256':sha(endpoint),'ema_metadata_sha256':sha(Path(rec['ema_metadata'])),'plain_ema_sha256':sha(Path(rec['plain_ema_model_state'])),'canonical_authority_sha256':sha(CANON),'source_cache_authority':auth,'code_sha256':hashes(),'family':FAMILY,'ridge':RIDGE,'scale_floor':SCALE_FLOOR,'fit_dtype':'float64','map_array_dtypes':{key:'float64' for key in ('p_mean','p_scale','y_mean','y_scale','weight','intercept')},'support_total_bins':sum(x['n_bins'] for x in rows.values()),'sessions':rows,'maps':str(maps),'maps_sha256':sha(maps),'applied_exports':{}}
 for label in ('selected','epoch12'):
  for surface in ('selection','complete'):
   source=Path(manifest['arms'][arm][label]['surfaces'][surface]['npz']);expect=manifest['arms'][arm][label]['surfaces'][surface]['sha256']
   if sha(source)!=expect:raise RuntimeError('input NPZ drift')
   data=np.load(source);raw=np.asarray(data['prediction_native_velocity']);transformed=np.empty_like(raw);sessions=np.asarray(data['session_id'])
   for n,x in fitted.items():transformed[sessions==n]=apply(x['map'],raw[sessions==n])
   if set(np.unique(sessions))!=set(fitted) or not np.isfinite(transformed).all():raise RuntimeError('coverage/nonfinite')
   body={k:data[k] for k in data.files};body['prediction_native_velocity']=transformed;path=out/f'{arm}_{label}_{surface}_m3mat7_native.npz';np.savez_compressed(path,**body);receipt['applied_exports'][f'{label}_{surface}']={'source_npz':str(source),'source_sha256':sha(source),'npz':str(path),'sha256':sha(path),'n_bins':len(transformed),'session_count':len(np.unique(sessions)),'preserved_fields':sorted(body),'r2_concat_float64_descriptive':r2(transformed,data['target_native_velocity'])}
 target=out/'receipt.json';target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'receipt':str(target),'sha256':sha(target)}))
if __name__=='__main__':main()
