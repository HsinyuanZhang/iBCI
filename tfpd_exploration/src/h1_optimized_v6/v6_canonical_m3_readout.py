"""Fixed canonical M3 MAT7 lambda-zero readout for sealed V6 e12 only."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,torch
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v2.score import _windows,CHUNK
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from tfpd_exploration.h1_series_20260830.src.h1_m3_readout_calibration_v1.core import fit,apply
from tfpd_exploration.src.h1_optimized_v5.v4full24_canonical_m3_readout import sha,ahash,r2,support_rows,CANON
from .model import make_matched_pair
from .ema import V6EMA
ATTEMPT=ROOT/'paired_v6_recency_queryonly_12ep_v1';EXPORT=ATTEMPT/'independent_score_export';OUT=ATTEMPT/'v6_canonical_m3_mat7_readout_v1';FAMILY='MAT7';RIDGE=0.;SCALE_FLOOR=1e-6
def code_hashes():
 here=Path(__file__).resolve();src=here.parents[1];files={'v6_fitter':here,'v6_model':here.with_name('model.py'),'v6_ema':here.with_name('ema.py'),'v6_train':here.with_name('paired_train.py'),'v6_export':here.with_name('export_predictions.py'),'m3_core':Path('tfpd_exploration/h1_series_20260830/src/h1_m3_readout_calibration_v1/core.py'),'source_data':src/'h1_temporal_decoder_quick_product_v1/data.py','cache':src/'h1_optimized_v2/cache.py'};return {k:sha(v) for k,v in files.items()}
def predict(model,cache,support,d):
 out={};model.eval()
 with torch.inference_mode():
  for n,item in support.items():
   b=H1Bank(*[cache['train'][n]['bank'][q].to(d) for q in ('E0','T','unit_mask')]);parts=[]
   for off in range(0,len(item['ends']),CHUNK):
    x=torch.as_tensor(_windows(item['rec'].neural,item['ends'][off:off+CHUNK]),device=d);parts.append((model.forward_last(x,b)/20).cpu().numpy().astype(np.float32,copy=False))
   out[n]=np.concatenate(parts)
 return out
def main():
 if OUT.exists():raise FileExistsError(OUT)
 need=[ATTEMPT/'input_authority.json',ATTEMPT/'selection_protocol_freeze.json',ATTEMPT/'selection_freeze.json',ATTEMPT/'final.json',EXPORT/'manifest.json',CANON]
 if any(not x.is_file() for x in need):raise FileNotFoundError('sealed V6 inputs missing')
 freeze=json.loads((ATTEMPT/'selection_freeze.json').read_text());endpoint=ATTEMPT/'t_v6_epoch_012.pt'
 if Path(freeze['selected']['checkpoint'])!=endpoint or freeze['selected']['epoch']!=12:raise RuntimeError('fixed fitter requires frozen V6 e12')
 manifest=json.loads((EXPORT/'manifest.json').read_text());rec=manifest['arms']['t']['epoch12'];
 if Path(rec['checkpoint'])!=endpoint:raise RuntimeError('V6 export checkpoint drift')
 protocol={'schema':'v6_canonical_m3_mat7_protocol_freeze_v1','frozen_before_fit':True,'model':'sealed V6 selected e12 only','family':'MAT7','ridge':0.,'scale_floor':1e-6,'support':'exact canonical held-in raw legal first3 TrialNum/eval-mask bins','prediction':'V6 EMA native /20 causal W700 zero startup','prohibitions':['no C2/V4 coefficients','no tuning','no epoch/model selection','no new training']};OUT.mkdir();(OUT/'protocol_freeze.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n')
 for name,digest in manifest['committed_input_bindings'].items():
  path=(ROOT/name) if name=='source_cache_authority.json' else ATTEMPT/name
  if sha(path)!=digest:raise RuntimeError('committed input drift '+name)
 if sha(endpoint)!=rec['checkpoint_sha256'] or sha(Path(rec['ema_metadata']))!=rec['ema_metadata_sha256'] or sha(Path(rec['plain_ema_model_state']))!=rec['plain_ema_model_state_sha256']:raise RuntimeError('export binding drift')
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);support=support_rows(cache);d=torch.device('cuda:0');m=make_matched_pair()[1].to(d);state=torch.load(endpoint,map_location='cpu',weights_only=False);m.load_state_dict(state['model'],strict=True)
 if int(m.frontend_contract_version)!=4 or int(m.temporal.temporal_contract_version)!=4:raise RuntimeError('V6 contract drift')
 ema=V6EMA(m,.9995);ema.load_checkpoint_state(state['ema']);plain=torch.load(rec['plain_ema_model_state'],map_location='cpu',weights_only=True);fitted={}
 def run(view):
  actual=view.state_dict()
  if actual.keys()!=plain.keys() or any(not torch.equal(actual[k].detach().cpu(),plain[k]) for k in actual):raise RuntimeError('EMA/plain mismatch')
  ps=predict(view,cache,support,d)
  for n,item in support.items():
   y=np.ascontiguousarray(item['rec'].velocity[item['ends']],np.float32);trial=np.ascontiguousarray(item['rec'].trial_num[item['ends']],np.float64);path=OUT/f'{n}_canonical_m3_support.npz';np.savez_compressed(path,endpoint=item['ends'].astype(np.int64),trial_id=trial,prediction_native_velocity=ps[n],target_native_velocity=y);saved=np.load(path);fitted[n]={'map':fit(saved['prediction_native_velocity'],saved['target_native_velocity'],family=FAMILY,ridge=RIDGE),'p':saved['prediction_native_velocity'],'y':saved['target_native_velocity'],'trial':saved['trial_id'],'path':path}
 ema.score_with_ema(m,run);maps=OUT/'per_session_maps.pt';torch.save({'schema':'v6_canonical_m3_mat7_maps_v1','family':FAMILY,'ridge':RIDGE,'maps':{n:{'family':x['map'].family,'ridge':x['map'].ridge,'p_mean':x['map'].p_mean,'p_scale':x['map'].p_scale,'y_mean':x['map'].y_mean,'y_scale':x['map'].y_scale,'weight':x['map'].weight,'intercept':x['map'].intercept} for n,x in fitted.items()}},maps)
 rows={n:{'first3':list(support[n]['first3']),'raw_nwb_sha256':support[n]['raw_nwb_sha256'],'n_bins':len(x['y']),'support_npz':str(x['path']),'support_npz_sha256':sha(x['path']),'endpoint_sha256':ahash(support[n]['ends'].astype(np.int64)),'trial_sha256':ahash(x['trial']),'prediction_sha256':ahash(x['p']),'target_sha256':ahash(x['y'])} for n,x in fitted.items()}
 receipt={'schema':'v6_canonical_m3_mat7_readout_v1','status':'FITTED_FIXED_CONTRACT','protocol_sha256':sha(OUT/'protocol_freeze.json'),'checkpoint_sha256':sha(endpoint),'ema_metadata_sha256':sha(Path(rec['ema_metadata'])),'plain_ema_sha256':sha(Path(rec['plain_ema_model_state'])),'canonical_authority_sha256':sha(CANON),'source_cache_authority':auth,'code_sha256':code_hashes(),'family':FAMILY,'ridge':RIDGE,'scale_floor':SCALE_FLOOR,'support_total_bins':sum(x['n_bins'] for x in rows.values()),'sessions':rows,'maps':str(maps),'maps_sha256':sha(maps),'applied_exports':{}}
 for label in ('selected','epoch12'):
  for surface in ('selection','complete'):
   source=Path(manifest['arms']['t'][label]['surfaces'][surface]['npz']);expect=manifest['arms']['t'][label]['surfaces'][surface]['sha256']
   if sha(source)!=expect:raise RuntimeError('input NPZ drift')
   data=np.load(source);p=np.asarray(data['prediction_native_velocity']);out=np.empty_like(p);sessions=np.asarray(data['session_id'])
   for n,x in fitted.items():out[sessions==n]=apply(x['map'],p[sessions==n])
   if set(np.unique(sessions))!=set(fitted) or not np.isfinite(out).all():raise RuntimeError('coverage/nonfinite')
   transformed={k:data[k] for k in data.files};transformed['prediction_native_velocity']=out;path=OUT/f't_{label}_{surface}_m3mat7_native.npz';np.savez_compressed(path,**transformed);receipt['applied_exports'][f'{label}_{surface}']={'source_npz':str(source),'source_sha256':sha(source),'npz':str(path),'sha256':sha(path),'n_bins':len(out),'session_count':len(np.unique(sessions)),'preserved_fields':sorted(transformed),'r2_concat_float64_descriptive':r2(out,data['target_native_velocity'])}
 target=OUT/'receipt.json';target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'receipt':str(target),'sha256':sha(target)}))
if __name__=='__main__':main()
